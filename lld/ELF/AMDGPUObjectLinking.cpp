//===- AMDGPUObjectLinking.cpp - AMDGPU link-time resolution --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements link-time resolution and patching for AMDGPU object linking.
//
// The linker:
//   1. Uses the linked image's resolved target e_flags
//   2. Collects SHN_AMDGPU_LDS and named-barrier symbols
//   3. Parses .amdgpu.info to build the cross-TU call graph, type-ID
//      signatures, LDS and named-barrier uses, and per-function resource usage
//   4. Resolves indirect call edges, function aliases, and kernel entries
//   5. Validates call-edge wave-size compatibility
//   6. Builds a shared SCC condensation graph for kernel-reachable functions
//   7. Computes per-kernel LDS and named-barrier reachability
//   8. Assigns LDS offsets and named-barrier IDs
//   9. Propagates resource usage across the SCC graph (MAX for registers, OR
//      for flags, caller scratch plus maximum callee scratch path)
//  10. Validates required ABI occupancy and wave-size metadata, call-edge ABI
//      compatibility, and each kernel's LDS usage against its occupancy
//
// After resolution, the linker patches kernel descriptors and HSA metadata
// with the resolved LDS size, named-barrier count, propagated register usage,
// scratch size, dynamic-stack flag, and related resource fields.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUObjectLinking.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/AMDGPUObjLinkingInfo.h"
#include "llvm/Support/AMDGPUObjectLinkingHelper.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#define DEBUG_TYPE "amdgpu-object-linking"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {

// One SHN_AMDGPU_LDS symbol after collection and before it is rewritten to a
// concrete offset.
struct LDSSymbolInfo {
  Symbol *sym = nullptr;
  uint64_t size = 0;
  Align alignment = Align(1);
  uint64_t assignedOffset = 0;
};

// One named-barrier pseudo-symbol. The linker assigns each barrier a hardware
// ID range and rewrites the symbol to the encoded barrier address.
struct NamedBarrierInfo {
  Symbol *sym = nullptr;
  uint32_t slotCount = 0;
  uint32_t assignedBarId = 0;
};

// Per-function resource usage parsed from .amdgpu.info, extended with
// linker-only validation fields.
struct CGResourceInfo : AMDGPU::RawLinkingResources {
  uint32_t occupancy = 0;
  uint32_t waveSize = 0;
  bool isCuMode = true;
};

// One function in the device call graph, plus the data computed for that
// function when it is a kernel entry.
struct CGNode {
  Symbol *sym = nullptr;
  bool isKernel = false;

  SmallVector<CGNode *, 4> callees;
  SmallVector<size_t, 2> ldsUseIndices;
  SmallVector<size_t, 2> barrierUseIndices;

  CGResourceInfo localRes;
  bool hasLocalRes = false;

  CGResourceInfo propagatedRes;
  bool hasPropagatedRes = false;

  DenseSet<size_t> reachableLDS;
  DenseSet<size_t> reachableBarriers;
  uint32_t ldsSize = 0;
  uint32_t numNamedBarrier = 0;
};

// Functions and indirect callers that share the same type-id encoding.
struct SignatureInfo {
  SmallVector<CGNode *, 4> functions;
  SmallVector<CGNode *, 4> indirectCallers;
};

// One strongly connected component in the kernel-reachable call graph. The SCC
// carries direct and propagated reachability/resource data so LDS,
// named-barrier, and resource passes can share the same condensed graph.
struct CallGraphSCC {
  SmallVector<CGNode *, 4> members;
  SmallVector<unsigned, 4> callees;

  DenseSet<size_t> reachableLDS;
  DenseSet<size_t> reachableBarriers;

  CGResourceInfo localRes;
  CGResourceInfo propagatedRes;

  bool hasPropagatedRes = false;
  bool isRecursive = false;
};

// Cross-object device call graph built from .amdgpu.info records. The graph
// keeps kernel order stable for diagnostics and later per-kernel patching.
// After construction, the graph can be condensed into an SCC DAG for
// reachability and resource propagation.
class AMDGPUCallGraph {
  SpecificBumpPtrAllocator<CGNode> alloc;
  DenseMap<Symbol *, CGNode *> symToNode;
  SmallVector<CGNode *> kernelNodes;
  bool hasLDSUseEntries = false;
  bool hasBarrierUseEntries = false;

  DenseSet<CGNode *> addressTakenNodes;
  StringMap<SignatureInfo> signatureMap;

  // SCC condensation state (populated by buildCondensedGraph).
  SmallVector<CallGraphSCC, 16> sccs;
  DenseMap<CGNode *, unsigned> nodeToSCC;

public:
  CGNode &getOrCreate(Symbol *sym) {
    auto [it, inserted] = symToNode.try_emplace(sym, nullptr);
    if (inserted) {
      it->second = new (alloc.Allocate()) CGNode();
      it->second->sym = sym;
    }
    return *it->second;
  }

  CGNode *lookup(Symbol *sym) const {
    auto it = symToNode.find(sym);
    return it != symToNode.end() ? it->second : nullptr;
  }

  void addKernel(CGNode &node) {
    if (node.isKernel)
      return;
    node.isKernel = true;
    kernelNodes.push_back(&node);
  }

  void markAddressTaken(CGNode *node) { addressTakenNodes.insert(node); }

  void addIndirectCall(CGNode *caller, StringRef encoding) {
    signatureMap[encoding].indirectCallers.push_back(caller);
  }

  void addSignature(CGNode *node, StringRef encoding) {
    signatureMap[encoding].functions.push_back(node);
  }

  // After all sections are parsed, resolve indirect call edges by matching
  // signature encodings: for each indirect call encoding, the potential callees
  // are address-taken functions with the same encoding.
  void buildIndirectEdges() {
    for (auto &[encoding, info] : signatureMap) {
      if (info.indirectCallers.empty())
        continue;

      SmallVector<CGNode *, 4> potentialCallees;
      for (CGNode *func : info.functions) {
        if (addressTakenNodes.count(func))
          potentialCallees.push_back(func);
      }

      if (potentialCallees.empty())
        continue;

      for (CGNode *caller : info.indirectCallers) {
        for (CGNode *callee : potentialCallees) {
          LLVM_DEBUG(dbgs() << "  indirect edge: " << caller->sym->getName()
                            << " -> " << callee->sym->getName()
                            << " (sig=" << encoding << ")\n");
          caller->callees.push_back(callee);
        }
      }
    }
  }

  // Resolve function aliases: multiple ELF symbols can point to the same
  // address (e.g. weak/strong pairs or constructor variants). The compiler
  // emits .amdgpu.info only for one of them. Redirect callee pointers from
  // alias nodes (no local resource info) to their canonical definition so
  // that reachability and resource propagation see real data.
  void resolveAliases() {
    DenseMap<std::pair<SectionBase *, uint64_t>, CGNode *> addrToNode;
    for (auto &[sym, node] : symToNode) {
      if (!node->hasLocalRes)
        continue;
      auto *d = dyn_cast<Defined>(sym);
      if (!d || !d->section)
        continue;
      addrToNode[{d->section, d->value}] = node;
    }

    DenseMap<CGNode *, CGNode *> aliasMap;
    for (auto &[sym, node] : symToNode) {
      if (node->hasLocalRes)
        continue;
      auto *d = dyn_cast<Defined>(sym);
      if (!d || !d->section)
        continue;
      auto it = addrToNode.find({d->section, d->value});
      if (it != addrToNode.end() && it->second != node) {
        LLVM_DEBUG(dbgs() << "  alias: " << sym->getName() << " -> "
                          << it->second->sym->getName() << "\n");
        aliasMap[node] = it->second;
      }
    }

    if (aliasMap.empty())
      return;

    for (auto &[sym, node] : symToNode) {
      for (CGNode *&callee : node->callees) {
        if (auto it = aliasMap.find(callee); it != aliasMap.end())
          callee = it->second;
      }
    }

    for (CGNode *&k : kernelNodes) {
      if (auto it = aliasMap.find(k); it != aliasMap.end())
        k = it->second;
    }
  }

  void setHasLDSUses() { hasLDSUseEntries = true; }
  bool hasLDSUses() const { return hasLDSUseEntries; }

  void setHasBarrierUses() { hasBarrierUseEntries = true; }
  bool hasBarrierUses() const { return hasBarrierUseEntries; }

  ArrayRef<CGNode *> kernels() const { return kernelNodes; }

  using const_iterator = DenseMap<Symbol *, CGNode *>::const_iterator;
  const_iterator begin() const { return symToNode.begin(); }
  const_iterator end() const { return symToNode.end(); }

  // Build the kernel-reachable SCC DAG. Tarjan emits SCCs in reverse
  // topological order so every outgoing edge points to an earlier SCC and
  // consumers can use a single forward sweep.
  bool buildCondensedGraph(Ctx &ctx, bool validateResources);

  // Propagate transitive LDS/barrier reachability over the SCC DAG into each
  // kernel node.
  void computeKernelReachability();

  // Propagate resource usage over the SCC DAG. Since SCCs are in reverse
  // topological order, each callee has already been propagated when its
  // caller is visited.
  void propagateResourceUsage();

private:
  bool buildSCCs(Ctx &ctx, CGNode *node,
                 DenseMap<CGNode *, unsigned> &nodeIndex,
                 DenseMap<CGNode *, unsigned> &lowLink,
                 DenseSet<CGNode *> &onStack, SmallVectorImpl<CGNode *> &stack,
                 unsigned &nextIndex, bool validateResources);

  void buildSCCEdges();
};

// The relocation payload needed to resolve a tagged .amdgpu.info entry.
struct RelocInfo {
  uint32_t symIdx = 0;
  int64_t addend = 0;
};

struct LDSAllocationTraits {
  bool hasUses(const AMDGPUCallGraph &cg) const { return cg.hasLDSUses(); }

  const DenseSet<size_t> &reachableResources(const CGNode &kernel) const {
    return kernel.reachableLDS;
  }

  uint64_t initialValue() const { return 0; }
  uint64_t size(const LDSSymbolInfo &lds) const { return lds.size; }
  Align alignment(const LDSSymbolInfo &lds) const { return lds.alignment; }
  uint64_t assigned(const LDSSymbolInfo &lds) const {
    return lds.assignedOffset;
  }

  void setAssigned(LDSSymbolInfo &lds, uint64_t value) const {
    lds.assignedOffset = value;
  }

  bool compare(const LDSSymbolInfo &a, const LDSSymbolInfo &b) const {
    if (a.alignment != b.alignment)
      return a.alignment > b.alignment;
    if (a.size != b.size)
      return a.size > b.size;
    return a.sym->getName() < b.sym->getName();
  }

  void printAllocation(const LDSSymbolInfo &lds, uint64_t offset,
                       size_t users) const {
    LLVM_DEBUG(dbgs() << "    allocate " << lds.sym->getName() << " at "
                      << offset << " size=" << lds.size << " users=" << users
                      << "\n");
  }
};

struct NamedBarrierAllocationTraits {
  bool hasUses(const AMDGPUCallGraph &cg) const { return cg.hasBarrierUses(); }

  const DenseSet<size_t> &reachableResources(const CGNode &kernel) const {
    return kernel.reachableBarriers;
  }

  uint64_t initialValue() const { return 1; }
  uint64_t size(const NamedBarrierInfo &bar) const { return bar.slotCount; }
  Align alignment(const NamedBarrierInfo &) const { return Align(1); }
  uint64_t assigned(const NamedBarrierInfo &bar) const {
    return bar.assignedBarId;
  }

  void setAssigned(NamedBarrierInfo &bar, uint64_t value) const {
    bar.assignedBarId = static_cast<uint32_t>(value);
  }

  bool compare(const NamedBarrierInfo &a, const NamedBarrierInfo &b) const {
    if (a.slotCount != b.slotCount)
      return a.slotCount > b.slotCount;
    return a.sym->getName() < b.sym->getName();
  }

  void printAllocation(const NamedBarrierInfo &bar, uint64_t barId,
                       size_t users) const {
    LLVM_DEBUG(dbgs() << "    allocate " << bar.sym->getName() << " at "
                      << barId << " slots=" << bar.slotCount
                      << " users=" << users << "\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Resource allocation helpers
//===----------------------------------------------------------------------===//

// Return the placement frontier shared by all kernels in users. Each kernel has
// an independent frontier, and a resource used by several kernels must be
// placed after every one of those kernels' existing live resources.
// initialValue is the resource-specific base frontier, e.g. LDS offsets start
// at 0 while named-barrier IDs start at 1.
static uint64_t
getMaxKernelFrontier(ArrayRef<CGNode *> users,
                     const DenseMap<CGNode *, uint64_t> &kernelFrontiers,
                     uint64_t initialValue) {
  uint64_t frontier = initialValue;
  for (CGNode *user : users) {
    auto it = kernelFrontiers.find(user);
    assert(it != kernelFrontiers.end() && "missing kernel frontier");
    frontier = std::max(frontier, it->second);
  }
  return frontier;
}

// Assign one packed, deterministic layout used by every kernel. This is the
// fallback when the input objects do not provide .amdgpu.info reachability for
// a resource kind. Non-zero-sized resources advance a single global frontier.
// Zero-sized resources are assigned to the first suitably aligned address after
// the fixed layout, matching the dynamic-LDS convention while also giving
// zero-slot barriers a stable value if they ever appear.
template <typename ResourceT, typename Traits>
static void assignUniversalResources(SmallVectorImpl<ResourceT> &resources,
                                     const Traits &traits) {
  llvm::sort(resources, [&](const ResourceT &a, const ResourceT &b) {
    return traits.compare(a, b);
  });
  uint64_t frontier = traits.initialValue();
  for (ResourceT &resource : resources) {
    if (traits.size(resource) == 0)
      continue;
    frontier = alignTo(frontier, traits.alignment(resource));
    traits.setAssigned(resource, frontier);
    frontier += traits.size(resource);
  }
  Align maxZeroAlign(1);
  for (ResourceT &resource : resources) {
    if (traits.size(resource) == 0)
      maxZeroAlign = std::max(maxZeroAlign, traits.alignment(resource));
  }
  uint64_t zeroBase = alignTo(frontier, maxZeroAlign);
  for (ResourceT &resource : resources) {
    if (traits.size(resource) == 0)
      traits.setAssigned(resource, zeroBase);
  }
}

#ifndef NDEBUG
template <typename ResourceT, typename Traits>
static void assertResourceLayoutInvariants(
    ArrayRef<ResourceT> resources, ArrayRef<CGNode *> kernels,
    const DenseMap<size_t, SmallVector<CGNode *, 4>> &resourceToUsers,
    const DenseSet<size_t> &assigned, const Traits &traits) {
  assert(assigned.size() == resources.size() && "not all resources assigned");
  for (size_t idx = 0, e = resources.size(); idx < e; ++idx)
    assert(assigned.count(idx) && "resource not assigned");

  for (const auto &[idx, users] : resourceToUsers) {
    uint64_t start = traits.assigned(resources[idx]);
    for (CGNode *user : users) {
      assert(traits.reachableResources(*user).count(idx) &&
             "inconsistent resource user map");
      assert(traits.assigned(resources[idx]) == start &&
             "resource has non-uniform assignment across kernels");
    }
  }

  for (CGNode *kernel : kernels) {
    SmallVector<std::pair<uint64_t, uint64_t>, 8> intervals;
    for (size_t idx : traits.reachableResources(*kernel)) {
      uint64_t size = traits.size(resources[idx]);
      if (size == 0)
        continue;
      uint64_t begin = traits.assigned(resources[idx]);
      intervals.push_back({begin, begin + size});
    }

    llvm::sort(intervals);
    for (size_t i = 1, e = intervals.size(); i < e; ++i)
      assert(intervals[i - 1].second <= intervals[i].first &&
             "kernel has overlapping resource intervals");
  }
}
#endif

// Assign a single global resource range to each resource while minimizing the
// per-kernel high-water mark. ResourceT is the linker-side record for one
// allocatable entity. It is currently either LDSSymbolInfo, where the assigned
// value is an LDS byte offset, or NamedBarrierInfo, where the assigned value is
// the first hardware named-barrier ID in the resource's ID range.
//
// The algorithm uses a per-kernel frontier. For a resource used by kernels K,
// place it at the maximum current frontier of K, rounded up to the resource's
// alignment, then advance only those kernels by the resource size. This keeps
// one uniform assignment per resource, while allowing resources reached by
// disjoint kernel sets to overlap. Resources are assigned in descending number
// of reaching kernels, with Traits::compare breaking ties in a
// resource-specific way.
//
// Traits supplies the resource-specific policy:
//   - hasUses(cg): whether .amdgpu.info reachability data is available.
//   - reachableResources(kernel): the DenseSet<size_t> of resource indices
//     reachable from the kernel.
//   - initialValue(): the first valid frontier value, e.g. 0 for LDS offsets
//     and 1 for named-barrier IDs.
//   - size(resource): the amount by which a non-zero resource advances a
//     kernel frontier.
//   - alignment(resource): required alignment of the assigned value.
//   - assigned(resource) / setAssigned(resource, value): accessors for the
//     resource's final assignment.
//   - compare(a, b): deterministic allocation priority for equal user counts.
//   - printAllocation(resource, value, users): debug logging for claimed
//     resources.
//
// If reachability is absent, all resources are laid out universally using the
// same Traits policy. If a resource is unreachable from every kernel, it still
// receives a valid assignment for relocation resolution, but it does not affect
// any kernel frontier.
template <typename ResourceT, typename Traits>
static void assignGroupedResources(SmallVectorImpl<ResourceT> &resources,
                                   AMDGPUCallGraph &cg, const Traits &traits) {
  if (!traits.hasUses(cg)) {
    assignUniversalResources(resources, traits);
    return;
  }

  SmallVector<CGNode *, 8> kernels(cg.kernels().begin(), cg.kernels().end());
  DenseMap<size_t, SmallVector<CGNode *, 4>> resourceToUsers;
  DenseMap<CGNode *, uint64_t> kernelFrontiers;
  for (CGNode *kernel : kernels) {
    kernelFrontiers[kernel] = traits.initialValue();
    for (size_t idx : traits.reachableResources(*kernel))
      resourceToUsers[idx].push_back(kernel);
  }

  auto idxCmp = [&](size_t a, size_t b) {
    return traits.compare(resources[a], resources[b]);
  };
  auto getUserCount = [&](size_t idx) {
    auto it = resourceToUsers.find(idx);
    return it == resourceToUsers.end() ? 0 : it->second.size();
  };
  auto claimedCmp = [&](size_t a, size_t b) {
    size_t aUsers = getUserCount(a);
    size_t bUsers = getUserCount(b);
    if (aUsers != bUsers)
      return aUsers > bUsers;
    return idxCmp(a, b);
  };

  DenseSet<size_t> assigned;
  // Split resources by two independent properties:
  //   - claimed resources are reachable from at least one kernel and therefore
  //     participate in per-kernel frontier allocation; unclaimed resources only
  //     need a valid standalone assignment for relocation resolution.
  //   - non-zero resources consume frontier space; zero-sized resources get a
  //     stable aligned assignment but must not advance any kernel frontier.
  //     In practice, zero-sized resources are dynamic LDS symbols. Named
  //     barriers should have nonzero slot counts, but the shared allocator
  //     keeps the zero-size handling generic.
  SmallVector<size_t, 8> claimedNonZero;
  SmallVector<size_t, 4> claimedZeroSize;
  SmallVector<size_t, 8> unclaimedNonZero;
  SmallVector<size_t, 4> unclaimedZeroSize;

  for (size_t i = 0, e = resources.size(); i < e; ++i) {
    bool hasUsers = getUserCount(i) != 0;
    if (hasUsers && traits.size(resources[i]) != 0)
      claimedNonZero.push_back(i);
    else if (hasUsers)
      claimedZeroSize.push_back(i);
    else if (traits.size(resources[i]) == 0)
      unclaimedZeroSize.push_back(i);
    else
      unclaimedNonZero.push_back(i);
  }

  llvm::sort(claimedNonZero, claimedCmp);
  llvm::sort(claimedZeroSize, claimedCmp);

  auto assignResource = [&](size_t idx, uint64_t start) {
    traits.setAssigned(resources[idx], start);
    [[maybe_unused]] bool inserted = assigned.insert(idx).second;
    assert(inserted && "resource assigned multiple times");
  };

  auto allocateClaimed = [&](size_t idx, bool advanceFrontier) {
    ArrayRef<CGNode *> users = resourceToUsers[idx];
    uint64_t start = alignTo(
        getMaxKernelFrontier(users, kernelFrontiers, traits.initialValue()),
        traits.alignment(resources[idx]));
    assignResource(idx, start);
    if (!advanceFrontier)
      return;

    uint64_t next = start + traits.size(resources[idx]);
    traits.printAllocation(resources[idx], start, users.size());
    for (CGNode *user : users)
      kernelFrontiers[user] = next;
  };

  for (size_t idx : claimedNonZero)
    allocateClaimed(idx, /*advanceFrontier=*/true);

  for (size_t idx : claimedZeroSize)
    allocateClaimed(idx, /*advanceFrontier=*/false);

  // Resources not reachable from any kernel still need valid addresses for
  // relocation resolution. Assign them independently from the initial frontier
  // because no kernel's resource total depends on them.
  if (!unclaimedNonZero.empty() || !unclaimedZeroSize.empty()) {
    llvm::sort(unclaimedNonZero, idxCmp);
    llvm::sort(unclaimedZeroSize, idxCmp);
    uint64_t frontier = traits.initialValue();
    for (size_t idx : unclaimedNonZero) {
      frontier = alignTo(frontier, traits.alignment(resources[idx]));
      assignResource(idx, frontier);
      frontier += traits.size(resources[idx]);
    }
    if (!unclaimedZeroSize.empty()) {
      Align maxZeroAlign(1);
      for (size_t idx : unclaimedZeroSize)
        maxZeroAlign = std::max(maxZeroAlign, traits.alignment(resources[idx]));
      uint64_t zeroBase = alignTo(frontier, maxZeroAlign);
      for (size_t idx : unclaimedZeroSize)
        assignResource(idx, zeroBase);
    }
  }

#ifndef NDEBUG
  assertResourceLayoutInvariants(ArrayRef<ResourceT>(resources), kernels,
                                 resourceToUsers, assigned, traits);
#endif
}

//===----------------------------------------------------------------------===//
// Section parsing
//===----------------------------------------------------------------------===//

// Build relocation map: byte offset -> {ELF symbol index, addend}.
// The addend is needed to resolve STT_SECTION symbols: ELF assemblers
// canonically convert relocations for local symbols in their own sections
// into section_symbol + addend form, so we must track the addend to map
// back to the actual function symbol (see resolveSecSymbol).
// Extract explicit RELA/CREL addends and treat REL entries as addend zero.
template <class RelTy> static int64_t getAddend(const RelTy &rel) {
  if constexpr (RelTy::HasAddend)
    return rel.r_addend;
  return 0;
}

// Build the relocation lookup for one .amdgpu.info section.
template <class ELFT>
static DenseMap<uint64_t, RelocInfo> buildRelocMap(ObjFile<ELFT> *obj,
                                                   uint32_t secIndex) {
  ArrayRef<typename ELFT::Shdr> objSections = obj->template getELFShdrs<ELFT>();
  const ELFFile<ELFT> &elfObj = obj->getObj();
  DenseMap<uint64_t, RelocInfo> relocMap;
  for (size_t i = 0, e = objSections.size(); i < e; ++i) {
    const auto &relSec = objSections[i];
    if (relSec.sh_info != secIndex)
      continue;
    if (relSec.sh_type == SHT_RELA) {
      for (const auto &rel :
           CHECK(elfObj.relas(relSec), "could not read rela section"))
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      break;
    }
    if (relSec.sh_type == SHT_REL) {
      for (const auto &rel :
           CHECK(elfObj.rels(relSec), "could not read rel section"))
        relocMap[rel.r_offset] = {rel.getSymbol(false), 0};
      break;
    }
    if (relSec.sh_type == SHT_CREL) {
      auto crels = CHECK(elfObj.crels(relSec), "could not read crel section");
      for (const auto &rel : crels.first)
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      for (const auto &rel : crels.second)
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      break;
    }
  }
  return relocMap;
}

// Map function definition addresses back to symbols so section-symbol
// relocations can recover the original named function.
template <class ELFT>
static DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *>
buildSymbolAddressMap(ObjFile<ELFT> *obj) {
  DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> addrToSym;
  for (Symbol *sym : obj->getSymbols()) {
    if (!sym || sym->getName().empty())
      continue;
    auto *def = dyn_cast<Defined>(sym);
    if (!def || !def->section)
      continue;
    addrToSym.try_emplace({def->section, def->value}, sym);
  }
  return addrToSym;
}

// Resolve an STT_SECTION symbol + addend to the named function symbol at that
// address. ELF assemblers replace relocations targeting local symbols with
// section_symbol + addend (standard ELF canonicalization). Since section
// symbols have empty names in LLD, the .amdgpu.info parser cannot identify
// the function from the section symbol alone. Use a per-object address map to
// find the named Defined symbol at the matching section and offset.
static Symbol *resolveSecSymbol(
    Symbol &secSym, int64_t addend,
    const DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> &addrToSym) {
  auto *secDef = dyn_cast<Defined>(&secSym);
  if (!secDef || !secDef->section)
    return nullptr;
  auto it = addrToSym.find({secDef->section, static_cast<uint64_t>(addend)});
  return it == addrToSym.end() ? nullptr : it->second;
}

using namespace llvm::AMDGPU;

// Resolve a relocation attached to a .amdgpu.info payload field. The result is
// either the referenced symbol or the named function represented by a
// section-symbol relocation.
template <class ELFT>
static Symbol *resolveRelocSym(
    ObjFile<ELFT> *obj, const DenseMap<uint64_t, RelocInfo> &relocMap,
    const DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> &addrToSym,
    uint64_t off, StringRef diagName) {
  auto it = relocMap.find(off);
  if (it == relocMap.end())
    return nullptr;
  Symbol *sym = &obj->getSymbol(it->second.symIdx);
  if (sym->getName().empty()) {
    sym = resolveSecSymbol(*sym, it->second.addend, addrToSym);
    if (!sym) {
      Warn(obj->ctx) << obj->getName() << ": .amdgpu.info " << diagName
                     << " at offset " << off
                     << " references a section symbol that could not be "
                        "resolved to a named function symbol";
    }
  }
  return sym;
}

// Return a string from .amdgpu.strtab without reading past the section if
// malformed input omits the trailing NUL.
static StringRef getInfoString(StringRef strtab, uint32_t offset) {
  if (offset >= strtab.size())
    return StringRef();
  StringRef tail = strtab.substr(offset);
  size_t nul = tail.find('\0');
  return tail.substr(0, nul);
}

// .amdgpu.info is emitted as a standalone section, so records from losing weak
// definitions or discarded COMDAT groups can survive. Only the prevailing
// definition may contribute local resources and edges. Non-prevailing scopes
// may still contribute type IDs for address-taken declarations.
static bool isPrevailingInfoFunction(ELFFileBase *obj, Symbol *sym) {
  auto *def = dyn_cast<Defined>(sym);
  if (!def || !def->section || def->section == &InputSection::discarded)
    return false;
  return def->section->file == obj;
}

// Check whether this object participates in AMDGPU object-linking metadata.
template <class ELFT> static bool hasAMDGPUInfoSection(ObjFile<ELFT> *obj) {
  return obj->amdgpuInfoSectionIndex != 0;
}

// Parse one object's .amdgpu.info and .amdgpu.strtab sections into the shared
// call graph, direct resource records, and direct LDS/barrier uses.
template <class ELFT>
static void parseInfoSection(ObjFile<ELFT> *obj, AMDGPUCallGraph &cg,
                             const DenseMap<Symbol *, size_t> &ldsSymToIndex,
                             const DenseMap<Symbol *, size_t> &barSymToIndex) {
  if (!hasAMDGPUInfoSection(obj))
    return;

  ArrayRef<typename ELFT::Shdr> objSections = obj->template getELFShdrs<ELFT>();
  const auto &sec = objSections[obj->amdgpuInfoSectionIndex];
  ArrayRef<uint8_t> data = CHECK(obj->getObj().getSectionContents(sec),
                                 "could not read .amdgpu.info section");
  if (data.empty())
    return;

  DenseMap<uint64_t, RelocInfo> relocMap =
      buildRelocMap(obj, obj->amdgpuInfoSectionIndex);
  DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> addrToSym =
      buildSymbolAddressMap(obj);

  StringRef strtab;
  if (obj->amdgpuStrtabSectionIndex != 0) {
    const auto &strtabSec = objSections[obj->amdgpuStrtabSectionIndex];
    ArrayRef<uint8_t> strtabData =
        CHECK(obj->getObj().getSectionContents(strtabSec),
              "could not read .amdgpu.strtab section");
    strtab = StringRef(reinterpret_cast<const char *>(strtabData.data()),
                       strtabData.size());
  }

  CGNode *curNode = nullptr;
  bool curFuncIsPrevailing = false;
  size_t pos = 0;
  while (pos + 2 <= data.size()) {
    uint8_t kind = data[pos];
    uint8_t len = data[pos + 1];
    size_t payloadOff = pos + 2;
    if (payloadOff + len > data.size())
      break;

    switch (kind) {
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_FUNC): {
      if (len < 8)
        break;
      Symbol *sym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "func");
      if (!sym || sym->getName().empty()) {
        curNode = nullptr;
        curFuncIsPrevailing = false;
        break;
      }
      curNode = &cg.getOrCreate(sym);
      curFuncIsPrevailing = isPrevailingInfoFunction(obj, sym);
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_FLAGS): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      AMDGPU::FuncInfoFlags flags = static_cast<AMDGPU::FuncInfoFlags>(
          read32le(data.data() + payloadOff));
      curNode->localRes.UsesVCC =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_USES_VCC);
      curNode->localRes.UsesFlatScratch =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_USES_FLAT_SCRATCH);
      curNode->localRes.HasDynSizedStack =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_HAS_DYN_STACK);
      curNode->localRes.isCuMode =
          !(flags & AMDGPU::FuncInfoFlags::FUNC_WGP_MODE);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_VGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.NumArchVGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_AGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.NumAccVGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_SGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.NumSGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_PRIVATE_SEGMENT_SIZE): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.ScratchSize = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_OCCUPANCY): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.occupancy = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_WAVE_SIZE): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.waveSize = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_USE): {
      if (!curNode || !curFuncIsPrevailing || len < 8)
        break;
      Symbol *resSym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "use");
      if (!resSym || resSym->getName().empty())
        break;
      auto barIt = barSymToIndex.find(resSym);
      if (barIt != barSymToIndex.end()) {
        LLVM_DEBUG(dbgs() << "  barrier use: " << curNode->sym->getName()
                          << " -> " << resSym->getName() << "\n");
        curNode->barrierUseIndices.push_back(barIt->second);
        cg.setHasBarrierUses();
        break;
      }
      auto ldsIt = ldsSymToIndex.find(resSym);
      if (ldsIt != ldsSymToIndex.end()) {
        LLVM_DEBUG(dbgs() << "  use: " << curNode->sym->getName() << " -> "
                          << resSym->getName() << "\n");
        curNode->ldsUseIndices.push_back(ldsIt->second);
        cg.setHasLDSUses();
      }
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_CALL): {
      if (!curNode || !curFuncIsPrevailing || len < 8)
        break;
      Symbol *dstSym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "call");
      if (!dstSym || dstSym->getName().empty())
        break;
      CGNode &dst = cg.getOrCreate(dstSym);
      LLVM_DEBUG(dbgs() << "  call: " << curNode->sym->getName() << " -> "
                        << dstSym->getName() << "\n");
      curNode->callees.push_back(&dst);
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_INDIRECT_CALL): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      uint32_t typeIdOff = read32le(data.data() + payloadOff);
      StringRef typeId = getInfoString(strtab, typeIdOff);
      if (!typeId.empty()) {
        LLVM_DEBUG(dbgs() << "  indirect-call: " << curNode->sym->getName()
                          << " enc=" << typeId << "\n");
        cg.addIndirectCall(curNode, typeId);
      }
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_TYPEID): {
      if (!curNode || len < 4)
        break;
      cg.markAddressTaken(curNode);
      uint32_t typeIdOff = read32le(data.data() + payloadOff);
      StringRef typeId = getInfoString(strtab, typeIdOff);
      if (!typeId.empty()) {
        LLVM_DEBUG(dbgs() << "  signature: " << curNode->sym->getName()
                          << " typeId=" << typeId << "\n");
        cg.addSignature(curNode, typeId);
      }
      break;
    }
    default:
      LLVM_DEBUG(dbgs() << "  unknown info kind " << (unsigned)kind
                        << " (len=" << (unsigned)len << "), skipping\n");
      break;
    }

    pos = payloadOff + len;
  }

  // Emit debug summary of parsed resources.
  if (curNode && curNode->hasLocalRes) {
    LLVM_DEBUG({
      for (auto &[sym, node] : cg) {
        if (!node->hasLocalRes)
          continue;
        dbgs() << "  resource: " << sym->getName()
               << " vgpr=" << node->localRes.NumArchVGPR
               << " agpr=" << node->localRes.NumAccVGPR
               << " sgpr=" << node->localRes.NumSGPR
               << " scratch=" << node->localRes.ScratchSize
               << " occupancy=" << node->localRes.occupancy << "\n";
      }
    });
  }
}

// Kernels are identified by the companion kernel descriptor symbol emitted as
// <kernel>.kd.
static bool hasKernelDescriptor(Ctx &ctx, Symbol *sym) {
  std::string kdName = (sym->getName() + ".kd").str();
  Symbol *kdSym = ctx.symtab->find(kdName);
  Defined *kdDef = dyn_cast_or_null<Defined>(kdSym);
  return kdDef && kdDef->section;
}

// Mark graph nodes with kernel descriptors as roots for reachability and
// resource propagation.
static void markKernelsWithDescriptors(Ctx &ctx, AMDGPUCallGraph &cg) {
  for (auto &[sym, node] : cg) {
    if (hasKernelDescriptor(ctx, sym)) {
      LLVM_DEBUG(dbgs() << "  kernel: " << sym->getName() << "\n");
      cg.addKernel(*node);
    }
  }
}

//===----------------------------------------------------------------------===//
// Call graph SCC condensation, reachability, and resource propagation
//===----------------------------------------------------------------------===//

// Validate local resource metadata for a kernel-reachable node before resource
// propagation depends on it.
static bool validateResourceNode(Ctx &ctx, CGNode *node) {
  assert(node->hasLocalRes && "missing resource usage after alias resolution");

  if (node->localRes.occupancy == 0) {
    Err(ctx) << "AMDGPU: function '" << node->sym->getName()
             << "' has invalid ABI occupancy 0";
    LLVM_DEBUG(dbgs() << "    resolve " << node->sym->getName()
                      << " (invalid occupancy)\n");
    return false;
  }

  if (node->localRes.waveSize == 0) {
    Err(ctx) << "AMDGPU: function '" << node->sym->getName()
             << "' has invalid ABI wave size 0";
    LLVM_DEBUG(dbgs() << "    resolve " << node->sym->getName()
                      << " (invalid wave size)\n");
    return false;
  }

  return true;
}

// Merge resource usage from another CGResourceInfo into the result using
// element-wise max for register/scratch counts and OR for boolean flags.
static void mergeResourceInfo(CGResourceInfo &result,
                              const CGResourceInfo &info) {
  result.NumArchVGPR = std::max(result.NumArchVGPR, info.NumArchVGPR);
  result.NumAccVGPR = std::max(result.NumAccVGPR, info.NumAccVGPR);
  result.NumSGPR = std::max(result.NumSGPR, info.NumSGPR);
  result.ScratchSize = std::max(result.ScratchSize, info.ScratchSize);
  result.UsesVCC |= info.UsesVCC;
  result.UsesFlatScratch |= info.UsesFlatScratch;
  result.HasDynSizedStack |= info.HasDynSizedStack;
}

bool AMDGPUCallGraph::buildCondensedGraph(Ctx &ctx, bool validateResources) {
  DenseMap<CGNode *, unsigned> nodeIndex;
  DenseMap<CGNode *, unsigned> lowLink;
  DenseSet<CGNode *> onStack;
  SmallVector<CGNode *, 16> stack;
  unsigned nextIndex = 0;

  // Start from kernels so resource diagnostics are limited to code that can
  // execute. A validation failure aborts the build; callers do not inspect the
  // partially constructed SCC state.
  for (CGNode *kernel : kernels()) {
    if (nodeIndex.count(kernel))
      continue;
    if (!buildSCCs(ctx, kernel, nodeIndex, lowLink, onStack, stack, nextIndex,
                   validateResources))
      return false;
  }

  buildSCCEdges();
  return true;
}

bool AMDGPUCallGraph::buildSCCs(Ctx &ctx, CGNode *node,
                                DenseMap<CGNode *, unsigned> &nodeIndex,
                                DenseMap<CGNode *, unsigned> &lowLink,
                                DenseSet<CGNode *> &onStack,
                                SmallVectorImpl<CGNode *> &stack,
                                unsigned &nextIndex, bool validateResources) {
  if (validateResources && !validateResourceNode(ctx, node))
    return false;

  nodeIndex[node] = nextIndex;
  lowLink[node] = nextIndex;
  ++nextIndex;
  stack.push_back(node);
  onStack.insert(node);

  if (validateResources) {
    const CGResourceInfo &info = node->localRes;
    LLVM_DEBUG(dbgs() << "    resolve " << node->sym->getName()
                      << " (has local res: vgpr=" << info.NumArchVGPR
                      << " sgpr=" << info.NumSGPR
                      << " scratch=" << info.ScratchSize << ")\n");
  }

  if (!node->callees.empty()) {
    LLVM_DEBUG({
      if (validateResources) {
        dbgs() << "    " << node->sym->getName() << " has "
               << node->callees.size() << " callees:";
        for (CGNode *c : node->callees)
          dbgs() << " " << c->sym->getName();
        dbgs() << "\n";
      }
    });

    for (CGNode *callee : node->callees) {
      if (validateResources) {
        if (!validateResourceNode(ctx, callee))
          return false;

        if (callee->localRes.occupancy < node->localRes.occupancy) {
          Err(ctx) << "AMDGPU: incompatible ABI occupancy: function '"
                   << node->sym->getName() << "' requires occupancy "
                   << node->localRes.occupancy << " but calls '"
                   << callee->sym->getName() << "' with occupancy "
                   << callee->localRes.occupancy;
          return false;
        }
      }

      auto calleeIndex = nodeIndex.find(callee);
      if (calleeIndex == nodeIndex.end()) {
        if (!buildSCCs(ctx, callee, nodeIndex, lowLink, onStack, stack,
                       nextIndex, validateResources))
          return false;
        lowLink[node] = std::min(lowLink[node], lowLink[callee]);
      } else if (onStack.count(callee)) {
        lowLink[node] = std::min(lowLink[node], calleeIndex->second);
      }
    }
  }

  if (lowLink[node] != nodeIndex[node])
    return true;

  unsigned sccId = sccs.size();
  sccs.emplace_back();
  CallGraphSCC &scc = sccs.back();

  for (;;) {
    CGNode *member = stack.pop_back_val();
    onStack.erase(member);
    scc.members.push_back(member);
    nodeToSCC[member] = sccId;

    if (validateResources) {
      if (scc.members.size() == 1)
        scc.localRes = member->localRes;
      else
        mergeResourceInfo(scc.localRes, member->localRes);
    }

    for (size_t idx : member->ldsUseIndices)
      scc.reachableLDS.insert(idx);

    for (size_t idx : member->barrierUseIndices)
      scc.reachableBarriers.insert(idx);

    if (member == node)
      break;
  }

  return true;
}

void AMDGPUCallGraph::buildSCCEdges() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    if (scc.members.size() > 1) {
      scc.isRecursive = true;
      scc.localRes.HasDynSizedStack = true;
    }

    DenseSet<unsigned> seenCallees;
    for (CGNode *member : scc.members) {
      for (CGNode *callee : member->callees) {
        auto it = nodeToSCC.find(callee);
        assert(it != nodeToSCC.end() && "callee should be in call graph SCC");

        unsigned calleeSCC = it->second;
        if (calleeSCC == sccId) {
          scc.isRecursive = true;
          scc.localRes.HasDynSizedStack = true;
          continue;
        }

        assert(calleeSCC < sccId &&
               "sccs should be in reverse topological order");

        if (seenCallees.insert(calleeSCC).second)
          scc.callees.push_back(calleeSCC);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// LDS resolution
//===----------------------------------------------------------------------===//

// Collect every distinct SHN_AMDGPU_LDS common symbol that needs a final
// link-time offset. Named barrier symbols (identified by the
// __amdgpu_named_barrier prefix) are routed to a separate barriers array.
static void collectLDSSymbols(Ctx &ctx,
                              SmallVectorImpl<LDSSymbolInfo> &ldsSymbols,
                              SmallVectorImpl<NamedBarrierInfo> &barriers) {
  DenseSet<Symbol *> seen;
  for (ELFFileBase *file : ctx.objectFiles) {
    if (!file->hasCommonSyms)
      continue;
    for (Symbol *sym : file->getGlobalSymbols()) {
      if (!sym->isAMDGPULDS || !seen.insert(sym).second)
        continue;
      auto *cs = dyn_cast<CommonSymbol>(sym);
      if (!cs)
        continue;
      if (sym->getName().starts_with("__amdgpu_named_barrier")) {
        LLVM_DEBUG(dbgs() << "  collected named barrier: " << sym->getName()
                          << " size=" << cs->size << "\n");
        barriers.push_back({sym, static_cast<uint32_t>(cs->size / 16)});
      } else {
        LLVM_DEBUG(dbgs() << "  collected LDS symbol: " << sym->getName()
                          << " size=" << cs->size << " align=" << cs->alignment
                          << "\n");
        ldsSymbols.push_back(
            {sym, cs->size, Align(cs->alignment), /*assignedOffset=*/0});
      }
    }
  }
}

void AMDGPUCallGraph::computeKernelReachability() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    for (unsigned calleeSCC : scc.callees) {
      assert(calleeSCC < sccId &&
             "sccs should be in reverse topological order");
      const CallGraphSCC &callee = sccs[calleeSCC];
      scc.reachableLDS.insert(callee.reachableLDS.begin(),
                              callee.reachableLDS.end());
      scc.reachableBarriers.insert(callee.reachableBarriers.begin(),
                                   callee.reachableBarriers.end());
    }
  }

  for (CGNode *kernel : kernels()) {
    auto it = nodeToSCC.find(kernel);
    assert(it != nodeToSCC.end() && "kernel should be in call graph SCC");
    const CallGraphSCC &scc = sccs[it->second];
    kernel->reachableLDS.clear();
    kernel->reachableLDS.insert(scc.reachableLDS.begin(),
                                scc.reachableLDS.end());
    kernel->reachableBarriers.clear();
    kernel->reachableBarriers.insert(scc.reachableBarriers.begin(),
                                     scc.reachableBarriers.end());
  }
}

// Compute each kernel's fixed LDS usage from the final offsets of the LDS
// objects reachable from that kernel.
static void computeKernelLDSSizes(ArrayRef<LDSSymbolInfo> ldsSymbols,
                                  AMDGPUCallGraph &cg) {
  if (!cg.hasLDSUses()) {
    uint32_t totalSize = 0;
    for (const LDSSymbolInfo &lds : ldsSymbols)
      totalSize = std::max<uint64_t>(totalSize, lds.assignedOffset + lds.size);
    for (CGNode *kernel : cg.kernels())
      kernel->ldsSize = totalSize;
    return;
  }

  for (CGNode *kernel : cg.kernels()) {
    uint32_t maxEnd = 0;
    for (size_t idx : kernel->reachableLDS)
      maxEnd = std::max<uint64_t>(maxEnd, ldsSymbols[idx].assignedOffset +
                                              ldsSymbols[idx].size);
    kernel->ldsSize = maxEnd;
  }
}

// Assign LDS offsets, rewrite LDS symbols to those offsets, and record each
// kernel's group segment size.
static void resolveLDS(Ctx &ctx, SmallVectorImpl<LDSSymbolInfo> &ldsSymbols,
                       AMDGPUCallGraph &cg) {
  LLVM_DEBUG(dbgs() << "AMDGPU LDS: assigning grouped offsets\n");
  assignGroupedResources(ldsSymbols, cg, LDSAllocationTraits());
  computeKernelLDSSizes(ldsSymbols, cg);

  // Symbol::overwrite preserves the old symbol's visibility, so for shared-
  // object links (AMDGPU code objects use -shared) we must explicitly force
  // STV_HIDDEN to prevent the symbol from being preemptible, which would cause
  // R_AMDGPU_ABS32_LO relocations to be rejected by the relocation scanner.
  LLVM_DEBUG(dbgs() << "AMDGPU LDS: final symbol assignments:\n");
  for (const LDSSymbolInfo &lds : ldsSymbols) {
    LLVM_DEBUG(dbgs() << "  " << lds.sym->getName() << " -> offset="
                      << lds.assignedOffset << " size=" << lds.size << "\n");
    Defined(ctx, ctx.internalFile, lds.sym->getName(), STB_GLOBAL, STV_HIDDEN,
            STT_NOTYPE, lds.assignedOffset, lds.size, nullptr)
        .overwrite(*lds.sym);
    if (ctx.arg.shared)
      lds.sym->stOther = (lds.sym->stOther & ~3) | STV_HIDDEN;
  }

  LLVM_DEBUG({
    dbgs() << "AMDGPU LDS: per-kernel LDS sizes:\n";
    for (CGNode *kernel : cg.kernels())
      dbgs() << "  " << kernel->sym->getName() << " -> " << kernel->ldsSize
             << " bytes\n";
  });
}

//===----------------------------------------------------------------------===//
// Named barrier resolution
//===----------------------------------------------------------------------===//

static uint32_t computeMaxNamedBarrierEnd(ArrayRef<NamedBarrierInfo> barriers) {
  uint32_t maxBarEnd = 0;
  for (const NamedBarrierInfo &bar : barriers) {
    if (bar.slotCount == 0)
      continue;
    maxBarEnd = std::max(maxBarEnd, bar.assignedBarId + bar.slotCount - 1);
  }
  return maxBarEnd;
}

static void computeKernelNamedBarrierCounts(ArrayRef<NamedBarrierInfo> barriers,
                                            AMDGPUCallGraph &cg) {
  if (!cg.hasBarrierUses()) {
    uint32_t totalCount = computeMaxNamedBarrierEnd(barriers);
    for (CGNode *kernel : cg.kernels())
      kernel->numNamedBarrier = totalCount;
    return;
  }

  for (CGNode *kernel : cg.kernels()) {
    uint32_t maxBarEnd = 0;
    for (size_t idx : kernel->reachableBarriers) {
      if (barriers[idx].slotCount == 0)
        continue;
      uint32_t end = barriers[idx].assignedBarId + barriers[idx].slotCount - 1;
      maxBarEnd = std::max(maxBarEnd, end);
    }
    kernel->numNamedBarrier = maxBarEnd;
  }
}

static void checkNamedBarrierCounts(Ctx &ctx, AMDGPUCallGraph &cg) {
  for (CGNode *kernel : cg.kernels()) {
    if (kernel->numNamedBarrier <= 31)
      continue;
    Err(ctx) << "AMDGPU: named barrier ID overflow (max ID "
             << kernel->numNamedBarrier << " exceeds limit of 31) in kernel '"
             << kernel->sym->getName() << "'";
  }
}

// Assign named-barrier hardware IDs, rewrite named-barrier symbols to encoded
// barrier addresses, and record each kernel's named-barrier count.
static void resolveNamedBarriers(Ctx &ctx,
                                 SmallVectorImpl<NamedBarrierInfo> &barriers,
                                 AMDGPUCallGraph &cg) {
  LLVM_DEBUG(dbgs() << "AMDGPU Named Barriers: assigning grouped IDs\n");
  assignGroupedResources(barriers, cg, NamedBarrierAllocationTraits());
  computeKernelNamedBarrierCounts(barriers, cg);
  checkNamedBarrierCounts(ctx, cg);

  constexpr uint32_t barScope = 0; // BARRIER_SCOPE_WORKGROUP
  LLVM_DEBUG(dbgs() << "AMDGPU Named Barriers: final assignments:\n");
  for (NamedBarrierInfo &bar : barriers) {
    if (bar.assignedBarId == 0)
      continue;
    uint32_t addr = 0x802000u | (barScope << 9) | (bar.assignedBarId << 4);
    LLVM_DEBUG(dbgs() << "  " << bar.sym->getName()
                      << " -> barId=" << bar.assignedBarId << " addr=0x"
                      << Twine::utohexstr(addr) << "\n");
    Defined(ctx, ctx.internalFile, bar.sym->getName(), STB_GLOBAL, STV_HIDDEN,
            STT_NOTYPE, addr, 0, nullptr)
        .overwrite(*bar.sym);
    if (ctx.arg.shared)
      bar.sym->stOther = (bar.sym->stOther & ~3) | STV_HIDDEN;
  }

  for (CGNode *kernel : cg.kernels()) {
    LLVM_DEBUG(dbgs() << "  " << kernel->sym->getName() << " numNamedBarrier="
                      << kernel->numNamedBarrier << "\n");
  }
}

// Check that each kernel's final LDS usage satisfies the ABI occupancy recorded
// in .amdgpu.info.
static bool
validateKernelLDSOccupancy(Ctx &ctx, AMDGPUCallGraph &cg,
                           const AMDGPU::ObjectLinkingTargetInfo &targetInfo) {
  for (CGNode *kernel : cg.kernels()) {
    assert(kernel->hasLocalRes &&
           "kernel resource usage should have been validated");
    assert(kernel->localRes.occupancy != 0 &&
           "kernel occupancy should have been validated");
    if (AMDGPU::isLDSSizeCompatibleWithOccupancy(
            targetInfo, kernel->localRes.waveSize, kernel->localRes.isCuMode,
            kernel->ldsSize, kernel->localRes.occupancy))
      continue;

    Err(ctx) << "AMDGPU: kernel '" << kernel->sym->getName() << "' uses "
             << kernel->ldsSize << " bytes of LDS, which does not meet ABI "
             << "occupancy " << kernel->localRes.occupancy;
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Resource usage propagation
//===----------------------------------------------------------------------===//

void AMDGPUCallGraph::propagateResourceUsage() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    CGResourceInfo result = scc.localRes;
    uint32_t maxCalleeScratch = 0;
    for (unsigned calleeSCC : scc.callees) {
      assert(calleeSCC < sccId &&
             "sccs should be in reverse topological order");
      const CallGraphSCC &callee = sccs[calleeSCC];
      assert(callee.hasPropagatedRes && "callee resource should be propagated");
      mergeResourceInfo(result, callee.propagatedRes);
      maxCalleeScratch =
          std::max(maxCalleeScratch, callee.propagatedRes.ScratchSize);
    }
    // For recursive SCCs, scratchSize remains the finite fixed-frame
    // contribution. The unbounded recursive depth is represented by
    // hasDynSizedStack, which forces dynamic stack handling.
    result.ScratchSize = scc.localRes.ScratchSize + maxCalleeScratch;

    scc.propagatedRes = result;
    scc.hasPropagatedRes = true;
    for (CGNode *member : scc.members) {
      member->propagatedRes = result;
      member->hasPropagatedRes = true;
    }
  }

  for (CGNode *kernel : kernels()) {
    LLVM_DEBUG(dbgs() << "  propagated " << kernel->sym->getName()
                      << ": vgpr=" << kernel->propagatedRes.NumArchVGPR
                      << " agpr=" << kernel->propagatedRes.NumAccVGPR
                      << " sgpr=" << kernel->propagatedRes.NumSGPR
                      << " scratch=" << kernel->propagatedRes.ScratchSize
                      << " dynstack=" << kernel->propagatedRes.HasDynSizedStack
                      << "\n");
  }
}

// Build raw resource information from the linker's internal CGNode data.
static AMDGPU::RawLinkingResources buildRawResources(const CGNode &kernel) {
  assert(kernel.hasPropagatedRes &&
         "kernel resource usage should have been propagated");
  AMDGPU::RawLinkingResources info = kernel.propagatedRes;
  info.LDSSize = kernel.ldsSize;
  info.NumNamedBarrier = kernel.numNamedBarrier;
  return info;
}

// Patch kernel descriptors in-place with resolved LDS, named-barrier, and
// propagated resource fields.
static void
patchKernelDescriptors(Ctx &ctx, AMDGPUCallGraph &cg,
                       const AMDGPU::ObjectLinkingTargetInfo &targetInfo,
                       bool hasLDS) {
  // Track sections that have been copied to writable memory so we don't
  // allocate redundant copies when multiple KDs share the same section.
  DenseSet<InputSection *> copiedSections;

  for (CGNode *kernel : cg.kernels()) {
    StringRef name = kernel->sym->getName();
    std::string kdName = (name + ".kd").str();
    Symbol *kdSym = ctx.symtab->find(kdName);
    if (!kdSym)
      continue;
    auto *kdDef = dyn_cast<Defined>(kdSym);
    if (!kdDef || !kdDef->section)
      continue;
    auto *isec = dyn_cast<InputSection>(kdDef->section);
    if (!isec)
      continue;

    uint64_t off = kdDef->value;
    if (off + sizeof(amdhsa::kernel_descriptor_t) > isec->size)
      continue;

    if (!hasLDS && !kernel->hasPropagatedRes)
      continue;

    // The section content may be in read-only mmap'd memory. Make a writable
    // copy the first time we need to patch a KD in this section.
    if (copiedSections.insert(isec).second) {
      auto *newBuf = ctx.bAlloc.Allocate<uint8_t>(isec->size);
      memcpy(newBuf, isec->content_, isec->size);
      isec->content_ = newBuf;
    }

    auto *buf = const_cast<uint8_t *>(isec->content_) + off;
    auto kdBuf =
        MutableArrayRef<uint8_t>(buf, sizeof(amdhsa::kernel_descriptor_t));

    AMDGPU::ResolvedLinkingResources info;
    if (kernel->hasPropagatedRes) {
      info = AMDGPU::resolveObjectLinkingResources(targetInfo,
                                                   buildRawResources(*kernel));
    } else {
      info.LDSSize = kernel->ldsSize;
    }

    AMDGPU::patchKernelDescriptor(kdBuf, targetInfo, info, hasLDS,
                                  kernel->hasPropagatedRes);

    LLVM_DEBUG(dbgs() << "  patched " << name << ".kd: lds=" << info.LDSSize
                      << " scratch=" << info.ScratchSize
                      << " dynstack=" << info.UsesDynamicStack << "\n");
  }
}

//===----------------------------------------------------------------------===//
// HSA metadata patching
//===----------------------------------------------------------------------===//

// Rewrite AMDHSA metadata notes so the YAML/msgpack-visible kernel resource
// fields match the patched kernel descriptors.
template <class ELFT>
static void patchHSAMetadata(Ctx &ctx, AMDGPUCallGraph &cg,
                             const AMDGPU::ObjectLinkingTargetInfo &targetInfo,
                             bool hasLDS) {
  bool hasRes = false;
  for (CGNode *k : cg.kernels())
    if (k->hasPropagatedRes) {
      hasRes = true;
      break;
    }
  if (!hasLDS && !hasRes)
    return;

  DenseMap<StringRef, CGNode *> nameToKernel;
  for (CGNode *kernel : cg.kernels())
    nameToKernel[kernel->sym->getName()] = kernel;

  for (InputSectionBase *sec : ctx.inputSections) {
    auto *isec = dyn_cast<InputSection>(sec);
    if (!isec || isec->type != SHT_NOTE)
      continue;
    if (isec->name != ".note")
      continue;

    ArrayRef<uint8_t> data = isec->contentMaybeDecompress();
    if (data.size() < 12)
      continue;

    uint32_t nameSize = read32le(data.data());
    uint32_t descSize = read32le(data.data() + 4);
    uint32_t noteType = read32le(data.data() + 8);

    // NT_AMDGPU_METADATA = 32
    if (noteType != 32)
      continue;

    uint32_t nameOff = 12;
    uint32_t namePadded = alignTo(nameSize, 4);
    uint32_t descOff = nameOff + namePadded;

    if (descOff + descSize > data.size())
      continue;

    ArrayRef<uint8_t> msgpackData = data.slice(descOff, descSize);
    msgpack::Document doc;
    if (!doc.readFromBlob(
            StringRef(reinterpret_cast<const char *>(msgpackData.data()),
                      msgpackData.size()),
            false))
      continue;

    msgpack::MapDocNode root = doc.getRoot().getMap();
    msgpack::DocNode kernelsNode = root["amdhsa.kernels"];
    if (kernelsNode.isEmpty())
      continue;

    bool modified = false;
    msgpack::ArrayDocNode kernelsArray = kernelsNode.getArray();
    for (size_t i = 0, e = kernelsArray.size(); i < e; ++i) {
      msgpack::MapDocNode kernMap = kernelsArray[i].getMap();
      msgpack::DocNode nameNode = kernMap[".name"];
      if (nameNode.isEmpty())
        continue;

      StringRef kernName = nameNode.getString();
      auto it = nameToKernel.find(kernName);
      if (it == nameToKernel.end())
        continue;
      CGNode *kernel = it->second;

      if (!hasLDS && !kernel->hasPropagatedRes)
        continue;

      AMDGPU::ResolvedLinkingResources info;
      if (kernel->hasPropagatedRes) {
        info = AMDGPU::resolveObjectLinkingResources(
            targetInfo, buildRawResources(*kernel));
      } else {
        info.LDSSize = kernel->ldsSize;
      }

      AMDGPU::patchHSAMetadataKernel(kernMap, doc, info, hasLDS,
                                     kernel->hasPropagatedRes);
      modified = true;
    }

    if (!modified)
      continue;

    std::string newMsgpack;
    doc.writeToBlob(newMsgpack);

    uint32_t newDescSize = newMsgpack.size();
    uint32_t newDescPadded = alignTo(newDescSize, 4);
    uint32_t newSize = nameOff + namePadded + newDescPadded;

    auto *buf = ctx.bAlloc.Allocate<uint8_t>(newSize);
    memset(buf, 0, newSize);
    write32le(buf, nameSize);
    write32le(buf + 4, newDescSize);
    write32le(buf + 8, noteType);
    memcpy(buf + nameOff, data.data() + nameOff, namePadded);
    memcpy(buf + nameOff + namePadded, newMsgpack.data(), newMsgpack.size());

    isec->content_ = buf;
    isec->size = newSize;
  }
}

// Validate call ABI compatibility across call edges. A caller and callee must
// use the same wavefront size and CU/WGP mode because the calling convention,
// register layout, and instruction semantics depend on these properties.
static bool validateCallABICompatibility(Ctx &ctx, AMDGPUCallGraph &cg) {
  bool valid = true;
  for (auto &[sym, node] : cg) {
    // Alias nodes remain in the graph map after resolveAliases, but all edges
    // to them have been redirected to the canonical node with resource info.
    if (!node->hasLocalRes)
      continue;
    for (CGNode *callee : node->callees) {
      assert(callee->hasLocalRes && "missing local resource info for callee");
      uint32_t callerWaveSize = node->localRes.waveSize;
      uint32_t calleeWaveSize = callee->localRes.waveSize;
      if (callerWaveSize != calleeWaveSize) {
        Err(ctx) << "AMDGPU object linking: wave size mismatch in call from '"
                 << node->sym->getName() << "' (wave" << callerWaveSize
                 << ") to '" << callee->sym->getName() << "' (wave"
                 << calleeWaveSize << ")";
        valid = false;
      }
      if (node->localRes.isCuMode != callee->localRes.isCuMode) {
        Err(ctx) << "AMDGPU object linking: CU/WGP mode mismatch in call from '"
                 << node->sym->getName() << "' ("
                 << (node->localRes.isCuMode ? "CU" : "WGP") << ") to '"
                 << callee->sym->getName() << "' ("
                 << (callee->localRes.isCuMode ? "CU" : "WGP") << ")";
        valid = false;
      }
    }
  }
  return valid;
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

// Resolve AMDGPU object-linking metadata for one ELF class. This is run after
// regular symbol resolution so all referenced symbols and kernel descriptors
// are available for graph construction and patching.
template <class ELFT> void elf::resolveAMDGPUObjectLinking(Ctx &ctx) {
  llvm::TimeTraceScope timeScope("Resolve AMDGPU Object Linking");

  LLVM_DEBUG(dbgs() << "AMDGPU: collecting LDS symbols\n");
  SmallVector<LDSSymbolInfo, 16> ldsSymbols;
  SmallVector<NamedBarrierInfo, 4> barriers;
  collectLDSSymbols(ctx, ldsSymbols, barriers);
  LLVM_DEBUG(dbgs() << "AMDGPU: found " << ldsSymbols.size() << " LDS symbols, "
                    << barriers.size() << " named barriers\n");

  // Build sym->index maps once (used by section parsers).
  DenseMap<Symbol *, size_t> ldsSymToIndex;
  for (size_t i = 0, e = ldsSymbols.size(); i < e; ++i)
    ldsSymToIndex[ldsSymbols[i].sym] = i;
  DenseMap<Symbol *, size_t> barSymToIndex;
  for (size_t i = 0, e = barriers.size(); i < e; ++i)
    barSymToIndex[barriers[i].sym] = i;

  AMDGPUCallGraph cg;

  LLVM_DEBUG(dbgs() << "AMDGPU: parsing sections\n");
  bool hasResourceUsage = false;
  for (ELFFileBase *file : ctx.objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    parseInfoSection(obj, cg, ldsSymToIndex, barSymToIndex);
    if (hasAMDGPUInfoSection(obj))
      hasResourceUsage = true;
  }
  LLVM_DEBUG(dbgs() << "AMDGPU: building indirect call edges\n");
  cg.buildIndirectEdges();

  LLVM_DEBUG(dbgs() << "AMDGPU: resolving function aliases\n");
  cg.resolveAliases();

  LLVM_DEBUG(dbgs() << "AMDGPU: validating call ABI compatibility\n");
  if (!validateCallABICompatibility(ctx, cg))
    return;

  LLVM_DEBUG(dbgs() << "AMDGPU: identifying kernels\n");
  markKernelsWithDescriptors(ctx, cg);

  // Detect resource usage from any node in the call graph.
  if (!hasResourceUsage)
    hasResourceUsage = cg.begin() != cg.end();

  LLVM_DEBUG(dbgs() << "AMDGPU: " << ldsSymbols.size() << " regular LDS, "
                    << barriers.size() << " named barriers\n");
  LLVM_DEBUG(dbgs() << "AMDGPU: " << cg.kernels().size() << " kernels\n");
  LLVM_DEBUG(if (hasResourceUsage) dbgs()
             << "AMDGPU: has resource usage data\n");

  if (ldsSymbols.empty() && barriers.empty() && !hasResourceUsage) {
    LLVM_DEBUG(dbgs() << "AMDGPU: nothing to resolve\n");
    return;
  }

  if (ctx.arg.osabi != ELF::ELFOSABI_AMDGPU_HSA) {
    Err(ctx) << "AMDGPU object linking is only supported for amdhsa (OSABI "
             << ctx.arg.osabi << ")";
    return;
  }

  uint32_t eflags = ctx.arg.eflags;
  StringRef cpu =
      ELF::getAMDGPUArchNameFromELFMach(eflags & ELF::EF_AMDGPU_MACH);
  AMDGPU::GPUKind kind = AMDGPU::parseArchAMDGCN(cpu);
  if (kind == AMDGPU::GK_NONE) {
    Err(ctx) << "AMDGPU object linking: unsupported GPU architecture";
    return;
  }

  uint32_t xnack = eflags & ELF::EF_AMDGPU_FEATURE_XNACK_V4;
  bool xnackOnOrAny = xnack == ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4 ||
                      xnack == ELF::EF_AMDGPU_FEATURE_XNACK_ANY_V4;
  AMDGPU::ObjectLinkingTargetInfo targetInfo =
      AMDGPU::ObjectLinkingTargetInfo::get(kind, xnackOnOrAny);

  bool hasLDS = !ldsSymbols.empty();
  bool hasBarriers = !barriers.empty();
  bool needsReachability =
      (cg.hasLDSUses() || cg.hasBarrierUses()) && !cg.kernels().empty();
  bool needsResource = hasResourceUsage && !cg.kernels().empty();

  // Alias resolution and kernel discovery must be complete before this point:
  // the shared SCC graph is intentionally kernel-reachable and is consumed by
  // both reachability propagation and resource propagation.
  if (needsReachability || needsResource) {
    LLVM_DEBUG(dbgs() << "AMDGPU: building condensed call graph\n");
    if (!cg.buildCondensedGraph(ctx, needsResource))
      return;
  }

  if (needsReachability) {
    LLVM_DEBUG(dbgs() << "AMDGPU: computing kernel reachability\n");
    cg.computeKernelReachability();
  }

  if (hasLDS)
    resolveLDS(ctx, ldsSymbols, cg);

  if (hasBarriers)
    resolveNamedBarriers(ctx, barriers, cg);

  if (needsResource) {
    LLVM_DEBUG(dbgs() << "AMDGPU: propagating resource usage\n");
    cg.propagateResourceUsage();
    if (!validateKernelLDSOccupancy(ctx, cg, targetInfo))
      return;
  }

  LLVM_DEBUG(dbgs() << "AMDGPU: patching kernel descriptors\n");
  patchKernelDescriptors(ctx, cg, targetInfo, hasLDS);

  LLVM_DEBUG(dbgs() << "AMDGPU: patching HSA metadata\n");
  patchHSAMetadata<ELFT>(ctx, cg, targetInfo, hasLDS);
}

template void elf::resolveAMDGPUObjectLinking<ELF32LE>(Ctx &);
template void elf::resolveAMDGPUObjectLinking<ELF64LE>(Ctx &);
