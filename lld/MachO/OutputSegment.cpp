//===- OutputSegment.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSegment.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Sections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

static uint32_t initProt(StringRef name) {
  auto it = find_if(
      config->segmentProtections,
      [&](const SegmentProtection &segprot) { return segprot.name == name; });
  if (it != config->segmentProtections.end())
    return it->initProt;

  if (name == segment_names::text)
    return VM_PROT_READ | VM_PROT_EXECUTE;
  if (name == segment_names::pageZero)
    return 0;
  if (name == segment_names::linkEdit)
    return VM_PROT_READ;
  return VM_PROT_READ | VM_PROT_WRITE;
}

static uint32_t maxProt(StringRef name) {
  assert(config->arch() != AK_i386 &&
         "TODO: i386 has different maxProt requirements");
  auto it = find_if(
      config->segmentProtections,
      [&](const SegmentProtection &segprot) { return segprot.name == name; });
  if (it != config->segmentProtections.end())
    return it->maxProt;

  return initProt(name);
}

static uint32_t flags(StringRef name) {
  // If we ever implement shared cache output support, SG_READ_ONLY should not
  // be used for dylibs that can be placed in it.
  return name == segment_names::dataConst ? (uint32_t)SG_READ_ONLY : 0;
}

size_t OutputSegment::numNonHiddenSections() const {
  size_t count = 0;
  for (const OutputSection *osec : sections)
    count += (!osec->isHidden() ? 1 : 0);
  return count;
}

void OutputSegment::addOutputSection(OutputSection *osec) {
  inputOrder = std::min(inputOrder, osec->inputOrder);

  osec->parent = this;
  sections.push_back(osec);

  for (const SectionAlign &sectAlign : config->sectionAlignments)
    if (sectAlign.segName == name && sectAlign.sectName == osec->name)
      osec->align = sectAlign.align;
}

template <typename T, typename F> static auto compareByOrder(F ord) {
  return [=](T a, T b) { return ord(a) < ord(b); };
}

static int segmentOrder(OutputSegment *seg) {
  return StringSwitch<int>(seg->name)
      .Case(segment_names::pageZero, -4)
      .Case(segment_names::text, -3)
      .Case(segment_names::dataConst, -2)
      .Case(segment_names::data, -1)
      .Case(segment_names::llvm, std::numeric_limits<int>::max() - 1)
      // Make sure __LINKEDIT is the last segment (i.e. all its hidden
      // sections must be ordered after other sections).
      .Case(segment_names::linkEdit, std::numeric_limits<int>::max())
      .Default(seg->inputOrder);
}

static int sectionOrder(OutputSection *osec) {
  StringRef segname = osec->parent->name;
  // Sections are uniquely identified by their segment + section name.
  if (segname == segment_names::text) {
    if (osec->name == section_names::header)
      return -7;
    // `__text` needs to precede the other code sections since its
    // expected to be the largest. This means in effect that it will
    // be the section that determines whether we need thunks or not.
    if (osec->name == section_names::text)
      return -6;

    // Prioritize specific section ordering based on our knowledge. This ensures
    // that certain sections are placed in a particular order, even if they
    // are also categorized as code sections. This explicit ordering takes
    // precedence over the general code section ordering.
    int knownPriority =
        StringSwitch<int>(osec->name)
            .Case(section_names::stubs, -4)
            .Case(section_names::stubHelper, -3)
            .Case(section_names::objcStubs, -2)
            .Case(section_names::initOffsets, -1)
            .Case(section_names::unwindInfo,
                  std::numeric_limits<int>::max() - 1)
            .Case(section_names::ehFrame, std::numeric_limits<int>::max())
            .Default(0);

    if (knownPriority != 0)
      return knownPriority;

    // Ensure all code sections are contiguous with `__text` for thunk
    // calculations.
    if (sections::isCodeSection(osec->name, segment_names::text, osec->flags)) {
      return -5;
    }

    return osec->inputOrder;
  } else if (segname == segment_names::data ||
             segname == segment_names::dataConst) {
    // For each thread spawned, dyld will initialize its TLVs by copying the
    // address range from the start of the first thread-local data section to
    // the end of the last one. We therefore arrange these sections contiguously
    // to minimize the amount of memory used. Additionally, since zerofill
    // sections must be at the end of their segments, and since TLV data
    // sections can be zerofills, we end up putting all TLV data sections at the
    // end of the segment.
    switch (sectionType(osec->flags)) {
    case S_THREAD_LOCAL_VARIABLE_POINTERS:
      return std::numeric_limits<int>::max() - 3;
    case S_THREAD_LOCAL_REGULAR:
      return std::numeric_limits<int>::max() - 2;
    case S_THREAD_LOCAL_ZEROFILL:
      return std::numeric_limits<int>::max() - 1;
    case S_ZEROFILL:
      return std::numeric_limits<int>::max();
    default:
      return StringSwitch<int>(osec->name)
          .Case(section_names::got, -3)
          .Case(section_names::lazySymbolPtr, -2)
          .Case(section_names::const_, -1)
          .Default(osec->inputOrder);
    }
  } else if (segname == segment_names::linkEdit) {
    return StringSwitch<int>(osec->name)
        .Case(section_names::chainFixups, -11)
        .Case(section_names::rebase, -10)
        .Case(section_names::binding, -9)
        .Case(section_names::weakBinding, -8)
        .Case(section_names::lazyBinding, -7)
        .Case(section_names::export_, -6)
        .Case(section_names::functionStarts, -5)
        .Case(section_names::dataInCode, -4)
        .Case(section_names::symbolTable, -3)
        .Case(section_names::indirectSymbolTable, -2)
        .Case(section_names::stringTable, -1)
        .Case(section_names::codeSignature, std::numeric_limits<int>::max())
        .Default(osec->inputOrder);
  }
  // ZeroFill sections must always be the at the end of their segments:
  // dyld checks if a segment's file size is smaller than its in-memory
  // size to detect if a segment has zerofill sections, and if so it maps
  // the missing tail as zerofill.
  if (sectionType(osec->flags) == S_ZEROFILL)
    return std::numeric_limits<int>::max();
  return osec->inputOrder;
}

void OutputSegment::sortOutputSections() {
  // Must be stable_sort() to keep special sections such as
  // S_THREAD_LOCAL_REGULAR in input order.
  llvm::stable_sort(sections, compareByOrder<OutputSection *>(sectionOrder));
}

void OutputSegment::assignAddressesToStartEndSymbols() {
  for (Defined *d : segmentStartSymbols)
    d->value = addr;
  for (Defined *d : segmentEndSymbols)
    d->value = addr + vmSize;
}

void macho::sortOutputSegments() {
  llvm::stable_sort(outputSegments,
                    compareByOrder<OutputSegment *>(segmentOrder));
}

static DenseMap<StringRef, OutputSegment *> nameToOutputSegment;
std::vector<OutputSegment *> macho::outputSegments;

void macho::resetOutputSegments() {
  outputSegments.clear();
  nameToOutputSegment.clear();
}

static StringRef maybeRenameSegment(StringRef name) {
  auto newName = config->segmentRenameMap.find(name);
  if (newName != config->segmentRenameMap.end())
    return newName->second;
  return name;
}

OutputSegment *macho::getOrCreateOutputSegment(StringRef name) {
  name = maybeRenameSegment(name);

  OutputSegment *&segRef = nameToOutputSegment[name];
  if (segRef)
    return segRef;

  segRef = make<OutputSegment>();
  segRef->name = name;
  segRef->maxProt = maxProt(name);
  segRef->initProt = initProt(name);
  segRef->flags = flags(name);

  outputSegments.push_back(segRef);
  return segRef;
}

namespace {
class BoundaryTable;
struct BranchExtension;

// Allocated only for boundaries that have hosted a proposed extension.
struct BoundaryPlan {
  uint32_t reservedSize = 0;
  uint32_t plannedSize = 0;
  SmallVector<BranchExtension *, 2> extensions;
};
} // namespace

namespace lld::macho {
/// Models an extender insertion point after an input section.
///
/// \code
/// current isec (i)  planned extensions   padding      next isec (i+1)
/// |----------------|--- boundary (i) ---|............|
/// ^                ^                    ^
/// getInputVA()     getInputEndVA()      getNextExtensionVA()
/// \endcode
class InputSectionBoundary {
public:
  InputSectionBoundary(ConcatInputSection *isec, ConcatOutputSection *section)
      : isec(isec), section(section) {}

  uint64_t getInputEndVA() const { return getInputVA() + isec->getSize(); }
  uint64_t getNextExtensionVA(const BranchExtender &extender) const {
    return alignToPowerOf2(getInputEndVA() + getPlannedSize(),
                           extender.getAlign());
  }
  ConcatInputSection *const isec;
  ConcatOutputSection *const section;
  uint64_t getInputVA() const { return isec->parent->addr + isec->outSecOff; }

  // Space retained from prior proposals to account for extensions when
  // computing callsite and callee addresses for the next planning pass.
  uint32_t getReservedSize() const { return plan ? plan->reservedSize : 0; }
  // Space used by this pass's proposed extensions, including alignment padding.
  // Recomputed each pass; it may be smaller or larger than the reservation.
  uint32_t getPlannedSize() const { return plan ? plan->plannedSize : 0; }
  ArrayRef<BranchExtension *> getPlannedExtensions() const {
    return plan ? ArrayRef<BranchExtension *>(plan->extensions)
                : ArrayRef<BranchExtension *>();
  }

private:
  friend class ::BoundaryTable;
  BoundaryPlan *plan = nullptr;
};
} // namespace lld::macho

namespace {
// Prefer the first usable boundary in ascending (lowest) or descending
// (highest) address order.
enum class BoundaryPreference { lowest, highest };

// Tracks eligible boundaries in output-segment layout order and supports
// searching for extension placements.
class BoundaryTable {
public:
  BoundaryTable(ArrayRef<OutputSection *> sections,
                const OutputSectionAttributesMap &osecAttrs);

  auto begin() { return boundaries.begin(); }
  auto end() { return boundaries.end(); }
  bool empty() const { return boundaries.empty(); }
  size_t size() const { return boundaries.size(); }

  InputSectionBoundary *
  findExtenderPlacementInRange(uint64_t lowVA, uint64_t highVA,
                               BoundaryPreference preference,
                               const BranchExtender &extender);
  BoundaryPlan &getOrCreatePlan(InputSectionBoundary &boundary);
  void resetPlannedSpace() {
    for (const auto &plan : activePlans) {
      plan->plannedSize = 0;
      plan->extensions.clear();
    }
  }
  bool growReservations() {
    bool grew = false;
    for (const auto &plan : activePlans)
      if (plan->reservedSize < plan->plannedSize) {
        plan->reservedSize = plan->plannedSize;
        grew = true;
      }
    return grew;
  }

private:
  // Search coarse boundaries before the full table to concentrate extensions at
  // shared boundaries for locality.
  SmallVector<InputSectionBoundary, 0> boundaries;
  SmallVector<size_t, 0> coarseBoundaryIndices;
  // Each plan is registered once on first allocation. Inactive boundaries
  // need no per-pass reset or reservation-growth work.
  SmallVector<std::unique_ptr<BoundaryPlan>, 0> activePlans;
};

static bool inBranchRange(uint64_t from, uint64_t to) {
  return to >= from ? to - from <= target->forwardBranchRange
                    : from - to <= target->backwardBranchRange;
}

BoundaryTable::BoundaryTable(ArrayRef<OutputSection *> sections,
                             const OutputSectionAttributesMap &osecAttrs) {
  // Concentrate extensions at shared insertion points to improve their code
  // locality. Select these points from input sizes and alignment before any
  // extensions are reserved so the candidate set stays stable across retries.
  // The backend controls spacing; searches fall back to all input boundaries
  // when no coarse candidate fits, preserving placements in narrow ranges.
  uint64_t va = sections.front()->parent->addr;
  uint64_t lastVA = va;
  for (auto *osec : sections) {
    if (!osec->isNeeded())
      continue;
    va = alignToPowerOf2(va, osec->align);
    auto *section = dyn_cast<ConcatOutputSection>(osec);
    if (!section || !osecAttrs.lookup(osec).containsBranchSource) {
      va += osecAttrs.lookup(osec).size;
      continue;
    }
    uint64_t sectionVA = va;
    for (auto *isec : section->inputs) {
      boundaries.emplace_back(isec, section);
      va = sectionVA + alignToPowerOf2(va - sectionVA, isec->align) +
           isec->getSize();
      if (coarseBoundaryIndices.empty() ||
          va - lastVA >= target->coarseBoundarySpacing) {
        coarseBoundaryIndices.push_back(boundaries.size() - 1);
        lastVA = va;
      }
    }
  }
  assert(!boundaries.empty() && "segment must contain a code input section");
  if (coarseBoundaryIndices.back() != boundaries.size() - 1)
    coarseBoundaryIndices.push_back(boundaries.size() - 1);
}

BoundaryPlan &BoundaryTable::getOrCreatePlan(InputSectionBoundary &boundary) {
  if (!boundary.plan) {
    activePlans.push_back(std::make_unique<BoundaryPlan>());
    boundary.plan = activePlans.back().get();
  }
  return *boundary.plan;
}

template <typename BoundaryRange, typename Predicate>
static InputSectionBoundary *findInDirection(BoundaryRange &&boundaries,
                                             BoundaryPreference preference,
                                             Predicate predicate) {
  for (auto *boundary : llvm::reverse_conditionally(
           boundaries, preference == BoundaryPreference::highest))
    if (predicate(*boundary))
      return boundary;
  return nullptr;
}

InputSectionBoundary *
BoundaryTable::findExtenderPlacementInRange(uint64_t lowVA, uint64_t highVA,
                                            BoundaryPreference preference,
                                            const BranchExtender &extender) {
  if (lowVA > highVA)
    return nullptr;

  auto isUsable = [&](const InputSectionBoundary &boundary) {
    uint64_t va = boundary.getNextExtensionVA(extender);
    return lowVA <= va && va <= highVA;
  };

  auto coarseBegin = llvm::lower_bound(
      coarseBoundaryIndices, lowVA, [&](size_t index, uint64_t va) {
        return boundaries[index].getInputEndVA() < va;
      });
  auto coarseEnd = llvm::upper_bound(
      coarseBoundaryIndices, highVA, [&](uint64_t va, size_t index) {
        return va < boundaries[index].getInputEndVA();
      });
  auto coarseRange =
      llvm::map_range(llvm::make_range(coarseBegin, coarseEnd),
                      [&](size_t index) { return &boundaries[index]; });
  if (auto *boundary = findInDirection(coarseRange, preference, isUsable))
    return boundary;

  auto begin =
      partition_point(boundaries, [&](const InputSectionBoundary &boundary) {
        return boundary.getInputEndVA() < lowVA;
      });
  auto end =
      partition_point(boundaries, [&](const InputSectionBoundary &boundary) {
        return boundary.getInputEndVA() <= highVA;
      });
  return findInDirection(make_pointer_range(make_range(begin, end)), preference,
                         isUsable);
}

struct CallSite {
  uint64_t getVA() const {
    return isec.parent->addr + isec.outSecOff + reloc.offset;
  }

  const ConcatInputSection &isec;
  Relocation &reloc;
  // Selected first hop for this proposal. Null means a direct call.
  BranchExtension *extension = nullptr;
};

struct Callee {
  Callee(Symbol *sym, int64_t addend) : sym(sym), addend(addend) {}

  Symbol *const sym;
  const int64_t addend;
  MutableArrayRef<CallSite> callsites;
  // First callsite at or above the callee in the reserved layout.
  // Unknown callee VAs use zero.
  size_t splitIndex = 0;
  // Planned instances are rebuilt for each layout proposal.
  SmallVector<std::unique_ptr<BranchExtension>, 0> extensions;
  // Used for branch extension planning. If unknown, pessimistically choose
  // one hop of the enabled extender with the greatest reach. Otherwise,
  // optimistically try extenders in ascending order of reach.
  std::optional<uint64_t> va;
};

struct BranchExtension {
  BranchExtension(Callee &callee, InputSectionBoundary &boundary,
                  const BranchExtender &extender,
                  BranchExtension *inwardChainedExtension, uint64_t va)
      : callee(callee), boundary(boundary), extender(extender),
        inwardChainedExtension(inwardChainedExtension), va(va) {}

  size_t size() const { return extender.getSize(); }
  uint32_t align() const { return extender.getAlign(); }

  // The ultimate destination is needed to materialize the extension body.
  Callee &callee;

  // The insertion boundary connects the proposal to its owning text section.
  InputSectionBoundary &boundary;
  const BranchExtender &extender;

  // The next extension toward the callee. A null pointer means this extension
  // branches directly to the final target.
  BranchExtension *inwardChainedExtension;

  // The proposed or exact address is used for branch-range validation.
  uint64_t va = 0;
  ConcatInputSection *isec = nullptr;
  Defined *sym = nullptr;
};

// Canonical callee identity has two forms: <isec*, offset> for non-binding
// definitions, or <sym*, addend> for other targets.
using CalleeKey = std::pair<const void *, uint64_t>;

static CalleeKey makeCalleeKey(Symbol *sym, int64_t addend) {
  if (const auto *defined = dyn_cast<Defined>(sym);
      defined && !needsBinding(sym))
    return {defined->isec(), defined->value + addend};
  return {sym, static_cast<uint64_t>(addend)};
}

class BranchExtensionPlanner {
public:
  BranchExtensionPlanner(OutputSegment &segment,
                         const OutputSectionAttributesMap &osecAttrs);

  void run();

private:
  OutputSegment &segment;
  BoundaryTable boundaries;
  BumpPtrAllocator callsiteAllocator;
  SmallVector<std::unique_ptr<Callee>, 0> callees;
  // Identifies branch-source sections that can host extenders and provides
  // fixed sizes for other sections when evaluating tentative layouts.
  const OutputSectionAttributesMap &osecAttrs;
  // Direct relocs valid in reservation layout may be invalid in proposal.
  // Register these relocs to use extension so that reservation may be updated.
  DenseSet<const Relocation *> requiredExtenderRelocs;

  template <typename Fn> void forEachExtenderHostOutputSections(Fn fn) {
    for (auto *osec : segment.getSections())
      if (auto *section = dyn_cast<ConcatOutputSection>(osec);
          section && section->isNeeded() &&
          osecAttrs.lookup(osec).containsBranchSource)
        fn(section);
  }

  std::optional<uint64_t> resolveCalleeVA(const Callee &) const;
  void
  evaluateLayout(function_ref<void(const InputSectionBoundary &, uint64_t &)>
                     preserveSpaceForBoundary);
  void computeCallsiteSplits();

  InputSectionBoundary *
  findChainBoundaryTowardCallsite(uint64_t callVA, uint64_t chainTipVA,
                                  const BranchExtender &extender);
  BranchExtension *planRoute(Callee &callee, uint64_t callVA,
                             BoundaryPreference preference,
                             const BranchExtender &extender);
  template <typename CallSiteRange>
  void planSide(Callee &callee, CallSiteRange &&callsites,
                BoundaryPreference preference);
  void planExtensionsForCallee(Callee &callee);
  void buildProposal();

  bool recoverFromInvalidProposal();
  bool promoteOutOfRangeDirectCalls();
  bool isLayoutValid();
  void instantiateExtension(BranchExtension &extension, StringRef name,
                            bool externalSymbol);
  void finalize();
};

// Bind branch targets and collect their callsites once. Identity and callsite
// locations survive retries; addresses and routes belong to each proposal.
BranchExtensionPlanner::BranchExtensionPlanner(
    OutputSegment &segment, const OutputSectionAttributesMap &osecAttrs)
    : segment(segment), boundaries(segment.getSections(), osecAttrs),
      osecAttrs(osecAttrs) {
  static_assert(std::is_trivially_destructible_v<CallSite>);
  // Stage calls in one temporary vector so we can allocate each callee's
  // callsite array at its final size, avoiding repeated allocation and copying
  // as individual callee vectors grow. This trades higher peak memory usage
  // for less allocation churn and allows parallel initialization of the final
  // arrays. Recorded indices preserve callsite order.
  struct PendingCall {
    // The `index`-th callsite of `callee` defined by `isec` and `reloc`
    Callee *callee;
    ConcatInputSection *isec;
    Relocation *reloc;
    size_t index;
  };
  SmallVector<PendingCall, 0> pendingCalls;
  DenseMap<CalleeKey, std::pair<Callee *, size_t>> calleesByKey;
  for (auto *osec : segment.getSections()) {
    auto *section = dyn_cast<ConcatOutputSection>(osec);
    if (!section || !section->isNeeded())
      continue;
    uint64_t offset = 0;
    SmallVector<Relocation *, 40> branches;
    for (auto *isec : section->inputs) {
      // ConcatOutputSections that cannot contain extenders defer input layout
      // until finalization, and evaluateLayout() skips their inputs. Initialize
      // their offsets here so planning can resolve their callsites and callees.
      // For sections that can contain extenders, evaluateLayout() replaces
      // them.
      isec->outSecOff = alignToPowerOf2(offset, isec->align);
      offset = isec->outSecOff + isec->getSize();
      branches.clear();
      for (auto &reloc : isec->relocs)
        if (target->hasAttr(reloc.type, RelocAttrBits::BRANCH))
          branches.push_back(&reloc);
      // Preserve callsite order for canonical target selection and demand.
      stable_sort(branches, [](const Relocation *a, const Relocation *b) {
        return a->offset < b->offset;
      });
      for (auto *reloc : branches) {
        Symbol *sym = cast<Symbol *>(reloc->referent);
        auto &[callee, count] = calleesByKey[makeCalleeKey(sym, reloc->addend)];
        if (!callee) {
          callees.push_back(std::make_unique<Callee>(sym, reloc->addend));
          callee = callees.back().get();
        }
        pendingCalls.push_back({callee, isec, reloc, count++});
      }
    }
  }

  for (auto &[callee, callsiteCount] : make_second_range(calleesByKey)) {
    CallSite *storage = callsiteAllocator.Allocate<CallSite>(callsiteCount);
    callee->callsites = {storage, callsiteCount};
  }

  parallelForEach(pendingCalls, [](const PendingCall &call) {
    new (call.callee->callsites.data() + call.index)
        CallSite{*call.isec, *call.reloc};
  });
}

// Resolve callees against the current layout, including stubs anywhere in the
// segment, without relying on their addresses having been finalized.
std::optional<uint64_t>
BranchExtensionPlanner::resolveCalleeVA(const Callee &callee) const {
  // Try to resolve targets in the current segment, plus absolute symbols and
  // finalized stubs; return nullopt for other targets.
  Symbol *sym = callee.sym;
  int64_t addend = callee.addend;

  if (sym->isInStubs() && addend == 0) {
    if (in.stubs->isFinal)
      return sym->getStubVA();
    if (in.stubs->isNeeded() && in.stubs->parent == &segment)
      return in.stubs->addr + uint64_t(sym->stubsIndex) * target->stubSize;
    return std::nullopt;
  }

  auto *defined = dyn_cast<Defined>(sym);
  if (!defined)
    return std::nullopt;

  if (in.objcStubs && defined->isec() == in.objcStubs->isec &&
      in.objcStubs->isNeeded()) {
    if (in.objcStubs->isec->isFinal)
      return defined->getVA() + addend;
    if (in.objcStubs->parent == &segment)
      return in.objcStubs->addr + defined->value + addend;
    return std::nullopt;
  }

  if (defined->isAbsolute())
    return defined->getVA() + addend;
  if (auto *isec = dyn_cast_or_null<ConcatInputSection>(defined->isec());
      isec && isec->parent && isec->parent->isNeeded() &&
      isec->parent->parent == &segment)
    return isec->parent->addr + isec->outSecOff + defined->value + addend;
  return std::nullopt;
}

// Recompute section and input addresses, then resolve callee VAs.
// Addresses based on reserved space feed proposal building; addresses based
// on planned extensions feed validation.
void BranchExtensionPlanner::evaluateLayout(
    function_ref<void(const InputSectionBoundary &, uint64_t &)>
        preserveSpaceForBoundary) {
  uint64_t va = segment.addr;
  auto boundaryIt = boundaries.begin();
  for (auto *osec : segment.getSections()) {
    if (!osec->isNeeded())
      continue;
    va = alignToPowerOf2(va, osec->align);
    uint64_t sectionVA = va;
    osec->addr = sectionVA;
    if (!osecAttrs.lookup(osec).containsBranchSource) {
      va += osecAttrs.lookup(osec).size;
      continue;
    }

    while (boundaryIt != boundaries.end() && boundaryIt->section == osec) {
      auto &boundary = *boundaryIt++;

      va = sectionVA + alignToPowerOf2(va - sectionVA, boundary.isec->align);
      boundary.isec->outSecOff = va - sectionVA;
      va += boundary.isec->getSize();

      preserveSpaceForBoundary(boundary, va);
    }
  }
  assert(boundaryIt == boundaries.end());

  parallelForEach(callees, [&](const auto &callee) {
    callee->va = resolveCalleeVA(*callee);
  });
}

void BranchExtensionPlanner::computeCallsiteSplits() {
  parallelForEach(callees, [](const auto &callee) {
    callee->splitIndex = 0;
    if (!callee->va)
      return;
    auto split =
        llvm::partition_point(callee->callsites, [&](const CallSite &callsite) {
          return callsite.getVA() < *callee->va;
        });
    callee->splitIndex = split - callee->callsites.begin();
  });
}

// Find a boundary between the callsite and chain tip where a new extender
// can reach the tip, preferring placements toward the callsite.
InputSectionBoundary *BranchExtensionPlanner::findChainBoundaryTowardCallsite(
    uint64_t callVA, uint64_t chainTipVA, const BranchExtender &extender) {
  if (callVA == chainTipVA)
    return nullptr;
  bool upper = callVA > chainTipVA;
  auto [low, high] = extender.getSourceRange(chainTipVA);
  // callVA < chainTipVA: [low, high] = sourceRange ∩ [callVA + 1, chainTipVA
  // - 1]. chainTipVA < callVA: [low, high] = sourceRange ∩ [chainTipVA + 1,
  // callVA - 1].
  low = std::max(low, SaturatingAdd(std::min(callVA, chainTipVA), uint64_t{1}));
  high = std::min(high, std::max(callVA, chainTipVA) - 1);
  return boundaries.findExtenderPlacementInRange(
      low, high,
      upper ? BoundaryPreference::highest : BoundaryPreference::lowest,
      extender);
}

BranchExtension *
BranchExtensionPlanner::planRoute(Callee &callee, uint64_t callVA,
                                  BoundaryPreference preference,
                                  const BranchExtender &extender) {
  if (extender.maxHops == 0)
    return nullptr;
  if (callee.va) {
    uint64_t distance =
        callVA < *callee.va ? *callee.va - callVA : callVA - *callee.va;
    // Optimistic: placement boundaries and alignment can require more hops.
    if (distance / extender.maxReach > extender.maxHops)
      return nullptr;
  }
  // An unknown target cannot anchor a chain. Use a single hop even when the
  // configured limit permits more; final relocations validate its reach.
  const uint32_t maxHops = callee.va ? extender.maxHops : 1;
  uint64_t tipVA = callee.va.value_or(0);
  // The head of the extension chain.
  BranchExtension *tip = nullptr;
  SmallVector<std::unique_ptr<BranchExtension>, 4> route;
  for (uint32_t depth = 0; depth < maxHops; ++depth) {
    InputSectionBoundary *boundary;
    if (maxHops > 1) {
      boundary = findChainBoundaryTowardCallsite(callVA, tipVA, extender);
    } else {
      auto [low, high] = callee.va
                             ? extender.getSourceRange(tipVA)
                             : std::pair<uint64_t, uint64_t>{0, UINT64_MAX};
      // Known callee VA: [lowVA, highVA] = callsiteReachableRange ∩
      // extenderSourceRange(tipVA). Unknown callee VA: [lowVA, highVA] =
      // callsiteReachableRange.
      boundary = boundaries.findExtenderPlacementInRange(
          std::max(low, callVA - std::min(callVA, target->backwardBranchRange)),
          std::min(high, SaturatingAdd(callVA, target->forwardBranchRange)),
          preference, extender);
    }
    if (!boundary)
      break;
    tipVA = boundary->getNextExtensionVA(extender);
    route.push_back(std::make_unique<BranchExtension>(callee, *boundary,
                                                      extender, tip, tipVA));
    tip = route.back().get();
    if (inBranchRange(callVA, tipVA)) {
      // Each hop uses a distinct boundary. Commit only the complete route.
      for (auto &extension : route) {
        auto &plan = boundaries.getOrCreatePlan(extension->boundary);
        plan.plannedSize = extension->va + extension->size() -
                           extension->boundary.getInputEndVA();
        plan.extensions.push_back(extension.get());
        callee.extensions.push_back(std::move(extension));
      }
      return tip;
    }
  }
  return nullptr;
}

template <typename CallSiteRange>
void BranchExtensionPlanner::planSide(Callee &callee, CallSiteRange &&callsites,
                                      BoundaryPreference preference) {
  std::optional<uint64_t> calleeVA = callee.va;
  // Without a target VA we cannot plan a chain. Pessimistically choose the
  // enabled extender with the greatest reach and use one hop. Place it toward
  // the remaining callsites to maximize sharing. Final relocations check reach.
  ArrayRef<const BranchExtender *> candidates = target->branchExtenders;
  if (!calleeVA) {
    while (!candidates.empty() && candidates.back()->maxHops == 0)
      candidates = candidates.drop_back();
    if (!candidates.empty())
      candidates = candidates.take_back(1);
  }
  for (auto &callsite : callsites) {
    // Clear the previous proposal's extension so a newly reachable direct
    // call does not retain a stale pointer.
    callsite.extension = nullptr;
    uint64_t callVA = callsite.getVA();
    if (calleeVA && inBranchRange(callVA, *calleeVA) &&
        !requiredExtenderRelocs.contains(&callsite.reloc))
      continue;

    for (const BranchExtender *extender : candidates) {
      if (extender->maxHops == 0)
        continue;
      for (const auto &extension : callee.extensions)
        if (&extension->extender == extender &&
            inBranchRange(callVA, extension->va)) {
          callsite.extension = extension.get();
          break;
        }
      if (!callsite.extension)
        callsite.extension = planRoute(callee, callVA, preference, *extender);
      if (callsite.extension)
        break;
    }
    if (!callsite.extension)
      fatal("cannot place branch extension for " + toString(*callee.sym));
  }
}

void BranchExtensionPlanner::planExtensionsForCallee(Callee &callee) {
  std::optional<uint64_t> calleeVA = callee.va;
  auto lower = callee.callsites.take_front(callee.splitIndex);
  auto upper = callee.callsites.drop_front(callee.splitIndex);

  // Sweep each side from farthest to nearest. Push single-hop extensions toward
  // the callsites still to be visited so they can share them.
  planSide(callee, lower, BoundaryPreference::highest);
  if (calleeVA)
    planSide(callee, reverse(upper), BoundaryPreference::lowest);
  else
    // Without a target VA, use a deterministic ascending sweep.
    planSide(callee, upper, BoundaryPreference::highest);
}

void BranchExtensionPlanner::buildProposal() {
  for (const auto &callee : callees)
    callee->extensions.clear();
  boundaries.resetPlannedSpace();

  // use reservedSize to update boundaries and callees' VA
  evaluateLayout([](const InputSectionBoundary &boundary, uint64_t &va) {
    va += boundary.getReservedSize();
  });

  computeCallsiteSplits();

  // once VAs are updated, plan branch extensions
  for (const auto &callee : callees)
    planExtensionsForCallee(*callee);

  // update VAs to reflect actual planned extensions
  evaluateLayout([](const InputSectionBoundary &boundary, uint64_t &va) {
    for (auto *extension : boundary.getPlannedExtensions()) {
      va = alignToPowerOf2(va, extension->align());
      extension->va = va;
      va += extension->size();
    }
  });
}

bool BranchExtensionPlanner::isLayoutValid() {
  std::atomic_bool valid{true};
  parallelForEach(callees, [&](const auto &callee) {
    if (!valid.load(std::memory_order_relaxed))
      return;
    std::optional<uint64_t> calleeVA = callee->va;

    // Every extension must still reach its next target after layout changes.
    for (const auto &extension : callee->extensions) {
      // Unknown final targets are checked by relocation validation once all
      // segments have addresses. Callsite-to-extension reach is checked below.
      if (!calleeVA && !extension->inwardChainedExtension)
        continue;
      uint64_t targetVA = extension->inwardChainedExtension
                              ? extension->inwardChainedExtension->va
                              : *calleeVA;
      if (!extension->extender.canReach(extension->va, targetVA)) {
        valid.store(false, std::memory_order_relaxed);
        return;
      }
    }

    // Every callsite must reach its selected target: an extension for a
    // rewritten relocation or the callee for a direct relocation.
    for (const auto &callsite : callee->callsites) {
      if (callsite.extension) {
        if (!inBranchRange(callsite.getVA(), callsite.extension->va)) {
          valid.store(false, std::memory_order_relaxed);
          return;
        }
      } else if (calleeVA && !inBranchRange(callsite.getVA(), *calleeVA)) {
        valid.store(false, std::memory_order_relaxed);
        return;
      }
    }
  });
  return valid.load(std::memory_order_relaxed);
}

void BranchExtensionPlanner::instantiateExtension(BranchExtension &extension,
                                                  StringRef name,
                                                  bool externalSymbol) {
  auto *section = extension.boundary.section;
  auto *isec = makeSyntheticInputSection(segment.name, section->name);
  isec->parent = section;
  Defined *sym;
  if (externalSymbol)
    sym = symtab->addDefined(
        name, /*file=*/nullptr, isec, /*value=*/0, /*size=*/extension.size(),
        /*isWeakDef=*/false, /*isPrivateExtern=*/true,
        /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
        /*isWeakDefCanBeHidden=*/false);
  else
    sym = make<Defined>(
        name, /*file=*/nullptr, isec, /*value=*/0, /*size=*/extension.size(),
        /*isWeakDef=*/false, /*isExternal=*/false,
        /*isPrivateExtern=*/true, /*includeInSymtab=*/true,
        /*isReferencedDynamically=*/false, /*noDeadStrip=*/false,
        /*canOverrideWeakDef=*/false);
  sym->used = true;
  sym->branchExtensionThunk = true;
  extension.isec = isec;
  extension.sym = sym;
}

void BranchExtensionPlanner::finalize() {
  parallelForEach(callees, [&](const auto &callee) {
    auto byVA = [](const auto &lhs, const auto &rhs) {
      return lhs->va < rhs->va;
    };
    sort(callee->extensions, byVA);
  });

  // Declare every extension before populating any body so chains may
  // refer an existing extension via the corresponding isec.
  for (const auto &callee : callees) {
    Symbol *targetSym = callee->sym;
    int64_t addend = callee->addend;
    bool external = true;
    if (const auto *defined = dyn_cast<Defined>(targetSym))
      external = defined->isExternal();
    int sequence = 0;
    for (const auto &extension : callee->extensions) {
      SmallString<128> nameStorage;
      // Each segment planner starts numbering at zero. Qualify global symbols
      // outside __TEXT so extenders in different segments cannot collide.
      if (external && segment.name != segment_names::text) {
        nameStorage += segment.name;
        nameStorage += ".";
      }
      nameStorage += targetSym->getName();
      if (addend != 0) {
        if (addend > 0)
          nameStorage += "+";
        nameStorage += Twine(addend).str();
      }
      nameStorage += extension->extender.getSuffix();
      nameStorage += Twine(sequence++).str();
      StringRef name = saver().save(StringRef(nameStorage));
      instantiateExtension(*extension, name, external);
    }
  }

  auto boundaryIt = boundaries.begin();
  // Each host starts at its planned address, accounting for intervening data
  // and alignment. Record each section's boundary range for finalization.
  forEachExtenderHostOutputSections([&](ConcatOutputSection *section) {
    uint64_t va = section->addr;
    auto boundaryStart = boundaryIt;
    while (boundaryIt != boundaries.end() && boundaryIt->section == section) {
      const auto &boundary = *boundaryIt++;
      for (auto *extension : boundary.getPlannedExtensions()) {
        Symbol *bodyTarget = extension->inwardChainedExtension
                                 ? extension->inwardChainedExtension->sym
                                 : extension->callee.sym;
        int64_t bodyAddend =
            extension->inwardChainedExtension ? 0 : extension->callee.addend;
        extension->extender.populate(extension->isec, bodyTarget, bodyAddend);
      }
      va = boundary.getInputEndVA() + boundary.getPlannedSize();
    }
    section->setBoundaries(ArrayRef(boundaryStart, boundaryIt),
                           va - section->addr);
  });
  assert(boundaryIt == boundaries.end());

  parallelForEach(callees, [](const auto &callee) {
    for (const auto &callsite : callee->callsites)
      if (callsite.extension) {
        callsite.reloc.referent =
            static_cast<Symbol *>(callsite.extension->sym);
        callsite.reloc.addend = 0;
      }
  });
}

bool BranchExtensionPlanner::promoteOutOfRangeDirectCalls() {
  bool promoted = false;
  for (const auto &callee : callees) {
    std::optional<uint64_t> calleeVA = callee->va;
    if (!calleeVA)
      continue;

    for (const auto &callsite : callee->callsites)
      if (!callsite.extension && !inBranchRange(callsite.getVA(), *calleeVA)) {
        bool inserted = requiredExtenderRelocs.insert(&callsite.reloc).second;
        assert(inserted && "direct call already requires an extension");
        promoted |= inserted;
      }
  }
  return promoted;
}

// A proposed layout is invalid if an extension cannot reach its next
// target or a callsite cannot reach its selected target.
bool BranchExtensionPlanner::recoverFromInvalidProposal() {
  // The extensions in a rejected proposal are discarded before replanning, so
  // grow the reservations for those extensions to preserve their space.
  if (boundaries.growReservations()) {
    // Registered callsites are specific to a fixed reservation layout.
    // Reconsider direct calls from scratch after growth changes that layout.
    requiredExtenderRelocs.clear();
    return true;
  }
  // If a direct callsite-to-callee edge becomes invalid, routing the call
  // through an extension may recover the layout.
  if (promoteOutOfRangeDirectCalls())
    return true;
  return false;
}

void BranchExtensionPlanner::run() {
  TimeTraceScope timeScope("Branch extension");
  constexpr size_t maxPassCount = 30;
  size_t pass = 1;
  for (; pass <= maxPassCount; ++pass) {
    buildProposal();
    if (isLayoutValid())
      break;
    if (!recoverFromInvalidProposal())
      fatal("branch extension can't recover from failure");
  }
  if (pass > maxPassCount)
    fatal("branch extension did not converge");
  finalize();
  size_t extensionTargetCount = count_if(
      callees, [&](const auto &callee) { return !callee->extensions.empty(); });
  size_t extensionCount = 0;
  for (const InputSectionBoundary &boundary : boundaries)
    extensionCount += boundary.getPlannedExtensions().size();
  log("branch extender for " + segment.name + ": passes = " +
      std::to_string(pass) + ", inputs = " + std::to_string(boundaries.size()) +
      ", targets = " + std::to_string(extensionTargetCount) +
      ", total extenders = " + std::to_string(extensionCount));
}
} // namespace

void ConcatOutputSection::setBoundaries(
    ArrayRef<InputSectionBoundary> boundaries, uint64_t expectedSize) {
  assert(extenders.empty() && "extenders have already been finalized");
  assert(!plannedSize && "boundaries have already been set");
  this->boundaries = boundaries;
  plannedSize = expectedSize;
}

void ConcatOutputSection::finalizeContents() {
  assert(size == 0 && fileSize == 0 && extenders.empty() &&
         "concat section has already been finalized");
  if (plannedSize) {
    assert(boundaries.size() == inputs.size());
    for (const InputSectionBoundary &boundary : boundaries) {
      finalizeOne(boundary.isec);
      for (const auto *extension : boundary.getPlannedExtensions()) {
        finalizeOne(extension->isec);
        extenders.push_back(extension->isec);
      }
    }
    assert(size == *plannedSize &&
           "finalized section size differs from extender plan");
    boundaries = {};
  } else {
    for (auto *isec : inputs)
      finalizeOne(isec);
  }
}

bool OutputSegment::needsBranchExtension(
    const OutputSectionAttributesMap &osecAttrs) const {
  assert(target->needsBranchExtension());
  // Estimate branch space from the start of fist section to the
  // end of the last section which contains both a branch source or
  // branch target, including everything between them.
  bool canHostExtenders = false;
  uint64_t firstCodeVA = UINT64_MAX;
  uint64_t lastCodeEndVA = 0;
  uint64_t endVA = addr;
  for (const OutputSection *osec : sections) {
    if (!osec->isNeeded())
      continue;
    auto attrs = osecAttrs.lookup(osec);
    canHostExtenders |= attrs.containsBranchSource;
    endVA = alignToPowerOf2(endVA, osec->align);
    uint64_t sectionVA = endVA;
    endVA += attrs.size;
    if (attrs.containsBranchSource || attrs.containsBranchTarget) {
      firstCodeVA = std::min(firstCodeVA, sectionVA);
      lastCodeEndVA = endVA;
    }
  }
  if (!canHostExtenders || firstCodeVA == UINT64_MAX)
    return false;
  // Include intervening data and alignment, but not data outside the code span.
  uint64_t span = lastCodeEndVA - firstCodeVA;
  return span > target->backwardBranchRange ||
         span > target->forwardBranchRange;
}

void OutputSegment::computeOsecAttrs(OutputSectionAttributesMap &osecAttrs) {
  TimeTraceScope timeScope("Compute section sizes and branch endpoints");
  struct SectionBucket {
    const OutputSection *section;
    OutputSectionAttributes attrs;
    DenseSet<const OutputSection *> branchTargets;
  };

  SmallVector<SectionBucket, 0> buckets;
  buckets.reserve(sections.size());
  for (const OutputSection *osec : sections)
    if (osec->isNeeded())
      buckets.push_back({osec, {}, {}});

  // Each worker owns its section's size, source flag, and target set.
  parallelForEach(buckets, [](SectionBucket &bucket) {
    auto *section = dyn_cast<ConcatOutputSection>(bucket.section);
    if (!section) {
      bucket.attrs.size = bucket.section->getSize();
      return;
    }

    uint64_t size = 0;
    for (const auto *isec : section->inputs) {
      size = alignToPowerOf2(size, isec->align) + isec->getSize();
      for (const auto &reloc : isec->relocs) {
        if (!target->hasAttr(reloc.type, RelocAttrBits::BRANCH))
          continue;

        bucket.attrs.containsBranchSource = true;
        Symbol *sym = cast<Symbol *>(reloc.referent);
        if (sym->isInStubs() && reloc.addend == 0) {
          bucket.branchTargets.insert(in.stubs);
        } else if (auto *defined = dyn_cast<Defined>(sym)) {
          if (auto *targetIsec = defined->isec())
            bucket.branchTargets.insert(targetIsec->parent);
        }
      }
    }
    bucket.attrs.size = size;
  });

  // Populate attributes before merging cross-section target flags.
  osecAttrs.reserve(buckets.size());
  for (const SectionBucket &bucket : buckets)
    osecAttrs[bucket.section] = bucket.attrs;
  for (const SectionBucket &bucket : buckets)
    for (const OutputSection *osec : bucket.branchTargets)
      osecAttrs[osec].containsBranchTarget = true;
}

void OutputSegment::finalize(const OutputSectionAttributesMap *osecAttrs) {
  const bool supportsBranchExtension = target->needsBranchExtension();

  const bool planExtenders =
      supportsBranchExtension && osecAttrs && needsBranchExtension(*osecAttrs);
  std::optional<BranchExtensionPlanner> planner;
  if (planExtenders)
    planner.emplace(*this, *osecAttrs).run();

  if (osecAttrs) {
    // Finalize the branch-source sections deferred by Writer.
    for (OutputSection *osec : sections) {
      if (!osec->isNeeded() || !osecAttrs->lookup(osec).containsBranchSource)
        continue;
      if (auto *section = dyn_cast<ConcatOutputSection>(osec))
        section->finalizeContents();
    }
  }

  uint64_t sectionAddr = addr;
  uint64_t sectionFileOff = fileOff;
  for (OutputSection *osec : sections) {
    if (!osec->isNeeded())
      continue;
    sectionAddr = alignToPowerOf2(sectionAddr, osec->align);
    sectionFileOff = alignToPowerOf2(sectionFileOff, osec->align);
    if (planExtenders && osecAttrs->lookup(osec).containsBranchSource)
      assert(osec->addr == sectionAddr);
    osec->addr = sectionAddr;
    osec->fileOff = isZeroFill(osec->flags) ? 0 : sectionFileOff;
    osec->finalize();
    osec->assignAddressesToStartEndSymbols();
    sectionAddr += osec->getSize();
    sectionFileOff += osec->getFileSize();
  }
  vmSize = sectionAddr - addr;
  fileSize = sectionFileOff - fileOff;
}
