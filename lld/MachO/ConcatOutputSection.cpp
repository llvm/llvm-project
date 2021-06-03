//===- ConcatOutputSection.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConcatOutputSection.h"
#include "Config.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/ScopedPrinter.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

void ConcatOutputSection::addInput(InputSection *input) {
  if (inputs.empty()) {
    align = input->align;
    flags = input->flags;
  } else {
    align = std::max(align, input->align);
    mergeFlags(input);
  }
  inputs.push_back(input);
  input->parent = this;
}

// Branch-range extension can be implemented in two ways, either through ...
//
// (1) Branch islands: Single branch instructions (also of limited range),
//     that might be chained in multiple hops to reach the desired
//     destination. On ARM64, as 16 branch islands are needed to hop between
//     opposite ends of a 2 GiB program. LD64 uses branch islands exclusively,
//     even when it needs excessive hops.
//
// (2) Thunks: Instruction(s) to load the destination address into a scratch
//     register, followed by a register-indirect branch. Thunks are
//     constructed to reach any arbitrary address, so need not be
//     chained. Although thunks need not be chained, a program might need
//     multiple thunks to the same destination distributed throughout a large
//     program so that all call sites can have one within range.
//
// The optimal approach is to mix islands for distinations within two hops,
// and use thunks for destinations at greater distance. For now, we only
// implement thunks. TODO: Adding support for branch islands!
//
// Internally -- as expressed in LLD's data structures -- a
// branch-range-extension thunk comprises ...
//
// (1) new Defined privateExtern symbol for the thunk named
//     <FUNCTION>.thunk.<SEQUENCE>, which references ...
// (2) new InputSection, which contains ...
// (3.1) new data for the instructions to load & branch to the far address +
// (3.2) new Relocs on instructions to load the far address, which reference ...
// (4.1) existing Defined extern symbol for the real function in __text, or
// (4.2) existing DylibSymbol for the real function in a dylib
//
// Nearly-optimal thunk-placement algorithm features:
//
// * Single pass: O(n) on the number of call sites.
//
// * Accounts for the exact space overhead of thunks - no heuristics
//
// * Exploits the full range of call instructions - forward & backward
//
// Data:
//
// * DenseMap<Symbol *, ThunkInfo> thunkMap: Maps the function symbol
//   to its thunk bookkeeper.
//
// * struct ThunkInfo (bookkeeper): Call instructions have limited range, and
//   distant call sites might be unable to reach the same thunk, so multiple
//   thunks are necessary to serve all call sites in a very large program. A
//   thunkInfo stores state for all thunks associated with a particular
//   function: (a) thunk symbol, (b) input section containing stub code, and
//   (c) sequence number for the active thunk incarnation. When an old thunk
//   goes out of range, we increment the sequence number and create a new
//   thunk named <FUNCTION>.thunk.<SEQUENCE>.
//
// * A thunk incarnation comprises (a) private-extern Defined symbol pointing
//   to (b) an InputSection holding machine instructions (similar to a MachO
//   stub), and (c) Reloc(s) that reference the real function for fixing-up
//   the stub code.
//
// * std::vector<InputSection *> MergedInputSection::thunks: A vector parallel
//   to the inputs vector. We store new thunks via cheap vector append, rather
//   than costly insertion into the inputs vector.
//
// Control Flow:
//
// * During address assignment, MergedInputSection::finalize() examines call
//   sites by ascending address and creates thunks.  When a function is beyond
//   the range of a call site, we need a thunk. Place it at the largest
//   available forward address from the call site. Call sites increase
//   monotonically and thunks are always placed as far forward as possible;
//   thus, we place thunks at monotonically increasing addresses. Once a thunk
//   is placed, it and all previous input-section addresses are final.
//
// * MergedInputSection::finalize() and MergedInputSection::writeTo() merge
//   the inputs and thunks vectors (both ordered by ascending address), which
//   is simple and cheap.

DenseMap<Symbol *, ThunkInfo> lld::macho::thunkMap;

// Determine whether we need thunks, which depends on the target arch -- RISC
// (i.e., ARM) generally does because it has limited-range branch/call
// instructions, whereas CISC (i.e., x86) generally doesn't. RISC only needs
// thunks for programs so large that branch source & destination addresses
// might differ more than the range of branch instruction(s).
bool ConcatOutputSection::needsThunks() const {
  if (!target->usesThunks())
    return false;
  uint64_t isecAddr = addr;
  for (InputSection *isec : inputs)
    isecAddr = alignTo(isecAddr, isec->align) + isec->getSize();
  if (isecAddr - addr + in.stubs->getSize() <= target->branchRange)
    return false;
  // Yes, this program is large enough to need thunks.
  for (InputSection *isec : inputs) {
    for (Reloc &r : isec->relocs) {
      if (!target->hasAttr(r.type, RelocAttrBits::BRANCH))
        continue;
      auto *sym = r.referent.get<Symbol *>();
      // Pre-populate the thunkMap and memoize call site counts for every
      // InputSection and ThunkInfo. We do this for the benefit of
      // ConcatOutputSection::estimateStubsInRangeVA()
      ThunkInfo &thunkInfo = thunkMap[sym];
      // Knowing ThunkInfo call site count will help us know whether or not we
      // might need to create more for this referent at the time we are
      // estimating distance to __stubs in .
      ++thunkInfo.callSiteCount;
      // Knowing InputSection call site count will help us avoid work on those
      // that have no BRANCH relocs.
      ++isec->callSiteCount;
    }
  }
  return true;
}

// Since __stubs is placed after __text, we must estimate the address
// beyond which stubs are within range of a simple forward branch.
uint64_t ConcatOutputSection::estimateStubsInRangeVA(size_t callIdx) const {
  uint64_t branchRange = target->branchRange;
  size_t endIdx = inputs.size();
  InputSection *isec = inputs[callIdx];
  uint64_t isecVA = isec->getVA();
  // Tally the non-stub functions which still have call sites
  // remaining to process, which yields the maximum number
  // of thunks we might yet place.
  size_t maxPotentialThunks = 0;
  for (auto &tp : thunkMap) {
    ThunkInfo &ti = tp.second;
    maxPotentialThunks +=
        !tp.first->isInStubs() && ti.callSitesUsed < ti.callSiteCount;
  }
  // Tally the total size of input sections remaining to process.
  uint64_t isecEnd = isec->getVA();
  for (size_t i = callIdx; i < endIdx; i++) {
    InputSection *isec = inputs[i];
    isecEnd = alignTo(isecEnd, isec->align) + isec->getSize();
  }
  // Estimate the address after which call sites can safely call stubs
  // directly rather than through intermediary thunks.
  uint64_t stubsInRangeVA = isecEnd + maxPotentialThunks * target->thunkSize +
                            in.stubs->getSize() - branchRange;
  log("thunks = " + std::to_string(thunkMap.size()) +
      ", potential = " + std::to_string(maxPotentialThunks) +
      ", stubs = " + std::to_string(in.stubs->getSize()) + ", isecVA = " +
      to_hexString(isecVA) + ", threshold = " + to_hexString(stubsInRangeVA) +
      ", isecEnd = " + to_hexString(isecEnd) +
      ", tail = " + to_hexString(isecEnd - isecVA) +
      ", slop = " + to_hexString(branchRange - (isecEnd - isecVA)));
  return stubsInRangeVA;
}

void ConcatOutputSection::finalize() {
  uint64_t isecAddr = addr;
  uint64_t isecFileOff = fileOff;
  auto finalizeOne = [&](InputSection *isec) {
    isecAddr = alignTo(isecAddr, isec->align);
    isecFileOff = alignTo(isecFileOff, isec->align);
    isec->outSecOff = isecAddr - addr;
    isec->outSecFileOff = isecFileOff - fileOff;
    isec->isFinal = true;
    isecAddr += isec->getSize();
    isecFileOff += isec->getFileSize();
  };

  if (!needsThunks()) {
    for (InputSection *isec : inputs)
      finalizeOne(isec);
    size = isecAddr - addr;
    fileSize = isecFileOff - fileOff;
    return;
  }

  uint64_t branchRange = target->branchRange;
  uint64_t stubsInRangeVA = TargetInfo::outOfRangeVA;
  size_t thunkSize = target->thunkSize;
  size_t relocCount = 0;
  size_t callSiteCount = 0;
  size_t thunkCallCount = 0;
  size_t thunkCount = 0;

  // inputs[finalIdx] is for finalization (address-assignment)
  size_t finalIdx = 0;
  // Kick-off by ensuring that the first input section has an address
  for (size_t callIdx = 0, endIdx = inputs.size(); callIdx < endIdx;
       ++callIdx) {
    if (finalIdx == callIdx)
      finalizeOne(inputs[finalIdx++]);
    InputSection *isec = inputs[callIdx];
    assert(isec->isFinal);
    uint64_t isecVA = isec->getVA();
    // Assign addresses up-to the forward branch-range limit
    while (finalIdx < endIdx &&
           isecAddr + inputs[finalIdx]->getSize() < isecVA + branchRange)
      finalizeOne(inputs[finalIdx++]);
    if (isec->callSiteCount == 0)
      continue;
    if (finalIdx == endIdx && stubsInRangeVA == TargetInfo::outOfRangeVA) {
      // When we have finalized all input sections, __stubs (destined
      // to follow __text) comes within range of forward branches and
      // we can estimate the threshold address after which we can
      // reach any stub with a forward branch. Note that although it
      // sits in the middle of a loop, this code executes only once.
      // It is in the loop because we need to call it at the proper
      // time: the earliest call site from which the end of __text
      // (and start of __stubs) comes within range of a forward branch.
      stubsInRangeVA = estimateStubsInRangeVA(callIdx);
    }
    // Process relocs by ascending address, i.e., ascending offset within isec
    std::vector<Reloc> &relocs = isec->relocs;
    assert(is_sorted(relocs,
                     [](Reloc &a, Reloc &b) { return a.offset > b.offset; }));
    for (Reloc &r : reverse(relocs)) {
      ++relocCount;
      if (!target->hasAttr(r.type, RelocAttrBits::BRANCH))
        continue;
      ++callSiteCount;
      // Calculate branch reachability boundaries
      uint64_t callVA = isecVA + r.offset;
      uint64_t lowVA = branchRange < callVA ? callVA - branchRange : 0;
      uint64_t highVA = callVA + branchRange;
      // Calculate our call referent address
      auto *funcSym = r.referent.get<Symbol *>();
      ThunkInfo &thunkInfo = thunkMap[funcSym];
      // The referent is not reachable, so we need to use a thunk ...
      if (funcSym->isInStubs() && callVA >= stubsInRangeVA) {
        // ... Oh, wait! We are close enough to the end that __stubs
        // are now within range of a simple forward branch.
        continue;
      }
      uint64_t funcVA = funcSym->resolveBranchVA();
      ++thunkInfo.callSitesUsed;
      if (lowVA < funcVA && funcVA < highVA) {
        // The referent is reachable with a simple call instruction.
        continue;
      }
      ++thunkInfo.thunkCallCount;
      ++thunkCallCount;
      // If an existing thunk is reachable, use it ...
      if (thunkInfo.sym) {
        uint64_t thunkVA = thunkInfo.isec->getVA();
        if (lowVA < thunkVA && thunkVA < highVA) {
          r.referent = thunkInfo.sym;
          continue;
        }
      }
      // ... otherwise, create a new thunk
      if (isecAddr > highVA) {
        // When there is small-to-no margin between highVA and
        // isecAddr and the distance between subsequent call sites is
        // smaller than thunkSize, then a new thunk can go out of
        // range.  Fix by unfinalizing inputs[finalIdx] to reduce the
        // distance between callVA and highVA, then shift some thunks
        // to occupy address-space formerly occupied by the
        // unfinalized inputs[finalIdx].
        fatal(Twine(__FUNCTION__) + ": FIXME: thunk range overrun");
      }
      thunkInfo.isec = make<InputSection>();
      thunkInfo.isec->name = isec->name;
      thunkInfo.isec->segname = isec->segname;
      thunkInfo.isec->parent = this;
      StringRef thunkName = saver.save(funcSym->getName() + ".thunk." +
                                       std::to_string(thunkInfo.sequence++));
      r.referent = thunkInfo.sym = symtab->addDefined(
          thunkName, /*file=*/nullptr, thunkInfo.isec, /*value=*/0,
          /*size=*/thunkSize, /*isWeakDef=*/false, /*isPrivateExtern=*/true,
          /*isThumb=*/false, /*isReferencedDynamically=*/false,
          /*noDeadStrip=*/false);
      target->populateThunk(thunkInfo.isec, funcSym);
      finalizeOne(thunkInfo.isec);
      thunks.push_back(thunkInfo.isec);
      ++thunkCount;
    }
  }
  size = isecAddr - addr;
  fileSize = isecFileOff - fileOff;

  log("thunks for " + parent->name + "," + name +
      ": funcs = " + std::to_string(thunkMap.size()) +
      ", relocs = " + std::to_string(relocCount) +
      ", all calls = " + std::to_string(callSiteCount) +
      ", thunk calls = " + std::to_string(thunkCallCount) +
      ", thunks = " + std::to_string(thunkCount));
}

void ConcatOutputSection::writeTo(uint8_t *buf) const {
  // Merge input sections from thunk & ordinary vectors
  size_t i = 0, ie = inputs.size();
  size_t t = 0, te = thunks.size();
  while (i < ie || t < te) {
    while (i < ie && (t == te || inputs[i]->getSize() == 0 ||
                      inputs[i]->outSecOff < thunks[t]->outSecOff)) {
      inputs[i]->writeTo(buf + inputs[i]->outSecFileOff);
      ++i;
    }
    while (t < te && (i == ie || thunks[t]->outSecOff < inputs[i]->outSecOff)) {
      thunks[t]->writeTo(buf + thunks[t]->outSecFileOff);
      ++t;
    }
  }
}

// TODO: this is most likely wrong; reconsider how section flags
// are actually merged. The logic presented here was written without
// any form of informed research.
void ConcatOutputSection::mergeFlags(InputSection *input) {
  uint8_t baseType = flags & SECTION_TYPE;
  uint8_t inputType = input->flags & SECTION_TYPE;
  if (baseType != inputType)
    error("Cannot merge section " + input->name + " (type=0x" +
          to_hexString(inputType) + ") into " + name + " (type=0x" +
          to_hexString(baseType) + "): inconsistent types");

  constexpr uint32_t strictFlags = S_ATTR_DEBUG | S_ATTR_STRIP_STATIC_SYMS |
                                   S_ATTR_NO_DEAD_STRIP | S_ATTR_LIVE_SUPPORT;
  if ((input->flags ^ flags) & strictFlags)
    error("Cannot merge section " + input->name + " (flags=0x" +
          to_hexString(input->flags) + ") into " + name + " (flags=0x" +
          to_hexString(flags) + "): strict flags differ");

  // Negate pure instruction presence if any section isn't pure.
  uint32_t pureMask = ~S_ATTR_PURE_INSTRUCTIONS | (input->flags & flags);

  // Merge the rest
  flags |= input->flags;
  flags &= pureMask;
}
