//===- bolt/Passes/LongJmp.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LongJmpPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/LongJmp.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/BranchLivenessUtils.h"
#include "bolt/Passes/RegAnalysis.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "longjmp"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> UseOldText;

static cl::opt<bool> GroupStubs("group-stubs",
                                cl::desc("share stubs across functions"),
                                cl::init(true), cl::cat(BoltOptCategory));

static cl::opt<bool>
    ExperimentalRelaxation("relax-exp",
                           cl::desc("run experimental relaxation pass"),
                           cl::init(false), cl::cat(BoltOptCategory));

static cl::opt<unsigned long long> MaxClusterSize(
    "max-cluster-size",
    cl::desc("maximum estimated size of a function fragment cluster in bytes"),
    cl::init(124 * 1024 * 1024), cl::cat(BoltOptCategory));

static cl::opt<unsigned> MaxThunkChainLength(
    "max-thunk-chain-length",
    cl::desc("maximum number of B-only thunks to use for call relaxation"),
    cl::init(1), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<unsigned> MaxThunkRemeasure(
    "max-thunk-remeasure",
    cl::desc("maximum number of thunk-island remeasurement iterations (0 "
             "keeps maximum island reservations)"),
    cl::init(1), cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

static const Align ColdFragmentAlignment(16);

static void relaxStubToShortJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  InstructionListType Seq;
  BC.MIB->createShortJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
  if (BC.usesBTI())
    BC.MIB->applyBTIFixupToTarget(StubBB);
}

static void relaxStubToLongJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  InstructionListType Seq;
  BC.MIB->createLongJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
  if (BC.usesBTI())
    BC.MIB->applyBTIFixupToTarget(StubBB);
}

static BinaryBasicBlock *getBBAtHotColdSplitPoint(BinaryFunction &Func) {
  if (!Func.isSplit() || Func.empty())
    return nullptr;

  assert(!(*Func.begin()).isCold() && "Entry cannot be cold");
  for (auto I = Func.getLayout().block_begin(),
            E = Func.getLayout().block_end();
       I != E; ++I) {
    auto Next = std::next(I);
    if (Next != E && (*Next)->isCold())
      return *I;
  }
  llvm_unreachable("No hot-cold split point found");
}

static bool mayNeedStub(const BinaryContext &BC, const MCInst &Inst) {
  if (BC.isAArch64() && BC.MIB->isShortRangeBranch(Inst) &&
      !opts::CompactCodeModel) {
    BC.errs() << "BOLT-ERROR: short range branch not supported"
              << " outside compact code model\n";
    BC.printInstruction(BC.errs(), Inst);
    exit(1);
  }
  return (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst)) &&
         !BC.MIB->isIndirectBranch(Inst) && !BC.MIB->isIndirectCall(Inst);
}

std::pair<std::unique_ptr<BinaryBasicBlock>, MCSymbol *>
LongJmpPass::createNewStub(BinaryBasicBlock &SourceBB, const MCSymbol *TgtSym,
                           bool TgtIsFunc, uint64_t AtAddress) {
  BinaryFunction &Func = *SourceBB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  const bool IsCold = SourceBB.isCold();
  MCSymbol *StubSym = BC.Ctx->createNamedTempSymbol("Stub");
  std::unique_ptr<BinaryBasicBlock> StubBB = Func.createBasicBlock(StubSym);
  MCInst Inst;
  BC.MIB->createUncondBranch(Inst, TgtSym, BC.Ctx.get());
  if (TgtIsFunc)
    BC.MIB->convertJmpToTailCall(Inst);
  StubBB->addInstruction(Inst);
  StubBB->setExecutionCount(0);

  // Register this in stubs maps
  auto registerInMap = [&](StubGroupsTy &Map) {
    StubGroupTy &StubGroup = Map[TgtSym];
    StubGroup.insert(
        llvm::lower_bound(
            StubGroup, std::make_pair(AtAddress, nullptr),
            [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
                const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
              return LHS.first < RHS.first;
            }),
        std::make_pair(AtAddress, StubBB.get()));
  };

  Stubs[&Func].insert(StubBB.get());
  StubBits[StubBB.get()] = BC.MIB->getUncondBranchEncodingSize();
  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: LongJmp: creating " << (IsCold ? "cold" : "main")
             << " stub " << StubSym->getName() << " in " << Func.getPrintName()
             << " at current layout address 0x" << Twine::utohexstr(AtAddress)
             << " for " << TgtSym->getName() << '\n');

  if (IsCold) {
    registerInMap(ColdLocalStubs[&Func]);
    if (opts::GroupStubs && TgtIsFunc)
      registerInMap(ColdStubGroups);
    ++NumColdStubs;
  } else {
    registerInMap(HotLocalStubs[&Func]);
    if (opts::GroupStubs && TgtIsFunc)
      registerInMap(HotStubGroups);
    ++NumHotStubs;
  }

  return std::make_pair(std::move(StubBB), StubSym);
}

BinaryBasicBlock *LongJmpPass::lookupStubFromGroup(
    const StubGroupsTy &StubGroups, const BinaryFunction &Func,
    const MCInst &Inst, const MCSymbol *TgtSym, uint64_t DotAddress) const {
  const BinaryContext &BC = Func.getBinaryContext();
  auto CandidatesIter = StubGroups.find(TgtSym);
  if (CandidatesIter == StubGroups.end())
    return nullptr;
  const StubGroupTy &Candidates = CandidatesIter->second;
  if (Candidates.empty())
    return nullptr;
  auto Cand = llvm::lower_bound(
      Candidates, std::make_pair(DotAddress, nullptr),
      [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
          const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
        return LHS.first < RHS.first;
      });
  if (Cand == Candidates.end()) {
    Cand = std::prev(Cand);
  } else if (Cand != Candidates.begin()) {
    const StubTy *LeftCand = std::prev(Cand);
    if (Cand->first - DotAddress > DotAddress - LeftCand->first)
      Cand = LeftCand;
  }
  int BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
  assert(BitsAvail < 63 && "PCRelEncodingSize is too large to use int64_t to"
                           "check for out-of-bounds.");
  int64_t MaxVal = (1ULL << BitsAvail) - 1;
  int64_t MinVal = -(1ULL << BitsAvail);
  uint64_t PCRelTgtAddress = Cand->first;
  int64_t PCOffset = (int64_t)(PCRelTgtAddress - DotAddress);

  LLVM_DEBUG({
    if (Candidates.size() > 1)
      dbgs() << "BOLT-DEBUG: LongJmp: considering stub group with "
             << Candidates.size() << " candidates at 0x"
             << Twine::utohexstr(DotAddress) << "; selected candidate at 0x"
             << Twine::utohexstr(Cand->first) << '\n';
  });
  return (PCOffset < MinVal || PCOffset > MaxVal) ? nullptr : Cand->second;
}

BinaryBasicBlock *
LongJmpPass::lookupGlobalStub(const BinaryBasicBlock &SourceBB,
                              const MCInst &Inst, const MCSymbol *TgtSym,
                              uint64_t DotAddress) const {
  const BinaryFunction &Func = *SourceBB.getFunction();
  const StubGroupsTy &StubGroups =
      SourceBB.isCold() ? ColdStubGroups : HotStubGroups;
  return lookupStubFromGroup(StubGroups, Func, Inst, TgtSym, DotAddress);
}

BinaryBasicBlock *LongJmpPass::lookupLocalStub(const BinaryBasicBlock &SourceBB,
                                               const MCInst &Inst,
                                               const MCSymbol *TgtSym,
                                               uint64_t DotAddress) const {
  const BinaryFunction &Func = *SourceBB.getFunction();
  const DenseMap<const BinaryFunction *, StubGroupsTy> &StubGroups =
      SourceBB.isCold() ? ColdLocalStubs : HotLocalStubs;
  const auto Iter = StubGroups.find(&Func);
  if (Iter == StubGroups.end())
    return nullptr;
  return lookupStubFromGroup(Iter->second, Func, Inst, TgtSym, DotAddress);
}

std::unique_ptr<BinaryBasicBlock>
LongJmpPass::replaceTargetWithStub(BinaryBasicBlock &BB, MCInst &Inst,
                                   uint64_t DotAddress,
                                   uint64_t StubCreationAddress) {
  const BinaryFunction &Func = *BB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  std::unique_ptr<BinaryBasicBlock> NewBB;
  const MCSymbol *TgtSym = BC.MIB->getTargetSymbol(Inst);
  assert(TgtSym && "getTargetSymbol failed");

  BinaryBasicBlock::BinaryBranchInfo BI{0, 0};
  BinaryBasicBlock *TgtBB = BB.getSuccessor(TgtSym, BI);
  auto LocalStubsIter = Stubs.find(&Func);

  // If already using stub and the stub is from another function, create a local
  // stub, since the foreign stub is now out of range
  if (!TgtBB) {
    auto SSIter = SharedStubs.find(TgtSym);
    if (SSIter != SharedStubs.end()) {
      TgtSym = BC.MIB->getTargetSymbol(*SSIter->second->begin());
      --NumSharedStubs;
    }
  } else if (LocalStubsIter != Stubs.end() &&
             LocalStubsIter->second.count(TgtBB)) {
    // The TgtBB and TgtSym now are the local out-of-range stub and its label.
    // So, we are attempting to restore BB to its previous state without using
    // this stub.
    TgtSym = BC.MIB->getTargetSymbol(*TgtBB->begin());
    assert(TgtSym &&
           "First instruction is expected to contain a target symbol.");
    BinaryBasicBlock *TgtBBSucc = TgtBB->getSuccessor(TgtSym, BI);

    // TgtBB might have no successor. e.g. a stub for a function call.
    if (TgtBBSucc) {
      BB.replaceSuccessor(TgtBB, TgtBBSucc, BI.Count, BI.MispredictedCount);
      assert(TgtBB->getExecutionCount() >= BI.Count &&
             "At least equal or greater than the branch count.");
      TgtBB->setExecutionCount(TgtBB->getExecutionCount() - BI.Count);
    }

    TgtBB = TgtBBSucc;
  }

  BinaryBasicBlock *StubBB = lookupLocalStub(BB, Inst, TgtSym, DotAddress);
  // If not found, look it up in globally shared stub maps if it is a function
  // call (TgtBB is not set)
  if (!StubBB && !TgtBB) {
    StubBB = lookupGlobalStub(BB, Inst, TgtSym, DotAddress);
    if (StubBB) {
      SharedStubs[StubBB->getLabel()] = StubBB;
      ++NumSharedStubs;
    }
  }
  MCSymbol *StubSymbol = StubBB ? StubBB->getLabel() : nullptr;

  if (!StubBB) {
    std::tie(NewBB, StubSymbol) =
        createNewStub(BB, TgtSym, /*is func?*/ !TgtBB, StubCreationAddress);
    StubBB = NewBB.get();
  }

  // Local branch
  if (TgtBB) {
    uint64_t OrigCount = BI.Count;
    uint64_t OrigMispreds = BI.MispredictedCount;
    BB.replaceSuccessor(TgtBB, StubBB, OrigCount, OrigMispreds);
    StubBB->setExecutionCount(StubBB->getExecutionCount() + OrigCount);
    if (NewBB) {
      StubBB->addSuccessor(TgtBB, OrigCount, OrigMispreds);
      StubBB->setIsCold(BB.isCold());
    }
    // Call / tail call
  } else {
    StubBB->setExecutionCount(StubBB->getExecutionCount() +
                              BB.getExecutionCount());
    if (NewBB) {
      assert(TgtBB == nullptr);
      StubBB->setIsCold(BB.isCold());
      // Set as entry point because this block is valid but we have no preds
      StubBB->getFunction()->addEntryPoint(*StubBB);
    }
  }
  BC.MIB->replaceBranchTarget(Inst, StubSymbol, BC.Ctx.get());

  return NewBB;
}

void LongJmpPass::updateStubGroups() {
  auto update = [&](StubGroupsTy &StubGroups) {
    for (auto &KeyVal : StubGroups) {
      for (StubTy &Elem : KeyVal.second)
        Elem.first = BBAddresses.at(Elem.second);
      llvm::sort(KeyVal.second, llvm::less_first());
    }
  };

  for (auto &KeyVal : HotLocalStubs)
    update(KeyVal.second);
  for (auto &KeyVal : ColdLocalStubs)
    update(KeyVal.second);
  update(HotStubGroups);
  update(ColdStubGroups);
}

uint64_t LongJmpPass::updateSectionAlignment(const BinaryContext &BC,
                                             const BinaryFunction &Func,
                                             const FunctionFragment &FF,
                                             uint64_t Alignment) const {
  if (BC.HasRelocations) {
    // BinaryEmitter::emitFunction() raises every emitted code section to at
    // least BC.AlignFunctions in relocation mode.
    Alignment = std::max<uint64_t>(Alignment, BC.AlignFunctions);

    // BinaryEmitter::emitAll() sets the main text section to BC.AlignText.
    if (Func.getCodeSectionName(FF.getFragmentNum()) ==
        BC.getMainCodeSectionName())
      Alignment = std::max<uint64_t>(Alignment, BC.AlignText);

    // BinaryEmitter::emitFunction() emits the mandatory minimum function
    // alignment first.
    Alignment = std::max<uint64_t>(Alignment, Func.getMinAlignment());

    // BinaryEmitter::emitFunction() emits the preferred function alignment
    // only when the corresponding maximum padding is nonzero.
    const uint16_t MaxAlignBytes = FF.isSplitFragment()
                                       ? Func.getMaxColdAlignmentBytes()
                                       : Func.getMaxAlignmentBytes();
    if (MaxAlignBytes > 0)
      Alignment = std::max<uint64_t>(Alignment, Func.getAlignment());
  } else {
    // In non-relocation mode BinaryEmitter emits only the preferred function
    // alignment. This path is used for newly allocated injected sections.
    Alignment = std::max<uint64_t>(Alignment, Func.getAlignment());
  }

  // BinaryEmitter::emitFunctionBody() emits enabled basic-block alignment
  // directives.
  if (BC.AlignBlocks || BC.PreserveBlocksAlignment)
    for (const BinaryBasicBlock *BB : FF)
      if (BB->getAlignment() > 1)
        Alignment = std::max<uint64_t>(Alignment, BB->getAlignment());

  // BinaryEmitter::emitConstantIslands() aligns owned and cloned islands
  // using the host function's constant-island alignment.
  if (Func.hasIslandsInfo())
    Alignment =
        std::max<uint64_t>(Alignment, Func.getConstantIslandAlignment());

  return Alignment;
}

void LongJmpPass::assignFunctionFragmentToSection(const BinaryContext &BC,
                                                  const BinaryFunction &Func,
                                                  const FunctionFragment &FF) {
  const StringRef SectionName = Func.getCodeSectionName(FF.getFragmentNum());
  auto It = llvm::find_if(Sections, [&](const SectionPlacement &Section) {
    return StringRef(Section.Name) == SectionName;
  });
  if (It == Sections.end()) {
    // AArch64ELFStreamer::changeSection() gives every text section a
    // four-byte minimum alignment before BinaryEmitter raises it further.
    Sections.push_back({SmallString<32>(SectionName),
                        {},
                        updateSectionAlignment(BC, Func, FF, 4)});
    It = std::prev(Sections.end());
  } else {
    It->Alignment = updateSectionAlignment(BC, Func, FF, It->Alignment);
  }

  It->Fragments.push_back({&Func, FF.getFragmentNum()});
}

void LongJmpPass::assignFunctionsToSections(
    const BinaryContext &BC, const BinaryFunctionListType &SortedFunctions) {
  // Mirror BinaryEmitter::emitFunctions(): emit each main fragment followed
  // immediately by its split fragments, preserving that order per section.
  for (const BinaryFunction *Func : SortedFunctions) {
    // Do not assign functions for which BinaryEmitter::emitFunction()
    // returns before selecting a section.
    if (!shouldEmitFunctionFragment(BC, *Func))
      continue;

    assert((BC.HasRelocations || Func->getLayout().isHotColdSplit()) &&
           "non-relocation mode supports only hot/cold splitting");

    // RewriteInstance ultimately excludes every code section with a
    // pre-assigned output address. At this point, those are the sections of
    // fixed-address injected functions.
    if (Func->isInjected() && Func->getOutputAddress())
      continue;

    // In relocation mode, process all remaining functions. In non-relocation
    // mode, process only non-fixed injected functions. Their sections are
    // allocated after the moved cold fragments and require the same alignment
    // calculation as relocation-mode sections.
    if (!BC.HasRelocations && !Func->isInjected())
      continue;

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp: collecting fragments for "
                      << Func->getPrintName() << " (#"
                      << Func->getFunctionNumber() << ")\n");

    const FunctionLayout &Layout = Func->getLayout();
    assignFunctionFragmentToSection(BC, *Func, Layout.getMainFragment());

    if (Func->isSplit()) {
      assert(!Func->isInjected() && "injected functions cannot be split");
      assert((Layout.fragment_size() == 1 || Func->isSimple()) &&
             "only simple functions can have multiple fragments");
      for (const FunctionFragment &FF : Layout.getSplitFragments()) {
        // BinaryEmitter::emitFunctions() skips an empty split fragment unless
        // the function carries a constant island.
        if (FF.empty() && !Func->hasConstantIsland())
          continue;
        assignFunctionFragmentToSection(BC, *Func, FF);
      }
    }
  }
}

/// Advance \p Offset using the rule from
/// MCObjectStreamer::emitCodeAlignment(). A zero maximum makes the alignment
/// mandatory; otherwise omit padding larger than \p MaxBytesToEmit.
static uint64_t applyCodeAlignment(uint64_t Offset, Align Alignment,
                                   uint64_t MaxBytesToEmit = 0) {
  const uint64_t Pad = offsetToAlignment(Offset, Alignment);
  return !MaxBytesToEmit || Pad <= MaxBytesToEmit ? Offset + Pad : Offset;
}

uint64_t LongJmpPass::layoutFunctionBody(const BinaryContext &BC,
                                         const BinaryFunction &Func,
                                         const FunctionFragment &FF,
                                         uint64_t DotAddress,
                                         bool RecordAddresses) {
  for (const BinaryBasicBlock *BB : FF) {
    // Mirror per-basic-block alignment in BinaryEmitter::emitFunctionBody().
    if ((BC.AlignBlocks || BC.PreserveBlocksAlignment) &&
        BB->getAlignment() > 1)
      DotAddress = applyCodeAlignment(DotAddress, BB->getAlign(),
                                      BB->getAlignmentMaxBytes());

    if (RecordAddresses) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp layout: basic block "
                        << BB->getName() << " in " << Func.getPrintName()
                        << " starts at 0x" << Twine::utohexstr(DotAddress)
                        << '\n');
      BBAddresses[BB] = DotAddress;
    }

#ifdef EXPENSIVE_CHECKS
    // computeCodeSize() skips all pseudo-instructions. Calling
    // computeInstructionSize() directly would not generally help: unless a
    // pseudo has an explicit size annotation, it also returns zero.
    // BinaryEmitter handles CFI pseudos separately by emitting unwind
    // directives that do not advance the code-section address, but passes
    // every other pseudo to emitInstruction(). No BOLT path is known to place
    // another pseudo in an emitted basic block; verify that assumption here.
    for (const MCInst &Instr : *BB)
      assert((!BC.MIB->isPseudo(Instr) || BC.MIB->isCFI(Instr)) &&
             "unexpected non-CFI pseudo in emitted function");
#endif
    DotAddress += BC.computeCodeSize(BB->begin(), BB->end());
  }

  // BinaryEmitter::emitFunctionBody() emits constant islands after the
  // fragment instructions.
  if (Func.hasIslandsInfo()) {
    DotAddress = alignTo(DotAddress, Func.getConstantIslandAlignment());
    DotAddress += Func.estimateConstantIslandSize();
  }

  return DotAddress;
}

uint64_t LongJmpPass::layoutFunctionFragment(const BinaryContext &BC,
                                             const BinaryFunction &Func,
                                             const FunctionFragment &FF,
                                             uint64_t DotAddress,
                                             bool RecordAddresses) {
  assert(shouldEmitFunctionFragment(BC, Func) &&
         "attempting to lay out a function BinaryEmitter will not emit");
  const FragmentNum Fragment = FF.getFragmentNum();
  const bool IsMain = FF.isMainFragment();
  const bool IsSplit = FF.isSplitFragment();

  const bool HasFixedOutputAddress =
      Func.isInjected() && Func.getOutputAddress();
  const bool NeedsRelocationAlignment =
      BC.HasRelocations && !HasFixedOutputAddress;
  const bool NeedsNonRelocInjectedAlignment =
      !BC.HasRelocations && Func.isInjected() && !HasFixedOutputAddress;
  const bool NeedsNonRelocColdAlignment =
      !BC.HasRelocations && !Func.isInjected() && IsSplit;

  // Apply the alignment that affects the fragment's mapped address.
  // Section-relative fragments mirror BinaryEmitter::emitFunction(); moved
  // cold fragments mirror mapCodeSectionsInPlace(). Ordinary non-relocation
  // main fragments and fixed-address injected functions already have exact
  // addresses.
  if (NeedsRelocationAlignment) {
    DotAddress = alignTo(DotAddress, Func.getMinAlignment());
    const uint16_t MaxAlignmentBytes =
        IsSplit ? Func.getMaxColdAlignmentBytes() : Func.getMaxAlignmentBytes();
    if (MaxAlignmentBytes > 0)
      DotAddress =
          applyCodeAlignment(DotAddress, Func.getAlign(), MaxAlignmentBytes);
  } else if (NeedsNonRelocInjectedAlignment) {
    // Newly allocated injected sections retain BinaryEmitter's regular
    // non-relocation function alignment.
    DotAddress = alignTo(DotAddress, Func.getAlign());
  } else if (NeedsNonRelocColdAlignment) {
    // mapCodeSectionsInPlace() aligns each moved cold fragment to a hard-coded
    // 16-byte boundary.
    DotAddress = alignTo(DotAddress, ColdFragmentAlignment);
  }

  // BinaryEmitter::emitFunction() places --pad-funcs-before after function
  // alignment and rejects nonzero padding in non-relocation mode.
  if (BC.HasRelocations)
    DotAddress += opts::padFunctionBefore(Func);

  const uint64_t FragmentAddress = DotAddress;

  // BinaryEmitter::emitFunction() emits the fragment entry symbols here.
  if (RecordAddresses && IsMain)
    HotAddresses[&Func] = DotAddress;

  // --break-funcs emits UD2 before the function body.
  DotAddress += opts::breakFunctionSize(Func);

  DotAddress = layoutFunctionBody(BC, Func, FF, DotAddress, RecordAddresses);

  // BinaryEmitter::emitFunction() emits --pad-funcs after the body in both
  // relocation and non-relocation modes.
  DotAddress += opts::padFunctionAfter(Func);

  // --mark-funcs emits the target-specific trap marker after the fragment.
  DotAddress += opts::markFunctionBytes(BC).size();

  if (RecordAddresses)
    FragmentAddresses[&FF] = {FragmentAddress, DotAddress};

  LLVM_DEBUG({
    if (RecordAddresses) {
      const StringRef FragmentName =
          IsMain                                                 ? "main"
          : BC.HasWarmSection && Fragment == FragmentNum::warm() ? "warm"
          : Fragment == FragmentNum::cold()                      ? "cold"
                                                                 : "split";
      dbgs() << "BOLT-DEBUG: LongJmp layout: " << FragmentName << " fragment "
             << Func.getPrintName();
      if (IsMain)
        dbgs() << " starts at 0x" << Twine::utohexstr(HotAddresses.at(&Func))
               << " and";
      dbgs() << " ends at 0x" << Twine::utohexstr(DotAddress) << '\n';
    }
  });

  return DotAddress;
}

uint64_t LongJmpPass::layoutSection(const BinaryContext &BC,
                                    const SectionPlacement &Section,
                                    uint64_t DotAddress, bool RecordAddresses) {
  LLVM_DEBUG({
    if (RecordAddresses)
      dbgs() << "BOLT-DEBUG: LongJmp layout: section " << Section.Name
             << " starts at 0x" << Twine::utohexstr(DotAddress)
             << ", alignment 0x" << Twine::utohexstr(Section.Alignment) << ", "
             << Section.Fragments.size() << " fragments\n";
  });

  for (const FunctionFragmentPlacement &Placement : Section.Fragments) {
    const BinaryFunction &Func = *Placement.Func;

    const FunctionFragment &FF =
        Func.getLayout().getFragment(Placement.Fragment);
    DotAddress =
        layoutFunctionFragment(BC, Func, FF, DotAddress, RecordAddresses);
  }

  LLVM_DEBUG({
    if (RecordAddresses)
      dbgs() << "BOLT-DEBUG: LongJmp layout: section " << Section.Name
             << " ends at 0x" << Twine::utohexstr(DotAddress) << '\n';
  });

  return DotAddress;
}

uint64_t LongJmpPass::layoutSectionsForward(const BinaryContext &BC,
                                            uint64_t DotAddress) {
  const StringRef LastNonColdSectionName = BC.HasWarmSection
                                               ? BC.getWarmCodeSectionName()
                                               : BC.getMainCodeSectionName();
  const bool AdjustLastNonColdSection =
      BC.HasRelocations &&
      (opts::HotText || (opts::Hugify && !BC.HasFixedLoadAddress));
  std::optional<uint64_t> LastNonColdSectionEnd = std::nullopt;

  // Mirror allocateAt() in RewriteInstance::mapCodeSections().
  for (const SectionPlacement &Section : Sections) {
    DotAddress = alignTo(DotAddress, Section.Alignment);
    DotAddress = layoutSection(BC, Section, DotAddress);

    if (AdjustLastNonColdSection &&
        StringRef(Section.Name) == LastNonColdSectionName) {
      if (opts::HotText)
        LastNonColdSectionEnd = DotAddress;
      // Mirror the extra post-hot-text alignment in allocateAt() for
      // --hugify. With CDSplit, warm code is part of hot text.
      if (opts::Hugify && !BC.HasFixedLoadAddress)
        DotAddress = alignTo(DotAddress, Section.Alignment);
    }
  }

  // Mirror RewriteInstance::mapCodeSections() padding used to accommodate
  // hot-text huge-page mapping. The hot-text end is the end of the warm
  // section when one exists, and the end of the main section otherwise.
  // RewriteInstance applies this adjustment only in allocateAt() to advance
  // the next free address; allocateBefore() starts from a fixed upper boundary
  // and has no corresponding adjustment.
  if (LastNonColdSectionEnd)
    DotAddress =
        std::max(DotAddress, alignTo(*LastNonColdSectionEnd, BC.PageAlign));

  return DotAddress;
}

bool LongJmpPass::layoutSectionsBackward(const BinaryContext &BC,
                                         uint64_t DotAddress) {
  SmallVector<uint64_t, 4> SectionAddresses(Sections.size());
  // Mirror allocateBefore() in RewriteInstance::mapCodeSections(): assign
  // section bases in reverse while preserving their sorted output order.
  for (size_t I = Sections.size(); I > 0; --I) {
    const SectionPlacement &Section = Sections[I - 1];
    uint64_t &SectionAddress = SectionAddresses[I - 1];
    // Match the BinarySection::getOutputSize() consumed by allocateBefore().
    const uint64_t SectionSize =
        layoutSection(BC, Section, 0, /*RecordAddresses=*/false);
    if (SectionSize > DotAddress)
      return false;
    DotAddress -= SectionSize;
    DotAddress = alignDown(DotAddress, Section.Alignment);
    if (DotAddress < BC.OldTextSectionAddress)
      return false;
    SectionAddress = DotAddress;
  }

  // Contents within every section are still laid out toward higher addresses.
  for (size_t I = 0; I < Sections.size(); ++I)
    layoutSection(BC, Sections[I], SectionAddresses[I]);

  return true;
}

void LongJmpPass::layoutFunctions(
    const BinaryContext &BC, const BinaryFunctionListType &SortedFunctions) {
  if (BC.HasRelocations) {
    // Mirror the old-text allocation choice in
    // RewriteInstance::mapCodeSections().
    bool AllocatedAtOldText = false;
    if (opts::UseOldText) {
      if (opts::HotFunctionsAtEnd) {
        AllocatedAtOldText = layoutSectionsBackward(
            BC, BC.OldTextSectionAddress + BC.OldTextSectionSize);
      } else {
        const uint64_t EndAddress =
            layoutSectionsForward(BC, BC.OldTextSectionAddress);
        AllocatedAtOldText =
            EndAddress <= BC.OldTextSectionAddress + BC.OldTextSectionSize;
      }

      if (!AllocatedAtOldText) {
        BC.errs() << "BOLT-WARNING: --use-old-text failed during LongJmp "
                     "layout. The original .text is too small to fit the new "
                     "code.\n";
        // Do not clear opts::UseOldText here. RewriteInstance also uses it
        // during emission to decide how to handle non-code sections such as
        // .eh_frame, and performs the authoritative fallback later.
      } else {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp: The layout fits into the "
                             "original .text section\n");
      }
    }

    // mapCodeSections() falls back to allocateAt() when old text is unused
    // or too small.
    if (!AllocatedAtOldText)
      layoutSectionsForward(BC, BC.LayoutStartAddress);
  } else {
    // Mirror RewriteInstance::mapCodeSectionsInPlace(). Main fragments retain
    // their input addresses, while split cold fragments are appended in
    // original-function order starting at the first free output address.
    uint64_t ColdAddress = BC.LayoutStartAddress;
    for (const auto &BFI : BC.getBinaryFunctions()) {
      const BinaryFunction &Func = BFI.second;

      // PopulateOutputFunctions excludes functions for which shouldEmit()
      // returns false. LongJmp never relaxes them, so they need no entries in
      // the function or basic-block address maps. Unlike the other layout
      // loops, this one visits getBinaryFunctions() to mirror
      // mapCodeSectionsInPlace(), and therefore needs an explicit check.
      if (!BC.shouldEmit(Func))
        continue;

      if (!shouldEmitFunctionFragment(BC, Func))
        continue;

      layoutFunctionFragment(BC, Func, Func.getLayout().getMainFragment(),
                             Func.getAddress());

      if (Func.isSplit()) {
        ColdAddress = layoutFunctionFragment(
            BC, Func, Func.getLayout().getFragment(FragmentNum::cold()),
            ColdAddress);
      }
    }

    // mapCodeSectionsInPlace() allocates non-fixed injected sections, normally
    // .text.injected, immediately after the moved cold fragments. These are
    // the only entries in Sections in non-relocation mode.
    layoutSectionsForward(BC, ColdAddress);
  }

  // Fixed-address injected functions are outside Sections and use their
  // pre-assigned output addresses.
  for (const BinaryFunction *Func : SortedFunctions) {
    if (!shouldEmitFunctionFragment(BC, *Func))
      continue;

    if (Func->isInjected() && Func->getOutputAddress()) {
      assert(!Func->isSplit() && "injected functions cannot be split");
      layoutFunctionFragment(BC, *Func, Func->getLayout().getMainFragment(),
                             Func->getOutputAddress());
    }
  }
}

void LongJmpPass::layout(const BinaryContext &BC,
                         const BinaryFunctionListType &SortedFunctions) {
  HotAddresses.clear();
  BBAddresses.clear();
  FragmentAddresses.clear();
  Sections.clear();

  LLVM_DEBUG(
      dbgs() << "BOLT-DEBUG: LongJmp layout starts at 0x"
             << Twine::utohexstr(BC.LayoutStartAddress) << ", text alignment 0x"
             << Twine::utohexstr(BC.AlignText) << ", function alignment 0x"
             << Twine::utohexstr(BC.AlignFunctions)
             << ", maximum main alignment 0x"
             << Twine::utohexstr(BC.MaxMainCodeAlignment.load())
             << ", maximum cold alignment 0x"
             << Twine::utohexstr(BC.MaxColdCodeAlignment.load()) << '\n');

  // Reproduce the code placement performed later by BinaryEmitter and
  // RewriteInstance. First catalogue fragments whose addresses are determined
  // by output-section placement. In relocation mode this includes all emitted
  // fragments except fixed-address injected patches. In non-relocation mode it
  // includes only non-fixed injected functions; ordinary main fragments remain
  // at their input addresses, while mapCodeSectionsInPlace() allocates moved
  // cold fragments directly.
  //
  // Section alignment depends on every fragment assigned to the section, so
  // the complete catalogue must be built before calculating any section base.
  // The layout phase then mirrors either mapCodeSections() or
  // mapCodeSectionsInPlace(), and finally records fixed injected patches whose
  // addresses do not come from the section catalogue.

  assignFunctionsToSections(BC, SortedFunctions);

  if (BC.HasRelocations) {
    // Mirror RewriteInstance::getCodeSections(). Sections not named explicitly
    // retain their first-emission order.
    llvm::stable_sort(
        Sections, [&](const SectionPlacement &A, const SectionPlacement &B) {
          return BC.compareSectionNames(A.Name, B.Name);
        });
  }

  layoutFunctions(BC, SortedFunctions);
}

bool LongJmpPass::usesStub(const BinaryFunction &Func,
                           const MCInst &Inst) const {
  const MCSymbol *TgtSym = Func.getBinaryContext().MIB->getTargetSymbol(Inst);
  const BinaryBasicBlock *TgtBB = Func.getBasicBlockForLabel(TgtSym);
  auto Iter = Stubs.find(&Func);
  if (Iter != Stubs.end())
    return Iter->second.count(TgtBB);
  return false;
}

uint64_t LongJmpPass::getSymbolAddress(const BinaryContext &BC,
                                       const MCSymbol *Target,
                                       const BinaryBasicBlock *TgtBB) const {
  if (TgtBB) {
    auto Iter = BBAddresses.find(TgtBB);
    assert(Iter != BBAddresses.end() && "Unrecognized BB");
    return Iter->second;
  }
  uint64_t EntryID = 0;
  const BinaryFunction *TargetFunc = BC.getFunctionForSymbol(Target, &EntryID);
  auto Iter = HotAddresses.find(TargetFunc);
  if (Iter == HotAddresses.end() || (TargetFunc && EntryID)) {
    // Look at BinaryContext's resolution for this symbol - this is a symbol not
    // mapped to a BinaryFunction
    ErrorOr<uint64_t> ValueOrError = BC.getSymbolValue(*Target);
    assert(ValueOrError && "Unrecognized symbol");
    return *ValueOrError;
  }
  return Iter->second;
}

Error LongJmpPass::relaxStub(BinaryBasicBlock &StubBB, bool &Modified) {
  BinaryFunction &Func = *StubBB.getFunction();
  BinaryContext &BC = Func.getBinaryContext();
  const int Bits = StubBits[&StubBB];
  // Already working with the largest range?
  if (Bits == static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8))
    return Error::success();

  const static int RangeShortJmp = BC.MIB->getShortJmpEncodingSize();
  const static int RangeSingleInstr = BC.MIB->getUncondBranchEncodingSize();
  const static uint64_t ShortJmpMask = ~((1ULL << RangeShortJmp) - 1);
  const static uint64_t SingleInstrMask =
      ~((1ULL << (RangeSingleInstr - 1)) - 1);

  const MCSymbol *RealTargetSym = BC.MIB->getTargetSymbol(*StubBB.begin());
  const BinaryBasicBlock *TgtBB = Func.getBasicBlockForLabel(RealTargetSym);
  uint64_t TgtAddress = getSymbolAddress(BC, RealTargetSym, TgtBB);
  uint64_t DotAddress = BBAddresses.at(&StubBB);
  uint64_t PCRelTgtAddress = DotAddress > TgtAddress ? DotAddress - TgtAddress
                                                     : TgtAddress - DotAddress;

  // If it fits in one instruction, do not relax
  if (!(PCRelTgtAddress & SingleInstrMask))
    return Error::success();

  // Fits short jmp
  if (!(PCRelTgtAddress & ShortJmpMask)) {
    if (Bits >= RangeShortJmp)
      return Error::success();

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp: relaxing stub to short jump; "
                      << "distance 0x" << Twine::utohexstr(PCRelTgtAddress)
                      << ", target " << RealTargetSym->getName() << '\n');
    relaxStubToShortJmp(StubBB, RealTargetSym);
    StubBits[&StubBB] = RangeShortJmp;
    Modified = true;
    return Error::success();
  }

  // The long jmp uses absolute address on AArch64
  // So we could not use it for PIC binaries
  if (BC.isAArch64() && !BC.HasFixedLoadAddress)
    return createFatalBOLTError(
        "BOLT-ERROR: Unable to relax stub for PIC binary\n");

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp: relaxing stub to long jump; "
                    << "distance 0x" << Twine::utohexstr(PCRelTgtAddress)
                    << ", target " << RealTargetSym->getName() << '\n');
  relaxStubToLongJmp(StubBB, RealTargetSym);
  StubBits[&StubBB] = static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8);
  Modified = true;
  return Error::success();
}

bool LongJmpPass::needsStub(const BinaryBasicBlock &BB, const MCInst &Inst,
                            uint64_t DotAddress) const {
  const BinaryFunction &Func = *BB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  const MCSymbol *TgtSym = BC.MIB->getTargetSymbol(Inst);
  assert(TgtSym && "getTargetSymbol failed");

  const BinaryBasicBlock *TgtBB = Func.getBasicBlockForLabel(TgtSym);
  // Check for shared stubs from foreign functions
  if (!TgtBB) {
    auto SSIter = SharedStubs.find(TgtSym);
    if (SSIter != SharedStubs.end())
      TgtBB = SSIter->second;
  }

  int BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
  assert(BitsAvail < 63 && "PCRelEncodingSize is too large to use int64_t to"
                           "check for out-of-bounds.");
  int64_t MaxVal = (1ULL << BitsAvail) - 1;
  int64_t MinVal = -(1ULL << BitsAvail);

  uint64_t PCRelTgtAddress = getSymbolAddress(BC, TgtSym, TgtBB);
  int64_t PCOffset = (int64_t)(PCRelTgtAddress - DotAddress);

  const bool Result = PCOffset < MinVal || PCOffset > MaxVal;
  LLVM_DEBUG({
    if (Result)
      dbgs() << "BOLT-DEBUG: LongJmp: out-of-range branch in "
             << Func.getPrintName() << ", basic block " << BB.getName()
             << ", source 0x" << Twine::utohexstr(DotAddress) << ", target "
             << TgtSym->getName() << " at 0x"
             << Twine::utohexstr(PCRelTgtAddress) << ", displacement "
             << PCOffset << ", range [" << MinVal << ", " << MaxVal << "]\n";
  });
  return Result;
}

Error LongJmpPass::relax(BinaryFunction &Func, bool &Modified) {
  const BinaryContext &BC = Func.getBinaryContext();

  assert(BC.isAArch64() && "Unsupported arch");
  // Keep the relaxation traversal consistent with layout(): functions that
  // BinaryEmitter will not emit have no entries in BBAddresses.
  if (!shouldEmitFunctionFragment(BC, Func))
    return Error::success();

  constexpr int InsnSize = 4; // AArch64
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      Insertions;

  BinaryBasicBlock *Frontier = getBBAtHotColdSplitPoint(Func);
  uint64_t FrontierAddress = Frontier ? BBAddresses.at(Frontier) : 0;
  if (FrontierAddress)
    FrontierAddress += Frontier->getNumNonPseudos() * InsnSize;

  // Add necessary stubs for branch targets we know we can't fit in the
  // instruction
  for (BinaryBasicBlock &BB : Func) {
    uint64_t DotAddress = BBAddresses.at(&BB);
    // Stubs themselves are relaxed on the next loop
    if (Stubs[&Func].count(&BB))
      continue;

    for (MCInst &Inst : BB) {
      if (BC.MIB->isPseudo(Inst))
        continue;

      if (!mayNeedStub(BC, Inst)) {
        DotAddress += InsnSize;
        continue;
      }

      // Check and relax direct branch or call
      if (!needsStub(BB, Inst, DotAddress)) {
        DotAddress += InsnSize;
        continue;
      }
      Modified = true;

      // Insert stubs close to the patched BB if call, but far away from the
      // hot path if a branch, since this branch target is the cold region
      // (but first check that the far away stub will be in range).
      BinaryBasicBlock *InsertionPoint = &BB;
      if (Func.isSimple() && !BC.MIB->isCall(Inst) && FrontierAddress &&
          !BB.isCold()) {
        int BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
        uint64_t Mask = ~((1ULL << BitsAvail) - 1);
        assert(FrontierAddress > DotAddress &&
               "Hot code should be before the frontier");
        uint64_t PCRelTgt = FrontierAddress - DotAddress;
        if (!(PCRelTgt & Mask))
          InsertionPoint = Frontier;
      }
      // Always put stubs at the end of the function if non-simple. We can't
      // change the layout of non-simple functions because it has jump tables
      // that we do not control.
      if (!Func.isSimple())
        InsertionPoint = &*std::prev(Func.end());

      // Create a stub to handle a far-away target
      Insertions.emplace_back(InsertionPoint,
                              replaceTargetWithStub(BB, Inst, DotAddress,
                                                    InsertionPoint == Frontier
                                                        ? FrontierAddress
                                                        : DotAddress));

      DotAddress += InsnSize;
    }
  }

  // Relax stubs if necessary
  for (BinaryBasicBlock &BB : Func) {
    if (!Stubs[&Func].count(&BB) || !BB.isValid())
      continue;

    if (auto E = relaxStub(BB, Modified))
      return Error(std::move(E));
  }

  for (std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>> &Elmt :
       Insertions) {
    if (!Elmt.second)
      continue;
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
    NewBBs.emplace_back(std::move(Elmt.second));
    Func.insertBasicBlocks(Elmt.first, std::move(NewBBs), true);
  }

  return Error::success();
}

bool LongJmpPass::isBranchOffsetInRange(const BinaryFunction &Func,
                                        const MCInst &Inst,
                                        int64_t Offset) const {
  const unsigned Bits = Func.getBinaryContext().MIB->getPCRelEncodingSize(Inst);
  return isIntN(Bits, Offset);
}

bool LongJmpPass::isBlockInRange(const BinaryFunction &Func, const MCInst &Inst,
                                 uint64_t InstAddress,
                                 const BinaryBasicBlock &BB) const {
  const int64_t Offset = BB.getOutputStartAddress() - InstAddress;
  return isBranchOffsetInRange(Func, Inst, Offset);
}

void LongJmpPass::adjustBasicBlockAddress(BinaryBasicBlock &BB,
                                          uint64_t Address,
                                          uint64_t Offset) const {
  if (BB.getOutputStartAddress() < Address)
    return;
  BB.setOutputStartAddress(BB.getOutputStartAddress() + Offset);
  BB.setOutputEndAddress(BB.getOutputEndAddress() + Offset);
}

BinaryBasicBlock *LongJmpPass::addLocalTrampoline(
    LocalBranchState &State, BinaryBasicBlock *BB, const MCSymbol *TargetSymbol,
    BinaryBasicBlock *TargetBB, uint64_t Count, uint64_t Offset) const {
  constexpr uint64_t TrampolineSize = 4;

  BinaryFunction &Func = State.Func;
  BinaryContext &BC = Func.getBinaryContext();

  State.FunctionTrampolines.emplace_back(BB ? BB : State.FF.back(),
                                         Func.createBasicBlock());
  BinaryBasicBlock *TrampolineBB =
      State.FunctionTrampolines.back().second.get();

  const uint64_t OldBBEnd = BB ? BB->getOutputEndAddress() : 0;
  if (BB && Offset)
    BB->setOutputEndAddress(OldBBEnd + Offset);
  Offset += TrampolineSize;

  MCInst Inst;
  {
    auto L = BC.scopeLock();
    BC.MIB->createUncondBranch(Inst, TargetSymbol, BC.Ctx.get());
  }
  TrampolineBB->addInstruction(Inst);
  if (TargetBB)
    TrampolineBB->addSuccessor(TargetBB, Count);
  TrampolineBB->setExecutionCount(Count);

  const uint64_t TrampolineAddress =
      BB ? BB->getOutputEndAddress() : State.FragmentSize;
  TrampolineBB->setOutputStartAddress(TrampolineAddress);
  TrampolineBB->setOutputEndAddress(TrampolineAddress + TrampolineSize);
  TrampolineBB->setFragmentNum(State.FF.getFragmentNum());

  if (TargetBB && !State.Trampolines.lookup(TargetBB))
    State.Trampolines[TargetBB] = TrampolineBB;

  // A split unconditional branch removes four bytes from its original block
  // and adds the same four bytes in the trampoline.
  if (!Offset)
    return TrampolineBB;

  State.FragmentSize += Offset;

  // A fragment-end trampoline cannot move blocks in another fragment.
  if (!BB)
    return TrampolineBB;

  for (BinaryBasicBlock *IBB : State.FF)
    adjustBasicBlockAddress(*IBB, OldBBEnd, Offset);

  // Pending trampolines are not in the function layout yet, so update their
  // recorded fragment-local addresses explicitly.
  for (LocalTrampolineTy &Insertion : State.FunctionTrampolines) {
    BinaryBasicBlock &IBB = *Insertion.second;
    if (IBB.getFragmentNum() != TrampolineBB->getFragmentNum() ||
        &IBB == TrampolineBB)
      continue;
    adjustBasicBlockAddress(IBB, OldBBEnd, Offset);
  }

  return TrampolineBB;
}

void LongJmpPass::splitUnconditionalBranches(LocalBranchState &State) const {
  BinaryFunction &Func = State.Func;
  const MCPlusBuilder &MIB = *Func.getBinaryContext().MIB;

  for (BinaryBasicBlock *BB : State.FF) {
    MCInst *Inst = BB->getLastNonPseudoInstr();
    if (!Inst || !MIB.isUnconditionalBranch(*Inst))
      continue;

    const MCSymbol *TargetSymbol = MIB.getTargetSymbol(*Inst);
    BB->eraseInstruction(BB->findInstruction(Inst));

    BinaryBasicBlock::BinaryBranchInfo BI;
    BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol, BI);

    // Erasing the unconditional branch shrinks BB by one instruction. The new
    // trampoline restores those four bytes at the same layout position.
    BinaryBasicBlock *TrampolineBB = addLocalTrampoline(
        State, BB, TargetBB->getLabel(), TargetBB, BI.Count, /*Offset=*/-4);
    BB->replaceSuccessor(TargetBB, TrampolineBB, BI.Count);
  }
}

bool LongJmpPass::relaxLocalBranch(LocalBranchState &State,
                                   BinaryBasicBlock &BB, MCInst &Inst,
                                   uint64_t InstAddress,
                                   BinaryBasicBlock &TargetBB) const {
  BinaryFunction &Func = State.Func;
  BinaryContext &BC = Func.getBinaryContext();
  const MCPlusBuilder &MIB = *BC.MIB;

  // Use branch taken count for optimal relaxation.
  const uint64_t Count = BB.getBranchInfo(TargetBB).Count;
  assert(Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
         "Expected valid branch execution count");

  // Try to reuse an existing trampoline without introducing any new code.
  BinaryBasicBlock *TrampolineBB = State.Trampolines.lookup(&TargetBB);
  if (TrampolineBB && isBlockInRange(Func, Inst, InstAddress, *TrampolineBB)) {
    BB.replaceSuccessor(&TargetBB, TrampolineBB, Count);
    TrampolineBB->setExecutionCount(TrampolineBB->getExecutionCount() + Count);
    auto L = BC.scopeLock();
    BC.MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
    return false;
  }

  // Keep a branch that was never taken out of the surrounding code when the
  // fragment-end trampoline is reachable.
  const int64_t OffsetToEnd = State.FragmentSize - InstAddress;
  if (Count == 0 && isBranchOffsetInRange(Func, Inst, OffsetToEnd)) {
    TrampolineBB = addLocalTrampoline(State, nullptr, TargetBB.getLabel(),
                                      &TargetBB, Count);
    BB.replaceSuccessor(&TargetBB, TrampolineBB, Count);
    auto L = BC.scopeLock();
    BC.MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
    return true;
  }

  // If the other successor is a fall-through, invert the condition code.
  BinaryBasicBlock *NextBB =
      Func.getLayout().getBasicBlockAfter(&BB, /*IgnoreSplits=*/false);
  const bool PreserveFlags =
      State.BLI ? State.BLI->mustPreserveFlags(Inst) : true;
  const bool IsReversibleBranch = MIB.isReversibleBranch(Inst, PreserveFlags);
  const bool ShouldReverseBranch = BB.getConditionalSuccessor(false) == NextBB;

  // If the condition cannot be inverted, preserve the fall-through with a
  // separate trampoline before inserting the taken-edge trampoline.
  if (ShouldReverseBranch && !IsReversibleBranch) {
    const uint64_t NextCount = BB.getBranchInfo(*NextBB).Count;
    BinaryBasicBlock *FallThrough =
        addLocalTrampoline(State, &BB, NextBB->getLabel(), NextBB, NextCount);
    BB.replaceSuccessor(NextBB, FallThrough, NextCount);
  }

  if (ShouldReverseBranch && IsReversibleBranch) {
    const uint64_t OldBBSize = BB.estimateSize();
    BB.swapConditionalSuccessors();
    {
      auto L = BC.scopeLock();
      if (State.BLI)
        State.BLI->removeAnnotation(Inst);
      InstructionListType Code = MIB.reverseBranchCondition(
          Inst, NextBB->getLabel(), BC.Ctx.get(), PreserveFlags);
      BB.replaceInstruction(BB.findInstruction(&Inst), Code);
    }
    const uint64_t NewBBSize = BB.estimateSize();

    TrampolineBB = addLocalTrampoline(State, &BB, TargetBB.getLabel(),
                                      &TargetBB, Count, NewBBSize - OldBBSize);
  } else {
    TrampolineBB =
        addLocalTrampoline(State, &BB, TargetBB.getLabel(), &TargetBB, Count);
    auto L = BC.scopeLock();
    BC.MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
  }
  BB.replaceSuccessor(&TargetBB, TrampolineBB, Count);
  return true;
}

bool LongJmpPass::relaxExternalBranch(
    LocalBranchState &State, BinaryBasicBlock &BB, MCInst &Inst,
    uint64_t InstAddress,
    DenseMap<const MCSymbol *, BinaryBasicBlock *> &SymbolTrampolines) const {
  BinaryFunction &Func = State.Func;
  BinaryContext &BC = Func.getBinaryContext();
  const MCPlusBuilder &MIB = *BC.MIB;
  assert(BC.HasRelocations &&
         "external branch relaxation requires relocation mode");
  const MCSymbol *TargetSymbol = MIB.getTargetSymbol(Inst);

  // Internal branches in non-simple functions retain their original layout.
  if (Func.getBasicBlockForLabel(TargetSymbol))
    return true;

  BinaryBasicBlock *TrampolineBB = SymbolTrampolines.lookup(TargetSymbol);
  if (TrampolineBB && isBlockInRange(Func, Inst, InstAddress, *TrampolineBB)) {
    auto L = BC.scopeLock();
    BC.MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
    return true;
  }

  // The layout of a non-simple function has to be preserved, so the fragment
  // end is the only available insertion point.
  const int64_t OffsetToEnd = State.FragmentSize - InstAddress;
  const unsigned BitsAvailable = MIB.getPCRelEncodingSize(Inst);
  if (!isBranchOffsetInRange(Func, Inst, OffsetToEnd)) {
    auto L = BC.scopeLock();
    BC.errs() << "BOLT-ERROR: cannot relax branch in non-simple function "
              << Func << ": a trampoline at the end of the function is "
              << OffsetToEnd << " bytes away, out of reach for a "
              << BitsAvailable << "-bit branch\n";
    BC.printInstruction(BC.errs(), Inst);
    return false;
  }

  TrampolineBB = addLocalTrampoline(State, /*BB=*/nullptr, TargetSymbol,
                                    /*TargetBB=*/nullptr, /*Count=*/0);
  SymbolTrampolines[TargetSymbol] = TrampolineBB;
  auto L = BC.scopeLock();
  BC.MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
  return true;
}

bool LongJmpPass::relaxLocalFragment(
    BinaryFunction &Func, FunctionFragment &FF, const BranchLivenessInfo *BLI,
    LocalTrampolineListTy &FunctionTrampolines) const {
  BinaryContext &BC = Func.getBinaryContext();
  const MCPlusBuilder &MIB = *BC.MIB;

  // Use output BB address ranges to store fragment-relative offsets.
  uint64_t FragmentSize = 0;
  for (BinaryBasicBlock *BB : FF) {
    BB->setOutputStartAddress(FragmentSize);
    FragmentSize += BB->estimateSize();
    BB->setOutputEndAddress(FragmentSize);
  }

  LocalBranchState State{
      Func,         FF,
      BLI,          FunctionTrampolines,
      FragmentSize, DenseMap<const BinaryBasicBlock *, BinaryBasicBlock *>()};

  // In simple functions, make terminating unconditional branches available
  // for reuse as local trampolines.
  if (Func.isSimple())
    splitUnconditionalBranches(State);

  // Non-simple functions can branch to external symbols, which are tracked by
  // symbol rather than by destination basic block.
  DenseMap<const MCSymbol *, BinaryBasicBlock *> SymbolTrampolines;

  bool MayNeedRelaxation;
  uint64_t NumIterations = 0;
  do {
    MayNeedRelaxation = false;
    ++NumIterations;
    for (BinaryBasicBlock *BB : FF) {
      uint64_t NextInstOffset = BB->getOutputStartAddress();
      // Branch reversal may replace the current instruction with a sequence.
      // Use an index so the next instruction is reloaded after the mutation.
      for (size_t I = 0; I < BB->size(); ++I) {
        MCInst &Inst = *(BB->begin() + I);
        const uint64_t InstAddress = NextInstOffset;
        assert((!MIB.isPseudo(Inst) || MIB.isCFI(Inst)) &&
               "unexpected non-CFI pseudo in function fragment");
        if (!MIB.isPseudo(Inst))
          NextInstOffset += 4;

        if (!mayNeedStub(BC, Inst))
          continue;

        const unsigned BitsAvailable = MIB.getPCRelEncodingSize(Inst);

        // The compact model assumes every +/-128MB branch is in range.
        if (BitsAvailable == LongestJumpBits)
          continue;

        const MCSymbol *TargetSymbol = MIB.getTargetSymbol(Inst);
        if (Func.isSimple()) {
          BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol);
          assert(TargetBB &&
                 "Basic block target expected for conditional branch.");

          if (TargetBB->getFragmentNum() != FF.getFragmentNum() ||
              !isBlockInRange(Func, Inst, InstAddress, *TargetBB))
            MayNeedRelaxation |=
                relaxLocalBranch(State, *BB, Inst, InstAddress, *TargetBB);
        } else if (!relaxExternalBranch(State, *BB, Inst, InstAddress,
                                        SymbolTrampolines))
          return false;
      }
    }

    // We may have added instructions, but every branch fits if the complete
    // fragment remains smaller than the shortest branch span.
    if (State.FragmentSize < ShortestJumpSpan)
      MayNeedRelaxation = false;
  } while (MayNeedRelaxation);

  LLVM_DEBUG({
    if (NumIterations > 2)
      dbgs() << "BOLT-DEBUG: relaxed fragment " << FF.getFragmentNum().get()
             << " of " << Func << " in " << NumIterations << " iterations\n";
  });
  (void)NumIterations;
  return true;
}

void LongJmpPass::commitLocalTrampolines(
    BinaryFunction &Func, LocalTrampolineListTy &FunctionTrampolines) const {
  DenseMap<BinaryBasicBlock *, std::vector<std::unique_ptr<BinaryBasicBlock>>>
      Insertions;
  for (LocalTrampolineTy &Insertion : FunctionTrampolines) {
    if (Insertion.second)
      Insertions[Insertion.first].emplace_back(std::move(Insertion.second));
  }

  for (auto &Insertion : Insertions)
    Func.insertBasicBlocks(Insertion.first, std::move(Insertion.second),
                           /*UpdateLayout=*/true, /*UpdateCFI=*/true,
                           /*RecomputeLPs=*/false);
}

bool LongJmpPass::relaxLocalBranches(BinaryFunction &BF,
                                     const BranchLivenessInfo *BLI) {
  // Quick path. Only valid for simple functions, where all branch targets are
  // basic blocks of the function itself. A non-simple function may branch to a
  // symbol outside of it that ends up out of range.
  if (BF.isSimple() && !BF.isSplit() && BF.estimateSize() < ShortestJumpSpan)
    return true;

  LocalTrampolineListTy FunctionTrampolines;
  for (FunctionFragment &FF : BF.getLayout().fragments())
    if (!relaxLocalFragment(BF, FF, BLI, FunctionTrampolines))
      return false;

  commitLocalTrampolines(BF, FunctionTrampolines);
  return true;
}

namespace {
class ClusteredRelaxation {
public:
  ClusteredRelaxation(
      BinaryContext &BC, BinaryFunctionListType &OutputFunctions,
      const DenseMap<const FunctionFragment *, std::pair<uint64_t, uint64_t>>
          &FragmentAddresses,
      const DenseMap<const BinaryBasicBlock *, uint64_t> &ExactBBAddresses,
      Align MaxLayoutAlignment)
      : BC(BC), OutputFunctions(OutputFunctions),
        FragmentAddresses(FragmentAddresses),
        ExactBBAddresses(ExactBBAddresses),
        MaxLayoutAlignment(MaxLayoutAlignment) {
    MCInst BranchInst;
    BC.MIB->createUncondBranch(BranchInst, BC.Ctx->createTempSymbol(),
                               BC.Ctx.get());
    BranchThunkSize = BC.computeInstructionSize(BranchInst);

    InstructionListType LongThunk;
    BC.MIB->createLongTailCall(LongThunk, BC.Ctx->createTempSymbol(),
                               BC.Ctx.get());
    LongThunkSize = BC.computeCodeSize(LongThunk.begin(), LongThunk.end());
  }

  bool run();

private:
  /// A group of function fragments that are located within the longest direct
  /// branch/call instruction distance. Jumps within the cluster do not require
  /// a thunk. The cluster may span output sections and include thunks for jumps
  /// to targets outside. Backward thunks are inserted before the cluster, while
  /// forward thunks are inserted after it.
  struct FragmentCluster {
    /// Output code sections containing the first and last cluster fragments.
    SmallString<32> StartSectionName;
    SmallString<32> EndSectionName;

    /// Estimated size of the cluster in bytes.
    uint64_t Size{0};

    /// Estimated output offset of the cluster.
    uint64_t StartOffset{0};

    /// Number of function fragments in the cluster.
    size_t NumFragments{0};

    /// The indices of the first and last functions contributing fragments to
    /// this cluster. Used as insertion points for adding thunks to the output
    /// function list.
    size_t FirstFunctionIndex = -1;
    size_t LastFunctionIndex = -1;

    /// Thunks located after this cluster.
    BinaryFunctionListType ForwardThunkList;

    /// Thunks located before this cluster.
    BinaryFunctionListType BackwardThunkList;

    /// Long call thunks emitted by this cluster.
    ///
    /// <Target Symbol> -> <Thunk Function>.
    DenseMap<const MCSymbol *, BinaryFunction *> LongThunks;

    /// B-only thunks emitted by this cluster.
    ///
    /// <Target Symbol> -> <Thunk Function>.
    DenseMap<const MCSymbol *, BinaryFunction *> BranchThunks;

    StringRef getThunkSectionName(bool IsForward) const {
      return IsForward ? EndSectionName : StartSectionName;
    }

    uint64_t getEndOffset() const { return StartOffset + Size; }
  };

  struct Position {
    unsigned Cluster;
    uint64_t Offset;
  };

  /// Relaxation info for out-of-range cross-cluster references.
  struct OutOfRangeRef {
    MCInst *Inst;
    const MCSymbol *TargetSymbol;
    uint64_t SourceOffset;
    uint64_t TargetOffset;
    unsigned SourceCluster;
    unsigned TargetCluster;
  };

  static bool isWithinClusterRange(uint64_t SourceOffset,
                                   uint64_t TargetOffset);
  bool initializeThunkIslandSize();
  static unsigned getClusterDistance(unsigned A, unsigned B);
  bool buildLayout();
  bool measureThunkIslands();
  void applyThunkIslandSizes();
  void clearReferences();
  bool collectOutOfRangeReferences(bool ConservativeCandidates);
  const MCSymbol *getOrCreateBranchThunkChain(const OutOfRangeRef &Ref,
                                              unsigned MaxThunks);
  void relaxCalls();
  bool relaxUnconditionalBranches();
  void insertThunks();

  BinaryContext &BC;
  BinaryFunctionListType &OutputFunctions;
  const DenseMap<const FunctionFragment *, std::pair<uint64_t, uint64_t>>
      &FragmentAddresses;
  const DenseMap<const BinaryBasicBlock *, uint64_t> &ExactBBAddresses;

  /// Maximum alignment affecting the full code layout. Virtual thunk islands
  /// must preserve it when shifting all subsequent addresses.
  const Align MaxLayoutAlignment;

  /// Fixed per-island reservation used by the initial conservative layout.
  uint64_t MaxThunkIslandSize{0};

  /// Emitted sizes of the two thunk forms, calculated once per relaxation.
  uint64_t BranchThunkSize{0};
  uint64_t LongThunkSize{0};

  /// Measured aligned thunk-island sizes at every cluster boundary.
  SmallVector<uint64_t, 4> ThunkIslandSizes;

  /// Island sizes currently reflected in the estimated cluster layout.
  SmallVector<uint64_t, 4> AppliedThunkIslandSizes;

  /// Estimated cluster layout and source/target position maps.
  SmallVector<FragmentCluster, 4> Clusters;
  DenseMap<const BinaryBasicBlock *, Position> BBLayout;
  DenseMap<const MCSymbol *, Position> SymLayout;

  /// Out-of-range calls and branches grouped for relaxation.
  SmallVector<OutOfRangeRef> OutOfLayoutCalls;
  SmallVector<SmallVector<OutOfRangeRef>, 4> CallsByDistance;
  SmallVector<OutOfRangeRef> Branches;

  /// Relaxation counters reported in BOLT-INFO.
  size_t NumShortThunkCalls = 0;
  size_t NumLongThunkCalls = 0;
  size_t NumShortThunks = 0;
  size_t NumShortThunksReused = 0;
  size_t NumLongThunks = 0;
  size_t NumLongThunksReused = 0;
  size_t NumBranchThunks = 0;
  size_t NumBranchThunksReused = 0;
};
} // namespace

static constexpr uint64_t Branch26Span = 1ULL << 27;

bool ClusteredRelaxation::isWithinClusterRange(uint64_t SourceOffset,
                                               uint64_t TargetOffset) {
  const uint64_t Distance = SourceOffset <= TargetOffset
                                ? TargetOffset - SourceOffset
                                : SourceOffset - TargetOffset;
  return Distance < Branch26Span;
}

bool ClusteredRelaxation::initializeThunkIslandSize() {
  if (opts::MaxClusterSize >= Branch26Span) {
    BC.errs() << "BOLT-ERROR: --max-cluster-size must be smaller than "
              << Branch26Span << " bytes\n";
    return false;
  }

  // Split the remaining Branch26 range equally between the thunk islands on
  // either side of a cluster. Round down so the reservation remains inside
  // that range while shifting subsequent sections and fragments by a multiple
  // of every alignment represented by the full layout.
  const uint64_t MaxIslandSize = (Branch26Span - opts::MaxClusterSize) / 2;
  MaxThunkIslandSize = alignDown(MaxIslandSize, MaxLayoutAlignment.value());
  if (MaxThunkIslandSize == 0) {
    BC.errs() << "BOLT-ERROR: --max-cluster-size leaves only " << MaxIslandSize
              << " bytes per thunk island, less than the required layout "
                 "alignment of "
              << MaxLayoutAlignment.value() << " bytes\n";
    return false;
  }
  return true;
}

unsigned ClusteredRelaxation::getClusterDistance(unsigned A, unsigned B) {
  return A > B ? A - B : B - A;
}

bool ClusteredRelaxation::buildLayout() {
  struct OutputFragment {
    const FunctionFragment *FF;
    size_t FunctionIndex;
    SmallString<32> SectionName;
    std::pair<uint64_t, uint64_t> Addresses;
  };

  struct FragmentRange {
    size_t Begin;
    size_t End;
  };

  if (!initializeThunkIslandSize())
    return false;

  SmallVector<SmallVector<OutputFragment>, 4> FragmentsBySection;
  StringMap<size_t> SectionToBucket;
  auto addOrderedFragment = [&](OutputFragment &&Fragment) {
    auto [It, Inserted] = SectionToBucket.try_emplace(
        Fragment.SectionName, FragmentsBySection.size());
    if (Inserted)
      FragmentsBySection.emplace_back();
    FragmentsBySection[It->second].push_back(std::move(Fragment));
  };

  for (size_t I = 0; I < OutputFunctions.size(); ++I) {
    BinaryFunction *BF = OutputFunctions[I];
    if (!shouldEmitFunctionFragment(BC, *BF) || BF->isPatch())
      continue;

    for (const FunctionFragment &FF : BF->getLayout().fragments()) {
      if (FF.empty() && !BF->hasConstantIsland())
        continue;

      const auto It = FragmentAddresses.find(&FF);
      assert(It != FragmentAddresses.end() &&
             "missing function fragment from full layout");
      const std::pair<uint64_t, uint64_t> Addresses = It->second;
      assert(Addresses.first <= Addresses.second && "invalid fragment range");
      if (Addresses.second - Addresses.first > opts::MaxClusterSize) {
        BC.errs() << "BOLT-ERROR: function fragment " << BF->getPrintName()
                  << " has size " << Addresses.second - Addresses.first
                  << " bytes, exceeding --max-cluster-size="
                  << opts::MaxClusterSize << '\n';
        return false;
      }
      addOrderedFragment(
          {&FF, I, BF->getCodeSectionName(FF.getFragmentNum()), Addresses});
    }
  }

  // The full layout supplies the section order through the address of each
  // section's first fragment. Sort only sections, retaining emission order for
  // the potentially thousands of fragments within each section.
  SmallVector<size_t, 4> SectionOrder;
  for (size_t I = 0; I < FragmentsBySection.size(); ++I)
    SectionOrder.push_back(I);

  llvm::sort(SectionOrder, [&](size_t A, size_t B) {
    const OutputFragment &FirstA = FragmentsBySection[A].front();
    const OutputFragment &FirstB = FragmentsBySection[B].front();
    if (FirstA.Addresses.first != FirstB.Addresses.first)
      return FirstA.Addresses.first < FirstB.Addresses.first;
    return BC.compareSectionNames(FirstA.SectionName, FirstB.SectionName);
  });

  SmallVector<const OutputFragment *> OrderedFragments;
  for (size_t SectionIndex : SectionOrder)
    for (const OutputFragment &Fragment : FragmentsBySection[SectionIndex])
      OrderedFragments.push_back(&Fragment);

  // Fragment clusters are built starting from hot code.
  auto buildClusterRanges = [&]() {
    SmallVector<FragmentRange> ClusterRanges;
    if (OrderedFragments.empty())
      return ClusterRanges;

    // Hot fragments appear first, so perform forward walk.
    if (!opts::HotFunctionsAtEnd) {
      size_t Begin = 0;
      uint64_t ClusterStart = OrderedFragments.front()->Addresses.first;
      uint64_t ClusterEnd = OrderedFragments.front()->Addresses.second;
      for (size_t I = 1; I < OrderedFragments.size(); ++I) {
        const OutputFragment &Fragment = *OrderedFragments[I];
        const uint64_t NewEnd = std::max(ClusterEnd, Fragment.Addresses.second);
        if (NewEnd - ClusterStart > opts::MaxClusterSize) {
          ClusterRanges.push_back({Begin, I});
          Begin = I;
          ClusterStart = Fragment.Addresses.first;
          ClusterEnd = Fragment.Addresses.second;
          continue;
        }
        ClusterEnd = NewEnd;
      }
      ClusterRanges.push_back({Begin, OrderedFragments.size()});
      return ClusterRanges;
    }

    // Hot fragments appear last, so perform reverse walk.
    size_t End = OrderedFragments.size();
    uint64_t ClusterEnd = OrderedFragments.back()->Addresses.second;
    for (size_t I = End - 1; I > 0;) {
      --I;
      const OutputFragment &Fragment = *OrderedFragments[I];
      const uint64_t NewStart = Fragment.Addresses.first;
      const uint64_t NewEnd = std::max(ClusterEnd, Fragment.Addresses.second);
      if (NewEnd - NewStart > opts::MaxClusterSize) {
        ClusterRanges.push_back({I + 1, End});
        End = I + 1;
        ClusterEnd = Fragment.Addresses.second;
        continue;
      }
      ClusterEnd = NewEnd;
    }
    ClusterRanges.push_back({0, End});
    std::reverse(ClusterRanges.begin(), ClusterRanges.end());
    return ClusterRanges;
  };

  auto addFragmentToCluster = [&](const OutputFragment &Fragment,
                                  uint64_t IslandOffset) {
    FragmentCluster &FC = Clusters.back();
    const unsigned ClusterNum = Clusters.size() - 1;
    BinaryFunction &BF = *OutputFunctions[Fragment.FunctionIndex];
    const FunctionFragment &FF = *Fragment.FF;
    const uint64_t FragmentAddress = Fragment.Addresses.first + IslandOffset;

    if (FC.NumFragments == 0) {
      FC.StartSectionName = Fragment.SectionName;
      FC.StartOffset = FragmentAddress;
      FC.FirstFunctionIndex = Fragment.FunctionIndex;
    }

    FC.EndSectionName = Fragment.SectionName;
    FC.LastFunctionIndex = Fragment.FunctionIndex;
    ++FC.NumFragments;

    // Map primary entry points.
    if (FF.isMainFragment())
      for (const MCSymbol *Symbol : BF.getSymbols())
        SymLayout[Symbol] = {ClusterNum, FragmentAddress};

    for (const BinaryBasicBlock *BB : FF) {
      const uint64_t BBAddress = ExactBBAddresses.at(BB) + IslandOffset;
      BBLayout[BB] = {ClusterNum, BBAddress};
      // Map the local BB label.
      if (const MCSymbol *Label = BB->getLabel())
        SymLayout[Label] = {ClusterNum, BBAddress};
      // Map the secondary entry point, which can differ from the BB label.
      if (MCSymbol *Label = BF.getLabelAtOffset(BB->getOffset()))
        if (MCSymbol *EntrySymbol = BF.getSecondaryEntryPointSymbol(Label))
          SymLayout[EntrySymbol] = {ClusterNum, BBAddress};
    }
  };

  uint64_t IslandOffset = 0;
  for (const FragmentRange &Range : buildClusterRanges()) {
    Clusters.emplace_back();
    uint64_t ClusterEnd =
        OrderedFragments[Range.Begin]->Addresses.second + IslandOffset;
    for (size_t I = Range.Begin; I < Range.End; ++I) {
      ClusterEnd = std::max(ClusterEnd, OrderedFragments[I]->Addresses.second +
                                            IslandOffset);
      addFragmentToCluster(*OrderedFragments[I], IslandOffset);
    }
    FragmentCluster &FC = Clusters.back();
    FC.Size = ClusterEnd - FC.StartOffset;

    // Reserve a virtual thunk island between consecutive clusters. Forward
    // thunks from this cluster and backward thunks from the next cluster share
    // this space. The reservation is not emitted as padding: actual thunks
    // replace part of it, so the virtual layout conservatively bounds their
    // effect on all following addresses. Its size is a multiple of the
    // maximum alignment affecting the code layout, preserving every alignment
    // decision already made by the full layout.
    IslandOffset += MaxThunkIslandSize;
  }

  if (Clusters.empty())
    return true;

  // Print cluster stats.
  BC.outs() << "BOLT-INFO: built " << Clusters.size()
            << " function fragment cluster(s)\n";
  for (size_t I = 0; I < Clusters.size(); ++I) {
    const FragmentCluster &FC = Clusters[I];
    BC.outs() << "BOLT-INFO: cluster: " << I << '\n'
              << "BOLT-INFO:   " << FC.NumFragments << " fragment(s)\n"
              << "BOLT-INFO:   " << FC.Size << " estimated bytes\n";
  }
  return true;
}

bool ClusteredRelaxation::measureThunkIslands() {
  // Boundary 0 precedes the first cluster, boundary N follows the last, and
  // boundary I is shared by forward thunks from cluster I - 1 and backward
  // thunks from cluster I.
  const size_t NumBoundaries = Clusters.size() + 1;
  SmallVector<DenseSet<const MCSymbol *>, 4> BranchThunkTargets(NumBoundaries);
  SmallVector<DenseSet<const MCSymbol *>, 4> LongThunkTargets(NumBoundaries);

  if (BranchThunkSize == 0 || LongThunkSize == 0) {
    BC.errs() << "BOLT-ERROR: unable to calculate emitted thunk sizes\n";
    return false;
  }

  auto reserveBranchChain = [&](const OutOfRangeRef &Ref) {
    if (Ref.TargetCluster == -1u) {
      if (Ref.TargetOffset < Ref.SourceOffset) {
        for (unsigned Boundary = 0; Boundary <= Ref.SourceCluster; ++Boundary)
          BranchThunkTargets[Boundary].insert(Ref.TargetSymbol);
      } else {
        for (unsigned Boundary = Ref.SourceCluster + 1;
             Boundary < NumBoundaries; ++Boundary)
          BranchThunkTargets[Boundary].insert(Ref.TargetSymbol);
      }
      return;
    }

    const unsigned First = std::min(Ref.SourceCluster, Ref.TargetCluster) + 1;
    const unsigned Last = std::max(Ref.SourceCluster, Ref.TargetCluster);
    for (unsigned Boundary = First; Boundary <= Last; ++Boundary)
      BranchThunkTargets[Boundary].insert(Ref.TargetSymbol);
  };

  auto reserveLongThunk = [&](const OutOfRangeRef &Ref) {
    const bool IsForward = Ref.TargetCluster == -1u
                               ? Ref.SourceOffset < Ref.TargetOffset
                               : Ref.SourceCluster < Ref.TargetCluster;
    const unsigned Boundary =
        IsForward ? Ref.SourceCluster + 1 : Ref.SourceCluster;
    LongThunkTargets[Boundary].insert(Ref.TargetSymbol);
  };

  // Reserve both possible call implementations. The final, smaller layout may
  // select different ladder rungs, but it cannot use more than one B thunk per
  // crossed boundary or more than one long thunk at the source boundary.
  for (const SmallVector<OutOfRangeRef> &Calls : CallsByDistance)
    for (const OutOfRangeRef &Call : Calls) {
      reserveBranchChain(Call);
      reserveLongThunk(Call);
    }

  for (const OutOfRangeRef &Call : OutOfLayoutCalls)
    reserveLongThunk(Call);

  for (const OutOfRangeRef &Branch : Branches)
    reserveBranchChain(Branch);

  ThunkIslandSizes.clear();
  ThunkIslandSizes.reserve(NumBoundaries);
  for (size_t I = 0; I < NumBoundaries; ++I) {
    const uint64_t NumBranchThunks = BranchThunkTargets[I].size();
    const uint64_t NumLongThunks = LongThunkTargets[I].size();
    if (NumBranchThunks > MaxThunkIslandSize / BranchThunkSize ||
        NumLongThunks > MaxThunkIslandSize / LongThunkSize) {
      BC.errs() << "BOLT-ERROR: too many candidate thunks for boundary " << I
                << '\n';
      return false;
    }
    const uint64_t BranchBytes = NumBranchThunks * BranchThunkSize;
    const uint64_t LongBytes = NumLongThunks * LongThunkSize;
    if (LongBytes > MaxThunkIslandSize - BranchBytes) {
      BC.errs() << "BOLT-ERROR: thunk island at boundary " << I
                << " requires more than its " << MaxThunkIslandSize
                << "-byte Branch26 reservation\n";
      return false;
    }
    const uint64_t Bytes = BranchBytes + LongBytes;
    const uint64_t Size = Bytes ? alignTo(Bytes, MaxLayoutAlignment) : 0;
    if (Size > MaxThunkIslandSize) {
      BC.errs() << "BOLT-ERROR: thunk island at boundary " << I << " requires "
                << Size << " aligned bytes, exceeding its "
                << MaxThunkIslandSize << "-byte Branch26 reservation\n";
      return false;
    }
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: thunk island at boundary " << I
                      << " reserves " << Size << " aligned bytes for "
                      << NumBranchThunks << " branch thunks and "
                      << NumLongThunks << " long thunks\n");
    ThunkIslandSizes.push_back(Size);
  }

  return true;
}

void ClusteredRelaxation::applyThunkIslandSizes() {
  assert(ThunkIslandSizes.size() == Clusters.size() + 1 &&
         "one thunk island expected at every cluster boundary");
  assert(AppliedThunkIslandSizes.size() == Clusters.size() + 1 &&
         "current thunk island size expected at every cluster boundary");

  SmallVector<uint64_t, 4> OldOffsets;
  SmallVector<uint64_t, 4> NewOffsets;
  OldOffsets.reserve(Clusters.size());
  NewOffsets.reserve(Clusters.size());

  uint64_t OldOffset = AppliedThunkIslandSizes.front();
  uint64_t NewOffset = ThunkIslandSizes.front();
  for (size_t I = 0; I < Clusters.size(); ++I) {
    OldOffsets.push_back(OldOffset);
    NewOffsets.push_back(NewOffset);
    OldOffset += AppliedThunkIslandSizes[I + 1];
    NewOffset += ThunkIslandSizes[I + 1];
  }

  auto adjustPosition = [&](Position &Pos) {
    assert(Pos.Cluster < Clusters.size() && "invalid position cluster");
    assert(Pos.Offset >= OldOffsets[Pos.Cluster] &&
           "fixed island offset exceeds position");
    Pos.Offset = Pos.Offset - OldOffsets[Pos.Cluster] + NewOffsets[Pos.Cluster];
  };

  for (auto &Entry : BBLayout)
    adjustPosition(Entry.second);
  for (auto &Entry : SymLayout)
    adjustPosition(Entry.second);

  for (size_t I = 0; I < Clusters.size(); ++I) {
    FragmentCluster &Cluster = Clusters[I];
    assert(Cluster.StartOffset >= OldOffsets[I] &&
           "fixed island offset exceeds cluster address");
    Cluster.StartOffset = Cluster.StartOffset - OldOffsets[I] + NewOffsets[I];
  }

  AppliedThunkIslandSizes = ThunkIslandSizes;
}

void ClusteredRelaxation::clearReferences() {
  OutOfLayoutCalls.clear();
  CallsByDistance.clear();
  Branches.clear();
}

bool ClusteredRelaxation::collectOutOfRangeReferences(
    bool ConservativeCandidates) {
  CallsByDistance.resize(Clusters.size());
  auto isPrimaryEntryTarget = [&](const MCSymbol *TargetSymbol) {
    uint64_t EntryID = 0;
    const BinaryFunction *BF = BC.getFunctionForSymbol(TargetSymbol, &EntryID);
    return BF && EntryID == 0;
  };

  // Walk all instructions once and collect both branches and calls.
  for (BinaryFunction *BF : OutputFunctions) {
    if (!BC.shouldEmit(*BF) || BF->isPatch())
      continue;

    for (BinaryBasicBlock &BB : *BF) {
      auto SourceIt = BBLayout.find(&BB);
      if (SourceIt == BBLayout.end())
        continue;
      const Position Source = SourceIt->second;
      uint64_t InstOffset = Source.Offset;

      for (MCInst &Inst : BB) {
        const uint64_t SourceOffset = InstOffset;
        if (!BC.MIB->isPseudo(Inst))
          InstOffset += 4;

        const bool IsCall = BC.MIB->isCall(Inst);
        const bool IsTailCall = BC.MIB->isTailCall(Inst);
        const bool IsUncondBranch = BC.MIB->isUnconditionalBranch(Inst);
        if (!IsCall && !IsUncondBranch)
          continue;

        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
        if (!TargetSymbol)
          continue;

        auto TargetIt = SymLayout.find(TargetSymbol);
        const bool Found = TargetIt != SymLayout.end();
        uint64_t TargetOffset = -1ULL;
        if (Found) {
          TargetOffset = TargetIt->second.Offset;
        } else if (ErrorOr<uint64_t> Value = BC.getSymbolValue(*TargetSymbol)) {
          TargetOffset = *Value;
        }
        const Position Target =
            Found ? TargetIt->second : Position{-1u, TargetOffset};

        // The fixed reservations conservatively bound distances within the
        // clustered output. Always retain references to targets outside that
        // layout during the sizing pass: replacing the reservations can shift
        // all output code relative to such a fixed target in either direction.
        const bool IsInRange =
            TargetOffset != -1ULL &&
            ((Found && Source.Cluster == Target.Cluster) ||
             isWithinClusterRange(SourceOffset, TargetOffset));
        if (IsInRange && (Found || !ConservativeCandidates))
          continue;

        const OutOfRangeRef Reference{&Inst,          TargetSymbol,
                                      SourceOffset,   Target.Offset,
                                      Source.Cluster, Target.Cluster};
        const bool UseBranchChain =
            IsUncondBranch ||
            (IsTailCall && !isPrimaryEntryTarget(TargetSymbol));
        if (UseBranchChain) {
          if (!Found && TargetOffset == -1ULL) {
            BC.errs() << "BOLT-ERROR: cannot build a branch thunk chain to "
                      << TargetSymbol->getName()
                      << " without a known target address\n";
            return false;
          }
          // A direct B may be annotated as a tail call, but it is not
          // necessarily an ABI call boundary. Use branch chains rather than
          // long call thunks that may clobber x16/x17.
          Branches.push_back(Reference);
          continue;
        }

        assert(IsCall && "expected call after branch-chain handling");

        if (Reference.TargetCluster == -1u) {
          OutOfLayoutCalls.push_back(Reference);
        } else {
          const unsigned Distance = getClusterDistance(Reference.SourceCluster,
                                                       Reference.TargetCluster);
          CallsByDistance[Distance].push_back(Reference);
        }
      }
    }
  }
  return true;
}

const MCSymbol *
ClusteredRelaxation::getOrCreateBranchThunkChain(const OutOfRangeRef &Ref,
                                                 unsigned MaxThunks) {
  const unsigned SourceCluster = Ref.SourceCluster;
  const unsigned TargetCluster = Ref.TargetCluster;
  const bool IsOutsideLayout = TargetCluster == -1u;
  const bool IsForward = IsOutsideLayout ? Ref.SourceOffset < Ref.TargetOffset
                                         : SourceCluster < TargetCluster;
  const bool IsCall = BC.MIB->isCall(*Ref.Inst);
  const unsigned NumHops =
      IsOutsideLayout
          ? (IsForward ? Clusters.size() - SourceCluster : SourceCluster + 1)
          : getClusterDistance(SourceCluster, TargetCluster);

  auto createBranchThunk = [&](const MCSymbol *TargetSymbol) {
    const size_t ThunkNumber = IsCall ? NumShortThunks++ : NumBranchThunks++;
    std::string ThunkName =
        (Twine("__AArch64_") + (IsForward ? "forward_" : "backward_") +
         "Thunk_" + (IsCall ? Twine(Ref.TargetSymbol->getName()) + "_" : "") +
         Twine(ThunkNumber))
            .str();

    BinaryFunction *ThunkBF = BC.createThunkBinaryFunction(ThunkName);
    MCInst Inst;
    BC.MIB->createUncondBranch(Inst, TargetSymbol, BC.Ctx.get());
    ThunkBF->addBasicBlock()->addInstruction(Inst);
    return ThunkBF;
  };

  auto registerBranchThunk = [&](FragmentCluster &Cluster,
                                 const MCSymbol *TargetSymbol,
                                 BinaryFunction *Thunk) {
    uint64_t EntryID = 0;
    const BinaryFunction *BF =
        IsCall ? BC.getFunctionForSymbol(TargetSymbol, &EntryID) : nullptr;
    // Call-style branch thunks may be reused through primary-entry aliases.
    if (BF && EntryID == 0)
      for (const MCSymbol *Symbol : BF->getSymbols())
        Cluster.BranchThunks[Symbol] = Thunk;
    else
      Cluster.BranchThunks[TargetSymbol] = Thunk;
  };

  auto getOrCreateBranchThunk = [&](FragmentCluster &Cluster,
                                    const MCSymbol *NextTarget) {
    auto It = Cluster.BranchThunks.find(Ref.TargetSymbol);
    if (It != Cluster.BranchThunks.end()) {
      if (IsCall)
        ++NumShortThunksReused;
      else
        ++NumBranchThunksReused;
      return It->second;
    }

    BinaryFunction *Thunk = createBranchThunk(NextTarget);
    Thunk->setCodeSectionName(Cluster.getThunkSectionName(IsForward));
    auto &ThunkList =
        IsForward ? Cluster.ForwardThunkList : Cluster.BackwardThunkList;
    ThunkList.push_back(Thunk);
    registerBranchThunk(Cluster, Ref.TargetSymbol, Thunk);
    return Thunk;
  };

  auto getClusterAtHop = [&](const unsigned Hop) {
    return IsForward ? SourceCluster + Hop : SourceCluster - Hop;
  };

  auto getThunkOffset = [&](const unsigned Cluster) {
    const FragmentCluster &FC = Clusters[Cluster];
    // BinaryEmitter aligns every injected thunk as a function. A backward
    // thunk is modeled at an already aligned fragment start; a forward thunk
    // following arbitrary fragment contents may require explicit padding.
    return IsForward ? alignTo(FC.getEndOffset(),
                               Align(BC.MIB->getMinFunctionAlignment()))
                     : FC.StartOffset;
  };

  SmallVector<unsigned> ThunkClusters;
  uint64_t CurrentOffset = Ref.SourceOffset;
  unsigned NextHop = 0;
  while (!isWithinClusterRange(CurrentOffset, Ref.TargetOffset)) {
    // Stop if the chain would exceed the configured thunk budget.
    if (ThunkClusters.size() == MaxThunks)
      return nullptr;

    unsigned BestHop = -1u;
    for (unsigned Hop = NextHop; Hop < NumHops; ++Hop) {
      const unsigned Cluster = getClusterAtHop(Hop);
      if (!isWithinClusterRange(CurrentOffset, getThunkOffset(Cluster)))
        break;
      BestHop = Hop;
    }

    if (BestHop == -1u)
      return nullptr;

    const unsigned BestCluster = getClusterAtHop(BestHop);
    ThunkClusters.push_back(BestCluster);
    CurrentOffset = getThunkOffset(BestCluster);
    NextHop = BestHop + 1;
  }

  BinaryFunction *FirstThunk = nullptr;
  const MCSymbol *NextTarget = Ref.TargetSymbol;
  // Reverse walk since each thunk depends on its target.
  for (const unsigned Cluster : llvm::reverse(ThunkClusters)) {
    FirstThunk = getOrCreateBranchThunk(Clusters[Cluster], NextTarget);
    NextTarget = FirstThunk->getSymbol();
  }

  return FirstThunk ? FirstThunk->getSymbol() : Ref.TargetSymbol;
}

void ClusteredRelaxation::relaxCalls() {
  auto createLongThunk = [&](const MCSymbol *TargetSymbol, bool IsForward) {
    BinaryFunction *Thunk = nullptr;
    const size_t ThunkNumber = NumLongThunks++;
    std::string ThunkName =
        (Twine("__AArch64_") + (IsForward ? "forward_" : "backward_") +
         "ADRPThunk_" + TargetSymbol->getName() + "_" + Twine(ThunkNumber))
            .str();
    Thunk = BC.createThunkBinaryFunction(ThunkName);
    InstructionListType Instructions;
    BC.MIB->createLongTailCall(Instructions, TargetSymbol, BC.Ctx.get());
    Thunk->addBasicBlock()->addInstructions(Instructions);
    return Thunk;
  };

  auto registerLongThunk = [&](FragmentCluster &FC, const MCSymbol *Callee,
                               BinaryFunction *Thunk) {
    uint64_t EntryID = 0;
    const BinaryFunction *BF = BC.getFunctionForSymbol(Callee, &EntryID);
    const bool IsPrimaryEntry = EntryID == 0;
    // Register thunks for all symbols associated with the function.
    if (BF && IsPrimaryEntry)
      for (const MCSymbol *Symbol : BF->getSymbols())
        FC.LongThunks[Symbol] = Thunk;
    else
      FC.LongThunks[Callee] = Thunk;
  };

  auto getOrCreateLongThunk = [&](const OutOfRangeRef &Call) {
    const MCSymbol *TargetSymbol = Call.TargetSymbol;
    const unsigned SourceCluster = Call.SourceCluster;
    const bool IsForward = Call.TargetCluster == -1u
                               ? Call.SourceOffset < Call.TargetOffset
                               : SourceCluster < Call.TargetCluster;
    FragmentCluster &FC = Clusters[SourceCluster];
    if (auto It = FC.LongThunks.find(TargetSymbol); It != FC.LongThunks.end()) {
      ++NumLongThunksReused;
      return It->second;
    }

    // Reuse long thunks hosted at the adjacent cluster boundary.
    FragmentCluster *ReuseCluster = nullptr;
    if (IsForward && SourceCluster > 0)
      ReuseCluster = &Clusters[SourceCluster - 1];
    else if (!IsForward && SourceCluster + 1 < Clusters.size())
      ReuseCluster = &Clusters[SourceCluster + 1];

    if (ReuseCluster) {
      auto It = ReuseCluster->LongThunks.find(TargetSymbol);
      if (It != ReuseCluster->LongThunks.end()) {
        BinaryFunction *Thunk = It->second;
        ++NumLongThunksReused;
        return Thunk;
      }
    }

    BinaryFunction *Thunk = createLongThunk(TargetSymbol, IsForward);

    Thunk->setCodeSectionName(FC.getThunkSectionName(IsForward));
    BinaryFunctionListType &ThunkList =
        IsForward ? FC.ForwardThunkList : FC.BackwardThunkList;
    ThunkList.push_back(Thunk);
    registerLongThunk(FC, TargetSymbol, Thunk);
    return Thunk;
  };

  auto relaxCallWithLongThunk = [&](OutOfRangeRef &Call) {
    BinaryFunction *Thunk = getOrCreateLongThunk(Call);
    BC.MIB->replaceBranchTarget(*Call.Inst, Thunk->getSymbol(), BC.Ctx.get());
  };

  // Process out-of-layout targets first.
  for (OutOfRangeRef &Call : OutOfLayoutCalls)
    relaxCallWithLongThunk(Call);

  NumLongThunkCalls += OutOfLayoutCalls.size();
  // Process known targets from farthest to nearest for maximizing thunk reuse.
  for (auto &Calls : llvm::reverse(CallsByDistance)) {
    for (OutOfRangeRef &Call : Calls) {
      // Attempt a branch thunk chain if length is below threshold.
      if (const MCSymbol *Target =
              getOrCreateBranchThunkChain(Call, opts::MaxThunkChainLength)) {
        BC.MIB->replaceBranchTarget(*Call.Inst, Target, BC.Ctx.get());
        ++NumShortThunkCalls;
        continue;
      }
      // Otherwise fall back to a long thunk.
      ++NumLongThunkCalls;
      relaxCallWithLongThunk(Call);
    }
  }

  if (NumShortThunkCalls)
    BC.outs() << "BOLT-INFO: relaxed " << NumShortThunkCalls
              << " calls with short thunks\n";

  if (NumLongThunkCalls)
    BC.outs() << "BOLT-INFO: relaxed " << NumLongThunkCalls
              << " calls with long thunks\n";

  if (NumShortThunks)
    BC.outs() << "BOLT-INFO: " << NumShortThunks << " short thunks created\n";

  if (NumLongThunks)
    BC.outs() << "BOLT-INFO: " << NumLongThunks << " long thunks created\n";

  if (NumShortThunksReused)
    BC.outs() << "BOLT-INFO: " << NumShortThunksReused
              << " short thunks reused\n";

  if (NumLongThunksReused)
    BC.outs() << "BOLT-INFO: " << NumLongThunksReused
              << " long thunks reused\n";
}

bool ClusteredRelaxation::relaxUnconditionalBranches() {
  for (const OutOfRangeRef &Branch : Branches) {
    if (BC.MIB->isTailCall(*Branch.Inst))
      BC.MIB->convertTailCallToJmp(*Branch.Inst);
    const MCSymbol *Target =
        getOrCreateBranchThunkChain(Branch, /*MaxThunks=*/-1u);
    if (!Target) {
      BC.errs() << "BOLT-ERROR: unable to build branch thunk chain from 0x"
                << Twine::utohexstr(Branch.SourceOffset) << " in cluster "
                << Branch.SourceCluster << " to "
                << Branch.TargetSymbol->getName() << " at 0x"
                << Twine::utohexstr(Branch.TargetOffset) << " in cluster ";
      if (Branch.TargetCluster == -1u)
        BC.errs() << "outside the clustered layout\n";
      else
        BC.errs() << Branch.TargetCluster << '\n';
      return false;
    }
    BC.MIB->replaceBranchTarget(*Branch.Inst, Target, BC.Ctx.get());
  }

  if (!Branches.empty())
    BC.outs() << "BOLT-INFO: relaxed " << Branches.size()
              << " unconditional branches\n";

  if (NumBranchThunks)
    BC.outs() << "BOLT-INFO: " << NumBranchThunks << " branch thunks created\n";

  if (NumBranchThunksReused)
    BC.outs() << "BOLT-INFO: " << NumBranchThunksReused
              << " branch thunks reused\n";
  return true;
}

void ClusteredRelaxation::insertThunks() {
  struct ThunkInsertion {
    size_t Position;
    bool IsForward;
    BinaryFunctionListType *ThunkList;
  };

  SmallVector<ThunkInsertion> Insertions;
  for (FragmentCluster &Cluster : Clusters) {
    if (!Cluster.BackwardThunkList.empty())
      Insertions.push_back({Cluster.FirstFunctionIndex, /*IsForward=*/false,
                            &Cluster.BackwardThunkList});

    if (!Cluster.ForwardThunkList.empty())
      Insertions.push_back({Cluster.LastFunctionIndex + 1,
                            /*IsForward=*/true, &Cluster.ForwardThunkList});
  }

  // Apply insertions from high to low indices so earlier insertions do not
  // invalidate later positions. At a shared boundary, repeated insertion at the
  // same index reverses application order, yielding forward thunks before
  // backward thunks.
  llvm::sort(Insertions, [](const ThunkInsertion &A, const ThunkInsertion &B) {
    if (A.Position != B.Position)
      return A.Position > B.Position;
    return !A.IsForward && B.IsForward;
  });

  for (ThunkInsertion &Insertion : Insertions)
    OutputFunctions.insert(
        std::next(OutputFunctions.begin(), Insertion.Position),
        Insertion.ThunkList->begin(), Insertion.ThunkList->end());
}

bool ClusteredRelaxation::run() {
  if (!buildLayout())
    return false;
  if (Clusters.empty())
    return true;

  // buildLayout() reserves no island before the first cluster and a maximum
  // sized island at every following boundary. The trailing size does not move
  // code, but keeping it here makes the boundary vectors uniform. With no
  // remeasurement requested, the final range checks use these reservations.
  AppliedThunkIslandSizes.assign(Clusters.size() + 1, MaxThunkIslandSize);
  AppliedThunkIslandSizes.front() = 0;

  // Use the fixed maximum reservations to find a conservative candidate set.
  // Repeatedly measure an upper bound for the required thunks and replace the
  // current reservations with the aligned measured islands. Every measured
  // internal island is no larger than the reservation it replaces, so no new
  // cross-cluster candidate can appear. References to fixed targets outside
  // the clustered output are retained unconditionally during every sizing
  // pass because moving the output can change their range in either direction.
  unsigned NumRemeasurements = 0;
  bool Stabilized = false;
  for (unsigned I = 0; I < opts::MaxThunkRemeasure; ++I) {
    if (!collectOutOfRangeReferences(/*ConservativeCandidates=*/true) ||
        !measureThunkIslands())
      return false;

    ++NumRemeasurements;
    const bool AtFixedPoint = ThunkIslandSizes == AppliedThunkIslandSizes;
    applyThunkIslandSizes();
    clearReferences();
    if (AtFixedPoint) {
      Stabilized = true;
      break;
    }
  }

  if (opts::MaxThunkRemeasure == 0) {
    BC.outs() << "BOLT-INFO: thunk island layout remeasurement disabled\n";
  } else {
    BC.outs() << "BOLT-INFO: thunk island layout "
              << (Stabilized ? "stabilized" : "did not stabilize") << " after "
              << NumRemeasurements << " remeasurement iteration"
              << (NumRemeasurements == 1 ? "" : "s") << '\n';
  }

  // Perform the final exact range checks against the last measured layout.
  if (!collectOutOfRangeReferences(/*ConservativeCandidates=*/false))
    return false;

  relaxCalls();
  if (!relaxUnconditionalBranches())
    return false;
  insertThunks();

  LLVM_DEBUG(dbgs() << "\nFunction layout with thunks:\n";
             for (const auto *BF : OutputFunctions) { dbgs() << *BF << '\n'; });

  return true;
}

bool LongJmpPass::relaxWithClusters(BinaryContext &BC) {
  BinaryFunctionListType OutputFunctions = BC.getOutputBinaryFunctions();
  layout(BC, OutputFunctions);

  // Inserting a virtual thunk island shifts all following addresses. Use the
  // maximum alignment affecting the full layout so that one aligned
  // reservation preserves its section, function, basic-block, and
  // constant-island alignment decisions.
  uint64_t MaxLayoutAlignment = BC.MIB->getMinFunctionAlignment();
  for (const BinaryFunction *Func : OutputFunctions) {
    if (!shouldEmitFunctionFragment(BC, *Func) || Func->isPatch())
      continue;

    for (const FunctionFragment &FF : Func->getLayout().fragments()) {
      if (FF.empty() && !Func->hasConstantIsland())
        continue;
      MaxLayoutAlignment =
          updateSectionAlignment(BC, *Func, FF, MaxLayoutAlignment);
      // RewriteInstance::mapCodeSectionsInPlace() uses this alignment for
      // moved cold fragments independently of BinaryEmitter's requirements.
      if (!BC.HasRelocations && FF.isSplitFragment())
        MaxLayoutAlignment =
            std::max(MaxLayoutAlignment, ColdFragmentAlignment.value());
    }
  }
  ClusteredRelaxation Relaxation(BC, OutputFunctions, FragmentAddresses,
                                 BBAddresses, Align(MaxLayoutAlignment));
  if (!Relaxation.run())
    return false;

  BC.updateOutputBinaryFunctions(std::move(OutputFunctions));
  return true;
}

Error LongJmpPass::runOnFunctions(BinaryContext &BC) {
  DenseMap<BinaryFunction *, BranchLivenessInfo> BranchLiveness;

  if (opts::FixBranchesWithLiveness) {
    SmallVector<BinaryFunction *> Candidates;
    for (auto &It : BC.getBinaryFunctions()) {
      BinaryFunction &BF = It.second;
      if (BC.shouldEmit(BF) && BF.isSimple() && needsBranchLiveness(BF))
        Candidates.push_back(&BF);
    }
    if (!Candidates.empty()) {
      RegAnalysis RA(BC, nullptr, nullptr);
      for (BinaryFunction *BF : Candidates)
        BranchLiveness.try_emplace(BF, computeBranchLiveness(*BF, RA));
    }
  }

  auto getBranchLiveness = [&](BinaryFunction &BF) {
    auto It = BranchLiveness.find(&BF);
    return It == BranchLiveness.end() ? nullptr : &It->second;
  };

  if (opts::CompactCodeModel || opts::ExperimentalRelaxation) {
    BC.outs()
        << "BOLT-INFO: relaxing branches for compact code model (<128MB)\n";

    std::atomic<bool> HasFatal{false};
    ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
      if (HasFatal)
        return;
      if (!relaxLocalBranches(BF, getBranchLiveness(BF)))
        HasFatal = true;
    };

    ParallelUtilities::PredicateTy SkipPredicate =
        [&](const BinaryFunction &BF) { return !BC.shouldEmit(BF); };

    ParallelUtilities::runOnEachFunction(
        BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
        SkipPredicate, "RelaxLocalBranches");

    // The error has already been reported by relaxLocalBranches().
    if (HasFatal)
      return createFatalBOLTError("branch relaxation failure");

    if (!opts::ExperimentalRelaxation)
      return Error::success();

    BC.outs() << "BOLT-INFO: starting experimental relaxation pass\n";
    if (!relaxWithClusters(BC))
      return createFatalBOLTError("clustered branch relaxation failure");
    return Error::success();
  }

  BC.outs() << "BOLT-INFO: Starting stub-insertion pass\n";
  BinaryFunctionListType Sorted = BC.getOutputBinaryFunctions();
  bool Modified;
  uint32_t Iterations = 0;
  do {
    ++Iterations;
    Modified = false;
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp: layout iteration " << Iterations
                      << '\n');
    layout(BC, Sorted);
    updateStubGroups();
    for (BinaryFunction *Func : Sorted) {
      if (auto E = relax(*Func, Modified))
        return Error(std::move(E));
      // Don't ruin non-simple functions, they can't afford to have the layout
      // changed.
      if (Modified && Func->isSimple())
        Func->fixBranches(getBranchLiveness(*Func));
    }
  } while (Modified);
  BC.outs() << "BOLT-INFO: Inserted " << NumHotStubs
            << " stubs in the hot area and " << NumColdStubs
            << " stubs in the cold area. Shared " << NumSharedStubs
            << " times, iterated " << Iterations << " times.\n";
  return Error::success();
}
} // namespace bolt
} // namespace llvm
