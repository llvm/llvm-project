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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
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

static cl::opt<bool> RelaxPLT("relax-plt",
                              cl::desc("indicate PLT proximity to hot text"),
                              cl::init(true), cl::cat(BoltOptCategory));
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
        assert(FF.getFragmentNum() == FragmentNum::cold() &&
               "LongJmp supports only main and cold function fragments");
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
  assert((Fragment == FragmentNum::main() || Fragment == FragmentNum::cold()) &&
         "LongJmp supports only main and cold function fragments");
  const bool IsCold = Fragment == FragmentNum::cold();

  const bool HasFixedOutputAddress =
      Func.isInjected() && Func.getOutputAddress();
  const bool NeedsRelocationAlignment =
      BC.HasRelocations && !HasFixedOutputAddress;
  const bool NeedsNonRelocInjectedAlignment =
      !BC.HasRelocations && Func.isInjected() && !HasFixedOutputAddress;
  const bool NeedsNonRelocColdAlignment =
      !BC.HasRelocations && !Func.isInjected() && IsCold;

  // Apply the alignment that affects the fragment's mapped address.
  // Section-relative fragments mirror BinaryEmitter::emitFunction(); moved
  // cold fragments mirror mapCodeSectionsInPlace(). Ordinary non-relocation
  // main fragments and fixed-address injected functions already have exact
  // addresses.
  if (NeedsRelocationAlignment) {
    DotAddress = alignTo(DotAddress, Func.getMinAlignment());
    const uint16_t MaxAlignmentBytes =
        IsCold ? Func.getMaxColdAlignmentBytes() : Func.getMaxAlignmentBytes();
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

  // BinaryEmitter::emitFunction() emits the fragment entry symbols here.
  if (RecordAddresses) {
    if (!IsCold)
      HotAddresses[&Func] = DotAddress;
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: LongJmp layout: "
                      << (IsCold ? "cold" : "main") << " fragment "
                      << Func.getPrintName() << " starts at 0x"
                      << Twine::utohexstr(DotAddress) << '\n');
  }

  // --break-funcs emits UD2 before the function body.
  DotAddress += opts::breakFunctionSize(Func);

  DotAddress = layoutFunctionBody(BC, Func, FF, DotAddress, RecordAddresses);

  // BinaryEmitter::emitFunction() emits --pad-funcs after the body in both
  // relocation and non-relocation modes.
  DotAddress += opts::padFunctionAfter(Func);

  // --mark-funcs emits the target-specific trap marker after the fragment.
  DotAddress += opts::markFunctionBytes(BC).size();

  LLVM_DEBUG({
    if (RecordAddresses)
      dbgs() << "BOLT-DEBUG: LongJmp layout: " << (IsCold ? "cold" : "main")
             << " fragment " << Func.getPrintName() << " ends at 0x"
             << Twine::utohexstr(DotAddress) << '\n';
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
  const bool AdjustMainSection =
      BC.HasRelocations &&
      (opts::HotText || (opts::Hugify && !BC.HasFixedLoadAddress));
  std::optional<uint64_t> MainSectionEnd = std::nullopt;

  // Mirror allocateAt() in RewriteInstance::mapCodeSections().
  for (const SectionPlacement &Section : Sections) {
    DotAddress = alignTo(DotAddress, Section.Alignment);
    DotAddress = layoutSection(BC, Section, DotAddress);

    if (AdjustMainSection &&
        StringRef(Section.Name) == BC.getMainCodeSectionName()) {
      if (opts::HotText)
        MainSectionEnd = DotAddress;
      // LongJmp supports only main and cold fragments. Mirror the extra
      // post-main alignment in allocateAt() for --hugify.
      if (opts::Hugify && !BC.HasFixedLoadAddress)
        DotAddress = alignTo(DotAddress, Section.Alignment);
    }
  }

  // Mirror RewriteInstance::mapCodeSections() padding used to accommodate
  // hot-text huge-page mapping. LongJmp does not support warm fragments, so the
  // hot-text end is the end of the main section. RewriteInstance applies this
  // adjustment only in allocateAt() to advance the next free address;
  // allocateBefore() starts from a fixed upper boundary and has no
  // corresponding adjustment.
  if (MainSectionEnd)
    DotAddress = std::max(DotAddress, alignTo(*MainSectionEnd, BC.PageAlign));

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
        assert(Func.getLayout().isHotColdSplit() &&
               "non-relocation mode supports only hot/cold splitting");
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
    const CodeSectionOrder CompareSections(BC);
    llvm::stable_sort(
        Sections, [&](const SectionPlacement &A, const SectionPlacement &B) {
          return CompareSections(A.Name, B.Name);
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

bool LongJmpPass::relaxLocalBranches(BinaryFunction &BF,
                                     const BranchLivenessInfo *BLI) {
  BinaryContext &BC = BF.getBinaryContext();
  auto &MIB = BC.MIB;

  // Quick path. Only valid for simple functions, where all branch targets are
  // basic blocks of the function itself. A non-simple function may branch to a
  // symbol outside of it that ends up out of range.
  if (BF.isSimple() && !BF.isSplit() && BF.estimateSize() < ShortestJumpSpan)
    return true;

  auto isBranchOffsetInRange = [&](const MCInst &Inst, int64_t Offset) {
    const unsigned Bits = MIB->getPCRelEncodingSize(Inst);
    return isIntN(Bits, Offset);
  };

  auto isBlockInRange = [&](const MCInst &Inst, uint64_t InstAddress,
                            const BinaryBasicBlock &BB) {
    const int64_t Offset = BB.getOutputStartAddress() - InstAddress;
    return isBranchOffsetInRange(Inst, Offset);
  };

  // Keep track of *all* function trampolines that are going to be added to the
  // function layout at the end of relaxation.
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      FunctionTrampolines;

  // Function fragments are relaxed independently.
  for (FunctionFragment &FF : BF.getLayout().fragments()) {
    // Fill out code size estimation for the fragment. Use output BB address
    // ranges to store offsets from the start of the function fragment.
    uint64_t CodeSize = 0;
    for (BinaryBasicBlock *BB : FF) {
      BB->setOutputStartAddress(CodeSize);
      CodeSize += BB->estimateSize();
      BB->setOutputEndAddress(CodeSize);
    }

    // Dynamically-updated size of the fragment.
    uint64_t FragmentSize = CodeSize;

    // Size of the trampoline in bytes.
    constexpr uint64_t TrampolineSize = 4;

    // Trampolines created for the fragment. DestinationBB -> TrampolineBB.
    // NB: here we store only the first trampoline created for DestinationBB.
    DenseMap<const BinaryBasicBlock *, BinaryBasicBlock *> FragmentTrampolines;

    // Create a trampoline code after \p BB or at the end of the fragment if BB
    // is nullptr. The trampoline branches to \p TargetSym. If \p TargetBB is
    // set, it is added as a successor and registered in FragmentTrampolines.
    // \p Offset reflects the size delta of BB caused by splitting unconditional
    // branches, or replacing a branch with a longer instruction sequence. It is
    // used to update the output addresses of basic blocks following the
    // trampoline.
    auto addTrampolineAfter = [&](BinaryBasicBlock *BB,
                                  const MCSymbol *TargetSym,
                                  BinaryBasicBlock *TargetBB, uint64_t Count,
                                  uint64_t Offset = 0) {
      FunctionTrampolines.emplace_back(BB ? BB : FF.back(),
                                       BF.createBasicBlock());
      BinaryBasicBlock *TrampolineBB = FunctionTrampolines.back().second.get();
      const uint64_t OldBBEnd = BB ? BB->getOutputEndAddress() : 0;
      if (BB && Offset)
        BB->setOutputEndAddress(OldBBEnd + Offset);
      Offset += TrampolineSize;

      MCInst Inst;
      {
        auto L = BC.scopeLock();
        MIB->createUncondBranch(Inst, TargetSym, BC.Ctx.get());
      }
      TrampolineBB->addInstruction(Inst);
      if (TargetBB)
        TrampolineBB->addSuccessor(TargetBB, Count);
      TrampolineBB->setExecutionCount(Count);
      const uint64_t TrampolineAddress =
          BB ? BB->getOutputEndAddress() : FragmentSize;
      TrampolineBB->setOutputStartAddress(TrampolineAddress);
      TrampolineBB->setOutputEndAddress(TrampolineAddress + TrampolineSize);
      TrampolineBB->setFragmentNum(FF.getFragmentNum());

      // Shift the fragment-local output address range for blocks at or after
      // the old end address.
      auto adjustBasicBlockAddress = [](BinaryBasicBlock *BB, uint64_t Address,
                                        uint64_t Offset) {
        if (BB->getOutputStartAddress() < Address)
          return;
        BB->setOutputStartAddress(BB->getOutputStartAddress() + Offset);
        BB->setOutputEndAddress(BB->getOutputEndAddress() + Offset);
      };

      if (TargetBB && !FragmentTrampolines.lookup(TargetBB))
        FragmentTrampolines[TargetBB] = TrampolineBB;

      if (!Offset)
        return TrampolineBB;

      FragmentSize += Offset;

      // If the trampoline was added at the end of the fragment, offsets of
      // other fragments should stay intact.
      if (!BB)
        return TrampolineBB;

      // Update offsets for blocks after BB.
      for (BinaryBasicBlock *IBB : FF)
        adjustBasicBlockAddress(IBB, OldBBEnd, Offset);

      // Update offsets for trampolines in this fragment that are placed after
      // the new trampoline. Note that trampoline blocks are not part of the
      // function/fragment layout until we add them right before the return
      // from relaxLocalBranches().
      for (auto &Pair : FunctionTrampolines) {
        BinaryBasicBlock *IBB = Pair.second.get();
        if (IBB->getFragmentNum() != TrampolineBB->getFragmentNum())
          continue;
        if (IBB == TrampolineBB)
          continue;
        adjustBasicBlockAddress(IBB, OldBBEnd, Offset);
      }

      return TrampolineBB;
    };

    // Pre-populate trampolines by splitting unconditional branches from the
    // containing basic block. Skip for non-simple functions: this creates
    // trampolines for targets inside the function, while in a non-simple
    // function we only relax branches to targets outside of it.
    if (BF.isSimple()) {
      for (BinaryBasicBlock *BB : FF) {
        MCInst *Inst = BB->getLastNonPseudoInstr();
        if (!Inst || !MIB->isUnconditionalBranch(*Inst))
          continue;

        const MCSymbol *TargetSymbol = MIB->getTargetSymbol(*Inst);
        BB->eraseInstruction(BB->findInstruction(Inst));

        BinaryBasicBlock::BinaryBranchInfo BI;
        BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol, BI);

        // Erasing the unconditional branch shrinks BB by one instruction.
        BinaryBasicBlock *TrampolineBB =
            addTrampolineAfter(BB, TargetBB->getLabel(), TargetBB, BI.Count,
                               /*Offset=*/-4);
        BB->replaceSuccessor(TargetBB, TrampolineBB, BI.Count);
      }
    }

    /// Relax the branch \p Inst in basic block \p BB that targets \p TargetBB.
    /// \p InstAddress contains offset of the branch from the start of the
    /// containing function fragment.
    auto relaxBranch = [&](BinaryBasicBlock *BB, MCInst &Inst,
                           uint64_t InstAddress, BinaryBasicBlock *TargetBB) {
      BinaryFunction *BF = BB->getParent();

      // Use branch taken count for optimal relaxation.
      const uint64_t Count = BB->getBranchInfo(*TargetBB).Count;
      assert(Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "Expected valid branch execution count");

      // Try to reuse an existing trampoline without introducing any new code.
      BinaryBasicBlock *TrampolineBB = FragmentTrampolines.lookup(TargetBB);
      if (TrampolineBB && isBlockInRange(Inst, InstAddress, *TrampolineBB)) {
        BB->replaceSuccessor(TargetBB, TrampolineBB, Count);
        TrampolineBB->setExecutionCount(TrampolineBB->getExecutionCount() +
                                        Count);
        auto L = BC.scopeLock();
        MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
        return;
      }

      // For cold branches, check if we can introduce a trampoline at the end
      // of the fragment that is within the branch reach. Note that such
      // trampoline may change address later and become unreachable in which
      // case we will need further relaxation.
      const int64_t OffsetToEnd = FragmentSize - InstAddress;
      if (Count == 0 && isBranchOffsetInRange(Inst, OffsetToEnd)) {
        TrampolineBB =
            addTrampolineAfter(nullptr, TargetBB->getLabel(), TargetBB, Count);
        BB->replaceSuccessor(TargetBB, TrampolineBB, Count);
        auto L = BC.scopeLock();
        MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());

        return;
      }

      // If the other successor is a fall-through, invert the condition code.
      BinaryBasicBlock *NextBB =
          BF->getLayout().getBasicBlockAfter(BB, /*IgnoreSplits*/ false);
      bool PreserveFlags = BLI ? BLI->mustPreserveFlags(Inst) : true;
      bool IsReversibleBranch = MIB->isReversibleBranch(Inst, PreserveFlags);
      bool ShouldReverseBranch = BB->getConditionalSuccessor(false) == NextBB;

      // Create a trampoline basic block for the fall-through target of the
      // branch if its condition cannot be inverted.
      if (ShouldReverseBranch && !IsReversibleBranch) {
        const uint64_t NextCount = BB->getBranchInfo(*NextBB).Count;
        BinaryBasicBlock *FallThrough =
            addTrampolineAfter(BB, NextBB->getLabel(), NextBB, NextCount);
        BB->replaceSuccessor(NextBB, FallThrough, NextCount);
      }

      if (ShouldReverseBranch && IsReversibleBranch) {
        const uint64_t OldBBSize = BB->estimateSize();
        BB->swapConditionalSuccessors();
        {
          auto L = BC.scopeLock();
          if (BLI)
            BLI->removeAnnotation(Inst);
          InstructionListType Code = MIB->reverseBranchCondition(
              Inst, NextBB->getLabel(), BC.Ctx.get(), PreserveFlags);
          BB->replaceInstruction(BB->findInstruction(&Inst), Code);
        }
        const uint64_t NewBBSize = BB->estimateSize();

        // Create a trampoline basic block for the original taken target.
        TrampolineBB = addTrampolineAfter(BB, TargetBB->getLabel(), TargetBB,
                                          Count, NewBBSize - OldBBSize);
      } else {
        // Create a trampoline basic block for the taken target of the branch.
        TrampolineBB =
            addTrampolineAfter(BB, TargetBB->getLabel(), TargetBB, Count);
        auto L = BC.scopeLock();
        MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(), BC.Ctx.get());
      }
      BB->replaceSuccessor(TargetBB, TrampolineBB, Count);
    };

    // For non-simple functions, branch targets may be different functions,
    // so we track trampolines by symbol rather than by basic block.
    DenseMap<const MCSymbol *, BinaryBasicBlock *> SymbolTrampolines;

    bool MayNeedRelaxation;
    uint64_t NumIterations = 0;
    do {
      MayNeedRelaxation = false;
      ++NumIterations;
      for (auto BBI = FF.begin(); BBI != FF.end(); ++BBI) {
        BinaryBasicBlock *BB = *BBI;
        uint64_t NextInstOffset = BB->getOutputStartAddress();
        // Branch reversal may replace the current instruction with a sequence.
        // Use an index so the next instruction is reloaded after the mutation.
        for (size_t I = 0; I < BB->size(); ++I) {
          MCInst &Inst = *(BB->begin() + I);
          const size_t InstAddress = NextInstOffset;
          if (!MIB->isPseudo(Inst))
            NextInstOffset += 4;

          if (!mayNeedStub(BF.getBinaryContext(), Inst))
            continue;

          const size_t BitsAvailable = MIB->getPCRelEncodingSize(Inst);

          // Span of +/-128MB.
          if (BitsAvailable == LongestJumpBits)
            continue;

          const MCSymbol *TargetSymbol = MIB->getTargetSymbol(Inst);

          if (BF.isSimple()) {
            BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol);
            assert(TargetBB &&
                   "Basic block target expected for conditional branch.");

            // Check if the relaxation is needed.
            if (TargetBB->getFragmentNum() == FF.getFragmentNum() &&
                isBlockInRange(Inst, InstAddress, *TargetBB))
              continue;

            relaxBranch(BB, Inst, InstAddress, TargetBB);
            MayNeedRelaxation = true;
          } else {
            // Skip if the target is within this function.
            if (BF.getBasicBlockForLabel(TargetSymbol))
              continue;

            // Try to reuse an existing trampoline for this symbol.
            BinaryBasicBlock *TrampolineBB =
                SymbolTrampolines.lookup(TargetSymbol);
            if (TrampolineBB &&
                isBlockInRange(Inst, InstAddress, *TrampolineBB)) {
              auto L = BC.scopeLock();
              MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(),
                                       BC.Ctx.get());
              continue;
            }

            // Create a trampoline at the end of the function. Since the layout
            // of a non-simple function has to be preserved, the end of the
            // function is the only place where we can put it.
            const int64_t OffsetToEnd = FragmentSize - InstAddress;
            if (!isBranchOffsetInRange(Inst, OffsetToEnd)) {
              auto L = BC.scopeLock();
              BC.errs() << "BOLT-ERROR: cannot relax branch in non-simple "
                           "function "
                        << BF << ": a trampoline at the end of the function is "
                        << OffsetToEnd << " bytes away, out of reach for a "
                        << BitsAvailable << "-bit branch\n";
              BC.printInstruction(BC.errs(), Inst);
              return false;
            }

            TrampolineBB = addTrampolineAfter(/*BB=*/nullptr, TargetSymbol,
                                              /*TargetBB=*/nullptr,
                                              /*Count=*/0);
            SymbolTrampolines[TargetSymbol] = TrampolineBB;
            auto L = BC.scopeLock();
            MIB->replaceBranchTarget(Inst, TrampolineBB->getLabel(),
                                     BC.Ctx.get());
          }
        }
      }

      // We may have added new instructions, but the whole fragment is less than
      // the minimum branch span.
      if (FragmentSize < ShortestJumpSpan)
        MayNeedRelaxation = false;

    } while (MayNeedRelaxation);

    LLVM_DEBUG({
      if (NumIterations > 2) {
        dbgs() << "BOLT-DEBUG: relaxed fragment " << FF.getFragmentNum().get()
               << " of " << BF << " in " << NumIterations << " iterations\n";
      }
    });
    (void)NumIterations;
  }

  // Add trampoline blocks from all fragments to the layout.
  DenseMap<BinaryBasicBlock *, std::vector<std::unique_ptr<BinaryBasicBlock>>>
      Insertions;
  for (std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>> &Pair :
       FunctionTrampolines) {
    if (!Pair.second)
      continue;
    Insertions[Pair.first].emplace_back(std::move(Pair.second));
  }

  for (auto &Pair : Insertions) {
    BF.insertBasicBlocks(Pair.first, std::move(Pair.second),
                         /*UpdateLayout*/ true, /*UpdateCFI*/ true,
                         /*RecomputeLPs*/ false);
  }

  return true;
}

void LongJmpPass::relaxCalls(BinaryContext &BC) {
  // Operate on a copy of binary functions. We are going to manually insert new
  // thunks and update the list.
  BinaryFunctionListType OutputFunctions = BC.getOutputBinaryFunctions();

  // Conservatively estimate emitted function size. Assume the worst case
  // alignment.
  auto estimateFunctionSize = [&](const BinaryFunction &BF) -> uint64_t {
    if (!BC.shouldEmit(BF))
      return 0;
    uint64_t Size = BF.estimateSize() + BF.getMaxAlignmentBytes();

    // Each additional fragment can attribute extra bytes due to its alignment
    // requirements.
    for ([[maybe_unused]] const FunctionFragment &FF :
         BF.getLayout().getSplitFragments())
      Size += BF.getMaxColdAlignmentBytes();

    if (BF.hasIslandsInfo()) {
      Size += BF.estimateConstantIslandSize();
      if (BF.getConstantIslandAlignment() > BF.getMinAlignment())
        Size += BF.getConstantIslandAlignment() - BF.getMinAlignment();
    }

    return Size;
  };

  // Map every function to its direct callees. Note that this is different from
  // the regular call graph as here we completely ignore indirect calls.
  uint64_t EstimatedSize = 0;
  DenseMap<BinaryFunction *, std::set<const MCSymbol *>> CallMap;
  for (BinaryFunction *BF : OutputFunctions) {
    if (!BC.shouldEmit(*BF) || BF->isPatch())
      continue;

    EstimatedSize += estimateFunctionSize(*BF);

    for (const BinaryBasicBlock &BB : *BF) {
      for (const MCInst &Inst : BB) {
        if (!BC.MIB->isCall(Inst) || BC.MIB->isIndirectCall(Inst) ||
            BC.MIB->isIndirectBranch(Inst))
          continue;
        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
        assert(TargetSymbol);

        // Ignore internal calls that use basic block labels as a destination.
        if (!BC.getFunctionForSymbol(TargetSymbol))
          continue;

        CallMap[BF].insert(TargetSymbol);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "LongJmp: estimated code size : " << EstimatedSize
                    << '\n');

  // Build clusters in the order the functions will appear in the output.
  std::vector<FunctionCluster> Clusters;
  for (size_t Index = 0, NumFuncs = OutputFunctions.size(); Index < NumFuncs;
       ++Index) {
    const size_t BFIndex =
        opts::HotFunctionsAtEnd ? NumFuncs - Index - 1 : Index;
    BinaryFunction *BF = OutputFunctions[BFIndex];
    if (!BC.shouldEmit(*BF) || BF->isPatch())
      continue;

    const uint64_t BFSize = estimateFunctionSize(*BF);
    if (Clusters.empty() || Clusters.back().Size + BFSize > MaxClusterSize) {
      Clusters.emplace_back(FunctionCluster());
      Clusters.back().FirstFunctionIndex = BFIndex;
    }

    FunctionCluster &FC = Clusters.back();
    FC.Functions.insert(BF);

    // When a function is added to the cluster, we have to remove all of its
    // symbols from the cluster callee list. These include alternative symbols
    // (e.g. after ICF) and secondary entry point symbols.
    for (const MCSymbol *Symbol : BF->getSymbols()) {
      auto It = FC.Callees.find(Symbol);
      if (It != FC.Callees.end())
        FC.Callees.erase(It);
    }
    BF->forEachEntryPoint(
        [&FC](uint64_t Offset, const MCSymbol *EntrySymbol) -> bool {
          auto It = FC.Callees.find(EntrySymbol);
          if (It != FC.Callees.end())
            FC.Callees.erase(It);
          return true;
        });

    // Update cluster callee list with added function callees.
    for (const MCSymbol *CalleeSymbol : CallMap[BF]) {
      BinaryFunction *Callee = BC.getFunctionForSymbol(CalleeSymbol);
      if (!FC.Functions.count(Callee)) {
        FC.Callees.insert(CalleeSymbol);
      }
    }

    FC.Size += BFSize;
    FC.LastFunctionIndex = BFIndex;
  }

  if (opts::HotFunctionsAtEnd) {
    std::reverse(Clusters.begin(), Clusters.end());
    llvm::for_each(Clusters, [](FunctionCluster &FC) {
      std::swap(FC.LastFunctionIndex, FC.FirstFunctionIndex);
    });
  }

  if (Clusters.empty())
    return;

  // Print cluster stats.
  BC.outs() << "BOLT-INFO: built " << Clusters.size()
            << " function cluster(s)\n";
  uint64_t ClusterIndex = 0;
  for (const FunctionCluster &FC : Clusters) {
    BC.outs() << "BOLT-INFO: cluster: " << ClusterIndex++ << '\n'
              << "BOLT-INFO:   " << FC.Functions.size() << " function(s)\n"
              << "BOLT-INFO:   " << FC.Callees.size() << " callee(s)\n"
              << "BOLT-INFO:   " << FC.Size << " estimated bytes\n";
  }

  if (opts::RelaxPLT) {
    // Populate one of the clusters with PLT functions based on the proximity of
    // the PLT section to avoid unneeded thunk redirection.
    const size_t PLTClusterNum = opts::UseOldText ? Clusters.size() - 1 : 0;
    auto &PLTCluster = Clusters[PLTClusterNum];
    for (BinaryFunction &BF :
         llvm::make_second_range(BC.getBinaryFunctions())) {
      if (BF.isPLTFunction()) {
        PLTCluster.Functions.insert(&BF);
        auto It = PLTCluster.Callees.find(BF.getSymbol());
        if (It != PLTCluster.Callees.end())
          PLTCluster.Callees.erase(It);
      }
    }
  }

  // Create a thunk with +-128MB span.
  size_t NumShortThunks = 0;
  auto createShortThunk = [&](const MCSymbol *TargetSymbol) {
    ++NumShortThunks;
    BinaryFunction *ThunkBF = BC.createThunkBinaryFunction(
        "__AArch64Thunk_" + TargetSymbol->getName().str());
    MCInst Inst;
    BC.MIB->createTailCall(Inst, TargetSymbol, BC.Ctx.get());
    ThunkBF->addBasicBlock()->addInstruction(Inst);

    return ThunkBF;
  };

  // Create a thunk with +-4GB span.
  size_t NumLongThunks = 0;
  auto createLongThunk = [&](const MCSymbol *TargetSymbol) {
    ++NumLongThunks;
    BinaryFunction *ThunkBF = BC.createThunkBinaryFunction(
        "__AArch64ADRPThunk_" + TargetSymbol->getName().str());
    InstructionListType Instructions;
    BC.MIB->createLongTailCall(Instructions, TargetSymbol, BC.Ctx.get());
    ThunkBF->addBasicBlock()->addInstructions(Instructions);

    return ThunkBF;
  };

  for (unsigned ClusterNum = 0; ClusterNum < Clusters.size(); ++ClusterNum) {
    FunctionCluster &FC = Clusters[ClusterNum];
    SmallVector<const MCSymbol *, 16> Callees(FC.Callees.begin(),
                                              FC.Callees.end());

    // Generate thunks in deterministic order.
    llvm::sort(Callees, [&BC](const MCSymbol *A, const MCSymbol *B) {
      uint64_t EntryA;
      uint64_t EntryB;
      BinaryFunction *BFA = BC.getFunctionForSymbol(A, &EntryA);
      BinaryFunction *BFB = BC.getFunctionForSymbol(B, &EntryB);
      if (BFA == BFB) {
        if (EntryA != EntryB)
          return EntryA < EntryB;

        // Use lexicographical order for ICF'ed symbols.
        return A->getName() < B->getName();
      }
      return compareBinaryFunctionByIndex(BFA, BFB);
    });

    // Return index of adjacent cluster containing the function.
    auto getAdjClusterWithFunction =
        [&](const BinaryFunction *BF) -> std::optional<unsigned> {
      if (ClusterNum > 0 && Clusters[ClusterNum - 1].Functions.count(BF))
        return ClusterNum - 1;
      if (ClusterNum + 1 < Clusters.size() &&
          Clusters[ClusterNum + 1].Functions.count(BF))
        return ClusterNum + 1;
      return std::nullopt;
    };

    const FunctionCluster *PrevCluster =
        ClusterNum ? &Clusters[ClusterNum - 1] : nullptr;

    // Create short thunks for callees in adjacent clusters and long thunks
    // for callees outside.
    for (const MCSymbol *Callee : Callees) {
      if (FC.Thunks.count(Callee))
        continue;

      BinaryFunction *Thunk = 0;
      std::optional<unsigned> AdjCluster =
          getAdjClusterWithFunction(BC.getFunctionForSymbol(Callee));
      if (AdjCluster) {
        Thunk = createShortThunk(Callee);
      } else {
        // Previous cluster may already have a long thunk that can be reused.
        if (PrevCluster) {
          auto It = PrevCluster->Thunks.find(Callee);
          // Reuse only if previous cluster hosts this thunk.
          if (It != PrevCluster->Thunks.end() &&
              llvm::is_contained(PrevCluster->ThunkList, It->second)) {
            FC.Thunks[Callee] = It->second;
            continue;
          }
        }
        Thunk = createLongThunk(Callee);
      }

      // The cluster that will host this thunk. If the current cluster is the
      // last one, try to use the previous one. Matters when we want to have hot
      // functions at higher addresses under HotFunctionsAtEnd.
      FunctionCluster *ThunkCluster = &Clusters[ClusterNum];
      if ((AdjCluster && *AdjCluster == ClusterNum - 1) ||
          (ClusterNum && ClusterNum == Clusters.size() - 1))
        ThunkCluster = &Clusters[ClusterNum - 1];
      ThunkCluster->ThunkList.push_back(Thunk);

      // Register thunks for all symbols associated with the function.
      uint64_t EntryID = 0;
      const BinaryFunction *BF = BC.getFunctionForSymbol(Callee, &EntryID);
      if (EntryID != 0) {
        FC.Thunks[Callee] = Thunk;
      } else {
        for (const MCSymbol *Symbol : BF->getSymbols()) {
          FC.Thunks[Symbol] = Thunk;
        }
      }
    }
  }

  if (NumShortThunks)
    BC.outs() << "BOLT-INFO: " << NumShortThunks << " short thunks created\n";

  if (NumLongThunks)
    BC.outs() << "BOLT-INFO: " << NumLongThunks << " long thunks created\n";

  // Replace callees with thunks.
  for (FunctionCluster &FC : Clusters) {
    for (BinaryFunction *BF : FC.Functions) {
      if (!CallMap.count(BF))
        continue;

      for (BinaryBasicBlock &BB : *BF) {
        for (MCInst &Inst : BB) {
          if (!BC.MIB->isCall(Inst) || BC.MIB->isIndirectCall(Inst) ||
              BC.MIB->isIndirectBranch(Inst))
            continue;
          const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
          assert(TargetSymbol);

          auto It = FC.Thunks.find(TargetSymbol);
          if (It != FC.Thunks.end())
            BC.MIB->replaceBranchTarget(Inst, It->second->getSymbol(),
                                        BC.Ctx.get());
        }
      }
    }
  }

  // Add thunks to the function list and assign a section name matching the
  // function they follow.
  for (const FunctionCluster &FC : llvm::reverse(Clusters)) {
    std::string SectionName =
        OutputFunctions[FC.LastFunctionIndex]->getCodeSectionName().str().str();
    for (BinaryFunction *Thunk : FC.ThunkList) {
      Thunk->setCodeSectionName(SectionName);
    }

    OutputFunctions.insert(
        std::next(OutputFunctions.begin(), FC.LastFunctionIndex + 1),
        FC.ThunkList.begin(), FC.ThunkList.end());
  }

  LLVM_DEBUG(dbgs() << "\nFunction layout with thunks:\n";
             for (const auto *BF : OutputFunctions) { dbgs() << *BF << '\n'; });

  BC.updateOutputBinaryFunctions(std::move(OutputFunctions));
}

Error LongJmpPass::runOnFunctions(BinaryContext &BC) {

  assert((opts::CompactCodeModel || opts::ExperimentalRelaxation ||
          opts::SplitStrategy != opts::SplitFunctionsStrategy::CDSplit) &&
         "LongJmp cannot work with functions split in more than two fragments");

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
    relaxCalls(BC);
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
