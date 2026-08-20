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
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/BranchLivenessUtils.h"
#include "bolt/Passes/RegAnalysis.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "longjmp"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> UseOldText;
extern cl::opt<bool> HotFunctionsAtEnd;

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
}

namespace llvm {
namespace bolt {

constexpr unsigned ColdFragAlign = 16;

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
      dbgs() << "Considering stub group with " << Candidates.size()
             << " candidates. DotAddress is " << Twine::utohexstr(DotAddress)
             << ", chosen candidate address is "
             << Twine::utohexstr(Cand->first) << "\n";
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
        Elem.first = BBAddresses[Elem.second];
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

void LongJmpPass::tentativeBBLayout(const BinaryFunction &Func) {
  const BinaryContext &BC = Func.getBinaryContext();
  uint64_t HotDot = HotAddresses[&Func];
  uint64_t ColdDot = ColdAddresses[&Func];
  bool Cold = false;
  for (const BinaryBasicBlock *BB : Func.getLayout().blocks()) {
    if (Cold || BB->isCold()) {
      Cold = true;
      BBAddresses[BB] = ColdDot;
      ColdDot += BC.computeCodeSize(BB->begin(), BB->end());
    } else {
      BBAddresses[BB] = HotDot;
      HotDot += BC.computeCodeSize(BB->begin(), BB->end());
    }
  }
}

uint64_t LongJmpPass::tentativeLayoutRelocColdPart(
    const BinaryContext &BC, BinaryFunctionListType &SortedFunctions,
    uint64_t DotAddress) {
  DotAddress =
      alignTo(DotAddress, std::max<uint64_t>(BC.AlignFunctions,
                                             BC.MaxColdCodeAlignment.load()));
  for (BinaryFunction *Func : SortedFunctions) {
    if (!Func->isSplit())
      continue;
    DotAddress = alignTo(DotAddress, Func->getMinAlignment());
    uint64_t Pad =
        offsetToAlignment(DotAddress, llvm::Align(Func->getAlignment()));
    if (Pad <= Func->getMaxColdAlignmentBytes())
      DotAddress += Pad;
    ColdAddresses[Func] = DotAddress;
    LLVM_DEBUG(dbgs() << Func->getPrintName() << " cold tentative: "
                      << Twine::utohexstr(DotAddress) << "\n");
    DotAddress += Func->estimateColdSize();
    if (uint64_t IslandSize = Func->estimateConstantIslandSize()) {
      DotAddress = alignTo(DotAddress, Func->getConstantIslandAlignment());
      DotAddress += IslandSize;
    }
  }
  return DotAddress;
}

uint64_t
LongJmpPass::tentativeLayoutRelocMode(const BinaryContext &BC,
                                      BinaryFunctionListType &SortedFunctions,
                                      uint64_t DotAddress) {
  // Compute hot cold frontier
  int64_t LastHotIndex = -1u;
  uint32_t CurrentIndex = 0;
  if (opts::HotFunctionsAtEnd) {
    for (BinaryFunction *BF : SortedFunctions) {
      if (BF->hasValidIndex()) {
        LastHotIndex = CurrentIndex;
        break;
      }

      ++CurrentIndex;
    }
  } else {
    for (BinaryFunction *BF : SortedFunctions) {
      if (!BF->hasValidIndex()) {
        LastHotIndex = CurrentIndex;
        break;
      }

      ++CurrentIndex;
    }
  }

  // Hot
  CurrentIndex = 0;
  bool ColdLayoutDone = false;
  auto runColdLayout = [&]() {
    // Mirror the extra hugify alignment inserted by final section allocation
    // after the last non-cold section. Account for it before assigning cold
    // fragment addresses so range checks see the hot-to-cold gap.
    if (opts::Hugify && !BC.HasFixedLoadAddress && !opts::HotFunctionsAtEnd)
      DotAddress = alignTo(DotAddress, BC.AlignText);
    DotAddress = tentativeLayoutRelocColdPart(BC, SortedFunctions, DotAddress);
    ColdLayoutDone = true;
    if (opts::HotFunctionsAtEnd)
      DotAddress = alignTo(DotAddress, BC.AlignText);
  };
  for (BinaryFunction *Func : SortedFunctions) {
    if (!BC.shouldEmit(*Func)) {
      HotAddresses[Func] = Func->getAddress();
      continue;
    }

    if (!ColdLayoutDone && CurrentIndex >= LastHotIndex)
      runColdLayout();

    DotAddress = alignTo(DotAddress, Func->getMinAlignment());
    uint64_t Pad =
        offsetToAlignment(DotAddress, llvm::Align(Func->getAlignment()));
    if (Pad <= Func->getMaxAlignmentBytes())
      DotAddress += Pad;
    HotAddresses[Func] = DotAddress;
    LLVM_DEBUG(dbgs() << Func->getPrintName() << " tentative: "
                      << Twine::utohexstr(DotAddress) << "\n");
    if (!Func->isSplit())
      DotAddress += Func->estimateSize();
    else
      DotAddress += Func->estimateHotSize();

    if (uint64_t IslandSize = Func->estimateConstantIslandSize()) {
      DotAddress = alignTo(DotAddress, Func->getConstantIslandAlignment());
      DotAddress += IslandSize;
    }
    ++CurrentIndex;
  }

  // Ensure that tentative code layout always runs for cold blocks.
  if (!ColdLayoutDone)
    runColdLayout();

  // BBs
  for (BinaryFunction *Func : SortedFunctions)
    tentativeBBLayout(*Func);

  return DotAddress;
}

void LongJmpPass::tentativeLayout(const BinaryContext &BC,
                                  BinaryFunctionListType &SortedFunctions) {
  uint64_t DotAddress = BC.LayoutStartAddress;

  if (!BC.HasRelocations) {
    for (BinaryFunction *Func : SortedFunctions) {
      HotAddresses[Func] = Func->getAddress();
      DotAddress = alignTo(DotAddress, ColdFragAlign);
      ColdAddresses[Func] = DotAddress;
      if (Func->isSplit())
        DotAddress += Func->estimateColdSize();
      tentativeBBLayout(*Func);
    }

    return;
  }

  // Relocation mode
  uint64_t EstimatedTextSize = 0;
  if (opts::UseOldText) {
    EstimatedTextSize = tentativeLayoutRelocMode(BC, SortedFunctions, 0);

    // Initial padding
    if (EstimatedTextSize <= BC.OldTextSectionSize) {
      DotAddress = BC.OldTextSectionAddress;
      uint64_t Pad = offsetToAlignment(DotAddress, llvm::Align(BC.AlignText));
      if (Pad + EstimatedTextSize <= BC.OldTextSectionSize) {
        DotAddress += Pad;
      }
    }
  }

  if (!EstimatedTextSize || EstimatedTextSize > BC.OldTextSectionSize) {
    uint64_t TextAlign =
        std::max<uint64_t>(BC.AlignText, BC.MaxMainCodeAlignment.load());
    DotAddress = alignTo(BC.LayoutStartAddress, TextAlign);
  }

  tentativeLayoutRelocMode(BC, SortedFunctions, DotAddress);
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
  uint64_t DotAddress = BBAddresses[&StubBB];
  uint64_t PCRelTgtAddress = DotAddress > TgtAddress ? DotAddress - TgtAddress
                                                     : TgtAddress - DotAddress;

  // If it fits in one instruction, do not relax
  if (!(PCRelTgtAddress & SingleInstrMask))
    return Error::success();

  // Fits short jmp
  if (!(PCRelTgtAddress & ShortJmpMask)) {
    if (Bits >= RangeShortJmp)
      return Error::success();

    LLVM_DEBUG(dbgs() << "Relaxing stub to short jump. PCRelTgtAddress = "
                      << Twine::utohexstr(PCRelTgtAddress)
                      << " RealTargetSym = " << RealTargetSym->getName()
                      << "\n");
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

  LLVM_DEBUG(dbgs() << "Relaxing stub to long jump. PCRelTgtAddress = "
                    << Twine::utohexstr(PCRelTgtAddress)
                    << " RealTargetSym = " << RealTargetSym->getName() << "\n");
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

  return PCOffset < MinVal || PCOffset > MaxVal;
}

Error LongJmpPass::relax(BinaryFunction &Func, bool &Modified) {
  const BinaryContext &BC = Func.getBinaryContext();

  assert(BC.isAArch64() && "Unsupported arch");
  constexpr int InsnSize = 4; // AArch64
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      Insertions;

  BinaryBasicBlock *Frontier = getBBAtHotColdSplitPoint(Func);
  uint64_t FrontierAddress = Frontier ? BBAddresses[Frontier] : 0;
  if (FrontierAddress)
    FrontierAddress += Frontier->getNumNonPseudos() * InsnSize;

  // Add necessary stubs for branch targets we know we can't fit in the
  // instruction
  for (BinaryBasicBlock &BB : Func) {
    uint64_t DotAddress = BBAddresses[&BB];
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

  DenseMap<const MCInst *, MCPhysReg> ScratchRegs;
  if (BC.isRISCV()) {
    const unsigned NumRegs = BC.MRI->getNumRegs();
    DenseMap<const BinaryBasicBlock *, BitVector> LiveIns;
    DenseMap<const BinaryBasicBlock *, BitVector> LiveOuts;
    for (const BinaryBasicBlock &BB : BF) {
      LiveIns.try_emplace(&BB, NumRegs, false);
      LiveOuts.try_emplace(&BB, NumRegs, false);
    }

    BitVector ABIExitState(NumRegs, false);
    MIB->getDefaultLiveOut(ABIExitState);
    MIB->getCalleeSavedRegs(ABIExitState);

    auto transfer = [&](const MCInst &Inst, BitVector State) {
      if (MIB->isCFI(Inst))
        return State;

      BitVector Written(NumRegs, false);
      BitVector Used(NumRegs, false);
      MIB->getWrittenRegs(Inst, Written);
      MIB->getUsedRegs(Inst, Used);
      if (MIB->isCall(Inst)) {
        BitVector CallClobbered(NumRegs, false);
        MIB->getGPRegs(CallClobbered, /*IncludeAlias=*/true);
        BitVector Preserved(NumRegs, false);
        MIB->getCalleeSavedRegs(Preserved);
        Preserved.flip();
        CallClobbered &= Preserved;
        Written |= CallClobbered;
        Used |= MIB->getRegsUsedAsParams();
      }
      Written.flip();
      State &= Written;
      State |= Used;
      return State;
    };

    bool Changed;
    do {
      Changed = false;
      for (BinaryBasicBlock &BB : reverse(BF)) {
        BitVector LiveOut(NumRegs, false);
        if (BB.succ_size() == 0)
          LiveOut = ABIExitState;
        else
          for (const BinaryBasicBlock *Succ : BB.successors())
            LiveOut |= LiveIns[Succ];

        BitVector LiveIn = LiveOut;
        for (const MCInst &Inst : reverse(BB))
          LiveIn = transfer(Inst, std::move(LiveIn));

        if (LiveOuts[&BB] != LiveOut) {
          LiveOuts[&BB] = std::move(LiveOut);
          Changed = true;
        }
        if (LiveIns[&BB] != LiveIn) {
          LiveIns[&BB] = std::move(LiveIn);
          Changed = true;
        }
      }
    } while (Changed);

    bool CanRelax = true;
    for (BinaryBasicBlock &BB : BF) {
      BitVector Live = LiveOuts[&BB];
      for (MCInst &Inst : reverse(BB)) {
        if (!MIB->isBranch(Inst) || MIB->isIndirectBranch(Inst))
          Live = transfer(Inst, std::move(Live));
        else {
          const MCSymbol *TargetSymbol = MIB->getTargetSymbol(Inst);
          BinaryBasicBlock *TargetBB = BB.getSuccessor(TargetSymbol);
          if (TargetBB && TargetBB->getFragmentNum() != BB.getFragmentNum()) {
            BitVector Available = Live;
            Available.flip();
            BitVector GPRegs(NumRegs, false);
            MIB->getGPRegs(GPRegs, /*IncludeAlias=*/false);
            Available &= GPRegs;
            MIB->removeNonScavengeableRegs(Available);
            const int Reg = Available.find_first();
            if (Reg == -1) {
              CanRelax = false;
              break;
            }
            ScratchRegs[&Inst] = Reg;
          }
          Live = transfer(Inst, std::move(Live));
        }
      }
      if (!CanRelax)
        break;
    }

    // Unlike AArch64, RISC-V has no ABI-reserved linker scratch register. If
    // every GPR is live across a cross-fragment edge, keep this function in a
    // single fragment rather than silently clobbering program state.
    if (!CanRelax) {
      BC.errs() << "BOLT-WARNING: keeping " << BF
                << " unsplit: no dead register for a RISC-V long jump\n";
      BinaryFunction::BasicBlockOrderType Layout(BF.getLayout().block_begin(),
                                                 BF.getLayout().block_end());
      for (BinaryBasicBlock &BB : BF)
        BB.setFragmentNum(FragmentNum::main());
      BF.getLayout().update(Layout);
      BF.fixBranches();
      return true;
    }
  }

  auto isBranchOffsetInRange = [&](const MCInst &Inst, int64_t Offset) {
    const unsigned Bits = MIB->getPCRelEncodingSize(Inst);
    return isIntN(Bits, Offset);
  };

  // Output address ranges are persistent metadata used later for translating
  // secondary entry points. Keep RISC-V's temporary relaxation offsets
  // separate so fragment-relative offsets cannot leak into symbol rewriting.
  DenseMap<const BinaryBasicBlock *, uint64_t> EstimatedStart;
  DenseMap<const BinaryBasicBlock *, uint64_t> EstimatedEnd;
  auto getEstimatedStart = [&](const BinaryBasicBlock *BB) {
    return BC.isRISCV() ? EstimatedStart.lookup(BB)
                        : BB->getOutputStartAddress();
  };
  auto getEstimatedEnd = [&](const BinaryBasicBlock *BB) {
    return BC.isRISCV() ? EstimatedEnd.lookup(BB) : BB->getOutputEndAddress();
  };
  auto setEstimatedRange = [&](BinaryBasicBlock *BB, uint64_t Start,
                               uint64_t End) {
    if (BC.isRISCV()) {
      EstimatedStart[BB] = Start;
      EstimatedEnd[BB] = End;
    } else {
      BB->setOutputStartAddress(Start);
      BB->setOutputEndAddress(End);
    }
  };

  auto isBlockInRange = [&](const MCInst &Inst, uint64_t InstAddress,
                            const BinaryBasicBlock &BB) {
    const int64_t Offset = getEstimatedStart(&BB) - InstAddress;
    return isBranchOffsetInRange(Inst, Offset);
  };

  // Keep track of *all* function trampolines that are going to be added to the
  // function layout at the end of relaxation.
  std::vector<std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>>>
      FunctionTrampolines;

  // Function fragments are relaxed independently.
  for (FunctionFragment &FF : BF.getLayout().fragments()) {
    // Fill out code size estimation for the fragment.
    uint64_t CodeSize = 0;
    for (BinaryBasicBlock *BB : FF) {
      const uint64_t Start = CodeSize;
      CodeSize += BB->estimateSize();
      setEstimatedRange(BB, Start, CodeSize);
    }

    // Dynamically-updated size of the fragment.
    uint64_t FragmentSize = CodeSize;

    // AArch64 trampolines start as one direct branch. RISC-V trampolines use
    // AUIPC+JALR so that split fragments can be placed outside the +/-1 MiB
    // JAL range.
    const uint64_t TrampolineSize = BC.isRISCV() ? 8 : 4;

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
                                  int64_t BBSizeDelta = 0,
                                  MCPhysReg ScratchReg = 0) {
      FunctionTrampolines.emplace_back(BB ? BB : FF.back(),
                                       BF.createBasicBlock());
      BinaryBasicBlock *TrampolineBB = FunctionTrampolines.back().second.get();
      const uint64_t OldBBEnd = BB ? getEstimatedEnd(BB) : 0;
      if (BB && BBSizeDelta)
        setEstimatedRange(BB, getEstimatedStart(BB),
                          getEstimatedEnd(BB) + BBSizeDelta);
      const int64_t Offset = BBSizeDelta + TrampolineSize;

      InstructionListType Seq;
      {
        auto L = BC.scopeLock();
        if (BC.isRISCV()) {
          assert(TargetBB && "RISC-V trampoline requires a basic block target");
          MIB->createLongJmp(Seq, TargetSym, BC.Ctx.get(),
                             /*IsTailCall=*/false, ScratchReg);
        } else {
          Seq.emplace_back();
          MIB->createUncondBranch(Seq.back(), TargetSym, BC.Ctx.get());
        }
      }
      TrampolineBB->addInstructions(Seq.begin(), Seq.end());
      if (TargetBB)
        TrampolineBB->addSuccessor(TargetBB, Count);
      TrampolineBB->setExecutionCount(Count);
      const uint64_t TrampolineAddress =
          BB ? getEstimatedEnd(BB) : FragmentSize;
      setEstimatedRange(TrampolineBB, TrampolineAddress,
                        TrampolineAddress + TrampolineSize);
      TrampolineBB->setFragmentNum(FF.getFragmentNum());

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
      for (BinaryBasicBlock *IBB : FF) {
        const uint64_t Start = getEstimatedStart(IBB);
        if (Start >= OldBBEnd)
          setEstimatedRange(IBB, Start + Offset, getEstimatedEnd(IBB) + Offset);
      }

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
        const uint64_t Start = getEstimatedStart(IBB);
        if (Start >= OldBBEnd)
          setEstimatedRange(IBB, Start + Offset, getEstimatedEnd(IBB) + Offset);
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
        BinaryBasicBlock::BinaryBranchInfo BI;
        BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol, BI);
        if (!TargetBB || (BC.isRISCV() &&
                          TargetBB->getFragmentNum() == BB->getFragmentNum()))
          continue;

        const uint64_t BranchSize =
            BC.isRISCV() ? BC.computeCodeSize(Inst, Inst + 1) : 4;
        const MCPhysReg ScratchReg = ScratchRegs.lookup(Inst);
        BB->eraseInstruction(BB->findInstruction(Inst));

        BinaryBasicBlock *TrampolineBB =
            addTrampolineAfter(BB, TargetBB->getLabel(), TargetBB, BI.Count,
                               -static_cast<int64_t>(BranchSize), ScratchReg);
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
      if (!BC.isRISCV() && TrampolineBB &&
          isBlockInRange(Inst, InstAddress, *TrampolineBB)) {
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
      const MCPhysReg ScratchReg = ScratchRegs.lookup(&Inst);
      const int64_t OffsetToEnd = FragmentSize - InstAddress;
      if (Count == 0 && isBranchOffsetInRange(Inst, OffsetToEnd)) {
        TrampolineBB =
            addTrampolineAfter(nullptr, TargetBB->getLabel(), TargetBB, Count,
                               /*BBSizeDelta=*/0, ScratchReg);
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
            addTrampolineAfter(BB, NextBB->getLabel(), NextBB, NextCount,
                               /*BBSizeDelta=*/0, ScratchReg);
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
        TrampolineBB = addTrampolineAfter(
            BB, TargetBB->getLabel(), TargetBB, Count,
            static_cast<int64_t>(NewBBSize) - static_cast<int64_t>(OldBBSize),
            ScratchReg);
      } else {
        // Create a trampoline basic block for the taken target of the branch.
        TrampolineBB =
            addTrampolineAfter(BB, TargetBB->getLabel(), TargetBB, Count,
                               /*BBSizeDelta=*/0, ScratchReg);
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
        uint64_t NextInstOffset = getEstimatedStart(BB);
        // Branch reversal may replace the current instruction with a sequence.
        // Use an index so the next instruction is reloaded after the mutation.
        for (size_t I = 0; I < BB->size(); ++I) {
          MCInst &Inst = *(BB->begin() + I);
          const size_t InstAddress = NextInstOffset;
          if (!MIB->isPseudo(Inst))
            NextInstOffset +=
                BC.isRISCV() ? BC.computeCodeSize(&Inst, &Inst + 1) : 4;

          if (!mayNeedStub(BF.getBinaryContext(), Inst))
            continue;

          const MCSymbol *TargetSymbol = MIB->getTargetSymbol(Inst);
          const size_t BitsAvailable = MIB->getPCRelEncodingSize(Inst);

          // AArch64 compact code model keeps fragments within the range of B.
          if (!BC.isRISCV() && BitsAvailable == LongestJumpBits)
            continue;

          if (BF.isSimple()) {
            BinaryBasicBlock *TargetBB = BB->getSuccessor(TargetSymbol);
            assert(TargetBB &&
                   "Basic block target expected for conditional branch.");

            // Existing intra-fragment RISC-V branches are handled by JITLink's
            // normal branch relaxation. This pass is responsible for edges
            // that become unrepresentable specifically because of a function
            // split.
            if (BC.isRISCV() &&
                TargetBB->getFragmentNum() == FF.getFragmentNum())
              continue;

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
  if (BC.isRISCV()) {
    BC.outs() << "BOLT-INFO: relaxing RISC-V cross-fragment branches\n";
    for (BinaryFunction *BF : BC.getOutputBinaryFunctions()) {
      if (!BC.shouldEmit(*BF) || !BF->isSimple() || !BF->isSplit())
        continue;
      relaxLocalBranches(*BF);
    }
    return Error::success();
  }

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
    tentativeLayout(BC, Sorted);
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
