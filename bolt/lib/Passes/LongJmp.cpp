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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

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

static cl::opt<unsigned long long> MaxClusterSize(
    "max-cluster-size",
    cl::desc("maximum estimated size of a function fragment cluster in bytes"),
    cl::init(124 * 1024 * 1024), cl::cat(BoltOptCategory));

static cl::opt<unsigned> MaxThunkChainLength(
    "max-thunk-chain-length",
    cl::desc("maximum number of B-only thunks to use for call relaxation"),
    cl::init(1), cl::Hidden, cl::cat(BoltOptCategory));
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

namespace {
class ClusteredRelaxation {
public:
  ClusteredRelaxation(BinaryContext &BC,
                      BinaryFunctionListType &OutputFunctions)
      : BC(BC), OutputFunctions(OutputFunctions) {}

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
  static unsigned getClusterDistance(unsigned A, unsigned B);
  static uint64_t estimateFragmentSize(const BinaryFunction &BF,
                                       const FunctionFragment &FF);
  void buildLayout();
  void collectOutOfRangeReferences();
  const MCSymbol *getOrCreateBranchThunkChain(const OutOfRangeRef &Ref,
                                              unsigned MaxThunks);
  void relaxCalls();
  void relaxUnconditionalBranches();
  void insertThunks();

  BinaryContext &BC;
  BinaryFunctionListType &OutputFunctions;

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

bool ClusteredRelaxation::isWithinClusterRange(uint64_t SourceOffset,
                                               uint64_t TargetOffset) {
  const uint64_t Distance = SourceOffset <= TargetOffset
                                ? TargetOffset - SourceOffset
                                : SourceOffset - TargetOffset;
  return Distance < opts::MaxClusterSize;
}

unsigned ClusteredRelaxation::getClusterDistance(unsigned A, unsigned B) {
  return A > B ? A - B : B - A;
}

uint64_t ClusteredRelaxation::estimateFragmentSize(const BinaryFunction &BF,
                                                   const FunctionFragment &FF) {
  uint64_t Size = 0;
  for (const BinaryBasicBlock *BB : FF)
    Size += BB->estimateSize();

  if (BF.hasIslandsInfo()) {
    Size += BF.estimateConstantIslandSize();
    if (BF.getConstantIslandAlignment() > BF.getMinAlignment())
      Size += BF.getConstantIslandAlignment() - BF.getMinAlignment();
  }

  Size += FF.isSplitFragment() ? BF.getMaxColdAlignmentBytes()
                               : BF.getMaxAlignmentBytes();
  return Size;
}

void ClusteredRelaxation::buildLayout() {
  struct OutputFragment {
    const FunctionFragment *FF;
    size_t FunctionIndex;
    SmallString<32> SectionName;
    uint64_t Size;
  };

  struct FragmentRange {
    size_t Begin;
    size_t End;
  };

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
    if (!BC.shouldEmit(*BF) || BF->isPatch())
      continue;

    for (const FunctionFragment &FF : BF->getLayout().fragments()) {
      if (FF.empty() && !BF->hasConstantIsland())
        continue;

      addOrderedFragment({&FF, I, BF->getCodeSectionName(FF.getFragmentNum()),
                          estimateFragmentSize(*BF, FF)});
    }
  }

  // Model final output layout by grouping function fragments in output section
  // order. Within each section, fragments remain in OutputFunctions order.
  SmallVector<size_t, 4> SectionOrder;
  for (size_t I = 0; I < FragmentsBySection.size(); ++I)
    SectionOrder.push_back(I);

  llvm::sort(SectionOrder, [&](size_t A, size_t B) {
    return BC.compareSectionNames(FragmentsBySection[A].front().SectionName,
                                  FragmentsBySection[B].front().SectionName);
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
      uint64_t Size = OrderedFragments[0]->Size;
      for (size_t I = 1; I < OrderedFragments.size(); ++I) {
        if (Size + OrderedFragments[I]->Size > opts::MaxClusterSize) {
          ClusterRanges.push_back({Begin, I});
          Begin = I;
          Size = 0;
        }
        Size += OrderedFragments[I]->Size;
      }
      ClusterRanges.push_back({Begin, OrderedFragments.size()});
      return ClusterRanges;
    }

    // Hot fragments appear last, so perform reverse walk.
    size_t End = OrderedFragments.size();
    uint64_t Size = OrderedFragments.back()->Size;
    for (size_t I = End - 1; I > 0;) {
      --I;
      if (Size + OrderedFragments[I]->Size > opts::MaxClusterSize) {
        ClusterRanges.push_back({I + 1, End});
        End = I + 1;
        Size = 0;
      }
      Size += OrderedFragments[I]->Size;
    }
    ClusterRanges.push_back({0, End});
    std::reverse(ClusterRanges.begin(), ClusterRanges.end());
    return ClusterRanges;
  };

  uint64_t LayoutOffset = 0;
  auto addFragmentToCluster = [&](const OutputFragment &Fragment) {
    FragmentCluster &FC = Clusters.back();
    const unsigned ClusterNum = Clusters.size() - 1;
    BinaryFunction &BF = *OutputFunctions[Fragment.FunctionIndex];
    const FunctionFragment &FF = *Fragment.FF;
    const uint64_t FragmentOffset = LayoutOffset;

    if (FC.NumFragments == 0) {
      FC.StartSectionName = Fragment.SectionName;
      FC.StartOffset = FragmentOffset;
      FC.FirstFunctionIndex = Fragment.FunctionIndex;
    }

    FC.EndSectionName = Fragment.SectionName;
    FC.LastFunctionIndex = Fragment.FunctionIndex;
    ++FC.NumFragments;

    // Map primary entry points.
    if (FF.isMainFragment())
      for (const MCSymbol *Symbol : BF.getSymbols())
        SymLayout[Symbol] = {ClusterNum, FragmentOffset};

    uint64_t BBOffset = FragmentOffset;
    for (const BinaryBasicBlock *BB : FF) {
      BBLayout[BB] = {ClusterNum, BBOffset};
      // Map the local BB label.
      if (const MCSymbol *Label = BB->getLabel())
        SymLayout[Label] = {ClusterNum, BBOffset};
      // Map the secondary entry point, which can differ from the BB label.
      if (MCSymbol *Label = BF.getLabelAtOffset(BB->getOffset()))
        if (MCSymbol *EntrySymbol = BF.getSecondaryEntryPointSymbol(Label))
          SymLayout[EntrySymbol] = {ClusterNum, BBOffset};

      BBOffset += BB->estimateSize();
    }

    FC.Size += Fragment.Size;
    LayoutOffset += Fragment.Size;
  };

  for (const FragmentRange &Range : buildClusterRanges()) {
    Clusters.emplace_back();
    for (size_t I = Range.Begin; I < Range.End; ++I)
      addFragmentToCluster(*OrderedFragments[I]);
  }

  if (Clusters.empty())
    return;

  LLVM_DEBUG(dbgs() << "LongJmp: estimated code size : " << LayoutOffset
                    << '\n');

  // Print cluster stats.
  BC.outs() << "BOLT-INFO: built " << Clusters.size()
            << " function fragment cluster(s)\n";
  for (size_t I = 0; I < Clusters.size(); ++I) {
    const FragmentCluster &FC = Clusters[I];
    BC.outs() << "BOLT-INFO: cluster: " << I << '\n'
              << "BOLT-INFO:   " << FC.NumFragments << " fragment(s)\n"
              << "BOLT-INFO:   " << FC.Size << " estimated bytes\n";
  }
}

void ClusteredRelaxation::collectOutOfRangeReferences() {
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
        // Unmapped targets use maximum offset and are treated as out of range.
        const Position Target = Found ? TargetIt->second : Position{-1u, -1ULL};

        // Skip already in-range references that do not need relaxation.
        if (Source.Cluster == Target.Cluster ||
            isWithinClusterRange(SourceOffset, Target.Offset))
          continue;

        const OutOfRangeRef Reference{&Inst,          TargetSymbol,
                                      SourceOffset,   Target.Offset,
                                      Source.Cluster, Target.Cluster};
        const bool UseBranchChain =
            Found && (IsUncondBranch ||
                      (IsTailCall && !isPrimaryEntryTarget(TargetSymbol)));
        if (UseBranchChain) {
          // A direct B to a body entry may be annotated as a tail call. It is
          // not an ABI call boundary, so use branch chains instead of call
          // thunks that may clobber x16/x17.
          if (IsTailCall)
            BC.MIB->convertTailCallToJmp(Inst);

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
}

const MCSymbol *
ClusteredRelaxation::getOrCreateBranchThunkChain(const OutOfRangeRef &Ref,
                                                 unsigned MaxThunks) {
  const unsigned SourceCluster = Ref.SourceCluster;
  const unsigned TargetCluster = Ref.TargetCluster;
  const bool IsForward = SourceCluster < TargetCluster;
  const bool IsCall = BC.MIB->isCall(*Ref.Inst);
  const unsigned NumHops = getClusterDistance(SourceCluster, TargetCluster);

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
    return IsForward ? FC.getEndOffset() : FC.StartOffset;
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
    const bool IsForward = SourceCluster < Call.TargetCluster;
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

void ClusteredRelaxation::relaxUnconditionalBranches() {
  for (const OutOfRangeRef &Branch : Branches) {
    const MCSymbol *Target =
        getOrCreateBranchThunkChain(Branch, /*MaxThunks=*/-1u);
    assert(Target && "expected branch thunk chain");
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
  buildLayout();
  if (Clusters.empty())
    return false;

  collectOutOfRangeReferences();
  relaxCalls();
  relaxUnconditionalBranches();
  insertThunks();

  LLVM_DEBUG(dbgs() << "\nFunction layout with thunks:\n";
             for (const auto *BF : OutputFunctions) { dbgs() << *BF << '\n'; });

  return true;
}

void LongJmpPass::relaxWithClusters(BinaryContext &BC) {
  BinaryFunctionListType OutputFunctions = BC.getOutputBinaryFunctions();
  ClusteredRelaxation Relaxation(BC, OutputFunctions);
  if (!Relaxation.run())
    return;

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
    relaxWithClusters(BC);
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
