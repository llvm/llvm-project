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

#define DEBUG_TYPE "longjmp"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltOptCategory;
extern llvm::cl::opt<unsigned> AlignText;
extern cl::opt<unsigned> AlignFunctions;
extern cl::opt<bool> UseOldText;
extern cl::opt<bool> HotFunctionsAtEnd;

static cl::opt<bool> GroupStubs("group-stubs",
                                cl::desc("share stubs across functions"),
                                cl::init(true), cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

namespace {
constexpr unsigned ColdFragAlign = 16;

void relaxStubToShortJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  InstructionListType Seq;
  BC.MIB->createShortJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
}

void relaxStubToLongJmp(BinaryBasicBlock &StubBB, const MCSymbol *Tgt) {
  const BinaryContext &BC = StubBB.getFunction()->getBinaryContext();
  InstructionListType Seq;
  BC.MIB->createLongJmp(Seq, Tgt, BC.Ctx.get());
  StubBB.clear();
  StubBB.addInstructions(Seq.begin(), Seq.end());
}

BinaryBasicBlock *getBBAtHotColdSplitPoint(BinaryFunction &Func) {
  if (!Func.isSplit() || Func.empty())
    return nullptr;

  assert(!(*Func.begin()).isCold() && "Entry cannot be cold");
  for (auto I = Func.layout_begin(), E = Func.layout_end(); I != E; ++I) {
    auto Next = std::next(I);
    if (Next != E && (*Next)->isCold())
      return *I;
  }
  llvm_unreachable("No hot-colt split point found");
}

bool shouldInsertStub(const BinaryContext &BC, const MCInst &Inst) {
  return (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst)) &&
         !BC.MIB->isIndirectBranch(Inst) && !BC.MIB->isIndirectCall(Inst);
}

} // end anonymous namespace

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
        std::lower_bound(
            StubGroup.begin(), StubGroup.end(),
            std::make_pair(AtAddress, nullptr),
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
  auto Cand = std::lower_bound(
      Candidates.begin(), Candidates.end(), std::make_pair(DotAddress, nullptr),
      [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
          const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
        return LHS.first < RHS.first;
      });
  if (Cand == Candidates.end())
    return nullptr;
  if (Cand != Candidates.begin()) {
    const StubTy *LeftCand = std::prev(Cand);
    if (Cand->first - DotAddress > DotAddress - LeftCand->first)
      Cand = LeftCand;
  }
  int BitsAvail = BC.MIB->getPCRelEncodingSize(Inst) - 1;
  uint64_t Mask = ~((1ULL << BitsAvail) - 1);
  uint64_t PCRelTgtAddress = Cand->first;
  PCRelTgtAddress = DotAddress > PCRelTgtAddress ? DotAddress - PCRelTgtAddress
                                                 : PCRelTgtAddress - DotAddress;
  LLVM_DEBUG({
    if (Candidates.size() > 1)
      dbgs() << "Considering stub group with " << Candidates.size()
             << " candidates. DotAddress is " << Twine::utohexstr(DotAddress)
             << ", chosen candidate address is "
             << Twine::utohexstr(Cand->first) << "\n";
  });
  return PCRelTgtAddress & Mask ? nullptr : Cand->second;
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
    // If we are replacing a local stub (because it is now out of range),
    // use its target instead of creating a stub to jump to another stub
    TgtSym = BC.MIB->getTargetSymbol(*TgtBB->begin());
    TgtBB = BB.getSuccessor(TgtSym, BI);
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
      std::sort(KeyVal.second.begin(), KeyVal.second.end(),
                [&](const std::pair<uint64_t, BinaryBasicBlock *> &LHS,
                    const std::pair<uint64_t, BinaryBasicBlock *> &RHS) {
                  return LHS.first < RHS.first;
                });
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
  for (BinaryBasicBlock *BB : Func.layout()) {
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
    const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions,
    uint64_t DotAddress) {
  DotAddress = alignTo(DotAddress, llvm::Align(opts::AlignFunctions));
  for (BinaryFunction *Func : SortedFunctions) {
    if (!Func->isSplit())
      continue;
    DotAddress = alignTo(DotAddress, BinaryFunction::MinAlign);
    uint64_t Pad =
        offsetToAlignment(DotAddress, llvm::Align(Func->getAlignment()));
    if (Pad <= Func->getMaxColdAlignmentBytes())
      DotAddress += Pad;
    ColdAddresses[Func] = DotAddress;
    LLVM_DEBUG(dbgs() << Func->getPrintName() << " cold tentative: "
                      << Twine::utohexstr(DotAddress) << "\n");
    DotAddress += Func->estimateColdSize();
    DotAddress = alignTo(DotAddress, Func->getConstantIslandAlignment());
    DotAddress += Func->estimateConstantIslandSize();
  }
  return DotAddress;
}

uint64_t LongJmpPass::tentativeLayoutRelocMode(
    const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions,
    uint64_t DotAddress) {

  // Compute hot cold frontier
  uint32_t LastHotIndex = -1u;
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
  for (BinaryFunction *Func : SortedFunctions) {
    if (!BC.shouldEmit(*Func)) {
      HotAddresses[Func] = Func->getAddress();
      continue;
    }

    if (!ColdLayoutDone && CurrentIndex >= LastHotIndex) {
      DotAddress =
          tentativeLayoutRelocColdPart(BC, SortedFunctions, DotAddress);
      ColdLayoutDone = true;
      if (opts::HotFunctionsAtEnd)
        DotAddress = alignTo(DotAddress, opts::AlignText);
    }

    DotAddress = alignTo(DotAddress, BinaryFunction::MinAlign);
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

    DotAddress = alignTo(DotAddress, Func->getConstantIslandAlignment());
    DotAddress += Func->estimateConstantIslandSize();
    ++CurrentIndex;
  }
  // BBs
  for (BinaryFunction *Func : SortedFunctions)
    tentativeBBLayout(*Func);

  return DotAddress;
}

void LongJmpPass::tentativeLayout(
    const BinaryContext &BC, std::vector<BinaryFunction *> &SortedFunctions) {
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
      uint64_t Pad =
          offsetToAlignment(DotAddress, llvm::Align(opts::AlignText));
      if (Pad + EstimatedTextSize <= BC.OldTextSectionSize) {
        DotAddress += Pad;
      }
    }
  }

  if (!EstimatedTextSize || EstimatedTextSize > BC.OldTextSectionSize)
    DotAddress = alignTo(BC.LayoutStartAddress, opts::AlignText);

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

bool LongJmpPass::relaxStub(BinaryBasicBlock &StubBB) {
  const BinaryFunction &Func = *StubBB.getFunction();
  const BinaryContext &BC = Func.getBinaryContext();
  const int Bits = StubBits[&StubBB];
  // Already working with the largest range?
  if (Bits == static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8))
    return false;

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
    return false;

  // Fits short jmp
  if (!(PCRelTgtAddress & ShortJmpMask)) {
    if (Bits >= RangeShortJmp)
      return false;

    LLVM_DEBUG(dbgs() << "Relaxing stub to short jump. PCRelTgtAddress = "
                      << Twine::utohexstr(PCRelTgtAddress)
                      << " RealTargetSym = " << RealTargetSym->getName()
                      << "\n");
    relaxStubToShortJmp(StubBB, RealTargetSym);
    StubBits[&StubBB] = RangeShortJmp;
    return true;
  }

  // The long jmp uses absolute address on AArch64
  // So we could not use it for PIC binaries
  if (BC.isAArch64() && !BC.HasFixedLoadAddress) {
    errs() << "BOLT-ERROR: Unable to relax stub for PIC binary\n";
    exit(1);
  }

  LLVM_DEBUG(dbgs() << "Relaxing stub to long jump. PCRelTgtAddress = "
                    << Twine::utohexstr(PCRelTgtAddress)
                    << " RealTargetSym = " << RealTargetSym->getName() << "\n");
  relaxStubToLongJmp(StubBB, RealTargetSym);
  StubBits[&StubBB] = static_cast<int>(BC.AsmInfo->getCodePointerSize() * 8);
  return true;
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
  uint64_t Mask = ~((1ULL << BitsAvail) - 1);

  uint64_t PCRelTgtAddress = getSymbolAddress(BC, TgtSym, TgtBB);
  PCRelTgtAddress = DotAddress > PCRelTgtAddress ? DotAddress - PCRelTgtAddress
                                                 : PCRelTgtAddress - DotAddress;

  return PCRelTgtAddress & Mask;
}

bool LongJmpPass::relax(BinaryFunction &Func) {
  const BinaryContext &BC = Func.getBinaryContext();
  bool Modified = false;

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

      if (!shouldInsertStub(BC, Inst)) {
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

    Modified |= relaxStub(BB);
  }

  for (std::pair<BinaryBasicBlock *, std::unique_ptr<BinaryBasicBlock>> &Elmt :
       Insertions) {
    if (!Elmt.second)
      continue;
    std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
    NewBBs.emplace_back(std::move(Elmt.second));
    Func.insertBasicBlocks(Elmt.first, std::move(NewBBs), true);
  }

  return Modified;
}

void LongJmpPass::runOnFunctions(BinaryContext &BC) {
  outs() << "BOLT-INFO: Starting stub-insertion pass\n";
  std::vector<BinaryFunction *> Sorted = BC.getSortedFunctions();
  bool Modified;
  uint32_t Iterations = 0;
  do {
    ++Iterations;
    Modified = false;
    tentativeLayout(BC, Sorted);
    updateStubGroups();
    for (BinaryFunction *Func : Sorted) {
      if (relax(*Func)) {
        // Don't ruin non-simple functions, they can't afford to have the layout
        // changed.
        if (Func->isSimple())
          Func->fixBranches();
        Modified = true;
      }
    }
  } while (Modified);
  outs() << "BOLT-INFO: Inserted " << NumHotStubs
         << " stubs in the hot area and " << NumColdStubs
         << " stubs in the cold area. Shared " << NumSharedStubs
         << " times, iterated " << Iterations << " times.\n";
}
} // namespace bolt
} // namespace llvm
