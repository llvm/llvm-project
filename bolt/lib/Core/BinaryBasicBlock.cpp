//===- bolt/Core/BinaryBasicBlock.cpp - Low-level basic block -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinaryBasicBlock class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Errc.h"

#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

constexpr uint32_t BinaryBasicBlock::INVALID_OFFSET;

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS) {
  return LHS.Index < RHS.Index;
}

bool BinaryBasicBlock::hasCFG() const { return getParent()->hasCFG(); }

bool BinaryBasicBlock::isEntryPoint() const {
  return getParent()->isEntryPoint(*this);
}

bool BinaryBasicBlock::hasInstructions() const {
  return getParent()->hasInstructions();
}

const JumpTable *BinaryBasicBlock::getJumpTable() const {
  const MCInst *Inst = getLastNonPseudoInstr();
  const JumpTable *JT = Inst ? Function->getJumpTable(*Inst) : nullptr;
  return JT;
}

void BinaryBasicBlock::adjustNumPseudos(const MCInst &Inst, int Sign) {
  BinaryContext &BC = Function->getBinaryContext();
  if (BC.MIB->isPseudo(Inst))
    NumPseudos += Sign;
}

BinaryBasicBlock::iterator BinaryBasicBlock::getFirstNonPseudo() {
  const BinaryContext &BC = Function->getBinaryContext();
  for (auto II = Instructions.begin(), E = Instructions.end(); II != E; ++II) {
    if (!BC.MIB->isPseudo(*II))
      return II;
  }
  return end();
}

BinaryBasicBlock::reverse_iterator BinaryBasicBlock::getLastNonPseudo() {
  const BinaryContext &BC = Function->getBinaryContext();
  for (auto RII = Instructions.rbegin(), E = Instructions.rend(); RII != E;
       ++RII) {
    if (!BC.MIB->isPseudo(*RII))
      return RII;
  }
  return rend();
}

bool BinaryBasicBlock::validateSuccessorInvariants() {
  const MCInst *Inst = getLastNonPseudoInstr();
  const JumpTable *JT = Inst ? Function->getJumpTable(*Inst) : nullptr;
  BinaryContext &BC = Function->getBinaryContext();
  bool Valid = true;

  if (JT) {
    // Note: for now we assume that successors do not reference labels from
    // any overlapping jump tables.  We only look at the entries for the jump
    // table that is referenced at the last instruction.
    const auto Range = JT->getEntriesForAddress(BC.MIB->getJumpTable(*Inst));
    const std::vector<const MCSymbol *> Entries(
        std::next(JT->Entries.begin(), Range.first),
        std::next(JT->Entries.begin(), Range.second));
    std::set<const MCSymbol *> UniqueSyms(Entries.begin(), Entries.end());
    for (BinaryBasicBlock *Succ : Successors) {
      auto Itr = UniqueSyms.find(Succ->getLabel());
      if (Itr != UniqueSyms.end()) {
        UniqueSyms.erase(Itr);
      } else {
        // Work on the assumption that jump table blocks don't
        // have a conditional successor.
        Valid = false;
        BC.errs() << "BOLT-WARNING: Jump table successor " << Succ->getName()
                  << " not contained in the jump table.\n";
      }
    }
    // If there are any leftover entries in the jump table, they
    // must be one of the function end labels.
    if (Valid) {
      for (const MCSymbol *Sym : UniqueSyms) {
        Valid &= (Sym == Function->getFunctionEndLabel() ||
                  Sym == Function->getFunctionEndLabel(getFragmentNum()));
        if (!Valid) {
          const BinaryFunction *TargetBF = BC.getFunctionForSymbol(Sym);
          if (TargetBF) {
            // It's possible for another function to be in the jump table entry
            // as a result of built-in unreachable.
            Valid = true;
          } else {
            BC.errs() << "BOLT-WARNING: Jump table contains illegal entry: "
                      << Sym->getName() << "\n";
          }
        }
        if (!Valid)
          break;
      }
    }
  } else {
    // Unknown control flow.
    if (Inst && BC.MIB->isIndirectBranch(*Inst))
      return true;

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;

    if (analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
      switch (Successors.size()) {
      case 0:
        Valid = !CondBranch && !UncondBranch;
        break;
      case 1: {
        const bool HasCondBlock =
            CondBranch && Function->getBasicBlockForLabel(
                              BC.MIB->getTargetSymbol(*CondBranch));
        Valid = !CondBranch || !HasCondBlock;
        break;
      }
      case 2:
        Valid =
            CondBranch && TBB == getConditionalSuccessor(true)->getLabel() &&
            (UncondBranch ? FBB == getConditionalSuccessor(false)->getLabel()
                          : !FBB);
        break;
      }
    }
  }
  if (!Valid) {
    BC.errs() << "BOLT-WARNING: CFG invalid in " << *getFunction() << " @ "
              << getName() << "\n";
    if (JT) {
      BC.errs() << "Jump Table instruction addr = 0x"
                << Twine::utohexstr(BC.MIB->getJumpTable(*Inst)) << "\n";
      JT->print(errs());
    }
    getFunction()->dump();
  }
  return Valid;
}

BinaryBasicBlock *BinaryBasicBlock::getSuccessor(const MCSymbol *Label) const {
  if (!Label && succ_size() == 1)
    return *succ_begin();

  for (BinaryBasicBlock *BB : successors())
    if (BB->getLabel() == Label)
      return BB;

  return nullptr;
}

BinaryBasicBlock *BinaryBasicBlock::getSuccessor(const MCSymbol *Label,
                                                 BinaryBranchInfo &BI) const {
  auto BIIter = branch_info_begin();
  for (BinaryBasicBlock *BB : successors()) {
    if (BB->getLabel() == Label) {
      BI = *BIIter;
      return BB;
    }
    ++BIIter;
  }

  return nullptr;
}

BinaryBasicBlock *BinaryBasicBlock::getLandingPad(const MCSymbol *Label) const {
  for (BinaryBasicBlock *BB : landing_pads())
    if (BB->getLabel() == Label)
      return BB;

  return nullptr;
}

int32_t BinaryBasicBlock::getCFIStateAtInstr(const MCInst *Instr) const {
  assert(
      getFunction()->getState() >= BinaryFunction::State::CFG &&
      "can only calculate CFI state when function is in or past the CFG state");

  const BinaryFunction::CFIInstrMapType &FDEProgram =
      getFunction()->getFDEProgram();

  // Find the last CFI preceding Instr in this basic block.
  const MCInst *LastCFI = nullptr;
  bool InstrSeen = (Instr == nullptr);
  for (const MCInst &Inst : llvm::reverse(Instructions)) {
    if (!InstrSeen) {
      InstrSeen = (&Inst == Instr);
      continue;
    }
    if (Function->getBinaryContext().MIB->isCFI(Inst)) {
      LastCFI = &Inst;
      break;
    }
  }

  assert(InstrSeen && "instruction expected in basic block");

  // CFI state is the same as at basic block entry point.
  if (!LastCFI)
    return getCFIState();

  // Fold all RememberState/RestoreState sequences, such as for:
  //
  //   [ CFI #(K-1) ]
  //   RememberState (#K)
  //     ....
  //   RestoreState
  //   RememberState
  //     ....
  //   RestoreState
  //   [ GNU_args_size ]
  //   RememberState
  //     ....
  //   RestoreState   <- LastCFI
  //
  // we return K - the most efficient state to (re-)generate.
  int64_t State = LastCFI->getOperand(0).getImm();
  while (State >= 0 &&
         FDEProgram[State].getOperation() == MCCFIInstruction::OpRestoreState) {
    int32_t Depth = 1;
    --State;
    assert(State >= 0 && "first CFI cannot be RestoreState");
    while (Depth && State >= 0) {
      const MCCFIInstruction &CFIInstr = FDEProgram[State];
      if (CFIInstr.getOperation() == MCCFIInstruction::OpRestoreState)
        ++Depth;
      else if (CFIInstr.getOperation() == MCCFIInstruction::OpRememberState)
        --Depth;
      --State;
    }
    assert(Depth == 0 && "unbalanced RememberState/RestoreState stack");

    // Skip any GNU_args_size.
    while (State >= 0 && FDEProgram[State].getOperation() ==
                             MCCFIInstruction::OpGnuArgsSize) {
      --State;
    }
  }

  assert((State + 1 >= 0) && "miscalculated CFI state");
  return State + 1;
}

void BinaryBasicBlock::addSuccessor(BinaryBasicBlock *Succ, uint64_t Count,
                                    uint64_t MispredictedCount) {
  Successors.push_back(Succ);
  BranchInfo.push_back({Count, MispredictedCount});
  Succ->Predecessors.push_back(this);
}

void BinaryBasicBlock::replaceSuccessor(BinaryBasicBlock *Succ,
                                        BinaryBasicBlock *NewSucc,
                                        uint64_t Count,
                                        uint64_t MispredictedCount) {
  Succ->removePredecessor(this, /*Multiple=*/false);
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  *I = NewSucc;
  *BI = BinaryBranchInfo{Count, MispredictedCount};
  NewSucc->addPredecessor(this);
}

void BinaryBasicBlock::removeAllSuccessors() {
  SmallPtrSet<BinaryBasicBlock *, 2> UniqSuccessors(succ_begin(), succ_end());
  for (BinaryBasicBlock *SuccessorBB : UniqSuccessors)
    SuccessorBB->removePredecessor(this);
  Successors.clear();
  BranchInfo.clear();
}

void BinaryBasicBlock::removeSuccessor(BinaryBasicBlock *Succ) {
  Succ->removePredecessor(this, /*Multiple=*/false);
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  Successors.erase(I);
  BranchInfo.erase(BI);
}

void BinaryBasicBlock::addPredecessor(BinaryBasicBlock *Pred) {
  Predecessors.push_back(Pred);
}

void BinaryBasicBlock::removePredecessor(BinaryBasicBlock *Pred,
                                         bool Multiple) {
  // Note: the predecessor could be listed multiple times.
  bool Erased = false;
  for (auto PredI = Predecessors.begin(); PredI != Predecessors.end();) {
    if (*PredI == Pred) {
      Erased = true;
      PredI = Predecessors.erase(PredI);
      if (!Multiple)
        return;
    } else {
      ++PredI;
    }
  }
  assert(Erased && "Pred is not a predecessor of this block!");
  (void)Erased;
}

void BinaryBasicBlock::removeDuplicateConditionalSuccessor(MCInst *CondBranch) {
  assert(succ_size() == 2 && Successors[0] == Successors[1] &&
         "conditional successors expected");

  BinaryBasicBlock *Succ = Successors[0];
  const BinaryBranchInfo CondBI = BranchInfo[0];
  const BinaryBranchInfo UncondBI = BranchInfo[1];

  eraseInstruction(findInstruction(CondBranch));

  Successors.clear();
  BranchInfo.clear();

  Successors.push_back(Succ);

  uint64_t Count = COUNT_NO_PROFILE;
  if (CondBI.Count != COUNT_NO_PROFILE && UncondBI.Count != COUNT_NO_PROFILE)
    Count = CondBI.Count + UncondBI.Count;
  BranchInfo.push_back({Count, 0});
}

void BinaryBasicBlock::updateJumpTableSuccessors() {
  const JumpTable *JT = getJumpTable();
  assert(JT && "Expected jump table instruction.");

  // Clear existing successors.
  removeAllSuccessors();

  // Generate the list of successors in deterministic order without duplicates.
  SmallVector<BinaryBasicBlock *, 16> SuccessorBBs;
  for (const MCSymbol *Label : JT->Entries) {
    BinaryBasicBlock *BB = getFunction()->getBasicBlockForLabel(Label);
    // Ignore __builtin_unreachable()
    if (!BB) {
      assert(Label == getFunction()->getFunctionEndLabel() &&
             "JT label should match a block or end of function.");
      continue;
    }
    SuccessorBBs.emplace_back(BB);
  }
  llvm::sort(SuccessorBBs,
             [](const BinaryBasicBlock *BB1, const BinaryBasicBlock *BB2) {
               return BB1->getInputOffset() < BB2->getInputOffset();
             });
  SuccessorBBs.erase(llvm::unique(SuccessorBBs), SuccessorBBs.end());

  for (BinaryBasicBlock *BB : SuccessorBBs)
    addSuccessor(BB);
}

void BinaryBasicBlock::adjustExecutionCount(double Ratio) {
  auto adjustedCount = [&](uint64_t Count) -> uint64_t {
    double NewCount = Count * Ratio;
    if (!NewCount && Count && (Ratio > 0.0))
      NewCount = 1;
    return NewCount;
  };

  setExecutionCount(adjustedCount(getKnownExecutionCount()));
  for (BinaryBranchInfo &BI : branch_info()) {
    if (BI.Count != COUNT_NO_PROFILE)
      BI.Count = adjustedCount(BI.Count);
    if (BI.MispredictedCount != COUNT_INFERRED)
      BI.MispredictedCount = adjustedCount(BI.MispredictedCount);
  }
}

bool BinaryBasicBlock::analyzeBranch(const MCSymbol *&TBB, const MCSymbol *&FBB,
                                     MCInst *&CondBranch,
                                     MCInst *&UncondBranch) {
  auto &MIB = Function->getBinaryContext().MIB;
  return MIB->analyzeBranch(Instructions.begin(), Instructions.end(), TBB, FBB,
                            CondBranch, UncondBranch);
}

MCInst *BinaryBasicBlock::getTerminatorBefore(MCInst *Pos) {
  BinaryContext &BC = Function->getBinaryContext();
  auto Itr = rbegin();
  bool Check = Pos ? false : true;
  MCInst *FirstTerminator = nullptr;
  while (Itr != rend()) {
    if (!Check) {
      if (&*Itr == Pos)
        Check = true;
      ++Itr;
      continue;
    }
    if (BC.MIB->isTerminator(*Itr))
      FirstTerminator = &*Itr;
    ++Itr;
  }
  return FirstTerminator;
}

bool BinaryBasicBlock::hasTerminatorAfter(MCInst *Pos) {
  BinaryContext &BC = Function->getBinaryContext();
  auto Itr = rbegin();
  while (Itr != rend()) {
    if (&*Itr == Pos)
      return false;
    if (BC.MIB->isTerminator(*Itr))
      return true;
    ++Itr;
  }
  return false;
}

bool BinaryBasicBlock::swapConditionalSuccessors() {
  if (succ_size() != 2)
    return false;

  std::swap(Successors[0], Successors[1]);
  std::swap(BranchInfo[0], BranchInfo[1]);
  return true;
}

void BinaryBasicBlock::addBranchInstruction(const BinaryBasicBlock *Successor) {
  assert(isSuccessor(Successor));
  BinaryContext &BC = Function->getBinaryContext();
  MCInst NewInst;
  std::unique_lock<llvm::sys::RWMutex> Lock(BC.CtxMutex);
  BC.MIB->createUncondBranch(NewInst, Successor->getLabel(), BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

void BinaryBasicBlock::addTailCallInstruction(const MCSymbol *Target) {
  BinaryContext &BC = Function->getBinaryContext();
  MCInst NewInst;
  BC.MIB->createTailCall(NewInst, Target, BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

uint32_t BinaryBasicBlock::getNumCalls() const {
  uint32_t N = 0;
  BinaryContext &BC = Function->getBinaryContext();
  for (const MCInst &Instr : Instructions) {
    if (BC.MIB->isCall(Instr))
      ++N;
  }
  return N;
}

uint32_t BinaryBasicBlock::getNumPseudos() const {
#ifndef NDEBUG
  BinaryContext &BC = Function->getBinaryContext();
  uint32_t N = 0;
  for (const MCInst &Instr : Instructions)
    if (BC.MIB->isPseudo(Instr))
      ++N;

  if (N != NumPseudos) {
    BC.errs() << "BOLT-ERROR: instructions for basic block " << getName()
              << " in function " << *Function << ": calculated pseudos " << N
              << ", set pseudos " << NumPseudos << ", size " << size() << '\n';
    llvm_unreachable("pseudos mismatch");
  }
#endif
  return NumPseudos;
}

ErrorOr<std::pair<double, double>>
BinaryBasicBlock::getBranchStats(const BinaryBasicBlock *Succ) const {
  if (Function->hasValidProfile()) {
    uint64_t TotalCount = 0;
    uint64_t TotalMispreds = 0;
    for (const BinaryBranchInfo &BI : BranchInfo) {
      if (BI.Count != COUNT_NO_PROFILE) {
        TotalCount += BI.Count;
        TotalMispreds += BI.MispredictedCount;
      }
    }

    if (TotalCount > 0) {
      auto Itr = llvm::find(Successors, Succ);
      assert(Itr != Successors.end());
      const BinaryBranchInfo &BI = BranchInfo[Itr - Successors.begin()];
      if (BI.Count && BI.Count != COUNT_NO_PROFILE) {
        if (TotalMispreds == 0)
          TotalMispreds = 1;
        return std::make_pair(double(BI.Count) / TotalCount,
                              double(BI.MispredictedCount) / TotalMispreds);
      }
    }
  }
  return make_error_code(llvm::errc::result_out_of_range);
}

void BinaryBasicBlock::dump() const {
  BinaryContext &BC = Function->getBinaryContext();
  if (Label)
    BC.outs() << Label->getName() << ":\n";
  BC.printInstructions(BC.outs(), Instructions.begin(), Instructions.end(),
                       getOffset(), Function);
  BC.outs() << "preds:";
  for (auto itr = pred_begin(); itr != pred_end(); ++itr) {
    BC.outs() << " " << (*itr)->getName();
  }
  BC.outs() << "\nsuccs:";
  for (auto itr = succ_begin(); itr != succ_end(); ++itr) {
    BC.outs() << " " << (*itr)->getName();
  }
  BC.outs() << "\n";
}

uint64_t BinaryBasicBlock::estimateSize(const MCCodeEmitter *Emitter) const {
  return Function->getBinaryContext().computeCodeSize(begin(), end(), Emitter);
}

BinaryBasicBlock::BinaryBranchInfo &
BinaryBasicBlock::getBranchInfo(const BinaryBasicBlock &Succ) {
  return const_cast<BinaryBranchInfo &>(
      static_cast<const BinaryBasicBlock &>(*this).getBranchInfo(Succ));
}

const BinaryBasicBlock::BinaryBranchInfo &
BinaryBasicBlock::getBranchInfo(const BinaryBasicBlock &Succ) const {
  const auto Zip = llvm::zip(successors(), branch_info());
  const auto Result = llvm::find_if(
      Zip, [&](const auto &Tuple) { return std::get<0>(Tuple) == &Succ; });
  assert(Result != Zip.end() && "Cannot find target in successors");
  return std::get<1>(*Result);
}

BinaryBasicBlock *BinaryBasicBlock::splitAt(iterator II) {
  assert(II != end() && "expected iterator pointing to instruction");

  BinaryBasicBlock *NewBlock = getFunction()->addBasicBlock();

  // Adjust successors/predecessors and propagate the execution count.
  moveAllSuccessorsTo(NewBlock);
  addSuccessor(NewBlock, getExecutionCount(), 0);

  // Set correct CFI state for the new block.
  NewBlock->setCFIState(getCFIStateAtInstr(&*II));

  // Move instructions over.
  adjustNumPseudos(II, end(), -1);
  NewBlock->addInstructions(II, end());
  Instructions.erase(II, end());

  return NewBlock;
}

} // namespace bolt
} // namespace llvm
