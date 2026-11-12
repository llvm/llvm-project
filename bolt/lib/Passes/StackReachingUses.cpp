//===- bolt/Passes/StackReachingUses.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StackReachingUses class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/StackReachingUses.h"
#include "bolt/Passes/FrameAnalysis.h"

#define DEBUG_TYPE "sru"

namespace llvm {
namespace bolt {

bool StackReachingUses::isLoadedInDifferentReg(const FrameIndexEntry &StoreFIE,
                                               ExprIterator Candidates) const {
  for (auto I = Candidates; I != expr_end(); ++I) {
    const MCInst *ReachingInst = *I;
    if (ErrorOr<const FrameIndexEntry &> FIEY = FA.getFIEFor(*ReachingInst)) {
      assert(FIEY->IsLoad == 1);
      if (StoreFIE.StackOffset + StoreFIE.Size > FIEY->StackOffset &&
          StoreFIE.StackOffset < FIEY->StackOffset + FIEY->Size &&
          StoreFIE.RegOrImm != FIEY->RegOrImm)
        return true;
    }
  }
  return false;
}

bool StackReachingUses::isStoreUsed(const FrameIndexEntry &StoreFIE,
                                    ExprIterator Candidates,
                                    bool IncludeLocalAccesses) const {
  for (auto I = Candidates; I != expr_end(); ++I) {
    const MCInst *ReachingInst = *I;
    if (IncludeLocalAccesses) {
      if (ErrorOr<const FrameIndexEntry &> FIEY = FA.getFIEFor(*ReachingInst)) {
        assert(FIEY->IsLoad == 1);
        if (StoreFIE.StackOffset + StoreFIE.Size > FIEY->StackOffset &&
            StoreFIE.StackOffset < FIEY->StackOffset + FIEY->Size)
          return true;
      }
    }
    ErrorOr<const ArgAccesses &> Args = FA.getArgAccessesFor(*ReachingInst);
    if (!Args)
      continue;
    if (Args->AssumeEverything)
      return true;

    for (ArgInStackAccess FIEY : Args->Set)
      if (StoreFIE.StackOffset + StoreFIE.Size > FIEY.StackOffset &&
          StoreFIE.StackOffset < FIEY.StackOffset + FIEY.Size)
        return true;
  }
  return false;
}

void StackReachingUses::preflight() {
  LLVM_DEBUG(dbgs() << "Starting StackReachingUses on \"" << Func.getPrintName()
                    << "\"\n");

  // Populate our universe of tracked expressions. We are interested in
  // tracking reaching loads from frame position at any given point of the
  // program.
  for (BinaryBasicBlock &BB : Func) {
    for (MCInst &Inst : BB) {
      if (ErrorOr<const FrameIndexEntry &> FIE = FA.getFIEFor(Inst)) {
        if (FIE->IsLoad == true) {
          Expressions.push_back(&Inst);
          ExprToIdx[&Inst] = NumInstrs++;
          continue;
        }
      }
      ErrorOr<const ArgAccesses &> AA = FA.getArgAccessesFor(Inst);
      if (AA && (!AA->Set.empty() || AA->AssumeEverything)) {
        Expressions.push_back(&Inst);
        ExprToIdx[&Inst] = NumInstrs++;
      }
    }
  }
}

bool StackReachingUses::doesXKillsY(const MCInst *X, const MCInst *Y) {
  // if X is a store to the same stack location and the bytes fetched is a
  // superset of those bytes affected by the load in Y, return true
  ErrorOr<const FrameIndexEntry &> FIEX = FA.getFIEFor(*X);
  ErrorOr<const FrameIndexEntry &> FIEY = FA.getFIEFor(*Y);
  if (FIEX && FIEY) {
    if (FIEX->IsSimple == true && FIEY->IsSimple == true &&
        FIEX->IsStore == true && FIEY->IsLoad == true &&
        FIEX->StackOffset <= FIEY->StackOffset &&
        FIEX->StackOffset + FIEX->Size >= FIEY->StackOffset + FIEY->Size)
      return true;
  }
  return false;
}

BitVector StackReachingUses::computeNext(const MCInst &Point,
                                         const BitVector &Cur) {
  BitVector Next = Cur;
  // Kill
  for (auto I = expr_begin(Next), E = expr_end(); I != E; ++I) {
    assert(*I != nullptr && "Lost pointers");
    if (doesXKillsY(&Point, *I)) {
      LLVM_DEBUG(dbgs() << "\t\t\tKilling ");
      LLVM_DEBUG((*I)->dump());
      Next.reset(I.getBitVectorIndex());
    }
  };
  // Gen
  if (ErrorOr<const FrameIndexEntry &> FIE = FA.getFIEFor(Point)) {
    if (FIE->IsLoad == true)
      Next.set(ExprToIdx[&Point]);
  }
  ErrorOr<const ArgAccesses &> AA = FA.getArgAccessesFor(Point);
  if (AA && (!AA->Set.empty() || AA->AssumeEverything))
    Next.set(ExprToIdx[&Point]);
  return Next;
}

} // namespace bolt
} // namespace llvm
