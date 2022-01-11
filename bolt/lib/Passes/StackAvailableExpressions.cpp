//===- bolt/Passes/StackAvailableExpressions.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StackAvailableExpressions class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/StackAvailableExpressions.h"
#include "bolt/Passes/FrameAnalysis.h"
#include "bolt/Passes/RegAnalysis.h"

#define DEBUG_TYPE "sae"

namespace llvm {
namespace bolt {

StackAvailableExpressions::StackAvailableExpressions(const RegAnalysis &RA,
                                                     const FrameAnalysis &FA,
                                                     BinaryFunction &BF)
    : InstrsDataflowAnalysis(BF), RA(RA), FA(FA) {}

void StackAvailableExpressions::preflight() {
  LLVM_DEBUG(dbgs() << "Starting StackAvailableExpressions on \""
                    << Func.getPrintName() << "\"\n");

  // Populate our universe of tracked expressions. We are interested in
  // tracking available stores to frame position at any given point of the
  // program.
  for (BinaryBasicBlock &BB : Func) {
    for (MCInst &Inst : BB) {
      ErrorOr<const FrameIndexEntry &> FIE = FA.getFIEFor(Inst);
      if (!FIE)
        continue;
      if (FIE->IsStore == true && FIE->IsSimple == true) {
        Expressions.push_back(&Inst);
        ExprToIdx[&Inst] = NumInstrs++;
      }
    }
  }
}

BitVector
StackAvailableExpressions::getStartingStateAtBB(const BinaryBasicBlock &BB) {
  // Entry points start with empty set
  // All others start with the full set.
  if (BB.pred_size() == 0 && BB.throw_size() == 0)
    return BitVector(NumInstrs, false);
  return BitVector(NumInstrs, true);
}

BitVector
StackAvailableExpressions::getStartingStateAtPoint(const MCInst &Point) {
  return BitVector(NumInstrs, true);
}

void StackAvailableExpressions::doConfluence(BitVector &StateOut,
                                             const BitVector &StateIn) {
  StateOut &= StateIn;
}

namespace {

bool isLoadRedundant(const FrameIndexEntry &LoadFIE,
                     const FrameIndexEntry &StoreFIE) {
  if (LoadFIE.IsLoad == false || LoadFIE.IsSimple == false)
    return false;
  if (LoadFIE.StackOffset == StoreFIE.StackOffset &&
      LoadFIE.Size == StoreFIE.Size)
    return true;

  return false;
}
}

bool StackAvailableExpressions::doesXKillsY(const MCInst *X, const MCInst *Y) {
  // if both are stores, and both store to the same stack location, return
  // true
  ErrorOr<const FrameIndexEntry &> FIEX = FA.getFIEFor(*X);
  ErrorOr<const FrameIndexEntry &> FIEY = FA.getFIEFor(*Y);
  if (FIEX && FIEY) {
    if (isLoadRedundant(*FIEX, *FIEY))
      return false;
    if (FIEX->IsStore == true && FIEY->IsStore == true &&
        FIEX->StackOffset + FIEX->Size > FIEY->StackOffset &&
        FIEX->StackOffset < FIEY->StackOffset + FIEY->Size)
      return true;
  }
  // getClobberedRegs for X and Y. If they intersect, return true
  BitVector XClobbers = BitVector(BC.MRI->getNumRegs(), false);
  BitVector YClobbers = BitVector(BC.MRI->getNumRegs(), false);
  RA.getInstClobberList(*X, XClobbers);
  // If Y is a store to stack, its clobber list is its source reg. This is
  // different than the rest because we want to check if the store source
  // reaches its corresponding load untouched.
  if (FIEY && FIEY->IsStore == true && FIEY->IsStoreFromReg)
    YClobbers.set(FIEY->RegOrImm);
  else
    RA.getInstClobberList(*Y, YClobbers);

  XClobbers &= YClobbers;
  return XClobbers.any();
}

BitVector StackAvailableExpressions::computeNext(const MCInst &Point,
                                                 const BitVector &Cur) {
  BitVector Next = Cur;
  // Kill
  for (auto I = expr_begin(Next), E = expr_end(); I != E; ++I) {
    assert(*I != nullptr && "Lost pointers");
    LLVM_DEBUG(dbgs() << "\t\t\tDoes it kill ");
    LLVM_DEBUG((*I)->dump());
    if (doesXKillsY(&Point, *I)) {
      LLVM_DEBUG(dbgs() << "\t\t\t\tKilling ");
      LLVM_DEBUG((*I)->dump());
      Next.reset(I.getBitVectorIndex());
    }
  }
  // Gen
  if (ErrorOr<const FrameIndexEntry &> FIE = FA.getFIEFor(Point)) {
    if (FIE->IsStore == true && FIE->IsSimple == true)
      Next.set(ExprToIdx[&Point]);
  }
  return Next;
}

} // namespace bolt
} // namespace llvm
