//===- AssignmentQuery.cpp - C++ Lifetime Safety Checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements trackAssignmentHistory.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/AssignmentQuery.h"

namespace clang::lifetimes::internal {
void trackAssignmentHistory(
    const FactManager &FactMgr, const LoanPropagationAnalysis &LoanPropagation,
    llvm::SmallVectorImpl<AssignmentUnit> &AssignmentList,
    const CFGBlock *StartBlock, OriginID StartOID, const LoanID &EndLoanID) {
  const auto TryInsertPropagationChain = [&](const OriginFlowFact *OFF) {
    if (OFF->getDestOriginID() == StartOID &&
        LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF)
            .contains(EndLoanID)) {
      const OriginID SrcOriginID = OFF->getSrcOriginID();
      AssignmentList.push_back(FactMgr.getOriginMgr().getOrigin(SrcOriginID));
      StartOID = SrcOriginID;
    }
  };

  llvm::ArrayRef<const Fact *> Facts = FactMgr.getFacts(StartBlock);
  for (const Fact *F : llvm::reverse(Facts)) {
    if (const auto *OFF = F->getAs<OriginFlowFact>()) {
      TryInsertPropagationChain(OFF);
    } else if (const auto *IF = F->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID)
        return;
    }
  }
}
} // namespace clang::lifetimes::internal
