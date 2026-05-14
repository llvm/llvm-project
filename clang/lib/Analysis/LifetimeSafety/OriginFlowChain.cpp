//===- OriginFlowChain.cpp - C++ Lifetime Safety Checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements buildOriginFlowChain.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/OriginFlowChain.h"

namespace clang::lifetimes::internal {
llvm::SmallVector<OriginID> buildOriginFlowChain(
    const FactManager &FactMgr, const LoanPropagationAnalysis &LoanPropagation,
    ProgramPoint StartPoint, const OriginID StartOID, const LoanID TargetLoan) {

  const auto hasLoanAtOrigin = [&LoanPropagation](OriginID OID, LoanID LID,
                                                  ProgramPoint CurrPoint) {
    return LoanPropagation.getLoans(OID, CurrPoint).contains(LID);
  };

  assert(hasLoanAtOrigin(StartOID, TargetLoan, StartPoint) &&
         "TargetLoan must be present in the initial propagation point");

  OriginID CurrOID = StartOID;
  llvm::SmallVector<OriginID> OriginFlowChain;
  llvm::ArrayRef<const Fact *> Facts = FactMgr.getBlockContaining(StartPoint);

  for (const Fact *F : llvm::reverse(Facts)) {
    if (const auto *IF = F->getAs<IssueFact>())
      if (IF->getLoanID() == TargetLoan)
        return OriginFlowChain;

    const auto *OFF = F->getAs<OriginFlowFact>();
    if (!OFF)
      continue;
    const OriginID SrcOriginID = OFF->getSrcOriginID();
    if (OFF->getDestOriginID() != CurrOID)
      continue;
    if (!hasLoanAtOrigin(SrcOriginID, TargetLoan, OFF))
      continue;
    OriginFlowChain.push_back(SrcOriginID);
    CurrOID = SrcOriginID;
  }

  return {};
}
} // namespace clang::lifetimes::internal
