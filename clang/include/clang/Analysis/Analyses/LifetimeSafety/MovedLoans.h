//===- MovedLoans.h - Moved Loans Analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MovedLoansAnalysis, a forward dataflow analysis that
// tracks which loans have been moved out of their original storage location
// at each program point.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_MOVED_LOANS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_MOVED_LOANS_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"

namespace clang::lifetimes::internal {

// Map from a loan to an expression responsible for moving the borrowed storage.
using MovedLoansMap = llvm::ImmutableMap<LoanID, const Expr *>;

class MovedLoansAnalysis {
public:
  MovedLoansAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                     const LoanPropagationAnalysis &LoanPropagation,
                     const LiveOriginsAnalysis &LiveOrigins,
                     const LoanManager &LoanMgr,
                     MovedLoansMap::Factory &MovedLoansMapFactory);
  ~MovedLoansAnalysis();

  MovedLoansMap getMovedLoans(ProgramPoint P) const;

private:
  class Impl;
  std::unique_ptr<Impl> PImpl;
};

} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_MOVED_LOANS_H
