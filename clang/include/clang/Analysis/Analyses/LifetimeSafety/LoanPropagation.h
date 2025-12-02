//===- LoanPropagation.h - Loan Propagation Analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoanPropagationAnalysis, a forward dataflow analysis
// that tracks which loans each origin holds at each program point. Loans
// represent borrows of storage locations and are propagated through the
// program as pointers are copied or assigned.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"

namespace clang::lifetimes::internal {

// Using LLVM's immutable collections is efficient for dataflow analysis
// as it avoids deep copies during state transitions.
// TODO(opt): Consider using a bitset to represent the set of loans.
using LoanSet = llvm::ImmutableSet<LoanID>;
using OriginLoanMap = llvm::ImmutableMap<OriginID, LoanSet>;

class LoanPropagationAnalysis {
public:
  LoanPropagationAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                          OriginLoanMap::Factory &OriginLoanMapFactory,
                          LoanSet::Factory &LoanSetFactory);
  ~LoanPropagationAnalysis();

  LoanSet getLoans(OriginID OID, ProgramPoint P) const;

private:
  class Impl;
  std::unique_ptr<Impl> PImpl;
};

} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H
