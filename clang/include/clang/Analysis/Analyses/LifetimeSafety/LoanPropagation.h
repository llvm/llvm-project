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

#include "clang/Analysis/Analyses/LifetimeSafety/Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/Debug.h"

namespace clang::lifetimes::internal {

// Using LLVM's immutable collections is efficient for dataflow analysis
// as it avoids deep copies during state transitions.
// TODO(opt): Consider using a bitset to represent the set of loans.
using LoanSet = llvm::ImmutableSet<LoanID>;
using OriginLoanMap = llvm::ImmutableMap<OriginID, LoanSet>;

/// Represents the dataflow lattice for loan propagation.
///
/// This lattice tracks which loans each origin may hold at a given program
/// point.The lattice has a finite height: An origin's loan set is bounded by
/// the total number of loans in the function.
/// TODO(opt): To reduce the lattice size, propagate origins of declarations,
/// not expressions, because expressions are not visible across blocks.
struct LoanPropagationLattice {
  /// The map from an origin to the set of loans it contains.
  OriginLoanMap Origins = OriginLoanMap(nullptr);

  explicit LoanPropagationLattice(const OriginLoanMap &S) : Origins(S) {}
  LoanPropagationLattice() = default;

  bool operator==(const LoanPropagationLattice &Other) const {
    return Origins == Other.Origins;
  }
  bool operator!=(const LoanPropagationLattice &Other) const {
    return !(*this == Other);
  }

  void dump(llvm::raw_ostream &OS) const;
};

class LoanPropagationAnalysis
    : public DataflowAnalysis<LoanPropagationAnalysis, LoanPropagationLattice,
                              Direction::Forward> {
  OriginLoanMap::Factory &OriginLoanMapFactory;
  LoanSet::Factory &LoanSetFactory;

public:
  LoanPropagationAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                          OriginLoanMap::Factory &OriginLoanMapFactory,
                          LoanSet::Factory &LoanSetFactory)
      : DataflowAnalysis(C, AC, F), OriginLoanMapFactory(OriginLoanMapFactory),
        LoanSetFactory(LoanSetFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  Lattice join(Lattice A, Lattice B);

  Lattice transfer(Lattice In, const IssueFact &F);
  Lattice transfer(Lattice In, const OriginFlowFact &F);

  LoanSet getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) const {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H
