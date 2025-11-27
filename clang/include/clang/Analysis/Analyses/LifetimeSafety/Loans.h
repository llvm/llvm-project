//===- Loans.h - Loan and Access Path Definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Loan and AccessPath structures, which represent
// borrows of storage locations, and the LoanManager, which manages the
// creation and retrieval of loans during lifetime analysis.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable.
/// TODO: Model access paths of other types, e.g., s.field, heap and globals.
struct AccessPath {
  const clang::ValueDecl *D;

  AccessPath(const clang::ValueDecl *D) : D(D) {}
};

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is created.
struct Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
  LoanID ID;
  AccessPath Path;
  /// The expression that creates the loan, e.g., &x.
  const Expr *IssueExpr;

  Loan(LoanID id, AccessPath path, const Expr *IssueExpr)
      : ID(id), Path(path), IssueExpr(IssueExpr) {}

  void dump(llvm::raw_ostream &OS) const;
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
public:
  LoanManager() = default;

  Loan &addLoan(AccessPath Path, const Expr *IssueExpr) {
    AllLoans.emplace_back(getNextLoanID(), Path, IssueExpr);
    return AllLoans.back();
  }

  const Loan &getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<Loan> getLoans() const { return AllLoans; }

  void addPlaceholderLoan(LoanID LID, const ParmVarDecl *PVD) {
    PlaceholderLoans[LID] = PVD;
  }

  const llvm::DenseMap<LoanID, const ParmVarDecl *> &
  getPlaceholderLoans() const {
    return PlaceholderLoans;
  }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<Loan> AllLoans;
  /// Represents a map of placeholder LoanID to the function parameter.
  /// Placeholder loans are dummy loans created for each pointer or reference
  /// parameter to represent a borrow from the function's caller, which the
  /// analysis tracks to see if it unsafely escapes the function's scope.
  llvm::DenseMap<LoanID, const ParmVarDecl *> PlaceholderLoans;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
