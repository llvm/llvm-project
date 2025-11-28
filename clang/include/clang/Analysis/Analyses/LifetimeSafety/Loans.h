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

/// An abstract base class for a single borrow, or "Loan".
class Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
public:
  enum class Kind : uint8_t {
    /// A regular borrow of a variable within the function that has a path and
    /// can expire.
    Borrow,
    /// A non-expiring placeholder loan for a parameter, representing a borrow
    /// from the function's caller.
    Placeholder
  };

  Loan(Kind K, LoanID ID) : K(K), ID(ID) {}
  virtual ~Loan() = default;

  Kind getKind() const { return K; }
  LoanID getID() const { return ID; }

  virtual void dump(llvm::raw_ostream &OS) const = 0;

private:
  const Kind K;
  const LoanID ID;
};

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is created.
class BorrowLoan : public Loan {
  AccessPath Path;
  const Expr *IssueExpr;

public:
  BorrowLoan(LoanID ID, AccessPath Path, const Expr *IssueExpr)
      : Loan(Kind::Borrow, ID), Path(Path), IssueExpr(IssueExpr) {}

  const AccessPath &getAccessPath() const { return Path; }
  const Expr *getIssueExpr() const { return IssueExpr; }

  void dump(llvm::raw_ostream &OS) const override;

  static bool classof(const Loan *L) { return L->getKind() == Kind::Borrow; }
};

/// A concrete loan type for placeholder loans on parameters, representing a
/// borrow from the function's caller.
class ParameterLoan : public Loan {
  const ParmVarDecl *PVD;

public:
  ParameterLoan(LoanID ID, const ParmVarDecl *PVD)
      : Loan(Kind::Placeholder, ID), PVD(PVD) {}

  const ParmVarDecl *getParmVarDecl() const { return PVD; }

  void dump(llvm::raw_ostream &OS) const override;

  static bool classof(const Loan *L) {
    return L->getKind() == Kind::Placeholder;
  }
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
public:
  LoanManager() = default;

  template <typename LoanType, typename... Args>
  LoanType *createLoan(Args &&...args) {
    void *Mem = LoanAllocator.Allocate<LoanType>();
    auto *NewLoan =
        new (Mem) LoanType(getNextLoanID(), std::forward<Args>(args)...);
    AllLoans.push_back(NewLoan);
    return NewLoan;
  }

  const Loan *getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<const Loan *> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<const Loan *> AllLoans;
  llvm::BumpPtrAllocator LoanAllocator;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
