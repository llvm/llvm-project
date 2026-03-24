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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable or a field within it: var.field.*
///
/// An AccessPath consists of a root which is one of:
///   - ValueDecl: a local variable or global
///   - MaterializeTemporaryExpr: a temporary object
///   - ParmVarDecl: a function parameter (placeholder)
///   - CXXMethodDecl: the implicit 'this' object (placeholder)
///
/// Placeholder paths never expire within the function scope, as they represent
/// storage from the caller's scope.
///
/// TODO: Model access paths of other types, e.g. field, array subscript, heap
/// and globals.
class AccessPath {
public:
  enum class Kind : uint8_t {
    ValueDecl,
    MaterializeTemporary,
    PlaceholderParam,
    PlaceholderThis
  };

private:
  Kind K;
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *,
                           const ParmVarDecl *, const CXXMethodDecl *>
      Root;

public:
  AccessPath(const clang::ValueDecl *D) : K(Kind::ValueDecl), Root(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE)
      : K(Kind::MaterializeTemporary), Root(MTE) {}
  static AccessPath Placeholder(const ParmVarDecl *PVD) {
    return AccessPath(Kind::PlaceholderParam, PVD);
  }
  static AccessPath Placeholder(const CXXMethodDecl *MD) {
    return AccessPath(Kind::PlaceholderThis, MD);
  }
  AccessPath(const AccessPath &Other) : K(Other.K), Root(Other.Root) {}

  Kind getKind() const { return K; }

  const clang::ValueDecl *getAsValueDecl() const {
    return K == Kind::ValueDecl ? Root.dyn_cast<const clang::ValueDecl *>()
                                : nullptr;
  }
  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return K == Kind::MaterializeTemporary
               ? Root.dyn_cast<const clang::MaterializeTemporaryExpr *>()
               : nullptr;
  }
  const ParmVarDecl *getAsPlaceholderParam() const {
    return K == Kind::PlaceholderParam ? Root.dyn_cast<const ParmVarDecl *>()
                                       : nullptr;
  }
  const CXXMethodDecl *getAsPlaceholderThis() const {
    return K == Kind::PlaceholderThis ? Root.dyn_cast<const CXXMethodDecl *>()
                                      : nullptr;
  }

  bool operator==(const AccessPath &RHS) const {
    return K == RHS.K && Root == RHS.Root;
  }
  bool operator!=(const AccessPath &RHS) const { return !(*this == RHS); }
  void dump(llvm::raw_ostream &OS) const;

private:
  AccessPath(Kind K, const ParmVarDecl *PVD) : K(K), Root(PVD) {}
  AccessPath(Kind K, const CXXMethodDecl *MD) : K(K), Root(MD) {}
};

/// Represents lending a storage location.
///
/// A loan tracks the borrowing relationship created by operations like
/// taking a pointer/reference (&x), creating a view (std::string_view sv = s),
/// or receiving a parameter.
///
/// Examples:
///   - `int* p = &x;` creates a loan to `x`
///   - Parameter loans have no IssueExpr (created at function entry)
class Loan {
  const LoanID ID;
  const AccessPath Path;
  /// The expression that creates the loan, e.g., &x. Null for placeholder
  /// loans.
  const Expr *IssuingExpr;

public:
  Loan(LoanID ID, AccessPath Path, const Expr *IssuingExpr)
      : ID(ID), Path(Path), IssuingExpr(IssuingExpr) {}
  LoanID getID() const { return ID; }
  const AccessPath &getAccessPath() const { return Path; }
  const Expr *getIssuingExpr() const { return IssuingExpr; }
  void dump(llvm::raw_ostream &OS) const;
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
public:
  LoanManager() = default;

  Loan *createLoan(AccessPath Path, const Expr *IssueExpr) {
    void *Mem = LoanAllocator.Allocate<Loan>();
    auto *NewLoan = new (Mem) Loan(getNextLoanID(), Path, IssueExpr);
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
