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
/// variable.
/// TODO: Model access paths of other types, e.g., s.field, heap and globals.
struct AccessPath {
private:
  // An access path can be:
  // - ValueDecl * , to represent the storage location corresponding to the
  //   variable declared in ValueDecl.
  // - MaterializeTemporaryExpr * , to represent the storage location of the
  //   temporary object materialized via this MaterializeTemporaryExpr.
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *>
      P;

public:
  AccessPath(const clang::ValueDecl *D) : P(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE) : P(MTE) {}

  const clang::ValueDecl *getAsValueDecl() const {
    return P.dyn_cast<const clang::ValueDecl *>();
  }

  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return P.dyn_cast<const clang::MaterializeTemporaryExpr *>();
  }
};

/// An abstract base class for a single "Loan" which represents lending a
/// storage in memory.
class Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
public:
  enum class Kind : uint8_t {
    /// A loan with an access path to a storage location.
    Path,
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

/// PathLoan represents lending a storage location that is visible within the
/// function's scope (e.g., a local variable on stack).
class PathLoan : public Loan {
  AccessPath Path;
  /// The expression that creates the loan, e.g., &x.
  const Expr *IssueExpr;

public:
  PathLoan(LoanID ID, AccessPath Path, const Expr *IssueExpr)
      : Loan(Kind::Path, ID), Path(Path), IssueExpr(IssueExpr) {}

  const AccessPath &getAccessPath() const { return Path; }
  const Expr *getIssueExpr() const { return IssueExpr; }

  void dump(llvm::raw_ostream &OS) const override;

  static bool classof(const Loan *L) { return L->getKind() == Kind::Path; }
};

/// A placeholder loan held by a function parameter or an implicit 'this'
/// object, representing a borrow from the caller's scope.
///
/// Created at function entry for each pointer or reference parameter or for
/// the implicit 'this' parameter of instance methods, with an
/// origin. Unlike PathLoan, placeholder loans:
/// - Have no IssueExpr (created at function entry, not at a borrow site)
/// - Have no AccessPath (the borrowed object is not visible to the function)
/// - Do not currently expire, but may in the future when modeling function
///   invalidations (e.g., vector::push_back)
///
/// When a placeholder loan escapes the function (e.g., via return), it
/// indicates the parameter or method should be marked [[clang::lifetimebound]],
/// enabling lifetime annotation suggestions.
class PlaceholderLoan : public Loan {
  /// The function parameter or method (representing 'this') that holds this
  /// placeholder loan.
  llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *> ParamOrMethod;

public:
  PlaceholderLoan(LoanID ID, const ParmVarDecl *PVD)
      : Loan(Kind::Placeholder, ID), ParamOrMethod(PVD) {}

  PlaceholderLoan(LoanID ID, const CXXMethodDecl *MD)
      : Loan(Kind::Placeholder, ID), ParamOrMethod(MD) {}

  const ParmVarDecl *getParmVarDecl() const {
    return ParamOrMethod.dyn_cast<const ParmVarDecl *>();
  }

  const CXXMethodDecl *getMethodDecl() const {
    return ParamOrMethod.dyn_cast<const CXXMethodDecl *>();
  }

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
    static_assert(
        std::is_same_v<LoanType, PathLoan> ||
            std::is_same_v<LoanType, PlaceholderLoan>,
        "createLoan can only be used with PathLoan or PlaceholderLoan");
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
