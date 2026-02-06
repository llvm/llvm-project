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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents one step in an access path: either a field access or an
/// access to an unnamed interior region (denoted by '*').
///
/// Examples:
///   - Field access: `obj.field` has PathElement 'field'
///   - Interior access: `owner.*` has '*'
//    - In `std::string s; std::string_view v = s;`, v has loan to s.*
class PathElement {
public:
  enum class Kind { Field, Interior };

  static PathElement getField(const FieldDecl *FD) {
    return PathElement(Kind::Field, FD);
  }
  static PathElement getInterior() {
    return PathElement(Kind::Interior, nullptr);
  }

  bool isField() const { return K == Kind::Field; }
  bool isInterior() const { return K == Kind::Interior; }
  const FieldDecl *getFieldDecl() const { return FD; }

  bool operator==(const PathElement &Other) const {
    return K == Other.K && FD == Other.FD;
  }
  bool operator!=(const PathElement &Other) const { return !(*this == Other); }

  void dump(llvm::raw_ostream &OS) const {
    if (isField())
      OS << "." << FD->getNameAsString();
    else
      OS << ".*";
  }

private:
  PathElement(Kind K, const FieldDecl *FD) : K(K), FD(FD) {}
  Kind K;
  const FieldDecl *FD;
};

/// Represents the base of a placeholder access path, which is either a
/// function parameter or the implicit 'this' object of an instance method.
/// Placeholder paths never expire within the function scope, as they represent
/// storage from the caller's scope.
class PlaceholderBase : public llvm::FoldingSetNode {
  llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *> ParamOrMethod;

public:
  PlaceholderBase(const ParmVarDecl *PVD) : ParamOrMethod(PVD) {}
  PlaceholderBase(const CXXMethodDecl *MD) : ParamOrMethod(MD) {}

  const ParmVarDecl *getParmVarDecl() const {
    return ParamOrMethod.dyn_cast<const ParmVarDecl *>();
  }

  const CXXMethodDecl *getMethodDecl() const {
    return ParamOrMethod.dyn_cast<const CXXMethodDecl *>();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(ParamOrMethod.getOpaqueValue());
  }
};

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable or a field within it: var.field.*
///
/// An AccessPath consists of:
///   - A base: either a ValueDecl, MaterializeTemporaryExpr, or PlaceholderBase
///   - A sequence of PathElements representing field accesses or interior
///   regions
///
/// Examples:
///   - `x` -> Base=x, Elements=[]
///   - `x.field` -> Base=x, Elements=[.field]
///   - `x.*` (e.g., string_view from string) -> Base=x, Elements=[.*]
///   - `x.field.*` -> Base=x, Elements=[.field, .*]
///   - `$param.field` -> Base=$param, Elements=[.field]
///
/// TODO: Model access paths of other types, e.g. heap and globals.
class AccessPath {
  /// The base of the access path: a variable, temporary, or placeholder.
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *,
                           const PlaceholderBase *>
      Base;
  /// The path elements representing field accesses and access to unnamed
  /// interior regions.
  llvm::SmallVector<PathElement, 1> Elements;

public:
  AccessPath(const clang::ValueDecl *D) : Base(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE) : Base(MTE) {}
  AccessPath(const PlaceholderBase *PB) : Base(PB) {}

  /// Creates an extended access path by appending a path element.
  /// Example: AccessPath(x_path, field) creates path to `x.field`.
  AccessPath(const AccessPath &Other, PathElement E)
      : Base(Other.Base), Elements(Other.Elements) {
    Elements.push_back(E);
  }

  const clang::ValueDecl *getAsValueDecl() const {
    return Base.dyn_cast<const clang::ValueDecl *>();
  }

  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return Base.dyn_cast<const clang::MaterializeTemporaryExpr *>();
  }

  const PlaceholderBase *getAsPlaceholderBase() const {
    return Base.dyn_cast<const PlaceholderBase *>();
  }

  bool operator==(const AccessPath &RHS) const {
    return Base == RHS.Base && Elements == RHS.Elements;
  }

  /// Returns true if this path is a prefix of Other (or same as Other).
  /// Examples:
  ///   - `x` is a prefix of `x`, `x.field`, `x.field.*`
  ///   - `x.field` is a prefix of `x.field` and `x.field.nested`
  ///   - `x.field` is NOT a prefix of `x.other_field`
  bool isPrefixOf(const AccessPath &Other) const {
    if (Base != Other.Base || Elements.size() > Other.Elements.size())
      return false;
    for (size_t i = 0; i < Elements.size(); ++i)
      if (Elements[i] != Other.Elements[i])
        return false;
    return true;
  }

  /// Returns true if this path is a strict prefix of Other.
  /// Example:
  ///   - `x` is a strict prefix of `x.field` but NOT of `x`
  bool isStrictPrefixOf(const AccessPath &Other) const {
    return isPrefixOf(Other) && Elements.size() < Other.Elements.size();
  }
  llvm::ArrayRef<PathElement> getElements() const { return Elements; }

  void dump(llvm::raw_ostream &OS) const;
};

/// Represents lending a storage location.
//
/// A loan tracks the borrowing relationship created by operations like
/// taking a pointer/reference (&x), creating a view (std::string_view sv = s),
/// or receiving a parameter.
///
/// Examples:
///   - `int* p = &x;` creates a loan to `x`
///   - `std::string_view v = s;` creates a loan to `s.*` (interior)
///   - `int* p = &obj.field;` creates a loan to `obj.field`
///   - Parameter loans have no IssueExpr (created at function entry)
class Loan {
  const LoanID ID;
  const AccessPath Path;
  /// The expression that creates the loan, e.g., &x. Optional for placeholder
  /// loans.
  const Expr *IssueExpr;

public:
  Loan(LoanID ID, AccessPath Path, const Expr *IssueExpr = nullptr)
      : ID(ID), Path(Path), IssueExpr(IssueExpr) {}

  LoanID getID() const { return ID; }
  const AccessPath &getAccessPath() const { return Path; }
  const Expr *getIssueExpr() const { return IssueExpr; }

  void dump(llvm::raw_ostream &OS) const;
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
  using ExtensionCacheKey = std::pair<LoanID, PathElement>;

public:
  LoanManager() = default;

  Loan *createLoan(AccessPath Path, const Expr *IssueExpr = nullptr) {
    void *Mem = LoanAllocator.Allocate<Loan>();
    auto *NewLoan = new (Mem) Loan(getNextLoanID(), Path, IssueExpr);
    AllLoans.push_back(NewLoan);
    return NewLoan;
  }

  /// Gets or creates a placeholder base for a given parameter or method.
  const PlaceholderBase *getOrCreatePlaceholderBase(const ParmVarDecl *PVD);
  const PlaceholderBase *getOrCreatePlaceholderBase(const CXXMethodDecl *MD);

  /// Gets or creates a loan by extending BaseLoanID with Element.
  /// Caches the result to ensure convergence in LoanPropagation.
  Loan *getOrCreateExtendedLoan(LoanID BaseLoanID, PathElement Element);

  const Loan *getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<const Loan *> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};

  llvm::FoldingSet<PlaceholderBase> PlaceholderBases;
  /// Cache for extended loans. Maps (BaseLoanID, PathElement) to the extended
  /// loan. Ensures that extending the same loan with the same path element
  /// always returns the same loan object, which is necessary for dataflow
  /// analysis convergence.
  llvm::DenseMap<ExtensionCacheKey, Loan *> ExtensionCache;

  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<const Loan *> AllLoans;
  llvm::BumpPtrAllocator LoanAllocator;
};
} // namespace clang::lifetimes::internal

namespace llvm {
template <> struct DenseMapInfo<clang::lifetimes::internal::PathElement> {
  using PathElement = clang::lifetimes::internal::PathElement;
  static inline PathElement getEmptyKey() {
    return PathElement::getField(
        llvm::DenseMapInfo<const clang::FieldDecl *>::getEmptyKey());
  }
  static inline PathElement getTombstoneKey() {
    return PathElement::getField(
        llvm::DenseMapInfo<const clang::FieldDecl *>::getTombstoneKey());
  }
  static unsigned getHashValue(const PathElement &Val) {
    return llvm::hash_combine(Val.isInterior(), Val.getFieldDecl());
  }
  static bool isEqual(const PathElement &LHS, const PathElement &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
