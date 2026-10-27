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
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang::lifetimes::internal {

using LoanID = utils::ID<struct LoanTag>;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, LoanID ID) {
  return OS << ID.Value;
}

/// Represents one step (a projection) in an access path. Projections form the
/// "place ownership tree" of an object:
///
///   t                 the object itself
///   ├── t.f           Field: a sub-object, same lifetime as `t`
///   │   └── t.f.*     the interior of the field
///   ├── t.*           Interior: storage *owned* by `t` (a vector's buffer, a
///   │                 string's characters, a unique_ptr's pointee). Also acts
///   │                 as a wildcard for "some interior place of t".
///   └── t.keys        Place: a named interior region declared with
///                     [[clang::places(keys, ...)]].
///
/// Fields die with the object. Interiors and places can additionally be
/// invalidated while the object is still alive (e.g. `push_back`).
///
/// Examples:
///   - `obj.field` has PathElement 'field'
///   - `std::string s; std::string_view v = s;` gives v a loan to `s.*`
///   - `arr[i]` has PathElement '*' (same as interior access)
///   - `t.Key(0)` with `Key() [[clang::lifetimebound(keys)]]` gives `t.keys`
class PathElement {
public:
  enum class Kind { Field, Interior, Place };

  static PathElement getField(const FieldDecl &FD) {
    return PathElement(Kind::Field, &FD);
  }
  static PathElement getInterior() {
    return PathElement(Kind::Interior, nullptr);
  }
  static PathElement getPlace(const IdentifierInfo &Name) {
    return PathElement(Kind::Place, &Name);
  }

  Kind getKind() const { return K; }
  bool isField() const { return K == Kind::Field; }
  bool isInterior() const { return K == Kind::Interior; }
  bool isPlace() const { return K == Kind::Place; }
  /// True for the anonymous interior and for named places, i.e. for storage
  /// owned by (rather than part of) the object.
  bool isInteriorOrPlace() const { return !isField(); }

  const FieldDecl *getFieldDecl() const {
    return isField() ? static_cast<const FieldDecl *>(Ptr) : nullptr;
  }
  const IdentifierInfo *getPlaceName() const {
    return isPlace() ? static_cast<const IdentifierInfo *>(Ptr) : nullptr;
  }

  bool operator==(const PathElement &Other) const {
    return K == Other.K && Ptr == Other.Ptr;
  }
  bool operator!=(const PathElement &Other) const { return !(*this == Other); }

  /// Returns true if this element, used as part of an invalidated prefix,
  /// covers `Other` as the element at the same position of a borrowed path.
  /// The anonymous interior is a wildcard over named places in both
  /// directions: an invalidation of "any interior" hits every named place, and
  /// a borrow of "some interior" is hit by the invalidation of any place.
  bool covers(const PathElement &Other) const {
    if (*this == Other)
      return true;
    return isInteriorOrPlace() && Other.isInteriorOrPlace() &&
           (isInterior() || Other.isInterior());
  }

  void dump(llvm::raw_ostream &OS) const;

private:
  PathElement(Kind K, const void *Ptr) : K(K), Ptr(Ptr) {}
  Kind K;
  /// FieldDecl for fields, IdentifierInfo for places, null for the interior.
  const void *Ptr;
};

/// Represents the base of a placeholder access path, which is a function
/// parameter, the implicit 'this' object of an instance method, or a field of
/// the enclosing class. Placeholder paths never expire within the function
/// scope, as they represent storage from the caller's or object's scope.
class PlaceholderBase : public llvm::FoldingSetNode {
  llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *,
                     const FieldDecl *>
      ParamMethodOrField;

public:
  PlaceholderBase(const ParmVarDecl *PVD) : ParamMethodOrField(PVD) {}
  PlaceholderBase(const CXXMethodDecl *MD) : ParamMethodOrField(MD) {}
  PlaceholderBase(const FieldDecl *FD) : ParamMethodOrField(FD) {}

  const ParmVarDecl *getParmVarDecl() const {
    return dyn_cast<const ParmVarDecl *>(ParamMethodOrField);
  }

  const CXXMethodDecl *getImplicitThisParent() const {
    return dyn_cast<const CXXMethodDecl *>(ParamMethodOrField);
  }

  const FieldDecl *getFieldDecl() const {
    return dyn_cast<const FieldDecl *>(ParamMethodOrField);
  }

  using KeyTy = llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *,
                                   const FieldDecl *>;

  static void Profile(llvm::FoldingSetNodeID &ID, KeyTy ParamMethodOrField) {
    ID.AddPointer(ParamMethodOrField.getOpaqueValue());
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, ParamMethodOrField);
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
/// A path may be *truncated*: projecting it further would have exceeded the
/// bound that keeps the set of loans finite (see
/// `LoanManager::getOrCreateProjectedLoan`). A truncated path stands for
/// every path extending it, so matching against it is conservative.
///
/// TODO: Model access paths of other types, e.g. heap and globals.
class AccessPath {
  /// The base of the access path: a variable, temporary, or placeholder.
  const llvm::PointerUnion<const clang::ValueDecl *,
                           const clang::MaterializeTemporaryExpr *,
                           const PlaceholderBase *, const clang::CXXNewExpr *>
      Base;
  /// The path elements representing field accesses and access to unnamed
  /// interior regions.
  llvm::SmallVector<PathElement, 1> Elements;
  /// See the class comment.
  bool Truncated = false;

public:
  AccessPath(const clang::ValueDecl *D) : Base(D) {}
  AccessPath(const clang::MaterializeTemporaryExpr *MTE) : Base(MTE) {}
  AccessPath(const PlaceholderBase *PB) : Base(PB) {}
  AccessPath(const clang::CXXNewExpr *New) : Base(New) {}

  /// Creates an extended access path by appending a path element.
  /// Example: AccessPath(x_path, field) creates path to `x.field`.
  AccessPath(const AccessPath &Other, PathElement E)
      : Base(Other.Base), Elements(Other.Elements), Truncated(Other.Truncated) {
    Elements.push_back(E);
  }

  /// Creates a truncated copy of a path.
  static AccessPath getTruncated(const AccessPath &Other) {
    AccessPath Result(Other);
    Result.Truncated = true;
    return Result;
  }

  const clang::ValueDecl *getAsValueDecl() const {
    return dyn_cast<const clang::ValueDecl *>(Base);
  }

  const clang::MaterializeTemporaryExpr *getAsMaterializeTemporaryExpr() const {
    return dyn_cast<const clang::MaterializeTemporaryExpr *>(Base);
  }

  const PlaceholderBase *getAsPlaceholderBase() const {
    return dyn_cast<const PlaceholderBase *>(Base);
  }

  const clang::CXXNewExpr *getAsNewAllocation() const {
    return dyn_cast<const clang::CXXNewExpr *>(Base);
  }

  bool hasSameBase(const AccessPath &RHS) const { return Base == RHS.Base; }

  bool operator==(const AccessPath &RHS) const {
    return Base == RHS.Base && Elements == RHS.Elements &&
           Truncated == RHS.Truncated;
  }
  bool operator!=(const AccessPath &RHS) const { return !(*this == RHS); }

  /// Returns true if this path is a prefix of Other (or same as Other).
  /// Examples:
  ///   - `x` is a prefix of `x`, `x.field`, `x.field.*`
  ///   - `x.field` is a prefix of `x.field` and `x.field.nested`
  ///   - `x.field` is NOT a prefix of `x.other_field`
  bool isPrefixOf(const AccessPath &Other) const {
    if (Base != Other.Base || Elements.size() > Other.Elements.size())
      return false;
    return std::equal(Elements.begin(), Elements.end(), Other.Elements.begin());
  }

  /// Returns true if this path is a strict prefix of Other.
  /// Example:
  ///   - `x` is a strict prefix of `x.field` but NOT of `x`
  bool isStrictPrefixOf(const AccessPath &Other) const {
    return Elements.size() < Other.Elements.size() && isPrefixOf(Other);
  }

  /// Returns true if invalidating this path (the "invalidated prefix") hits the
  /// storage borrowed by `Borrowed`, under `PathElement::covers` wildcard
  /// matching of interiors and places. Also true when `Borrowed` is truncated
  /// and extends into this path, since it then may denote a deeper place.
  ///
  /// Example: `t.keys` covers `t.keys`, `t.keys.x` and `t.*` but not
  /// `t.values` or `t`.
  bool coversPrefixOf(const AccessPath &Borrowed) const {
    if (Base != Borrowed.Base)
      return false;
    size_t N = std::min(Elements.size(), Borrowed.Elements.size());
    for (size_t I = 0; I < N; ++I)
      if (!Elements[I].covers(Borrowed.Elements[I]))
        return false;
    return Elements.size() <= Borrowed.Elements.size() || Borrowed.Truncated;
  }

  /// Returns true if the path has an interior or place element at index
  /// `From` or later, i.e. if the storage it denotes is owned by (and not a
  /// sub-object of) the object denoted by its first `From` elements. A
  /// truncated path may extend into an interior, so it answers true.
  bool hasInteriorFrom(size_t From) const {
    if (Truncated)
      return true;
    for (size_t I = From; I < Elements.size(); ++I)
      if (Elements[I].isInteriorOrPlace())
        return true;
    return false;
  }

  bool isTruncated() const { return Truncated; }
  llvm::ArrayRef<PathElement> getElements() const { return Elements; }

  void dump(llvm::raw_ostream &OS) const;
};

/// Represents a component of an access path: either a named field access or an
/// abstract unnamed interior region (denoted by '*').
///
/// The interior access (`*`) represents the borrowable content of an object
/// without exposing its internal implementation details. It may abstract over
/// multiple underlying fields or memory regions.
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

public:
  LoanManager() = default;

  Loan *createLoan(AccessPath Path, const Expr *IssueExpr = nullptr) {
    void *Mem = LoanAllocator.Allocate<Loan>();
    auto *NewLoan = new (Mem) Loan(getNextLoanID(), Path, IssueExpr);
    AllLoans.push_back(NewLoan);
    return NewLoan;
  }

  Loan *createPlaceholderLoan(const ParmVarDecl *PVD) {
    return createLoan(AccessPath(getOrCreatePlaceholderBase(PVD)));
  }
  Loan *createPlaceholderLoan(const CXXMethodDecl *MD) {
    return createLoan(AccessPath(getOrCreatePlaceholderBase(MD)));
  }
  Loan *createPlaceholderLoan(const FieldDecl *FD) {
    return createLoan(AccessPath(getOrCreatePlaceholderBase(FD)));
  }

  /// Returns the loan for the path of `BaseLoanID` projected by `Element`
  /// (e.g. `x` projected by `.f` is `x.f`), creating it on first use. The
  /// result is interned per (base loan, element) so that repeated projections
  /// during the dataflow fixpoint yield the same loan.
  ///
  /// To keep the number of loans finite in loops over recursive structures
  /// (`p = &p->kids[0]`), a path never gains the same field twice nor more
  /// than `MaxInteriorDepth` interior elements; beyond that the path is marked
  /// truncated instead (see `AccessPath`).
  const Loan *getOrCreateProjectedLoan(LoanID BaseLoanID, PathElement Element);

  /// Returns the loan `ProjectedLoanID` was projected from, if any.
  std::optional<LoanID> getBaseLoan(LoanID ProjectedLoanID) const {
    auto It = BaseLoans.find(ProjectedLoanID);
    if (It == BaseLoans.end())
      return std::nullopt;
    return It->second;
  }

  const Loan *getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }

  llvm::ArrayRef<const Loan *> getLoans() const { return AllLoans; }

private:
  /// Maximum number of interior/place elements in a path before truncation.
  static constexpr unsigned MaxInteriorDepth = 4;

  LoanID getNextLoanID() { return NextLoanID++; }

  /// Gets or creates a placeholder base for a given parameter, method, or
  /// field.
  const PlaceholderBase *getOrCreatePlaceholderBase(const ParmVarDecl *PVD);
  const PlaceholderBase *getOrCreatePlaceholderBase(const CXXMethodDecl *MD);
  const PlaceholderBase *getOrCreatePlaceholderBase(const FieldDecl *FD);

  LoanID NextLoanID{0};

  llvm::FoldingSet<PlaceholderBase> PlaceholderBases;

  /// Interning table for projected loans, see `getOrCreateProjectedLoan`.
  llvm::DenseMap<std::pair<LoanID, PathElement>, const Loan *> ProjectedLoans;
  /// Maps a projected loan back to the loan it was projected from.
  llvm::DenseMap<LoanID, LoanID> BaseLoans;

  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<const Loan *> AllLoans;
  llvm::BumpPtrAllocator LoanAllocator;
};
} // namespace clang::lifetimes::internal

namespace llvm {
template <> struct DenseMapInfo<clang::lifetimes::internal::PathElement> {
  using PathElement = clang::lifetimes::internal::PathElement;
  static unsigned getHashValue(const PathElement &Val) {
    return llvm::hash_combine(static_cast<int>(Val.getKind()),
                              Val.getFieldDecl(), Val.getPlaceName());
  }
  static bool isEqual(const PathElement &LHS, const PathElement &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
