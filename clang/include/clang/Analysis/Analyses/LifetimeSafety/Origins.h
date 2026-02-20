//===- Origins.h - Origin and Origin Management ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Origins, which represent the set of possible loans a
// pointer-like object could hold, and the OriginManager, which manages the
// creation, storage, and retrieval of origins for variables and expressions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using OriginID = utils::ID<struct OriginTag>;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, OriginID ID) {
  return OS << ID.Value;
}

/// An Origin is a symbolic identifier that represents the set of possible
/// loans a pointer-like object could hold at any given time.
///
/// Each Origin corresponds to a single level of indirection. For complex types
/// with multiple levels of indirection (e.g., `int**`), multiple Origins are
/// organized into an OriginList structure (see below).
struct Origin {
  OriginID ID;
  /// A pointer to the AST node that this origin represents. This union
  /// distinguishes between origins from declarations (variables or parameters)
  /// and origins from expressions.
  llvm::PointerUnion<const clang::ValueDecl *, const clang::Expr *> Ptr;

  /// The type at this indirection level.
  ///
  /// For `int** pp`:
  ///   Root origin: QT = `int**` (what pp points to)
  ///   Pointee origin: QT = `int*` (what *pp points to)
  ///
  /// Null for synthetic lvalue origins (e.g., outer origin of DeclRefExpr).
  const Type *Ty;

  Origin(OriginID ID, const clang::ValueDecl *D, const Type *QT)
      : ID(ID), Ptr(D), Ty(QT) {}
  Origin(OriginID ID, const clang::Expr *E, const Type *QT)
      : ID(ID), Ptr(E), Ty(QT) {}

  const clang::ValueDecl *getDecl() const {
    return Ptr.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::Expr *getExpr() const {
    return Ptr.dyn_cast<const clang::Expr *>();
  }
};

/// A list of origins representing levels of indirection for pointer-like types.
///
/// Each node in the list contains an OriginID representing a level of
/// indirection. The list structure captures the multi-level nature of
/// pointer and reference types in the lifetime analysis.
///
/// Examples:
///   - For `int& x`, the list has size 2:
///     * Outer: origin for the reference storage itself (the lvalue `x`)
///     * Inner: origin for what `x` refers to
///
///   - For `int* p`, the list has size 2:
///     * Outer: origin for the pointer variable `p`
///     * Inner: origin for what `p` points to
///
///   - For `View v` (where View is gsl::Pointer), the list has size 2:
///     * Outer: origin for the view object itself
///     * Inner: origin for what the view refers to
///
///   - For `int** pp`, the list has size 3:
///     * Outer: origin for `pp` itself
///     * Inner: origin for `*pp` (what `pp` points to)
///     * Inner->Inner: origin for `**pp` (what `*pp` points to)
///
/// The list structure enables the analysis to track how loans flow through
/// different levels of indirection when assignments and dereferences occur.
class OriginList {
public:
  OriginList(OriginID OID) : OuterOID(OID) {}

  OriginList *peelOuterOrigin() const { return InnerList; }
  OriginID getOuterOriginID() const { return OuterOID; }

  void setInnerOriginList(OriginList *Inner) { InnerList = Inner; }

  // Used for assertion checks only (to ensure origin lists have matching
  // lengths).
  size_t getLength() const {
    size_t Length = 1;
    const OriginList *T = this;
    while (T->InnerList) {
      T = T->InnerList;
      Length++;
    }
    return Length;
  }

private:
  OriginID OuterOID;
  OriginList *InnerList = nullptr;
};

bool hasOrigins(QualType QT);
bool hasOrigins(const Expr *E);
bool doesDeclHaveStorage(const ValueDecl *D);

/// Manages the creation, storage, and retrieval of origins for pointer-like
/// variables and expressions.
class OriginManager {
public:
  explicit OriginManager(ASTContext &AST, const Decl *D);

  /// Gets or creates the OriginList for a given ValueDecl.
  ///
  /// Creates a list structure mirroring the levels of indirection in the
  /// declaration's type (e.g., `int** p` creates list of size 2).
  ///
  /// \returns The OriginList, or nullptr if the type is not pointer-like.
  OriginList *getOrCreateList(const ValueDecl *D);

  /// Gets or creates the OriginList for a given Expr.
  ///
  /// Creates a list based on the expression's type and value category:
  /// - Lvalues get an implicit reference level (modeling addressability)
  /// - Rvalues of non-pointer type return nullptr (no trackable origin)
  /// - DeclRefExpr may reuse the underlying declaration's list
  ///
  /// \returns The OriginList, or nullptr for non-pointer rvalues.
  OriginList *getOrCreateList(const Expr *E);

  /// Returns the OriginList for the implicit 'this' parameter if the current
  /// declaration is an instance method.
  std::optional<OriginList *> getThisOrigins() const { return ThisOrigins; }

  const Origin &getOrigin(OriginID ID) const;

  llvm::ArrayRef<Origin> getOrigins() const { return AllOrigins; }

  unsigned getNumOrigins() const { return NextOriginID.Value; }

  void dump(OriginID OID, llvm::raw_ostream &OS) const;

  /// Collects statistics about expressions that lack associated origins.
  void collectMissingOrigins(Stmt &FunctionBody, LifetimeSafetyStats &LSStats);

private:
  OriginID getNextOriginID() { return NextOriginID++; }

  OriginList *createNode(const ValueDecl *D, QualType QT);
  OriginList *createNode(const Expr *E, QualType QT);

  template <typename T>
  OriginList *buildListForType(QualType QT, const T *Node);

  ASTContext &AST;
  OriginID NextOriginID{0};
  /// TODO(opt): Profile and evaluate the usefulness of small buffer
  /// optimisation.
  llvm::SmallVector<Origin> AllOrigins;
  llvm::BumpPtrAllocator ListAllocator;
  llvm::DenseMap<const clang::ValueDecl *, OriginList *> DeclToList;
  llvm::DenseMap<const clang::Expr *, OriginList *> ExprToList;
  std::optional<OriginList *> ThisOrigins;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
