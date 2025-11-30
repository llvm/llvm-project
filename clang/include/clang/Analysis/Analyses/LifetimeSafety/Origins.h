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
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"

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
/// organized into an OriginTree structure (see below).
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
  QualType QT;

  Origin(OriginID ID, const clang::ValueDecl *D, QualType QT)
      : ID(ID), Ptr(D), QT(QT) {}
  Origin(OriginID ID, const clang::Expr *E, QualType QT)
      : ID(ID), Ptr(E), QT(QT) {}

  const clang::ValueDecl *getDecl() const {
    return Ptr.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::Expr *getExpr() const {
    return Ptr.dyn_cast<const clang::Expr *>();
  }
};

/// A tree of origins representing levels of indirection for pointer-like types.
///
/// Each node in the tree contains an OriginID representing a level of
/// indirection. The tree structure captures the multi-level nature of
/// pointer and reference types in the lifetime analysis.
///
/// Examples:
///   - For `int& x`, the tree has depth 2:
///     * Root: origin for the reference storage itself (the lvalue `x`)
///     * Pointee: origin for what `x` refers to
///
///   - For `int* p`, the tree has depth 2:
///     * Root: origin for the pointer variable `p`
///     * Pointee: origin for what `p` points to
///
///   - For `View v` (where View is gsl::Pointer), the tree has depth 2:
///     * Root: origin for the view object itself
///     * Pointee: origin for what the view refers to
///
///   - For `int** pp`, the tree has depth 3:
///     * Root: origin for `pp` itself
///     * Pointee: origin for `*pp` (what `pp` points to)
///     * Pointee->Pointee: origin for `**pp` (what `*pp` points to)
///
/// The tree structure enables the analysis to track how loans flow through
/// different levels of indirection when assignments and dereferences occur.
struct OriginTree {
  OriginID OID;
  OriginTree *Pointee = nullptr;

  OriginTree(OriginID OID) : OID(OID) {}

  size_t getDepth() const {
    size_t Depth = 1;
    const OriginTree *T = this;
    while (T->Pointee) {
      T = T->Pointee;
      Depth++;
    }
    return Depth;
  }
};

bool hasOrigins(QualType QT);
bool hasOrigins(const Expr *E);
bool doesDeclHaveStorage(const ValueDecl *D);

/// Manages the creation, storage, and retrieval of origins for pointer-like
/// variables and expressions.
class OriginManager {
public:
  explicit OriginManager(ASTContext &AST) : AST(AST) {}

  /// Gets or creates the OriginTree for a given ValueDecl.
  ///
  /// Creates a tree structure mirroring the levels of indirection in the
  /// declaration's type (e.g., `int** p` creates depth 2).
  ///
  /// \returns The OriginTree, or nullptr if the type is not pointer-like.
  OriginTree *getOrCreateTree(const ValueDecl *D);

  /// Gets or creates the OriginTree for a given Expr.
  ///
  /// Creates a tree based on the expression's type and value category:
  /// - Lvalues get an implicit reference level (modeling addressability)
  /// - Rvalues of non-pointer type return nullptr (no trackable origin)
  /// - DeclRefExpr may reuse the underlying declaration's tree
  ///
  /// \returns The OriginTree, or nullptr for non-pointer rvalues.
  OriginTree *getOrCreateTree(const Expr *E, size_t Depth = 0);

  const Origin &getOrigin(OriginID ID) const;

  llvm::ArrayRef<Origin> getOrigins() const { return AllOrigins; }

  unsigned getNumOrigins() const { return NextOriginID.Value; }

  void dump(OriginID OID, llvm::raw_ostream &OS) const;

private:
  OriginID getNextOriginID() { return NextOriginID++; }

  OriginTree *createNode(const ValueDecl *D, QualType QT) {
    OriginID NewID = getNextOriginID();
    AllOrigins.emplace_back(NewID, D, QT);
    return new (TreeAllocator.Allocate<OriginTree>()) OriginTree(NewID);
  }

  OriginTree *createNode(const Expr *E, QualType QT) {
    OriginID NewID = getNextOriginID();
    AllOrigins.emplace_back(NewID, E, QT);
    return new (TreeAllocator.Allocate<OriginTree>()) OriginTree(NewID);
  }

  template <typename T>
  OriginTree *buildTreeForType(QualType QT, const T *Node);

  ASTContext &AST;
  OriginID NextOriginID{0};
  /// TODO(opt): Profile and evaluate the usefulness of small buffer
  /// optimisation.
  llvm::SmallVector<Origin> AllOrigins;
  llvm::BumpPtrAllocator TreeAllocator;
  llvm::DenseMap<const clang::ValueDecl *, OriginTree *> DeclToTree;
  llvm::DenseMap<const clang::Expr *, OriginTree *> ExprToTree;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
