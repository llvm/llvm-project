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
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"

namespace clang::lifetimes::internal {

using OriginID = utils::ID<struct OriginTag>;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, OriginID ID) {
  return OS << ID.Value;
}

/// An Origin is a symbolic identifier that represents the set of possible
/// loans a pointer-like object could hold at any given time.
/// TODO: Enhance the origin model to handle complex types, pointer
/// indirection and reborrowing. The plan is to move from a single origin per
/// variable/expression to a "list of origins" governed by the Type.
/// For example, the type 'int**' would have two origins.
/// See discussion:
/// https://github.com/llvm/llvm-project/pull/142313/commits/0cd187b01e61b200d92ca0b640789c1586075142#r2137644238
struct Origin {
  OriginID ID;
  /// A pointer to the AST node that this origin represents. This union
  /// distinguishes between origins from declarations (variables or parameters)
  /// and origins from expressions.
  llvm::PointerUnion<const clang::ValueDecl *, const clang::Expr *> Ptr;

  Origin(OriginID ID, const clang::ValueDecl *D) : ID(ID), Ptr(D) {}
  Origin(OriginID ID, const clang::Expr *E) : ID(ID), Ptr(E) {}

  const clang::ValueDecl *getDecl() const {
    return Ptr.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::Expr *getExpr() const {
    return Ptr.dyn_cast<const clang::Expr *>();
  }
};

/// Manages the creation, storage, and retrieval of origins for pointer-like
/// variables and expressions.
class OriginManager {
public:
  OriginManager() = default;

  Origin &addOrigin(OriginID ID, const clang::ValueDecl &D);
  Origin &addOrigin(OriginID ID, const clang::Expr &E);

  // TODO: Mark this method as const once we remove the call to getOrCreate.
  OriginID get(const Expr &E);

  OriginID get(const ValueDecl &D);

  OriginID getOrCreate(const Expr &E);

  const Origin &getOrigin(OriginID ID) const;

  llvm::ArrayRef<Origin> getOrigins() const { return AllOrigins; }

  OriginID getOrCreate(const ValueDecl &D);

  void dump(OriginID OID, llvm::raw_ostream &OS) const;

private:
  OriginID getNextOriginID() { return NextOriginID++; }

  OriginID NextOriginID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<Origin> AllOrigins;
  llvm::DenseMap<const clang::ValueDecl *, OriginID> DeclToOriginID;
  llvm::DenseMap<const clang::Expr *, OriginID> ExprToOriginID;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
