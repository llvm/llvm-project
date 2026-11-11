//===--- InferAlloc.h - Allocation type inference ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for allocation-related type inference.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INFERALLOC_H
#define LLVM_CLANG_AST_INFERALLOC_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/AllocToken.h"
#include <optional>

namespace clang {

class Stmt;
class VarDecl;

namespace infer_alloc {

/// Lazily collects, for one function body, the types constructed by placement
/// new into memory returned by allocation calls. This lets the allocated type
/// be inferred for the common "allocate raw storage, then placement-new the
/// object into it" idiom, where neither a sizeof nor a cast names the type:
///
///   void *mem = alloc(size);
///   MyType *obj = new (mem) MyType();  // infers 'MyType' for alloc(size)
///
///   auto res = size_returning_alloc(size);
///   new (res.p) MyType();              // infers 'MyType' (via member .p)
///
///   new (alloc(size)) MyType();        // infers 'MyType'
///
/// The placement argument may be wrapped in parentheses, casts, or member
/// accesses of the call result (or of a local variable initialized with it).
/// If placement news of different types use the same allocation, no type is
/// inferred. Like the other inference heuristics, this ignores control flow
/// and reassignments of the variable.
class PlacementNewTypes {
public:
  explicit PlacementNewTypes(const Stmt *Body) : Body(Body) {}

  /// Returns the type placement-constructed into the result of allocation call
  /// \p E, or a null type if there is none or it is ambiguous.
  QualType lookup(const CallExpr *E);

private:
  void collect();

  const Stmt *Body;
  bool Collected = false;
  // Allocation call -> placement-new type (null if ambiguous).
  llvm::DenseMap<const CallExpr *, QualType> CallTypes;
  // Variable -> placement-new type of its contents (null if ambiguous).
  llvm::DenseMap<const VarDecl *, QualType> VarTypes;
  // Allocation call -> variable it initializes.
  llvm::DenseMap<const CallExpr *, const VarDecl *> CallVars;
};

/// Infer the possible allocated type from an allocation call expression.
/// \p CastE is the cast the call result is immediately used in, if any, and
/// \p PlacementTypes the placement-new types of the enclosing function, if
/// available.
QualType inferPossibleType(const CallExpr *E, const ASTContext &Ctx,
                           const CastExpr *CastE,
                           PlacementNewTypes *PlacementTypes = nullptr);

/// Get the information required for construction of an allocation token ID.
std::optional<llvm::AllocTokenMetadata>
getAllocTokenMetadata(QualType T, const ASTContext &Ctx);

} // namespace infer_alloc
} // namespace clang

#endif // LLVM_CLANG_AST_INFERALLOC_H
