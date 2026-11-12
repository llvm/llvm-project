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

namespace infer_alloc {

/// Infer the possible allocated type from an allocation call expression.
/// \p CastE is the cast the call result is immediately used in, if any.
QualType inferPossibleType(const CallExpr *E, const ASTContext &Ctx,
                           const CastExpr *CastE);

/// Maps the allocation calls in \p Body whose result is used as placement-new
/// storage, directly or via a local variable initialized with it, to the type
/// constructed (null if different types are constructed into the same
/// storage). This covers the "allocate raw storage, then construct in place"
/// idiom; the constructed type is more precise than a sizeof or a cast, which
/// may only describe the storage (a char buffer, a base class):
///
///   void *mem = alloc(size);
///   MyType *obj = new (mem) MyType();  // infers 'MyType' for alloc(size)
///
///   auto res = size_returning_alloc(size);
///   new (res.p) MyType();              // infers 'MyType' (via member .p)
///
/// Like the other inference heuristics, this ignores control flow and
/// reassignments of the variable.
llvm::DenseMap<const CallExpr *, QualType>
collectPlacementNewTypes(const Stmt *Body);

/// Get the information required for construction of an allocation token ID.
std::optional<llvm::AllocTokenMetadata>
getAllocTokenMetadata(QualType T, const ASTContext &Ctx);

} // namespace infer_alloc
} // namespace clang

#endif // LLVM_CLANG_AST_INFERALLOC_H
