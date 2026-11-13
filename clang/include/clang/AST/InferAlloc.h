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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/AllocToken.h"
#include <optional>

namespace clang {
namespace infer_alloc {

/// Infer the possible allocated type from an allocation call expression, in
/// this order: a sizeof in a size argument, a placement new into the result
/// (\p PlacementNewType, if given, maps the call to the type constructed into
/// its result, see collectPlacementNewTypes()), and finally a cast of the
/// result (\p CastE is the cast the call result is immediately used in, if
/// any).
QualType inferPossibleType(
    const CallExpr *E, const ASTContext &Ctx, const CastExpr *CastE,
    llvm::function_ref<QualType(const CallExpr *)> PlacementNewType = nullptr);

/// Maps the allocation calls in \p Body whose result is used as placement-new
/// storage, directly or via a local variable holding it, to the type
/// constructed (null if different types are constructed into the same
/// storage). This covers the "allocate raw storage, then construct in place"
/// idiom:
///
///   void *mem = alloc(size);
///   MyType *obj = new (mem) MyType();  // infers 'MyType' for alloc(size)
///
///   auto res = size_returning_alloc(size);
///   new (res.p) MyType();              // infers 'MyType' (via member .p)
///
/// A variable is only used if the function defines it exactly once and never
/// writes to it otherwise, so that its value is known regardless of control
/// flow; reassigned variables are not used for inference.
llvm::DenseMap<const CallExpr *, QualType>
collectPlacementNewTypes(const Stmt *Body);

/// Get the information required for construction of an allocation token ID.
std::optional<llvm::AllocTokenMetadata>
getAllocTokenMetadata(QualType T, const ASTContext &Ctx);

} // namespace infer_alloc
} // namespace clang

#endif // LLVM_CLANG_AST_INFERALLOC_H
