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
#include "llvm/Support/AllocToken.h"
#include <optional>

namespace clang {
namespace infer_alloc {

/// Infer the possible allocated type from an allocation call expression.
QualType inferPossibleType(const CallExpr *E, const ASTContext &Ctx,
                           const CastExpr *CastE);

/// Get the information required for construction of an allocation token ID.
std::optional<llvm::AllocTokenMetadata>
getAllocTokenMetadata(QualType T, const ASTContext &Ctx);

} // namespace infer_alloc
} // namespace clang

#endif // LLVM_CLANG_AST_INFERALLOC_H
