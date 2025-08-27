//===- LowLevelHelpers.h - helpers with pure AST interface ---- *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Collects a number of helpers that are used by matchers, but can be reused
// outside of them, e.g. when corresponding matchers cannot be used due to
// performance constraints.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ASTMATCHERS_LOWLEVELHELPERS_H
#define LLVM_CLANG_ASTMATCHERS_LOWLEVELHELPERS_H

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {
namespace ast_matchers {

void matchEachArgumentWithParamType(
    const CallExpr &Node,
    llvm::function_ref<void(QualType /*Param*/, const Expr * /*Arg*/)>
        OnParamAndArg);

void matchEachArgumentWithParamType(
    const CXXConstructExpr &Node,
    llvm::function_ref<void(QualType /*Param*/, const Expr * /*Arg*/)>
        OnParamAndArg);

} // namespace ast_matchers
} // namespace clang

#endif
