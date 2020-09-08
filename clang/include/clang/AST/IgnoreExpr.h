//===--- IgnoreExpr.h - Ignore intermediate Expressions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common functions to ignore intermediate expression nodes
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_IGNOREEXPR_H
#define LLVM_CLANG_AST_IGNOREEXPR_H

#include "clang/AST/Expr.h"

namespace clang {
namespace detail {
/// Given an expression E and functions Fn_1,...,Fn_n : Expr * -> Expr *,
/// Return Fn_n(...(Fn_1(E)))
inline Expr *IgnoreExprNodesImpl(Expr *E) { return E; }
template <typename FnTy, typename... FnTys>
Expr *IgnoreExprNodesImpl(Expr *E, FnTy &&Fn, FnTys &&... Fns) {
  return IgnoreExprNodesImpl(Fn(E), std::forward<FnTys>(Fns)...);
}
} // namespace detail

/// Given an expression E and functions Fn_1,...,Fn_n : Expr * -> Expr *,
/// Recursively apply each of the functions to E until reaching a fixed point.
/// Note that a null E is valid; in this case nothing is done.
template <typename... FnTys> Expr *IgnoreExprNodes(Expr *E, FnTys &&... Fns) {
  Expr *LastE = nullptr;
  while (E != LastE) {
    LastE = E;
    E = detail::IgnoreExprNodesImpl(E, std::forward<FnTys>(Fns)...);
  }
  return E;
}

Expr *IgnoreImplicitCastsSingleStep(Expr *E);

Expr *IgnoreImplicitCastsExtraSingleStep(Expr *E);

Expr *IgnoreCastsSingleStep(Expr *E);

Expr *IgnoreLValueCastsSingleStep(Expr *E);

Expr *IgnoreBaseCastsSingleStep(Expr *E);

Expr *IgnoreImplicitSingleStep(Expr *E);

Expr *IgnoreImplicitAsWrittenSingleStep(Expr *E);

Expr *IgnoreParensOnlySingleStep(Expr *E);

Expr *IgnoreParensSingleStep(Expr *E);

} // namespace clang

#endif // LLVM_CLANG_AST_IGNOREEXPR_H
