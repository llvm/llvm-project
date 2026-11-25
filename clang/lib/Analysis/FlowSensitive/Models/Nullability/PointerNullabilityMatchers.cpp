//===-- PointerNullabilityMatchers.cpp - AST matchers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerNullabilityMatchers.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::dataflow::nullability {

using ast_matchers::anyOf;
using ast_matchers::expr;
using ast_matchers::hasCastKind;
using ast_matchers::hasOperatorName;
using ast_matchers::hasType;
using ast_matchers::implicitCastExpr;
using ast_matchers::unaryOperator;
using ast_matchers::internal::Matcher;

Matcher<Stmt> isPointerExpr() { return expr(hasType(isSupportedRawPointer())); }

Matcher<Stmt> isNullPointerLiteral() {
  return implicitCastExpr(anyOf(hasCastKind(CK_NullToPointer),
                                hasCastKind(CK_NullToMemberPointer)));
}

Matcher<Stmt> isAddrOf() { return unaryOperator(hasOperatorName("&")); }

} // namespace clang::dataflow::nullability
