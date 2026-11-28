//===-- PointerNullabilityMatchers.h - AST matchers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYMATCHERS_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYMATCHERS_H

#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullability.h"

namespace clang::dataflow::nullability {

AST_MATCHER(QualType, isSupportedPointer) {
  return isSupportedPointerType(Node);
}

AST_MATCHER(QualType, isSupportedRawPointer) {
  return isSupportedRawPointerType(Node);
}

ast_matchers::internal::Matcher<Stmt> isPointerExpr();
ast_matchers::internal::Matcher<Stmt> isNullPointerLiteral();
ast_matchers::internal::Matcher<Stmt> isAddrOf();

} // namespace clang::dataflow::nullability

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYMATCHERS_H
