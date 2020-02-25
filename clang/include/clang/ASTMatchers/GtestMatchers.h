//===- GtestMatchers.h - AST Matchers for GTest -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements matchers specific to structures in the Googletest
//  (gtest) framework.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ASTMATCHERS_GTESTMATCHERS_H
#define LLVM_CLANG_ASTMATCHERS_GTESTMATCHERS_H

#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace ast_matchers {

/// Gtest's comparison operations.
enum class GtestCmp {
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
};

/// Matcher for gtest's ASSERT_... macros.
internal::BindableMatcher<Stmt> gtestAssert(GtestCmp Cmp, StatementMatcher Left,
                                            StatementMatcher Right);

/// Matcher for gtest's EXPECT_... macros.
internal::BindableMatcher<Stmt> gtestExpect(GtestCmp Cmp, StatementMatcher Left,
                                            StatementMatcher Right);

} // namespace ast_matchers
} // namespace clang

#endif // LLVM_CLANG_ASTMATCHERS_GTESTMATCHERS_H

