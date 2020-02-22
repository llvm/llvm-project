//===- GtestMatchers.cpp - AST Matchers for Gtest ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/GtestMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Timer.h"
#include <deque>
#include <memory>
#include <set>

namespace clang {
namespace ast_matchers {

static DeclarationMatcher getComparisonDecl(GtestCmp Cmp) {
  switch (Cmp) {
    case GtestCmp::Eq:
      return cxxMethodDecl(hasName("Compare"),
                           ofClass(cxxRecordDecl(isSameOrDerivedFrom(
                               hasName("::testing::internal::EqHelper")))));
    case GtestCmp::Ne:
      return functionDecl(hasName("::testing::internal::CmpHelperNE"));
    case GtestCmp::Ge:
      return functionDecl(hasName("::testing::internal::CmpHelperGE"));
    case GtestCmp::Gt:
      return functionDecl(hasName("::testing::internal::CmpHelperGT"));
    case GtestCmp::Le:
      return functionDecl(hasName("::testing::internal::CmpHelperLE"));
    case GtestCmp::Lt:
      return functionDecl(hasName("::testing::internal::CmpHelperLT"));
  }
  llvm_unreachable("Unhandled GtestCmp enum");
}

static llvm::StringRef getAssertMacro(GtestCmp Cmp) {
  switch (Cmp) {
    case GtestCmp::Eq:
      return "ASSERT_EQ";
    case GtestCmp::Ne:
      return "ASSERT_NE";
    case GtestCmp::Ge:
      return "ASSERT_GE";
    case GtestCmp::Gt:
      return "ASSERT_GT";
    case GtestCmp::Le:
      return "ASSERT_LE";
    case GtestCmp::Lt:
      return "ASSERT_LT";
  }
  llvm_unreachable("Unhandled GtestCmp enum");
}

static llvm::StringRef getExpectMacro(GtestCmp Cmp) {
  switch (Cmp) {
    case GtestCmp::Eq:
      return "EXPECT_EQ";
    case GtestCmp::Ne:
      return "EXPECT_NE";
    case GtestCmp::Ge:
      return "EXPECT_GE";
    case GtestCmp::Gt:
      return "EXPECT_GT";
    case GtestCmp::Le:
      return "EXPECT_LE";
    case GtestCmp::Lt:
      return "EXPECT_LT";
  }
  llvm_unreachable("Unhandled GtestCmp enum");
}

// In general, AST matchers cannot match calls to macros. However, we can
// simulate such matches if the macro definition has identifiable elements that
// themselves can be matched. In that case, we can match on those elements and
// then check that the match occurs within an expansion of the desired
// macro. The more uncommon the identified elements, the more efficient this
// process will be.
//
// We use this approach to implement the derived matchers gtestAssert and
// gtestExpect.
internal::BindableMatcher<Stmt> gtestAssert(GtestCmp Cmp, StatementMatcher Left,
                                            StatementMatcher Right) {
  return callExpr(callee(getComparisonDecl(Cmp)),
                  isExpandedFromMacro(getAssertMacro(Cmp)),
                  hasArgument(2, Left), hasArgument(3, Right));
}

internal::BindableMatcher<Stmt> gtestExpect(GtestCmp Cmp, StatementMatcher Left,
                                            StatementMatcher Right) {
  return callExpr(callee(getComparisonDecl(Cmp)),
                  isExpandedFromMacro(getExpectMacro(Cmp)),
                  hasArgument(2, Left), hasArgument(3, Right));
}

} // end namespace ast_matchers
} // end namespace clang
