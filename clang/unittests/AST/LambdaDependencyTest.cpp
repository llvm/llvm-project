//===- unittests/AST/LambdaDependencyTest.cpp --- Lambda dependency tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeclMatcher.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

namespace {

TEST(LambdaDependencyTest, NoexceptSpecifierOfAbbreviatedFunctionTemplate) {
  auto AST = buildASTFromCodeWithArgs(
      "void f(auto&& fn) noexcept(noexcept([_ = fn] {})) {}",
      /*Args=*/{"-std=c++20"});
  ASSERT_TRUE(AST);

  FirstDeclMatcher<CXXRecordDecl> Matcher;
  auto *Closure = Matcher.match(AST->getASTContext().getTranslationUnitDecl(),
                                cxxRecordDecl(isLambda()));
  ASSERT_NE(nullptr, Closure);
  EXPECT_TRUE(Closure->isDependentType());
}

} // namespace
