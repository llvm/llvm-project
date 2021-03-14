//===- unittest/Introspection/IntrospectionTest.cpp ----------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for AST location API introspection.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/NodeIntrospection.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

#if SKIP_INTROSPECTION_GENERATION

TEST(Introspection, NonFatalAPI) {
  auto AST = buildASTFromCode("void foo() {} void bar() { foo(); }", "foo.cpp",
                              std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          callExpr(callee(functionDecl(hasName("foo")))).bind("fooCall"))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  auto *FooCall = BoundNodes[0].getNodeAs<CallExpr>("fooCall");

  auto result = NodeIntrospection::GetLocations(FooCall);

  EXPECT_EQ(result.LocationAccessors.size(), 0u);
  EXPECT_EQ(result.RangeAccessors.size(), 0u);
}

#else

TEST(Introspection, SourceLocations) {
  auto AST = buildASTFromCode("void foo() {} void bar() { foo(); }", "foo.cpp",
                              std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          callExpr(callee(functionDecl(hasName("foo")))).bind("fooCall"))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  auto *FooCall = BoundNodes[0].getNodeAs<CallExpr>("fooCall");

  auto result = NodeIntrospection::GetLocations(FooCall);

  std::map<std::string, SourceLocation> ExpectedLocations;
  llvm::transform(result.LocationAccessors,
                  std::inserter(ExpectedLocations, ExpectedLocations.end()),
                  [](const auto &Accessor) {
                    return std::make_pair(
                        LocationCallFormatterCpp::format(Accessor.second.get()),
                        Accessor.first);
                  });

  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(Pair("getBeginLoc()", FooCall->getBeginLoc()),
                           Pair("getEndLoc()", FooCall->getEndLoc()),
                           Pair("getExprLoc()", FooCall->getExprLoc()),
                           Pair("getRParenLoc()", FooCall->getRParenLoc())));

  std::map<std::string, SourceRange> ExpectedRanges;
  llvm::transform(result.RangeAccessors,
                  std::inserter(ExpectedRanges, ExpectedRanges.end()),
                  [](const auto &Accessor) {
                    return std::make_pair(
                        LocationCallFormatterCpp::format(Accessor.second.get()),
                        Accessor.first);
                  });

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(
                  Pair("getSourceRange()", FooCall->getSourceRange())));
}
#endif
