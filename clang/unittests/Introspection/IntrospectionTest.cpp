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

template<typename T, typename MapType>
std::map<std::string, T>
FormatExpected(const MapType &Accessors) {
  std::map<std::string, T> Result;
  llvm::transform(Accessors,
                  std::inserter(Result, Result.end()),
                  [](const auto &Accessor) {
                    return std::make_pair(
                        LocationCallFormatterCpp::format(Accessor.second.get()),
                        Accessor.first);
                  });
  return Result;
}

#define STRING_LOCATION_PAIR(INSTANCE, LOC) Pair(#LOC, INSTANCE->LOC)

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

  auto Result = NodeIntrospection::GetLocations(FooCall);

  if (Result.LocationAccessors.empty() && Result.RangeAccessors.empty())
  {
    return;
  }

  auto ExpectedLocations =
    FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(STRING_LOCATION_PAIR(FooCall, getBeginLoc()),
                           STRING_LOCATION_PAIR(FooCall, getEndLoc()),
                           STRING_LOCATION_PAIR(FooCall, getExprLoc()),
                           STRING_LOCATION_PAIR(FooCall, getRParenLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(STRING_LOCATION_PAIR(
                                  FooCall, getSourceRange())));
}
