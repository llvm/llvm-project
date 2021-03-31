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

TEST(Introspection, SourceLocations_Stmt) {
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

TEST(Introspection, SourceLocations_Decl) {
  auto AST =
      buildASTFromCode(R"cpp(
namespace ns1 {
namespace ns2 {
template <typename T, typename U> struct Foo {};
template <typename T, typename U> struct Bar {
  struct Nested {
    template <typename A, typename B>
    Foo<A, B> method(int i, bool b) const noexcept(true);
  };
};
} // namespace ns2
} // namespace ns1

template <typename T, typename U>
template <typename A, typename B>
ns1::ns2::Foo<A, B> ns1::ns2::Bar<T, U>::Nested::method(int i, bool b) const
    noexcept(true) {}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          cxxMethodDecl(hasName("method")).bind("method"))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *MethodDecl = BoundNodes[0].getNodeAs<CXXMethodDecl>("method");

  auto Result = NodeIntrospection::GetLocations(MethodDecl);

  if (Result.LocationAccessors.empty() && Result.RangeAccessors.empty()) {
    return;
  }

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
                  STRING_LOCATION_PAIR(MethodDecl, getBeginLoc()),
                  STRING_LOCATION_PAIR(MethodDecl, getBodyRBrace()),
                  STRING_LOCATION_PAIR(MethodDecl, getEllipsisLoc()),
                  STRING_LOCATION_PAIR(MethodDecl, getInnerLocStart()),
                  STRING_LOCATION_PAIR(MethodDecl, getLocation()),
                  STRING_LOCATION_PAIR(MethodDecl, getOuterLocStart()),
                  STRING_LOCATION_PAIR(MethodDecl, getPointOfInstantiation()),
                  STRING_LOCATION_PAIR(MethodDecl, getTypeSpecEndLoc()),
                  STRING_LOCATION_PAIR(MethodDecl, getTypeSpecStartLoc()),
                  STRING_LOCATION_PAIR(MethodDecl, getEndLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(MethodDecl, getExceptionSpecSourceRange()),
          STRING_LOCATION_PAIR(MethodDecl, getParametersSourceRange()),
          STRING_LOCATION_PAIR(MethodDecl, getReturnTypeSourceRange()),
          STRING_LOCATION_PAIR(MethodDecl, getSourceRange())));
}
