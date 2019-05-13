//===- unittest/AST/ASTImporterTest.cpp - AST node import test ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type-parameterized tests for the correct import of Decls with different
// visibility.
//
//===----------------------------------------------------------------------===//

// Define this to have ::testing::Combine available.
// FIXME: Better solution for this?
#define GTEST_HAS_COMBINE 1

#include "ASTImporterFixtures.h"

namespace clang {
namespace ast_matchers {

using internal::BindableMatcher;

// Type parameters for type-parameterized test fixtures.
struct GetFunPattern {
  using DeclTy = FunctionDecl;
  BindableMatcher<Decl> operator()() { return functionDecl(hasName("f")); }
};
struct GetVarPattern {
  using DeclTy = VarDecl;
  BindableMatcher<Decl> operator()() { return varDecl(hasName("v")); }
};

// Values for the value-parameterized test fixtures.
// FunctionDecl:
const auto *ExternF = "void f();";
const auto *StaticF = "static void f();";
const auto *AnonF = "namespace { void f(); }";
// VarDecl:
const auto *ExternV = "extern int v;";
const auto *StaticV = "static int v;";
const auto *AnonV = "namespace { extern int v; }";

// First value in tuple: Compile options.
// Second value in tuple: Source code to be used in the test.
using ImportVisibilityChainParams =
    ::testing::WithParamInterface<std::tuple<ArgVector, const char *>>;
// Fixture to test the redecl chain of Decls with the same visibility. Gtest
// makes it possible to have either value-parameterized or type-parameterized
// fixtures. However, we cannot have both value- and type-parameterized test
// fixtures. This is a value-parameterized test fixture in the gtest sense. We
// intend to mimic gtest's type-parameters via the PatternFactory template
// parameter. We manually instantiate the different tests with the each types.
template <typename PatternFactory>
class ImportVisibilityChain
    : public ASTImporterTestBase, public ImportVisibilityChainParams {
protected:
  using DeclTy = typename PatternFactory::DeclTy;
  ArgVector getExtraArgs() const override { return std::get<0>(GetParam()); }
  std::string getCode() const { return std::get<1>(GetParam()); }
  BindableMatcher<Decl> getPattern() const { return PatternFactory()(); }

  // Type-parameterized test.
  void TypedTest_ImportChain() {
    std::string Code = getCode() + getCode();
    auto Pattern = getPattern();

    TranslationUnitDecl *FromTu = getTuDecl(Code, Lang_CXX, "input0.cc");

    auto *FromD0 = FirstDeclMatcher<DeclTy>().match(FromTu, Pattern);
    auto *FromD1 = LastDeclMatcher<DeclTy>().match(FromTu, Pattern);

    auto *ToD0 = Import(FromD0, Lang_CXX);
    auto *ToD1 = Import(FromD1, Lang_CXX);

    EXPECT_TRUE(ToD0);
    ASSERT_TRUE(ToD1);
    EXPECT_NE(ToD0, ToD1);
    EXPECT_EQ(ToD1->getPreviousDecl(), ToD0);
  }
};

// Manual instantiation of the fixture with each type.
using ImportFunctionsVisibilityChain = ImportVisibilityChain<GetFunPattern>;
using ImportVariablesVisibilityChain = ImportVisibilityChain<GetVarPattern>;
// Value-parameterized test for the first type.
TEST_P(ImportFunctionsVisibilityChain, ImportChain) {
  TypedTest_ImportChain();
}
// Value-parameterized test for the second type.
TEST_P(ImportVariablesVisibilityChain, ImportChain) {
  TypedTest_ImportChain();
}

// Automatic instantiation of the value-parameterized tests.
INSTANTIATE_TEST_CASE_P(ParameterizedTests, ImportFunctionsVisibilityChain,
                        ::testing::Combine(
                           DefaultTestValuesForRunOptions,
                           ::testing::Values(ExternF, StaticF, AnonF)), );
INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportVariablesVisibilityChain,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        // There is no point to instantiate with StaticV, because in C++ we can
        // forward declare a variable only with the 'extern' keyword.
        // Consequently, each fwd declared variable has external linkage.  This
        // is different in the C language where any declaration without an
        // initializer is a tentative definition, subsequent definitions may be
        // provided but they must have the same linkage.  See also the test
        // ImportVariableChainInC which test for this special C Lang case.
        ::testing::Values(ExternV, AnonV)), );

// First value in tuple: Compile options.
// Second value in tuple: Tuple with informations for the test.
// Code for first import (or initial code), code to import, whether the `f`
// functions are expected to be linked in a declaration chain.
// One value of this tuple is combined with every value of compile options.
// The test can have a single tuple as parameter only.
using ImportVisibilityParams = ::testing::WithParamInterface<
    std::tuple<ArgVector, std::tuple<const char *, const char *, bool>>>;

template <typename PatternFactory>
class ImportVisibility
    : public ASTImporterTestBase,
      public ImportVisibilityParams {
protected:
  using DeclTy = typename PatternFactory::DeclTy;
  ArgVector getExtraArgs() const override { return std::get<0>(GetParam()); }
  std::string getCode0() const { return std::get<0>(std::get<1>(GetParam())); }
  std::string getCode1() const { return std::get<1>(std::get<1>(GetParam())); }
  bool shouldBeLinked() const { return std::get<2>(std::get<1>(GetParam())); }
  BindableMatcher<Decl> getPattern() const { return PatternFactory()(); }

  void TypedTest_ImportAfter() {
    TranslationUnitDecl *ToTu = getToTuDecl(getCode0(), Lang_CXX);
    TranslationUnitDecl *FromTu = getTuDecl(getCode1(), Lang_CXX, "input1.cc");

    auto *ToD0 = FirstDeclMatcher<DeclTy>().match(ToTu, getPattern());
    auto *FromD1 = FirstDeclMatcher<DeclTy>().match(FromTu, getPattern());

    auto *ToD1 = Import(FromD1, Lang_CXX);

    ASSERT_TRUE(ToD0);
    ASSERT_TRUE(ToD1);
    EXPECT_NE(ToD0, ToD1);

    if (shouldBeLinked())
      EXPECT_EQ(ToD1->getPreviousDecl(), ToD0);
    else
      EXPECT_FALSE(ToD1->getPreviousDecl());
  }

  void TypedTest_ImportAfterImport() {
    TranslationUnitDecl *FromTu0 = getTuDecl(getCode0(), Lang_CXX, "input0.cc");
    TranslationUnitDecl *FromTu1 = getTuDecl(getCode1(), Lang_CXX, "input1.cc");
    auto *FromD0 = FirstDeclMatcher<DeclTy>().match(FromTu0, getPattern());
    auto *FromD1 = FirstDeclMatcher<DeclTy>().match(FromTu1, getPattern());
    auto *ToD0 = Import(FromD0, Lang_CXX);
    auto *ToD1 = Import(FromD1, Lang_CXX);
    ASSERT_TRUE(ToD0);
    ASSERT_TRUE(ToD1);
    EXPECT_NE(ToD0, ToD1);
    if (shouldBeLinked())
      EXPECT_EQ(ToD1->getPreviousDecl(), ToD0);
    else
      EXPECT_FALSE(ToD1->getPreviousDecl());
  }
};
using ImportFunctionsVisibility = ImportVisibility<GetFunPattern>;
using ImportVariablesVisibility = ImportVisibility<GetVarPattern>;

// FunctionDecl.
TEST_P(ImportFunctionsVisibility, ImportAfter) {
  TypedTest_ImportAfter();
}
TEST_P(ImportFunctionsVisibility, ImportAfterImport) {
  TypedTest_ImportAfterImport();
}
// VarDecl.
TEST_P(ImportVariablesVisibility, ImportAfter) {
  TypedTest_ImportAfter();
}
TEST_P(ImportVariablesVisibility, ImportAfterImport) {
  TypedTest_ImportAfterImport();
}

const bool ExpectLink = true;
const bool ExpectNotLink = false;

INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportFunctionsVisibility,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        ::testing::Values(std::make_tuple(ExternF, ExternF, ExpectLink),
                          std::make_tuple(ExternF, StaticF, ExpectNotLink),
                          std::make_tuple(ExternF, AnonF, ExpectNotLink),
                          std::make_tuple(StaticF, ExternF, ExpectNotLink),
                          std::make_tuple(StaticF, StaticF, ExpectNotLink),
                          std::make_tuple(StaticF, AnonF, ExpectNotLink),
                          std::make_tuple(AnonF, ExternF, ExpectNotLink),
                          std::make_tuple(AnonF, StaticF, ExpectNotLink),
                          std::make_tuple(AnonF, AnonF, ExpectNotLink))), );
INSTANTIATE_TEST_CASE_P(
    ParameterizedTests, ImportVariablesVisibility,
    ::testing::Combine(
        DefaultTestValuesForRunOptions,
        ::testing::Values(std::make_tuple(ExternV, ExternV, ExpectLink),
                          std::make_tuple(ExternV, StaticV, ExpectNotLink),
                          std::make_tuple(ExternV, AnonV, ExpectNotLink),
                          std::make_tuple(StaticV, ExternV, ExpectNotLink),
                          std::make_tuple(StaticV, StaticV, ExpectNotLink),
                          std::make_tuple(StaticV, AnonV, ExpectNotLink),
                          std::make_tuple(AnonV, ExternV, ExpectNotLink),
                          std::make_tuple(AnonV, StaticV, ExpectNotLink),
                          std::make_tuple(AnonV, AnonV, ExpectNotLink))), );

} // end namespace ast_matchers
} // end namespace clang
