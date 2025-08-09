//===- unittests/AST/TemplateNameTest.cpp --- Tests for TemplateName ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang {
namespace {
using namespace ast_matchers;

std::string printTemplateName(TemplateName TN, const PrintingPolicy &Policy,
                              TemplateName::Qualified Qual) {
  std::string Result;
  llvm::raw_string_ostream Out(Result);
  TN.print(Out, Policy, Qual);
  return Result;
}

TEST(TemplateName, PrintTemplate) {
  std::string Code = R"cpp(
    namespace std {
      template <typename> struct vector {};
    }
    template<template <typename> class T> class X;
    using A = X<std::vector>;
  )cpp";
  auto AST = tooling::buildASTFromCode(Code);
  ASTContext &Ctx = AST->getASTContext();
  // Match the template argument vector in X<std::vector>.
  auto MatchResults = match(templateArgumentLoc().bind("id"), Ctx);
  const auto *Template = selectFirst<TemplateArgumentLoc>("id", MatchResults);
  ASSERT_TRUE(Template);

  TemplateName TN = Template->getArgument().getAsTemplate();
  EXPECT_EQ(TN.getKind(), TemplateName::QualifiedTemplate);
  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::AsWritten),
            "std::vector");
  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::None),
            "vector");
}

TEST(TemplateName, PrintUsingTemplate) {
  std::string Code = R"cpp(
    namespace std {
      template <typename> struct vector {};
    }
    namespace absl { using std::vector; }

    template<template <typename> class T> class X;

    using absl::vector;
    using A = X<vector>;
  )cpp";
  auto AST = tooling::buildASTFromCode(Code);
  ASTContext &Ctx = AST->getASTContext();
  // Match the template argument vector in X<vector>.
  auto MatchResults = match(templateArgumentLoc().bind("id"), Ctx);
  const auto *Template = selectFirst<TemplateArgumentLoc>("id", MatchResults);
  ASSERT_TRUE(Template);

  TemplateName TN = Template->getArgument().getAsTemplate();
  EXPECT_EQ(TN.getKind(), TemplateName::QualifiedTemplate);
  UsingShadowDecl *USD = TN.getAsUsingShadowDecl();
  EXPECT_TRUE(USD != nullptr);
  EXPECT_EQ(USD->getTargetDecl(), TN.getAsTemplateDecl());

  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::AsWritten),
            "vector");
  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::None),
            "vector");
}

TEST(TemplateName, QualifiedUsingTemplate) {
  std::string Code = R"cpp(
    namespace std {
      template <typename> struct vector {};
    }
    namespace absl { using std::vector; }

    template<template <typename> class T> class X;

    using A = X<absl::vector>; // QualifiedTemplateName in a template argument.
  )cpp";
  auto AST = tooling::buildASTFromCode(Code);
  // Match the template argument absl::vector in X<absl::vector>.
  auto Matcher = templateArgumentLoc().bind("id");
  auto MatchResults = match(Matcher, AST->getASTContext());
  const auto *TAL = MatchResults.front().getNodeAs<TemplateArgumentLoc>("id");
  ASSERT_TRUE(TAL);
  TemplateName TN = TAL->getArgument().getAsTemplate();
  EXPECT_EQ(TN.getKind(), TemplateName::QualifiedTemplate);
  const auto *QTN = TN.getAsQualifiedTemplateName();
  // Verify that we have the Using template name in the QualifiedTemplateName.
  const auto *USD = QTN->getUnderlyingTemplate().getAsUsingShadowDecl();
  EXPECT_TRUE(USD);
  EXPECT_EQ(USD->getTargetDecl(), TN.getAsTemplateDecl());
  EXPECT_EQ(TN.getAsUsingShadowDecl(), USD);
}

TEST(TemplateName, UsingTemplate) {
  auto AST = tooling::buildASTFromCode(R"cpp(
    namespace std {
      template <typename T> struct vector { vector(T); };
    }
    namespace absl { using std::vector; }
    // The "absl::vector<int>" is an elaborated TemplateSpecializationType with
    // an inner Using TemplateName (not a Qualified TemplateName, the qualifiers
    // are rather part of the ElaboratedType)!
    absl::vector<int> v(123);
  )cpp");
  auto Matcher = elaboratedTypeLoc(
      hasNamedTypeLoc(loc(templateSpecializationType().bind("id"))));
  auto MatchResults = match(Matcher, AST->getASTContext());
  const auto *TST =
      MatchResults.front().getNodeAs<TemplateSpecializationType>("id");
  ASSERT_TRUE(TST);
  EXPECT_EQ(TST->getTemplateName().getKind(), TemplateName::QualifiedTemplate);
  EXPECT_TRUE(TST->getTemplateName().getAsUsingShadowDecl() != nullptr);

  AST = tooling::buildASTFromCodeWithArgs(R"cpp(
    namespace std {
      template <typename T> struct vector { vector(T); };
    }
    namespace absl { using std::vector; }
    // Similiar to the TemplateSpecializationType, absl::vector is an elaborated
    // DeducedTemplateSpecializationType with an inner Using TemplateName!
    absl::vector DTST(123);
    )cpp",
                                          {"-std=c++17"});
  Matcher = elaboratedTypeLoc(
      hasNamedTypeLoc(loc(deducedTemplateSpecializationType().bind("id"))));
  MatchResults = match(Matcher, AST->getASTContext());
  const auto *DTST =
      MatchResults.front().getNodeAs<DeducedTemplateSpecializationType>("id");
  ASSERT_TRUE(DTST);
  EXPECT_EQ(DTST->getTemplateName().getKind(), TemplateName::QualifiedTemplate);
  EXPECT_TRUE(DTST->getTemplateName().getAsUsingShadowDecl() != nullptr);
}

} // namespace
} // namespace clang
