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
  return Out.str();
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
  EXPECT_EQ(TN.getKind(), TemplateName::UsingTemplate);
  EXPECT_EQ(TN.getAsUsingShadowDecl()->getTargetDecl(), TN.getAsTemplateDecl());

  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::Fully),
            "std::vector");
  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::AsWritten),
            "vector");
  EXPECT_EQ(printTemplateName(TN, Ctx.getPrintingPolicy(),
                              TemplateName::Qualified::None),
            "vector");
}

} // namespace
} // namespace clang
