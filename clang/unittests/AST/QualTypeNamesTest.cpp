//===- unittests/AST/QualTypeNamesTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for helpers from QualTypeNames.h.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/QualTypeNames.h"
#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/TypeBase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

namespace clang {
namespace {

TEST(QualTypeNamesTest, TemplateParameters) {
  constexpr llvm::StringLiteral Code = R"cpp(
    template <template<class> class T> struct Foo {
      using type_of_interest = T<int>;
    };
  )cpp";
  auto AST = tooling::buildASTFromCode(Code);
  ASSERT_NE(AST, nullptr);

  auto &Ctx = AST->getASTContext();
  auto FooLR = Ctx.getTranslationUnitDecl()->lookup(
      DeclarationName(AST->getPreprocessor().getIdentifierInfo("Foo")));
  ASSERT_TRUE(FooLR.isSingleResult());

  auto TypeLR =
      llvm::cast<ClassTemplateDecl>(FooLR.front())
          ->getTemplatedDecl()
          ->lookup(DeclarationName(
              AST->getPreprocessor().getIdentifierInfo("type_of_interest")));
  ASSERT_TRUE(TypeLR.isSingleResult());

  auto Type = cast<TypeAliasDecl>(TypeLR.front())->getUnderlyingType();
  ASSERT_TRUE(isa<TemplateSpecializationType>(Type));

  EXPECT_EQ(TypeName::getFullyQualifiedName(Type, Ctx, Ctx.getPrintingPolicy()),
            "T<int>");
}

} // namespace
} // namespace clang
