//===- unittests/AST/DeclBaseTest.cpp --- Declaration tests----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Decl class in the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclBase.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using ::clang::BindingDecl;
using ::clang::ast_matchers::bindingDecl;
using ::clang::ast_matchers::hasName;
using ::clang::ast_matchers::match;
using ::clang::ast_matchers::selectFirst;

TEST(DeclGetFunctionType, BindingDecl) {
  llvm::StringRef Code = R"cpp(
    template <typename A, typename B>
    struct Pair {
      A AnA;
      B AB;
    };

    void target(int *i) {
      Pair<void (*)(int *), bool> P;
      auto [FunctionPointer, B] = P;
      FunctionPointer(i);
    }
  )cpp";

  auto AST =
      clang::tooling::buildASTFromCodeWithArgs(Code, /*Args=*/{"-std=c++20"});
  clang::ASTContext &Ctx = AST->getASTContext();

  auto *BD = selectFirst<clang::BindingDecl>(
      "FunctionPointer",
      match(bindingDecl(hasName("FunctionPointer")).bind("FunctionPointer"),
            Ctx));
  ASSERT_NE(BD, nullptr);

  EXPECT_NE(BD->getFunctionType(), nullptr);

  // Emulate a call before the BindingDecl has a bound type.
  const_cast<clang::BindingDecl *>(BD)->setBinding(clang::QualType(), nullptr);
  EXPECT_EQ(BD->getFunctionType(), nullptr);
}
