//== unittests/Sema/GslOwnerPointerInference.cpp - gsl::Owner/Pointer ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ASTMatchers/ASTMatchersTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "gtest/gtest.h"

namespace clang {
using namespace ast_matchers;

TEST(OwnerPointer, BothHaveAttributes) {
  EXPECT_TRUE(matches("template<class T>"
                      "class [[gsl::Owner]] C;"

                      "template<class T>"
                      "class [[gsl::Owner]] C {};"

                      "C<int> c;",
                      classTemplateSpecializationDecl(
                          hasName("C"), hasAttr(clang::attr::Owner))));
}

TEST(OwnerPointer, ForwardDeclOnly) {
  EXPECT_TRUE(matches("template<class T>"
                      "class [[gsl::Owner]] C;"

                      "template<class T>"
                      "class C {};"

                      "C<int> c;",
                      classTemplateSpecializationDecl(
                          hasName("C"), hasAttr(clang::attr::Owner))));
}

TEST(OwnerPointer, LateForwardDeclOnly) {
  EXPECT_TRUE(matches("template<class T>"
                      "class C;"

                      "template<class T>"
                      "class C {};"

                      "template<class T>"
                      "class [[gsl::Owner]] C;"

                      "C<int> c;",
                      classTemplateSpecializationDecl(
                          hasName("C"), hasAttr(clang::attr::Owner))));
}

} // namespace clang
