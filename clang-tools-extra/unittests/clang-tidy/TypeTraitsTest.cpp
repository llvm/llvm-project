//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/TypeTraits.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang::tidy::test {
namespace {

using namespace ast_matchers;

// Returns whether the type of the declaration named `DeclName` in `Code` is a
// std::initializer_list specialization.
bool declTypeIsStdInitializerList(StringRef Code, StringRef DeclName) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++17"});
  EXPECT_NE(AST, nullptr);
  auto Matches =
      match(valueDecl(hasName(DeclName)).bind("d"), AST->getASTContext());
  EXPECT_EQ(Matches.size(), 1u);
  const auto *D = Matches[0].getNodeAs<ValueDecl>("d");
  return utils::type_traits::isStdInitializerList(D->getType());
}

constexpr char InitializerListDecl[] =
    "namespace std { template <typename T> class initializer_list {}; }\n";

TEST(IsStdInitializerListTest, MatchesSpecialization) {
  EXPECT_TRUE(declTypeIsStdInitializerList(
      std::string(InitializerListDecl) + "std::initializer_list<int> v;", "v"));
}

TEST(IsStdInitializerListTest, MatchesDependentSpecialization) {
  EXPECT_TRUE(declTypeIsStdInitializerList(
      std::string(InitializerListDecl) +
          "template <typename T> void f(std::initializer_list<T> p);",
      "p"));
}

TEST(IsStdInitializerListTest, RejectsBuiltin) {
  EXPECT_FALSE(declTypeIsStdInitializerList("int v;", "v"));
}

TEST(IsStdInitializerListTest, RejectsNonStdInitializerList) {
  EXPECT_FALSE(declTypeIsStdInitializerList(
      "template <typename T> class initializer_list {}; "
      "initializer_list<int> v;",
      "v"));
}

} // namespace
} // namespace clang::tidy::test
