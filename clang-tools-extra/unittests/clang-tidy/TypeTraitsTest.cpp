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

// Returns the result of `Predicate` for the definition of class `S` in `Code`.
bool classSatisfies(bool (*Predicate)(const CXXRecordDecl &), StringRef Code) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-std=c++17"});
  EXPECT_NE(AST, nullptr);
  auto Matches = match(cxxRecordDecl(hasName("S"), isDefinition()).bind("s"),
                       AST->getASTContext());
  EXPECT_EQ(Matches.size(), 1u);
  return Predicate(*Matches[0].getNodeAs<CXXRecordDecl>("s"));
}

TEST(IsCopyConstructibleTest, Implicit) {
  EXPECT_TRUE(
      classSatisfies(utils::type_traits::isCopyConstructible, "struct S {};"));
}

TEST(IsCopyConstructibleTest, UserDeclared) {
  EXPECT_TRUE(classSatisfies(utils::type_traits::isCopyConstructible,
                             "struct S { S(const S &); };"));
}

TEST(IsCopyConstructibleTest, Deleted) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyConstructible,
                              "struct S { S(const S &) = delete; };"));
}

TEST(IsCopyConstructibleTest, DeletedByMoveConstructor) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyConstructible,
                              "struct S { S(S &&); };"));
}

TEST(IsCopyConstructibleTest, DeletedByMember) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyConstructible,
                              "struct M { M(const M &) = delete; };"
                              "struct S { M m; };"));
}

TEST(IsCopyConstructibleTest, DeletedByBase) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyConstructible,
                              "struct B { B(const B &) = delete; };"
                              "struct S : B {};"));
}

TEST(IsMoveConstructibleTest, Implicit) {
  EXPECT_TRUE(
      classSatisfies(utils::type_traits::isMoveConstructible, "struct S {};"));
}

TEST(IsMoveConstructibleTest, UserDeclared) {
  EXPECT_TRUE(classSatisfies(utils::type_traits::isMoveConstructible,
                             "struct S { S(S &&); };"));
}

TEST(IsMoveConstructibleTest, Deleted) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isMoveConstructible,
                              "struct S { S(S &&) = delete; };"));
}

TEST(IsMoveConstructibleTest, NotDeclared) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isMoveConstructible,
                              "struct S { S(const S &); };"));
}

TEST(IsMoveConstructibleTest, DeletedByMember) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isMoveConstructible,
                              "struct M { M(M &&) = delete; };"
                              "struct S { M m; };"));
}

TEST(IsCopyAssignableTest, Implicit) {
  EXPECT_TRUE(
      classSatisfies(utils::type_traits::isCopyAssignable, "struct S {};"));
}

TEST(IsCopyAssignableTest, UserDeclared) {
  EXPECT_TRUE(classSatisfies(utils::type_traits::isCopyAssignable,
                             "struct S { S &operator=(const S &); };"));
}

TEST(IsCopyAssignableTest, Deleted) {
  EXPECT_FALSE(
      classSatisfies(utils::type_traits::isCopyAssignable,
                     "struct S { S &operator=(const S &) = delete; };"));
}

TEST(IsCopyAssignableTest, DeletedByMoveAssignment) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyAssignable,
                              "struct S { S &operator=(S &&); };"));
}

TEST(IsCopyAssignableTest, DeletedByConstMember) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyAssignable,
                              "struct S { const int I; };"));
}

TEST(IsCopyAssignableTest, DeletedByReferenceMember) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isCopyAssignable,
                              "struct S { int &R; };"));
}

TEST(IsMoveAssignableTest, Implicit) {
  EXPECT_TRUE(
      classSatisfies(utils::type_traits::isMoveAssignable, "struct S {};"));
}

TEST(IsMoveAssignableTest, UserDeclared) {
  EXPECT_TRUE(classSatisfies(utils::type_traits::isMoveAssignable,
                             "struct S { S &operator=(S &&); };"));
}

TEST(IsMoveAssignableTest, Deleted) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isMoveAssignable,
                              "struct S { S &operator=(S &&) = delete; };"));
}

TEST(IsMoveAssignableTest, NotDeclared) {
  EXPECT_FALSE(classSatisfies(utils::type_traits::isMoveAssignable,
                              "struct S { S &operator=(const S &); };"));
}

} // namespace
} // namespace clang::tidy::test
