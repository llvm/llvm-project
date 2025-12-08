//===- unittests/Analysis/Scalable/ASTEntityMappingTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/ASTEntityMapping.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;

namespace clang::ssaf {
namespace {

// Helper function to find a declaration by name
template <typename DeclType>
const DeclType *findDecl(ASTContext &Ctx, StringRef Name) {
  auto Matcher = namedDecl(hasName(Name)).bind("decl");
  auto Matches = match(Matcher, Ctx);
  if (Matches.empty())
    return nullptr;
  if (auto Result = Matches[0].getNodeAs<DeclType>("decl"))
    return dyn_cast<DeclType>(Result->getCanonicalDecl());
  return nullptr;
}

TEST(ASTEntityMappingTest, FunctionDecl) {
  auto AST = tooling::buildASTFromCode("void foo() {}");
  auto &Ctx = AST->getASTContext();

  const auto *FD = findDecl<FunctionDecl>(Ctx, "foo");
  ASSERT_NE(FD, nullptr);

  auto EntityName = getEntityName(FD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, VarDecl) {
  auto AST = tooling::buildASTFromCode("int x = 42;");
  auto &Ctx = AST->getASTContext();

  const auto *VD = findDecl<VarDecl>(Ctx, "x");
  ASSERT_NE(VD, nullptr);

  auto EntityName = getEntityName(VD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, ParmVarDecl) {
  auto AST = tooling::buildASTFromCode("void foo(int x) {}");
  auto &Ctx = AST->getASTContext();

  const auto *FD = findDecl<FunctionDecl>(Ctx, "foo");
  ASSERT_NE(FD, nullptr);
  ASSERT_EQ(FD->param_size(), 1u);

  const auto *PVD = FD->getParamDecl(0);
  ASSERT_NE(PVD, nullptr);

  auto EntityName = getEntityName(PVD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, RecordDecl) {
  auto AST = tooling::buildASTFromCode("struct S {};");
  auto &Ctx = AST->getASTContext();

  const auto *RD = findDecl<RecordDecl>(Ctx, "S");
  ASSERT_NE(RD, nullptr);

  auto EntityName = getEntityName(RD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, FieldDecl) {
  auto AST = tooling::buildASTFromCode("struct S { int field; };");
  auto &Ctx = AST->getASTContext();

  const auto *FD = findDecl<FieldDecl>(Ctx, "field");
  ASSERT_NE(FD, nullptr);

  auto EntityName = getEntityName(FD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, NullDecl) {
  auto EntityName = getEntityName(nullptr);
  EXPECT_FALSE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, ImplicitDecl) {
  auto AST = tooling::buildASTFromCode(R"(
    struct S {
      S() = default;
    };
  )", "test.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();

  const auto *RD = findDecl<CXXRecordDecl>(Ctx, "S");
  ASSERT_NE(RD, nullptr);

  // Find the implicitly-declared copy constructor
  const CXXConstructorDecl * ImplCtor = nullptr;
  for (const auto *Ctor : RD->ctors()) {
    if (Ctor->isCopyConstructor() && Ctor->isImplicit()) {
      ImplCtor = Ctor;
      break;
    }
  }

  auto EntityName = getEntityName(ImplCtor);
  EXPECT_FALSE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, BuiltinFunction) {
  auto AST = tooling::buildASTFromCode(R"(
    void test() {
      __builtin_memcpy(0, 0, 0);
    }
  )");
  auto &Ctx = AST->getASTContext();

  // Find the builtin call
  auto Matcher = callExpr().bind("call");
  auto Matches = match(Matcher, Ctx);
  ASSERT_EQ(Matches.size(), 1ul);

  const auto *CE = Matches[0].getNodeAs<CallExpr>("call");
  ASSERT_NE(CE, nullptr);

  const auto *Callee = CE->getDirectCallee();
  ASSERT_NE(Callee, nullptr);
  ASSERT_NE(Callee->getBuiltinID(), 0u /* not a built-in */);

  auto EntityName = getEntityName(Callee);
  EXPECT_FALSE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, UnsupportedDecl) {
  auto AST = tooling::buildASTFromCode("namespace N {}");
  auto &Ctx = AST->getASTContext();

  const auto *ND = findDecl<NamespaceDecl>(Ctx, "N");
  ASSERT_NE(ND, nullptr);

  auto EntityName = getEntityName(ND);
  EXPECT_FALSE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, FunctionReturn) {
  auto AST = tooling::buildASTFromCode("int foo() { return 42; }");
  auto &Ctx = AST->getASTContext();

  const auto *FD = findDecl<FunctionDecl>(Ctx, "foo");
  ASSERT_NE(FD, nullptr);

  auto EntityName = getEntityNameForReturn(FD);
  EXPECT_TRUE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, FunctionReturnNull) {
  auto EntityName = getEntityNameForReturn(nullptr);
  EXPECT_FALSE(EntityName.has_value());
}

TEST(ASTEntityMappingTest, FunctionReturnBuiltin) {
  auto AST = tooling::buildASTFromCode(R"(
    void test() {
      __builtin_memcpy(0, 0, 0);
    }
  )");
  auto &Ctx = AST->getASTContext();

  // Find the builtin call
  auto Matcher = callExpr().bind("call");
  auto Matches = match(Matcher, Ctx);
  ASSERT_FALSE(Matches.empty());

  const auto *CE = Matches[0].getNodeAs<CallExpr>("call");
  ASSERT_NE(CE, nullptr);

  const auto *Callee = CE->getDirectCallee();
  if (Callee && Callee->getBuiltinID()) {
    auto EntityName = getEntityNameForReturn(Callee);
    EXPECT_FALSE(EntityName.has_value());
  }
}

TEST(ASTEntityMappingTest, DifferentFunctionsDifferentNames) {
  auto AST = tooling::buildASTFromCode(R"(
    void foo() {}
    void bar() {}
  )");
  auto &Ctx = AST->getASTContext();

  const auto *Foo = findDecl<FunctionDecl>(Ctx, "foo");
  const auto *Bar = findDecl<FunctionDecl>(Ctx, "bar");
  ASSERT_NE(Foo, nullptr);
  ASSERT_NE(Bar, nullptr);

  auto FooName = getEntityName(Foo);
  auto BarName = getEntityName(Bar);
  ASSERT_TRUE(FooName.has_value());
  ASSERT_TRUE(BarName.has_value());

  EXPECT_NE(*FooName, *BarName);
}

// Redeclaration tests

TEST(ASTEntityMappingTest, FunctionRedeclaration) {
  auto AST = tooling::buildASTFromCode(R"(
    void foo();
    void foo() {}
  )");
  auto &Ctx = AST->getASTContext();

  auto Matcher = functionDecl(hasName("foo")).bind("decl");
  auto Matches = match(Matcher, Ctx);
  ASSERT_GE(Matches.size(), 2u);

  const auto *FirstDecl = Matches[0].getNodeAs<FunctionDecl>("decl");
  ASSERT_NE(FirstDecl, nullptr);

  auto FirstName = getEntityName(FirstDecl);
  ASSERT_TRUE(FirstName.has_value());

  for (size_t I = 1; I < Matches.size(); ++I) {
    const auto *Decl = Matches[I].getNodeAs<FunctionDecl>("decl");
    ASSERT_NE(Decl, nullptr);

    auto Name = getEntityName(Decl);
    ASSERT_TRUE(Name.has_value());
    EXPECT_EQ(*FirstName, *Name);
  }
}

TEST(ASTEntityMappingTest, VarRedeclaration) {
  auto AST = tooling::buildASTFromCode(R"(
    extern int x;
    int x = 42;
  )");
  auto &Ctx = AST->getASTContext();

  auto Matcher = varDecl(hasName("x")).bind("decl");
  auto Matches = match(Matcher, Ctx);
  ASSERT_EQ(Matches.size(), 2u);

  const auto *FirstDecl = Matches[0].getNodeAs<VarDecl>("decl");
  ASSERT_NE(FirstDecl, nullptr);

  auto FirstName = getEntityName(FirstDecl);
  ASSERT_TRUE(FirstName.has_value());

  for (size_t I = 1; I < Matches.size(); ++I) {
    const auto *Decl = Matches[I].getNodeAs<VarDecl>("decl");
    ASSERT_NE(Decl, nullptr);

    auto Name = getEntityName(Decl);
    ASSERT_TRUE(Name.has_value());
    EXPECT_EQ(*FirstName, *Name);
  }
}

TEST(ASTEntityMappingTest, RecordRedeclaration) {
  auto AST = tooling::buildASTFromCode(R"(
    struct S;
    struct S {};
  )");
  auto &Ctx = AST->getASTContext();

  auto Matcher = recordDecl(hasName("S"), unless(isImplicit())).bind("decl");
  auto Matches = match(Matcher, Ctx);
  ASSERT_GE(Matches.size(), 2u);

  const auto *FirstDecl = Matches[0].getNodeAs<RecordDecl>("decl");
  ASSERT_NE(FirstDecl, nullptr);

  auto FirstName = getEntityName(FirstDecl);
  ASSERT_TRUE(FirstName.has_value());

  for (size_t I = 1; I < Matches.size(); ++I) {
    const auto *Decl = Matches[I].getNodeAs<RecordDecl>("decl");
    ASSERT_NE(Decl, nullptr);

    auto Name = getEntityName(Decl);
    ASSERT_TRUE(Name.has_value());
    EXPECT_EQ(*FirstName, *Name);
  }
}

TEST(ASTEntityMappingTest, ParmVarDeclRedeclaration) {
  auto AST = tooling::buildASTFromCode(R"(
    void foo(int);
    void foo(int x);
    void foo(int y);
    void foo(int x) {}
  )");
  auto &Ctx = AST->getASTContext();

  auto Matcher = functionDecl(hasName("foo")).bind("decl");
  auto Matches = match(Matcher, Ctx);
  ASSERT_GE(Matches.size(), 2u);

  const auto *FirstFuncDecl = Matches[0].getNodeAs<FunctionDecl>("decl");
  ASSERT_NE(FirstFuncDecl, nullptr);
  ASSERT_GT(FirstFuncDecl->param_size(), 0u);

  auto ParamEName = getEntityName(FirstFuncDecl->getParamDecl(0));
  ASSERT_TRUE(ParamEName.has_value());

  for (size_t I = 1; I < Matches.size(); ++I) {
    const auto *FDecl = Matches[I].getNodeAs<FunctionDecl>("decl");
    ASSERT_NE(FDecl, nullptr);
    ASSERT_GT(FDecl->param_size(), 0u);

    auto ParamRedeclEName = getEntityName(FDecl->getParamDecl(0));
    EXPECT_EQ(*ParamEName, *ParamRedeclEName);
  }
}

TEST(ASTEntityMappingTest, FunctionReturnRedeclaration) {
  auto AST = tooling::buildASTFromCode(R"(
    int foo();
    int foo() { return 42; }
  )");
  auto &Ctx = AST->getASTContext();

  auto Matcher = functionDecl(hasName("foo")).bind("decl");
  auto Matches = match(Matcher, Ctx);
  ASSERT_EQ(Matches.size(), 2u);

  const auto *Decl1 = Matches[0].getNodeAs<FunctionDecl>("decl");
  ASSERT_NE(Decl1, nullptr);
  auto Name1 = getEntityNameForReturn(Decl1);
  ASSERT_TRUE(Name1.has_value());

  for (size_t I = 1; I < Matches.size(); ++I) {
    const auto *FDecl = Matches[I].getNodeAs<FunctionDecl>("decl");
    ASSERT_NE(FDecl, nullptr);

    auto Name = getEntityNameForReturn(Decl1);
    ASSERT_TRUE(Name.has_value());
    EXPECT_EQ(*Name1, *Name);
  }
}

} // namespace
} // namespace clang::ssaf
