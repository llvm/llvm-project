//===- AnyCallTest.cpp - AnyCall unit tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnyCall.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace {

using namespace ast_matchers;

std::unique_ptr<ASTUnit> buildAST(llvm::StringRef Code,
                                  std::vector<std::string> Args = {
                                      "-fsyntax-only", "-std=c++17"}) {
  return tooling::buildASTFromCodeWithArgs(Code, Args);
}

const IntegerLiteral *asIntegerLiteral(const Expr *E) {
  return dyn_cast<IntegerLiteral>(E->IgnoreImplicit());
}

void expectIntegerArguments(const AnyCall &Call,
                            std::initializer_list<int> Expected) {
  ASSERT_EQ(Call.arg_size(), Expected.size());
  EXPECT_FALSE(Call.arg_empty());
  EXPECT_EQ(Call.arguments()[0], Call.getArg(0));
  EXPECT_EQ(*Call.arg_begin(), Call.getArg(0));

  unsigned Index = 0;
  for (int ExpectedValue : Expected) {
    const auto *Arg = asIntegerLiteral(Call.getArg(Index));
    ASSERT_NE(Arg, nullptr);
    EXPECT_EQ(Arg->getValue(), ExpectedValue);
    ++Index;
  }
}

void expectNoArguments(const AnyCall &Call) {
  EXPECT_TRUE(Call.arg_empty());
  EXPECT_EQ(Call.arg_size(), 0u);
  EXPECT_EQ(Call.arg_begin(), Call.arg_end());
}

TEST(AnyCallTest, ExposesFunctionParametersAndArguments) {
  auto AST = buildAST(R"cpp(
    void callee(int first, int second);
    void target() { callee(1, 2); }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *CE = selectFirst<CallExpr>(
      "call",
      match(callExpr(callee(functionDecl(hasName("callee")))).bind("call"),
            Ctx));
  ASSERT_NE(CE, nullptr);

  AnyCall Call(CE);
  const auto *Callee = cast<FunctionDecl>(Call.getDecl());
  EXPECT_FALSE(Call.param_empty());
  ASSERT_EQ(Call.param_size(), 2u);
  EXPECT_EQ(Call.parameters()[0], Callee->getParamDecl(0));
  EXPECT_EQ(*Call.param_begin(), Callee->getParamDecl(0));

  expectIntegerArguments(Call, {1, 2});
}

TEST(AnyCallTest, ExposesBlockCallArguments) {
  auto AST = buildAST(R"cpp(
    void target() {
      void (^block)(int, int) = ^(int, int) {};
      block(3, 4);
    }
  )cpp",
                      {"-fsyntax-only", "-std=c++17", "-fblocks"});
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *CE = selectFirst<CallExpr>(
      "call", match(callExpr(callee(expr(hasType(blockPointerType()))),
                             argumentCountIs(2))
                        .bind("call"),
                    Ctx));
  ASSERT_NE(CE, nullptr);

  AnyCall Call(CE);
  EXPECT_EQ(Call.getKind(), AnyCall::Block);
  expectIntegerArguments(Call, {3, 4});
}

TEST(AnyCallTest, ExposesObjCMethodArguments) {
  auto AST = buildAST(R"objc(
    @interface Receiver
    - (void)method:(int)x second:(int)y;
    @end

    void target(Receiver *R) {
      [R method:5 second:6];
    }
  )objc",
                      {"-fsyntax-only", "-x", "objective-c++", "-std=c++17"});
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *ME = selectFirst<ObjCMessageExpr>(
      "message",
      match(objcMessageExpr(callee(objcMethodDecl(hasName("method:second:"))),
                            argumentCountIs(2))
                .bind("message"),
            Ctx));
  ASSERT_NE(ME, nullptr);

  AnyCall Call(ME);
  EXPECT_EQ(Call.getKind(), AnyCall::ObjCMethod);
  expectIntegerArguments(Call, {5, 6});
}

TEST(AnyCallTest, ExposesConstructorArguments) {
  auto AST = buildAST(R"cpp(
    struct Widget {
      Widget(int, int);
    };
    void target() { Widget W(3, 4); }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *CtorExpr = selectFirst<CXXConstructExpr>(
      "ctor", match(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                                         ofClass(hasName("Widget")))),
                                     argumentCountIs(2))
                        .bind("ctor"),
                    Ctx));
  ASSERT_NE(CtorExpr, nullptr);

  AnyCall Call(CtorExpr);
  expectIntegerArguments(Call, {3, 4});
}

TEST(AnyCallTest, AllocatorCallsHaveNoArguments) {
  auto AST = buildAST(R"cpp(
    void target() {
      int *P = new int(5);
    }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *NE =
      selectFirst<CXXNewExpr>("new", match(cxxNewExpr().bind("new"), Ctx));
  ASSERT_NE(NE, nullptr);

  AnyCall Call(NE);
  EXPECT_EQ(Call.getKind(), AnyCall::Allocator);
  expectNoArguments(Call);
}

TEST(AnyCallTest, DeallocatorCallsHaveNoArguments) {
  auto AST = buildAST(R"cpp(
    void target(int *P) {
      delete P;
    }
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *DE = selectFirst<CXXDeleteExpr>(
      "delete", match(cxxDeleteExpr().bind("delete"), Ctx));
  ASSERT_NE(DE, nullptr);

  AnyCall Call(DE);
  EXPECT_EQ(Call.getKind(), AnyCall::Deallocator);
  expectNoArguments(Call);
}

TEST(AnyCallTest, InheritedConstructorCallsHaveNoArguments) {
  auto AST = buildAST(R"cpp(
    struct Base {
      Base(int) {}
    };
    struct Derived : Base {
      using Base::Base;
    };

    Derived D = Derived(0);
  )cpp",
                      {"-fsyntax-only", "-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *InheritedCtorInit = selectFirst<CXXInheritedCtorInitExpr>(
      "init",
      match(cxxConstructorDecl(hasAnyConstructorInitializer(
                cxxCtorInitializer(withInitializer(expr().bind("init"))))),
            Ctx));
  ASSERT_NE(InheritedCtorInit, nullptr);

  AnyCall Call(InheritedCtorInit);
  EXPECT_EQ(Call.getKind(), AnyCall::InheritedConstructor);
  expectNoArguments(Call);
}

TEST(AnyCallTest, DestructorDeclarationsHaveNoArguments) {
  auto AST = buildAST(R"cpp(
    struct Widget {
      ~Widget();
    };
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *Destructor = selectFirst<CXXDestructorDecl>(
      "destructor",
      match(cxxDestructorDecl(ofClass(hasName("Widget"))).bind("destructor"),
            Ctx));
  ASSERT_NE(Destructor, nullptr);

  AnyCall Call(Destructor);
  EXPECT_EQ(Call.getKind(), AnyCall::Destructor);
  expectNoArguments(Call);
}

TEST(AnyCallTest, DeclarationBackedCallsHaveNoArguments) {
  auto AST = buildAST(R"cpp(
    void callee(int first, int second);
  )cpp");
  ASTContext &Ctx = AST->getASTContext();
  ASSERT_EQ(Ctx.getDiagnostics().getClient()->getNumErrors(), 0U);

  const auto *Callee = selectFirst<FunctionDecl>(
      "callee", match(functionDecl(hasName("callee")).bind("callee"), Ctx));
  ASSERT_NE(Callee, nullptr);

  AnyCall Call(Callee);
  EXPECT_EQ(Call.getKind(), AnyCall::Function);
  expectNoArguments(Call);
}

} // namespace
} // namespace clang
