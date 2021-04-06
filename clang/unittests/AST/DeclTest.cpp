//===- unittests/AST/DeclTest.cpp --- Declaration tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Decl nodes in the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/DataLayout.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

TEST(Decl, CleansUpAPValues) {
  MatchFinder Finder;
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));

  // This is a regression test for a memory leak in APValues for structs that
  // allocate memory. This test only fails if run under valgrind with full leak
  // checking enabled.
  std::vector<std::string> Args(1, "-std=c++11");
  Args.push_back("-fno-ms-extensions");
  ASSERT_TRUE(runToolOnCodeWithArgs(
      Factory->create(),
      "struct X { int a; }; constexpr X x = { 42 };"
      "union Y { constexpr Y(int a) : a(a) {} int a; }; constexpr Y y = { 42 };"
      "constexpr int z[2] = { 42, 43 };"
      "constexpr int __attribute__((vector_size(16))) v1 = {};"
      "\n#ifdef __SIZEOF_INT128__\n"
      "constexpr __uint128_t large_int = 0xffffffffffffffff;"
      "constexpr __uint128_t small_int = 1;"
      "\n#endif\n"
      "constexpr double d1 = 42.42;"
      "constexpr long double d2 = 42.42;"
      "constexpr _Complex long double c1 = 42.0i;"
      "constexpr _Complex long double c2 = 42.0;"
      "template<int N> struct A : A<N-1> {};"
      "template<> struct A<0> { int n; }; A<50> a;"
      "constexpr int &r = a.n;"
      "constexpr int A<50>::*p = &A<50>::n;"
      "void f() { foo: bar: constexpr int k = __builtin_constant_p(0) ?"
      "                         (char*)&&foo - (char*)&&bar : 0; }",
      Args));

  // FIXME: Once this test starts breaking we can test APValue::needsCleanup
  // for ComplexInt.
  ASSERT_FALSE(runToolOnCodeWithArgs(
      Factory->create(),
      "constexpr _Complex __uint128_t c = 0xffffffffffffffff;",
      Args));
}

TEST(Decl, AsmLabelAttr) {
  // Create two method decls: `f` and `g`.
  StringRef Code = R"(
    struct S {
      void f() {}
      void g() {}
    };
  )";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-target", "i386-apple-darwin"});
  ASTContext &Ctx = AST->getASTContext();
  assert(Ctx.getTargetInfo().getDataLayout().getGlobalPrefix() &&
         "Expected target to have a global prefix");
  DiagnosticsEngine &Diags = AST->getDiagnostics();

  const auto *DeclS =
      selectFirst<CXXRecordDecl>("d", match(cxxRecordDecl().bind("d"), Ctx));
  NamedDecl *DeclF = *DeclS->method_begin();
  NamedDecl *DeclG = *(++DeclS->method_begin());

  // Attach asm labels to the decls: one literal, and one not.
  DeclF->addAttr(::new (Ctx) AsmLabelAttr(Ctx, SourceLocation(), "foo",
                                          /*LiteralLabel=*/true));
  DeclG->addAttr(::new (Ctx) AsmLabelAttr(Ctx, SourceLocation(), "goo",
                                          /*LiteralLabel=*/false));

  // Mangle the decl names.
  std::string MangleF, MangleG;
  std::unique_ptr<ItaniumMangleContext> MC(
      ItaniumMangleContext::create(Ctx, Diags));
  {
    llvm::raw_string_ostream OS_F(MangleF);
    llvm::raw_string_ostream OS_G(MangleG);
    MC->mangleName(DeclF, OS_F);
    MC->mangleName(DeclG, OS_G);
  }

  ASSERT_TRUE(0 == MangleF.compare("\x01" "foo"));
  ASSERT_TRUE(0 == MangleG.compare("goo"));
}

TEST(Decl, MangleDependentSizedArray) {
  StringRef Code = R"(
    template <int ...N>
    int A[] = {N...};

    template <typename T, int N>
    struct S {
      T B[N];
    };
  )";
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code, {"-target", "i386-apple-darwin"});
  ASTContext &Ctx = AST->getASTContext();
  assert(Ctx.getTargetInfo().getDataLayout().getGlobalPrefix() &&
         "Expected target to have a global prefix");
  DiagnosticsEngine &Diags = AST->getDiagnostics();

  const auto *DeclA =
      selectFirst<VarDecl>("A", match(varDecl().bind("A"), Ctx));
  const auto *DeclB =
      selectFirst<FieldDecl>("B", match(fieldDecl().bind("B"), Ctx));

  std::string MangleA, MangleB;
  llvm::raw_string_ostream OS_A(MangleA), OS_B(MangleB);
  std::unique_ptr<ItaniumMangleContext> MC(
      ItaniumMangleContext::create(Ctx, Diags));

  MC->mangleTypeName(DeclA->getType(), OS_A);
  MC->mangleTypeName(DeclB->getType(), OS_B);

  ASSERT_TRUE(0 == MangleA.compare("_ZTSA_i"));
  ASSERT_TRUE(0 == MangleB.compare("_ZTSAT0__T_"));
}
