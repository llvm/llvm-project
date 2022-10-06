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

#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Testing/Support/Annotations.h"
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
  assert(Ctx.getTargetInfo().getUserLabelPrefix() == StringRef("_") &&
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
  assert(Ctx.getTargetInfo().getUserLabelPrefix() == StringRef("_") &&
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

TEST(Decl, EnumDeclRange) {
  llvm::Annotations Code(R"(
    typedef int Foo;
    [[enum Bar : Foo]];)");
  auto AST = tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{});
  ASTContext &Ctx = AST->getASTContext();
  const auto &SM = Ctx.getSourceManager();

  const auto *Bar =
      selectFirst<TagDecl>("Bar", match(enumDecl().bind("Bar"), Ctx));
  auto BarRange =
      Lexer::getAsCharRange(Bar->getSourceRange(), SM, Ctx.getLangOpts());
  EXPECT_EQ(SM.getFileOffset(BarRange.getBegin()), Code.range().Begin);
  EXPECT_EQ(SM.getFileOffset(BarRange.getEnd()), Code.range().End);
}

TEST(Decl, IsInExportDeclContext) {
  llvm::Annotations Code(R"(
    export module m;
    export template <class T>
    void f() {})");
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  const auto *f =
      selectFirst<FunctionDecl>("f", match(functionDecl().bind("f"), Ctx));
  EXPECT_TRUE(f->isInExportDeclContext());
}

TEST(Decl, InConsistLinkageForTemplates) {
  llvm::Annotations Code(R"(
    export module m;
    export template <class T>
    void f() {}

    template <>
    void f<int>() {}

    export template <class T>
    class C {};

    template<>
    class C<int> {};
    )");

  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  llvm::SmallVector<ast_matchers::BoundNodes, 2> Funcs =
      match(functionDecl().bind("f"), Ctx);

  EXPECT_EQ(Funcs.size(), 2U);
  const FunctionDecl *TemplateF = Funcs[0].getNodeAs<FunctionDecl>("f");
  const FunctionDecl *SpecializedF = Funcs[1].getNodeAs<FunctionDecl>("f");
  EXPECT_EQ(TemplateF->getLinkageInternal(),
            SpecializedF->getLinkageInternal());

  llvm::SmallVector<ast_matchers::BoundNodes, 1> ClassTemplates =
      match(classTemplateDecl().bind("C"), Ctx);
  llvm::SmallVector<ast_matchers::BoundNodes, 1> ClassSpecializations =
      match(classTemplateSpecializationDecl().bind("C"), Ctx);

  EXPECT_EQ(ClassTemplates.size(), 1U);
  EXPECT_EQ(ClassSpecializations.size(), 1U);
  const NamedDecl *TemplatedC = ClassTemplates[0].getNodeAs<NamedDecl>("C");
  const NamedDecl *SpecializedC = ClassSpecializations[0].getNodeAs<NamedDecl>("C");
  EXPECT_EQ(TemplatedC->getLinkageInternal(),
            SpecializedC->getLinkageInternal());
}

TEST(Decl, ModuleAndInternalLinkage) {
  llvm::Annotations Code(R"(
    export module M;
    static int a;
    static int f(int x);

    int b;
    int g(int x);)");

  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  const auto *a =
      selectFirst<VarDecl>("a", match(varDecl(hasName("a")).bind("a"), Ctx));
  const auto *f = selectFirst<FunctionDecl>(
      "f", match(functionDecl(hasName("f")).bind("f"), Ctx));

  EXPECT_EQ(a->getLinkageInternal(), InternalLinkage);
  EXPECT_EQ(f->getLinkageInternal(), InternalLinkage);

  const auto *b =
      selectFirst<VarDecl>("b", match(varDecl(hasName("b")).bind("b"), Ctx));
  const auto *g = selectFirst<FunctionDecl>(
      "g", match(functionDecl(hasName("g")).bind("g"), Ctx));

  EXPECT_EQ(b->getLinkageInternal(), ModuleLinkage);
  EXPECT_EQ(g->getLinkageInternal(), ModuleLinkage);

  AST = tooling::buildASTFromCodeWithArgs(
      Code.code(), /*Args=*/{"-std=c++20", "-fmodules-ts"});
  ASTContext &CtxTS = AST->getASTContext();
  a = selectFirst<VarDecl>("a", match(varDecl(hasName("a")).bind("a"), CtxTS));
  f = selectFirst<FunctionDecl>(
      "f", match(functionDecl(hasName("f")).bind("f"), CtxTS));

  EXPECT_EQ(a->getLinkageInternal(), ModuleInternalLinkage);
  EXPECT_EQ(f->getLinkageInternal(), ModuleInternalLinkage);

  b = selectFirst<VarDecl>("b", match(varDecl(hasName("b")).bind("b"), CtxTS));
  g = selectFirst<FunctionDecl>(
      "g", match(functionDecl(hasName("g")).bind("g"), CtxTS));

  EXPECT_EQ(b->getLinkageInternal(), ModuleLinkage);
  EXPECT_EQ(g->getLinkageInternal(), ModuleLinkage);
}

TEST(Decl, GetNonTransparentDeclContext) {
  llvm::Annotations Code(R"(
    export module m3;
    export template <class> struct X {
      template <class Self> friend void f(Self &&self) {
        (Self&)self;
      }
    };)");

  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  auto *f = selectFirst<FunctionDecl>(
      "f", match(functionDecl(hasName("f")).bind("f"), Ctx));

  EXPECT_TRUE(f->getNonTransparentDeclContext()->isFileContext());
}

TEST(Decl, MemberFunctionInModules) {
  llvm::Annotations Code(R"(
    module;
    class G {
      void bar() {}
    };
    export module M;
    class A {
      void foo() {}
    };
    )");

  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.code(), /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  auto *foo = selectFirst<FunctionDecl>(
      "foo", match(functionDecl(hasName("foo")).bind("foo"), Ctx));

  // The function defined within a class definition is not implicitly inline
  // if it is not attached to global module
  EXPECT_FALSE(foo->isInlined());

  auto *bar = selectFirst<FunctionDecl>(
      "bar", match(functionDecl(hasName("bar")).bind("bar"), Ctx));

  // In global module, the function defined within a class definition is
  // implicitly inline.
  EXPECT_TRUE(bar->isInlined());
}

TEST(Decl, MemberFunctionInHeaderUnit) {
  llvm::Annotations Code(R"(
    class foo {
    public:
      int memFn() {
        return 43;
      }
    };
    )");

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code.code(), {"-std=c++20", " -xc++-user-header ", "-emit-header-unit"});
  ASTContext &Ctx = AST->getASTContext();

  auto *memFn = selectFirst<FunctionDecl>(
      "memFn", match(functionDecl(hasName("memFn")).bind("memFn"), Ctx));

  EXPECT_TRUE(memFn->isInlined());
}

TEST(Decl, FriendFunctionWithinClassInHeaderUnit) {
  llvm::Annotations Code(R"(
    class foo {
      int value;
    public:
      foo(int v) : value(v) {}

      friend int getFooValue(foo f) {
        return f.value;
      }
    };
    )");

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code.code(), {"-std=c++20", " -xc++-user-header ", "-emit-header-unit"});
  ASTContext &Ctx = AST->getASTContext();

  auto *getFooValue = selectFirst<FunctionDecl>(
      "getFooValue",
      match(functionDecl(hasName("getFooValue")).bind("getFooValue"), Ctx));

  EXPECT_TRUE(getFooValue->isInlined());
}

TEST(Decl, NoProtoFunctionDeclAttributes) {
  llvm::Annotations Code(R"(
    void f();
    )");

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code.code(),
      /*Args=*/{"-target", "i386-apple-darwin", "-x", "objective-c",
                "-std=c89"});
  ASTContext &Ctx = AST->getASTContext();

  auto *f = selectFirst<FunctionDecl>(
      "f", match(functionDecl(hasName("f")).bind("f"), Ctx));

  const auto *FPT = f->getType()->getAs<FunctionNoProtoType>();

  // Functions without prototypes always have 0 initialized qualifiers
  EXPECT_FALSE(FPT->isConst());
  EXPECT_FALSE(FPT->isVolatile());
  EXPECT_FALSE(FPT->isRestrict());
}
