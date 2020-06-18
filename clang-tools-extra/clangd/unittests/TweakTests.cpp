//===-- TweakTests.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TweakTesting.h"
#include "refactor/Tweak.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <string>
#include <utility>
#include <vector>

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

MATCHER_P2(FileWithContents, FileName, Contents, "") {
  return arg.first() == FileName && arg.second == Contents;
}

TEST(FileEdits, AbsolutePath) {
  auto RelPaths = {"a.h", "foo.cpp", "test/test.cpp"};

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> MemFS(
      new llvm::vfs::InMemoryFileSystem);
  MemFS->setCurrentWorkingDirectory(testRoot());
  for (auto Path : RelPaths)
    MemFS->addFile(Path, 0, llvm::MemoryBuffer::getMemBuffer("", Path));
  FileManager FM(FileSystemOptions(), MemFS);
  DiagnosticsEngine DE(new DiagnosticIDs, new DiagnosticOptions);
  SourceManager SM(DE, FM);

  for (auto Path : RelPaths) {
    auto FID = SM.createFileID(*FM.getFile(Path), SourceLocation(),
                               clang::SrcMgr::C_User);
    auto Res = Tweak::Effect::fileEdit(SM, FID, tooling::Replacements());
    ASSERT_THAT_EXPECTED(Res, llvm::Succeeded());
    EXPECT_EQ(Res->first, testPath(Path));
  }
}

TWEAK_TEST(SwapIfBranches);
TEST_F(SwapIfBranchesTest, Test) {
  Context = Function;
  EXPECT_EQ(apply("^if (true) {return;} else {(void)0;}"),
            "if (true) {(void)0;} else {return;}");
  EXPECT_EQ(apply("^if (/*error-ok*/) {return;} else {(void)0;}"),
            "if (/*error-ok*/) {(void)0;} else {return;}")
      << "broken condition";
  EXPECT_AVAILABLE("^i^f^^(^t^r^u^e^) { return; } ^e^l^s^e^ { return; }");
  EXPECT_UNAVAILABLE("if (true) {^return ^;^ } else { ^return^;^ }");
  // Available in subexpressions of the condition;
  EXPECT_THAT("if(2 + [[2]] + 2) { return; } else {return;}", isAvailable());
  // But not as part of the branches.
  EXPECT_THAT("if(2 + 2 + 2) { [[return]]; } else { return; }",
              Not(isAvailable()));
  // Range covers the "else" token, so available.
  EXPECT_THAT("if(2 + 2 + 2) { return[[; } else {return;]]}", isAvailable());
  // Not available in compound statements in condition.
  EXPECT_THAT("if([]{return [[true]];}()) { return; } else { return; }",
              Not(isAvailable()));
  // Not available if both sides aren't braced.
  EXPECT_THAT("^if (1) return; else { return; }", Not(isAvailable()));
  // Only one if statement is supported!
  EXPECT_THAT("[[if(1){}else{}if(2){}else{}]]", Not(isAvailable()));
}

TWEAK_TEST(RawStringLiteral);
TEST_F(RawStringLiteralTest, Test) {
  Context = Expression;
  EXPECT_AVAILABLE(R"cpp(^"^f^o^o^\^n^")cpp");
  EXPECT_AVAILABLE(R"cpp(R"(multi )" ^"token " "str\ning")cpp");
  EXPECT_UNAVAILABLE(R"cpp(^"f^o^o^o")cpp"); // no chars need escaping
  EXPECT_UNAVAILABLE(R"cpp(R"(multi )" ^"token " u8"str\ning")cpp"); // nonascii
  EXPECT_UNAVAILABLE(R"cpp(^R^"^(^multi )" "token " "str\ning")cpp"); // raw
  EXPECT_UNAVAILABLE(R"cpp(^"token\n" __FILE__)cpp"); // chunk is macro
  EXPECT_UNAVAILABLE(R"cpp(^"a\r\n";)cpp");           // forbidden escape char

  const char *Input = R"cpp(R"(multi
token)" "\nst^ring\n" "literal")cpp";
  const char *Output = R"cpp(R"(multi
token
string
literal)")cpp";
  EXPECT_EQ(apply(Input), Output);
}

TWEAK_TEST(ObjCLocalizeStringLiteral);
TEST_F(ObjCLocalizeStringLiteralTest, Test) {
  ExtraArgs.push_back("-x");
  ExtraArgs.push_back("objective-c");

  // Ensure the action can be initiated in the string literal.
  EXPECT_AVAILABLE(R"(id x = ^[[@[[^"^t^est^"]]]];)");

  // Ensure that the action can't be initiated in other places.
  EXPECT_UNAVAILABLE(R"([[i^d ^[[x]] ^= @"test";^]])");

  // Ensure that the action is not available for regular C strings.
  EXPECT_UNAVAILABLE(R"(const char * x= "^test";)");

  const char *Input = R"(id x = [[@"test"]];)";
  const char *Output = R"(id x = NSLocalizedString(@"test", @"");)";
  EXPECT_EQ(apply(Input), Output);
}

TWEAK_TEST(DumpAST);
TEST_F(DumpASTTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_UNAVAILABLE("/*c^omment*/ int foo() { return 2 ^ + 2; }");
  EXPECT_THAT(apply("int x = 2 ^+ 2;"),
              AllOf(StartsWith("message:"), HasSubstr("BinaryOperator"),
                    HasSubstr("'+'"), HasSubstr("|-IntegerLiteral"),
                    HasSubstr("<col:9> 'int' 2\n`-IntegerLiteral"),
                    HasSubstr("<col:13> 'int' 2")));
}

TWEAK_TEST(DumpSymbol);
TEST_F(DumpSymbolTest, Test) {
  std::string ID = R"("id":"CA2EBE44A1D76D2A")";
  std::string USR = R"("usr":"c:@F@foo#")";
  EXPECT_THAT(apply("void f^oo();"),
              AllOf(StartsWith("message:"), testing::HasSubstr(ID),
                    testing::HasSubstr(USR)));
}

TWEAK_TEST(ShowSelectionTree);
TEST_F(ShowSelectionTreeTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_AVAILABLE("/*c^omment*/ int foo() { return 2 ^ + 2; }");

  const char *Output = R"(message:
 TranslationUnitDecl 
   VarDecl int x = fcall(2 + 2)
    .CallExpr fcall(2 + 2)
       ImplicitCastExpr fcall
        .DeclRefExpr fcall
      .BinaryOperator 2 + 2
        *IntegerLiteral 2
)";
  EXPECT_EQ(apply("int fcall(int); int x = fca[[ll(2 +]]2);"), Output);

  Output = R"(message:
 TranslationUnitDecl 
   FunctionDecl void x()
     CompoundStmt { …
       ForStmt for (;;) …
        *BreakStmt break;
)";
  EXPECT_EQ(apply("void x() { for (;;) br^eak; }"), Output);
}

TWEAK_TEST(DumpRecordLayout);
TEST_F(DumpRecordLayoutTest, Test) {
  EXPECT_AVAILABLE("^s^truct ^X ^{ int x; ^};");
  EXPECT_THAT("struct X { int ^a; };", Not(isAvailable()));
  EXPECT_THAT("struct ^X;", Not(isAvailable()));
  EXPECT_THAT("template <typename T> struct ^X { T t; };", Not(isAvailable()));
  EXPECT_THAT("enum ^X {};", Not(isAvailable()));

  EXPECT_THAT(apply("struct ^X { int x; int y; };"),
              AllOf(StartsWith("message:"), HasSubstr("0 |   int x")));
}

TWEAK_TEST(ExtractVariable);
TEST_F(ExtractVariableTest, Test) {
  const char *AvailableCases = R"cpp(
    int xyz(int a = 1) {
      struct T {
        int bar(int a = 1);
        int z;
      } t;
      // return statement
      return [[[[t.b[[a]]r]](t.z)]];
    }
    void f() {
      int a = [[5 +]] [[4 * [[[[xyz]]()]]]];
      // multivariable initialization
      if(1)
        int x = [[1]], y = [[a + 1]], a = [[1]], z = a + 1;
      // if without else
      if([[1]])
        a = [[1]];
      // if with else
      if(a < [[3]])
        if(a == [[4]])
          a = [[5]];
        else
          a = [[5]];
      else if (a < [[4]])
        a = [[4]];
      else
        a = [[5]];
      // for loop
      for(a = [[1]]; a > [[[[3]] + [[4]]]]; a++)
        a = [[2]];
      // while
      while(a < [[1]])
        a = [[1]];
      // do while
      do
        a = [[1]];
      while(a < [[3]]);
    }
  )cpp";
  EXPECT_AVAILABLE(AvailableCases);

  ExtraArgs = {"-xc"};
  const char *AvailableButC = R"cpp(
    void foo() {
      int x = [[1]];
    })cpp";
  EXPECT_UNAVAILABLE(AvailableButC);
  ExtraArgs = {};

  const char *NoCrashCases = R"cpp(
    // error-ok: broken code, but shouldn't crash
    template<typename T, typename ...Args>
    struct Test<T, Args...> {
    Test(const T &v) :val[[(^]]) {}
      T val;
    };
  )cpp";
  EXPECT_UNAVAILABLE(NoCrashCases);

  const char *UnavailableCases = R"cpp(
    int xyz(int a = [[1]]) {
      struct T {
        int bar(int a = [[1]]);
        int z = [[1]];
      } t;
      return [[t]].bar([[[[t]].z]]);
    }
    void v() { return; }
    // function default argument
    void f(int b = [[1]]) {
      // empty selection
      int a = ^1 ^+ ^2;
      // void expressions
      auto i = new int, j = new int;
      [[[[delete i]], delete j]];
      [[v]]();
      // if
      if(1)
        int x = 1, y = a + 1, a = 1, z = [[a + 1]];
      if(int a = 1)
        if([[a + 1]] == 4)
          a = [[[[a]] +]] 1;
      // for loop
      for(int a = 1, b = 2, c = 3; a > [[b + c]]; [[a++]])
        a = [[a + 1]];
      // lambda
      auto lamb = [&[[a]], &[[b]]](int r = [[1]]) {return 1;};
      // assignment
      xyz([[a = 5]]);
      xyz([[a *= 5]]);
      // Variable DeclRefExpr
      a = [[b]];
      // statement expression
      [[xyz()]];
      while (a)
        [[++a]];
      // label statement
      goto label;
      label:
        a = [[1]];
    }
  )cpp";
  EXPECT_UNAVAILABLE(UnavailableCases);

  // vector of pairs of input and output strings
  const std::vector<std::pair<std::string, std::string>>
      InputOutputs = {
          // extraction from variable declaration/assignment
          {R"cpp(void varDecl() {
                   int a = 5 * (4 + (3 [[- 1)]]);
                 })cpp",
           R"cpp(void varDecl() {
                   auto dummy = (3 - 1); int a = 5 * (4 + dummy);
                 })cpp"},
          // FIXME: extraction from switch case
          /*{R"cpp(void f(int a) {
                   if(1)
                     while(a < 1)
                       switch (1) {
                           case 1:
                             a = [[1 + 2]];
                             break;
                           default:
                             break;
                       }
                 })cpp",
           R"cpp(void f(int a) {
                   auto dummy = 1 + 2; if(1)
                     while(a < 1)
                       switch (1) {
                           case 1:
                             a = dummy;
                             break;
                           default:
                             break;
                       }
                 })cpp"},*/
          // Macros
          {R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   int y = PLUS([[1+a]]);
                 })cpp",
           /*FIXME: It should be extracted like this.
            R"cpp(#define PLUS(x) x++
                  void f(int a) {
                    auto dummy = 1+a; int y = PLUS(dummy);
                  })cpp"},*/
           R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   auto dummy = PLUS(1+a); int y = dummy;
                 })cpp"},
          // ensure InsertionPoint isn't inside a macro
          {R"cpp(#define LOOP(x) while (1) {a = x;}
                 void f(int a) {
                   if(1)
                    LOOP(5 + [[3]])
                 })cpp",
            R"cpp(#define LOOP(x) while (1) {a = x;}
                 void f(int a) {
                   auto dummy = 3; if(1)
                    LOOP(5 + dummy)
                 })cpp"},
          {R"cpp(#define LOOP(x) do {x;} while(1);
                 void f(int a) {
                   if(1)
                    LOOP(5 + [[3]])
                 })cpp",
           R"cpp(#define LOOP(x) do {x;} while(1);
                 void f(int a) {
                   auto dummy = 3; if(1)
                    LOOP(5 + dummy)
                 })cpp"},
          // attribute testing
          {R"cpp(void f(int a) {
                    [ [gsl::suppress("type")] ] for (;;) a = [[1]];
                 })cpp",
           R"cpp(void f(int a) {
                    auto dummy = 1; [ [gsl::suppress("type")] ] for (;;) a = dummy;
                 })cpp"},
          // MemberExpr
          {R"cpp(class T {
                   T f() {
                     return [[T().f()]].f();
                   }
                 };)cpp",
           R"cpp(class T {
                   T f() {
                     auto dummy = T().f(); return dummy.f();
                   }
                 };)cpp"},
          // Function DeclRefExpr
          {R"cpp(int f() {
                   return [[f]]();
                 })cpp",
           R"cpp(int f() {
                   auto dummy = f(); return dummy;
                 })cpp"},
          // FIXME: Wrong result for \[\[clang::uninitialized\]\] int b = [[1]];
          // since the attr is inside the DeclStmt and the bounds of
          // DeclStmt don't cover the attribute.

          // Binary subexpressions
          {R"cpp(void f() {
                   int x = 1 + [[2 + 3 + 4]] + 5;
                 })cpp",
           R"cpp(void f() {
                   auto dummy = 2 + 3 + 4; int x = 1 + dummy + 5;
                 })cpp"},
          {R"cpp(void f() {
                   int x = [[1 + 2 + 3]] + 4 + 5;
                 })cpp",
           R"cpp(void f() {
                   auto dummy = 1 + 2 + 3; int x = dummy + 4 + 5;
                 })cpp"},
          {R"cpp(void f() {
                   int x = 1 + 2 + [[3 + 4 + 5]];
                 })cpp",
           R"cpp(void f() {
                   auto dummy = 3 + 4 + 5; int x = 1 + 2 + dummy;
                 })cpp"},
          // Non-associative operations have no special support
          {R"cpp(void f() {
                   int x = 1 - [[2 - 3 - 4]] - 5;
                 })cpp",
           R"cpp(void f() {
                   auto dummy = 1 - 2 - 3 - 4; int x = dummy - 5;
                 })cpp"},
          // A mix of associative operators isn't associative.
          {R"cpp(void f() {
                   int x = 0 + 1 * [[2 + 3]] * 4 + 5;
                 })cpp",
           R"cpp(void f() {
                   auto dummy = 1 * 2 + 3 * 4; int x = 0 + dummy + 5;
                 })cpp"},
          // Overloaded operators are supported, we assume associativity
          // as if they were built-in.
          {R"cpp(struct S {
                   S(int);
                 };
                 S operator+(S, S);

                 void f() {
                   S x = S(1) + [[S(2) + S(3) + S(4)]] + S(5);
                 })cpp",
           R"cpp(struct S {
                   S(int);
                 };
                 S operator+(S, S);

                 void f() {
                   auto dummy = S(2) + S(3) + S(4); S x = S(1) + dummy + S(5);
                 })cpp"},
          // Don't try to analyze across macro boundaries
          // FIXME: it'd be nice to do this someday (in a safe way)
          {R"cpp(#define ECHO(X) X
                 void f() {
                   int x = 1 + [[ECHO(2 + 3) + 4]] + 5;
                 })cpp",
           R"cpp(#define ECHO(X) X
                 void f() {
                   auto dummy = 1 + ECHO(2 + 3) + 4; int x = dummy + 5;
                 })cpp"},
          {R"cpp(#define ECHO(X) X
                 void f() {
                   int x = 1 + [[ECHO(2) + ECHO(3) + 4]] + 5;
                 })cpp",
           R"cpp(#define ECHO(X) X
                 void f() {
                   auto dummy = 1 + ECHO(2) + ECHO(3) + 4; int x = dummy + 5;
                 })cpp"},
      };
  for (const auto &IO : InputOutputs) {
    EXPECT_EQ(IO.second, apply(IO.first)) << IO.first;
  }
}

TWEAK_TEST(AnnotateHighlightings);
TEST_F(AnnotateHighlightingsTest, Test) {
  EXPECT_AVAILABLE("^vo^id^ ^f(^) {^}^"); // available everywhere.
  EXPECT_AVAILABLE("[[int a; int b;]]");
  EXPECT_EQ("void /* entity.name.function.cpp */f() {}", apply("void ^f() {}"));

  EXPECT_EQ(apply("[[void f1(); void f2();]]"),
            "void /* entity.name.function.cpp */f1(); "
            "void /* entity.name.function.cpp */f2();");

  EXPECT_EQ(apply("void f1(); void f2() {^}"),
            "void f1(); "
            "void /* entity.name.function.cpp */f2() {}");
}

TWEAK_TEST(ExpandMacro);
TEST_F(ExpandMacroTest, Test) {
  Header = R"cpp(
    // error-ok: not real c++, just token manipulation
    #define FOO 1 2 3
    #define FUNC(X) X+X+X
    #define EMPTY
    #define EMPTY_FN(X)
  )cpp";

  // Available on macro names, not available anywhere else.
  EXPECT_AVAILABLE("^F^O^O^ BAR ^F^O^O^");
  EXPECT_AVAILABLE("^F^U^N^C^(1)");
  EXPECT_UNAVAILABLE("^#^d^efine^ ^XY^Z 1 ^2 ^3^");
  EXPECT_UNAVAILABLE("FOO ^B^A^R^ FOO ^");
  EXPECT_UNAVAILABLE("FUNC(^1^)^");

  // Works as expected on object-like macros.
  EXPECT_EQ(apply("^FOO BAR FOO"), "1 2 3 BAR FOO");
  EXPECT_EQ(apply("FOO BAR ^FOO"), "FOO BAR 1 2 3");
  // And function-like macros.
  EXPECT_EQ(apply("F^UNC(2)"), "2 + 2 + 2");

  // Works on empty macros.
  EXPECT_EQ(apply("int a ^EMPTY;"), "int a ;");
  EXPECT_EQ(apply("int a ^EMPTY_FN(1 2 3);"), "int a ;");
  EXPECT_EQ(apply("int a = 123 ^EMPTY EMPTY_FN(1);"),
            "int a = 123  EMPTY_FN(1);");
  EXPECT_EQ(apply("int a = 123 ^EMPTY_FN(1) EMPTY;"), "int a = 123  EMPTY;");
  EXPECT_EQ(apply("int a = 123 EMPTY_FN(1) ^EMPTY;"),
            "int a = 123 EMPTY_FN(1) ;");
}

TWEAK_TEST(ExpandAutoType);
TEST_F(ExpandAutoTypeTest, Test) {

  Header = R"cpp(
    namespace ns {
      struct Class {
        struct Nested {};
      };
      void Func();
    }
    inline namespace inl_ns {
      namespace {
        struct Visible {};
      }
    }
  )cpp";

  EXPECT_AVAILABLE("^a^u^t^o^ i = 0;");
  EXPECT_UNAVAILABLE("auto ^i^ ^=^ ^0^;^");

  // check primitive type
  EXPECT_EQ(apply("[[auto]] i = 0;"), "int i = 0;");
  EXPECT_EQ(apply("au^to i = 0;"), "int i = 0;");
  // check classes and namespaces
  EXPECT_EQ(apply("^auto C = ns::Class::Nested();"),
            "ns::Class::Nested C = ns::Class::Nested();");
  // check that namespaces are shortened
  EXPECT_EQ(apply("namespace ns { void f() { ^auto C = Class(); } }"),
            "namespace ns { void f() { Class C = Class(); } }");
  // undefined functions should not be replaced
  EXPECT_THAT(apply("au^to x = doesnt_exist(); // error-ok"),
              StartsWith("fail: Could not deduce type for 'auto' type"));
  // function pointers should not be replaced
  EXPECT_THAT(apply("au^to x = &ns::Func;"),
              StartsWith("fail: Could not expand type of function pointer"));
  // lambda types are not replaced
  EXPECT_THAT(apply("au^to x = []{};"),
              StartsWith("fail: Could not expand type of lambda expression"));
  // inline namespaces
  EXPECT_EQ(apply("au^to x = inl_ns::Visible();"),
            "Visible x = inl_ns::Visible();");
  // local class
  EXPECT_EQ(apply("namespace x { void y() { struct S{}; ^auto z = S(); } }"),
            "namespace x { void y() { struct S{}; S z = S(); } }");
  // replace array types
  EXPECT_EQ(apply(R"cpp(au^to x = "test";)cpp"),
            R"cpp(const char * x = "test";)cpp");

  EXPECT_UNAVAILABLE("dec^ltype(au^to) x = 10;");

  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  // unknown types in a template should not be replaced
  EXPECT_THAT(apply("template <typename T> void x() { ^auto y = T::z(); }"),
              StartsWith("fail: Could not deduce type for 'auto' type"));
}

TWEAK_TEST(ExtractFunction);
TEST_F(ExtractFunctionTest, FunctionTest) {
  Context = Function;

  // Root statements should have common parent.
  EXPECT_EQ(apply("for(;;) [[1+2; 1+2;]]"), "unavailable");
  // Expressions aren't extracted.
  EXPECT_EQ(apply("int x = 0; [[x++;]]"), "unavailable");
  // We don't support extraction from lambdas.
  EXPECT_EQ(apply("auto lam = [](){ [[int x;]] }; "), "unavailable");
  // Partial statements aren't extracted.
  EXPECT_THAT(apply("int [[x = 0]];"), "unavailable");

  // Ensure that end of Zone and Beginning of PostZone being adjacent doesn't
  // lead to break being included in the extraction zone.
  EXPECT_THAT(apply("for(;;) { [[int x;]]break; }"), HasSubstr("extracted"));
  // FIXME: ExtractFunction should be unavailable inside loop construct
  // initializer/condition.
  EXPECT_THAT(apply(" for([[int i = 0;]];);"), HasSubstr("extracted"));
  // Don't extract because needs hoisting.
  EXPECT_THAT(apply(" [[int a = 5;]] a++; "), StartsWith("fail"));
  // Extract certain return
  EXPECT_THAT(apply(" if(true) [[{ return; }]] "), HasSubstr("extracted"));
  // Don't extract uncertain return
  EXPECT_THAT(apply(" if(true) [[if (false) return;]] "), StartsWith("fail"));
}

TEST_F(ExtractFunctionTest, FileTest) {
  // Check all parameters are in order
  std::string ParameterCheckInput = R"cpp(
struct Foo {
  int x;
};
void f(int a) {
  int b;
  int *ptr = &a;
  Foo foo;
  [[a += foo.x + b;
  *ptr++;]]
})cpp";
  std::string ParameterCheckOutput = R"cpp(
struct Foo {
  int x;
};
void extracted(int &a, int &b, int * &ptr, Foo &foo) {
a += foo.x + b;
  *ptr++;
}
void f(int a) {
  int b;
  int *ptr = &a;
  Foo foo;
  extracted(a, b, ptr, foo);
})cpp";
  EXPECT_EQ(apply(ParameterCheckInput), ParameterCheckOutput);

  // Check const qualifier
  std::string ConstCheckInput = R"cpp(
void f(const int c) {
  [[while(c) {}]]
})cpp";
  std::string ConstCheckOutput = R"cpp(
void extracted(const int &c) {
while(c) {}
}
void f(const int c) {
  extracted(c);
})cpp";
  EXPECT_EQ(apply(ConstCheckInput), ConstCheckOutput);

  // Don't extract when we need to make a function as a parameter.
  EXPECT_THAT(apply("void f() { [[int a; f();]] }"), StartsWith("fail"));

  // We don't extract from methods for now since they may involve multi-file
  // edits
  std::string MethodFailInput = R"cpp(
    class T {
      void f() {
        [[int x;]]
      }
    };
  )cpp";
  EXPECT_EQ(apply(MethodFailInput), "unavailable");

  // We don't extract from templated functions for now as templates are hard
  // to deal with.
  std::string TemplateFailInput = R"cpp(
    template<typename T>
    void f() {
      [[int x;]]
    }
  )cpp";
  EXPECT_EQ(apply(TemplateFailInput), "unavailable");

  std::string MacroInput = R"cpp(
    #define F(BODY) void f() { BODY }
    F ([[int x = 0;]])
  )cpp";
  std::string MacroOutput = R"cpp(
    #define F(BODY) void f() { BODY }
    void extracted() {
int x = 0;
}
F (extracted();)
  )cpp";
  EXPECT_EQ(apply(MacroInput), MacroOutput);

  // Shouldn't crash.
  EXPECT_EQ(apply("void f([[int a]]);"), "unavailable");
  // Don't extract if we select the entire function body (CompoundStmt).
  std::string CompoundFailInput = R"cpp(
    void f() [[{
      int a;
    }]]
  )cpp";
  EXPECT_EQ(apply(CompoundFailInput), "unavailable");
}

TEST_F(ExtractFunctionTest, ControlFlow) {
  Context = Function;
  // We should be able to extract break/continue with a parent loop/switch.
  EXPECT_THAT(apply(" [[for(;;) if(1) break;]] "), HasSubstr("extracted"));
  EXPECT_THAT(apply(" for(;;) [[while(1) break;]] "), HasSubstr("extracted"));
  EXPECT_THAT(apply(" [[switch(1) { break; }]]"), HasSubstr("extracted"));
  EXPECT_THAT(apply(" [[while(1) switch(1) { continue; }]]"),
              HasSubstr("extracted"));
  // Don't extract break and continue without a loop/switch parent.
  EXPECT_THAT(apply(" for(;;) [[if(1) continue;]] "), StartsWith("fail"));
  EXPECT_THAT(apply(" while(1) [[if(1) break;]] "), StartsWith("fail"));
  EXPECT_THAT(apply(" switch(1) { [[break;]] }"), StartsWith("fail"));
  EXPECT_THAT(apply(" for(;;) { [[while(1) break; break;]] }"),
              StartsWith("fail"));
}

TEST_F(ExtractFunctionTest, ExistingReturnStatement) {
  Context = File;
  const char* Before = R"cpp(
    bool lucky(int N);
    int getNum(bool Superstitious, int Min, int Max) {
      if (Superstitious) [[{
        for (int I = Min; I <= Max; ++I)
          if (lucky(I))
            return I;
        return -1;
      }]] else {
        return (Min + Max) / 2;
      }
    }
  )cpp";
  // FIXME: min/max should be by value.
  // FIXME: avoid emitting redundant braces
  const char* After = R"cpp(
    bool lucky(int N);
    int extracted(int &Min, int &Max) {
{
        for (int I = Min; I <= Max; ++I)
          if (lucky(I))
            return I;
        return -1;
      }
}
int getNum(bool Superstitious, int Min, int Max) {
      if (Superstitious) return extracted(Min, Max); else {
        return (Min + Max) / 2;
      }
    }
  )cpp";
  EXPECT_EQ(apply(Before), After);
}

TWEAK_TEST(RemoveUsingNamespace);
TEST_F(RemoveUsingNamespaceTest, All) {
  std::pair<llvm::StringRef /*Input*/, llvm::StringRef /*Expected*/> Cases[] = {
      {// Remove all occurrences of ns. Qualify only unqualified.
       R"cpp(
      namespace ns1 { struct vector {}; }
      namespace ns2 { struct map {}; }
      using namespace n^s1;
      using namespace ns2;
      using namespace ns1;
      int main() {
        ns1::vector v1;
        vector v2;
        map m1;
      }
    )cpp",
       R"cpp(
      namespace ns1 { struct vector {}; }
      namespace ns2 { struct map {}; }
      
      using namespace ns2;
      
      int main() {
        ns1::vector v1;
        ns1::vector v2;
        map m1;
      }
    )cpp"},
      {// Ident to be qualified is a macro arg.
       R"cpp(
      #define DECLARE(x, y) x y
      namespace ns { struct vector {}; }
      using namespace n^s;
      int main() {
        DECLARE(ns::vector, v1);
        DECLARE(vector, v2);
      }
    )cpp",
       R"cpp(
      #define DECLARE(x, y) x y
      namespace ns { struct vector {}; }
      
      int main() {
        DECLARE(ns::vector, v1);
        DECLARE(ns::vector, v2);
      }
    )cpp"},
      {// Nested namespace: Fully qualify ident from inner ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::b^b;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Nested namespace: Fully qualify ident from inner ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace a^a;
      int main() {
        bb::map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Typedef.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace a^a;
      typedef bb::map map;
      int main() { map M; }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      typedef aa::bb::map map;
      int main() { map M; }
    )cpp"},
      {// FIXME: Nested namespaces: Not aware of using ns decl of outer ns.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using name[[space aa::b]]b;
      using namespace aa;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      
      using namespace aa;
      int main() {
        aa::bb::map m;
      }
    )cpp"},
      {// Does not qualify ident from inner namespace.
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::bb;
      using namespace a^a;
      int main() {
        map m;
      }
    )cpp",
       R"cpp(
      namespace aa { namespace bb { struct map {}; }}
      using namespace aa::bb;
      
      int main() {
        map m;
      }
    )cpp"},
      {// Available only for top level namespace decl.
       R"cpp(
        namespace aa {
          namespace bb { struct map {}; }
          using namespace b^b;
        }
        int main() { aa::map m; }
    )cpp",
       "unavailable"},
      {// FIXME: Unavailable for namespaces containing using-namespace decl.
       R"cpp(
      namespace aa {
        namespace bb { struct map {}; }
        using namespace bb;
      }
      using namespace a^a;
      int main() {
        map m;
      }
    )cpp",
       "unavailable"},
      {R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      using namespace a::[[b]];
      using namespace b;
      int main() { Foo F;}
    )cpp",
       R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      
      
      int main() { a::b::Foo F;}
    )cpp"},
      {R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      using namespace a::b;
      using namespace [[b]];
      int main() { Foo F;}
    )cpp",
       R"cpp(
      namespace a::b { struct Foo {}; }
      using namespace a;
      
      
      int main() { b::Foo F;}
    )cpp"},
      {// Enumerators.
       R"cpp(
      namespace tokens {
      enum Token {
        comma, identifier, numeric
      };
      }
      using namespace tok^ens;
      int main() {
        auto x = comma;
      }
    )cpp",
       R"cpp(
      namespace tokens {
      enum Token {
        comma, identifier, numeric
      };
      }
      
      int main() {
        auto x = tokens::comma;
      }
    )cpp"},
      {// inline namespaces.
       R"cpp(
      namespace std { inline namespace ns1 { inline namespace ns2 { struct vector {}; }}}
      using namespace st^d;
      int main() {
        vector V;
      }
    )cpp",
       R"cpp(
      namespace std { inline namespace ns1 { inline namespace ns2 { struct vector {}; }}}
      
      int main() {
        std::vector V;
      }
    )cpp"}};
  for (auto C : Cases)
    EXPECT_EQ(C.second, apply(C.first)) << C.first;
}

TWEAK_TEST(DefineInline);
TEST_F(DefineInlineTest, TriggersOnFunctionDecl) {
  // Basic check for function body and signature.
  EXPECT_AVAILABLE(R"cpp(
  class Bar {
    void baz();
  };

  [[void [[Bar::[[b^a^z]]]]() [[{
    return;
  }]]]]

  void foo();
  [[void [[f^o^o]]() [[{
    return;
  }]]]]
  )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
  // Not a definition
  vo^i[[d^ ^f]]^oo();

  [[vo^id ]]foo[[()]] {[[
    [[(void)(5+3);
    return;]]
  }]]

  // Definition with no body.
  class Bar { Bar() = def^ault; };
  )cpp");
}

TEST_F(DefineInlineTest, NoForwardDecl) {
  Header = "void bar();";
  EXPECT_UNAVAILABLE(R"cpp(
  void bar() {
    return;
  }
  // FIXME: Generate a decl in the header.
  void fo^o() {
    return;
  })cpp");
}

TEST_F(DefineInlineTest, ReferencedDecls) {
  EXPECT_AVAILABLE(R"cpp(
    void bar();
    void foo(int test);

    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // Internal symbol usage.
  Header = "void foo(int test);";
  EXPECT_UNAVAILABLE(R"cpp(
    void bar();
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // Becomes available after making symbol visible.
  Header = "void bar();" + Header;
  EXPECT_AVAILABLE(R"cpp(
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");

  // FIXME: Move declaration below bar to make it visible.
  Header.clear();
  EXPECT_UNAVAILABLE(R"cpp(
    void foo();
    void bar();

    void fo^o() {
      bar();
    })cpp");

  // Order doesn't matter within a class.
  EXPECT_AVAILABLE(R"cpp(
    class Bar {
      void foo();
      void bar();
    };

    void Bar::fo^o() {
      bar();
    })cpp");

  // FIXME: Perform include insertion to make symbol visible.
  ExtraFiles["a.h"] = "void bar();";
  Header = "void foo(int test);";
  EXPECT_UNAVAILABLE(R"cpp(
    #include "a.h"
    void fo^o(int baz) {
      int x = 10;
      bar();
    })cpp");
}

TEST_F(DefineInlineTest, TemplateSpec) {
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> void foo();
    template<> void foo<char>();

    template<> void f^oo<int>() {
    })cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> void foo();

    template<> void f^oo<int>() {
    })cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename T> struct Foo { void foo(); };

    template <typename T> void Foo<T>::f^oo() {
    })cpp");
  EXPECT_AVAILABLE(R"cpp(
    template <typename T> void foo();
    void bar();
    template <> void foo<int>();

    template<> void f^oo<int>() {
      bar();
    })cpp");
}

TEST_F(DefineInlineTest, CheckForCanonDecl) {
  EXPECT_UNAVAILABLE(R"cpp(
    void foo();

    void bar() {}
    void f^oo() {
      // This bar normally refers to the definition just above, but it is not
      // visible from the forward declaration of foo.
      bar();
    })cpp");
  // Make it available with a forward decl.
  EXPECT_AVAILABLE(R"cpp(
    void bar();
    void foo();

    void bar() {}
    void f^oo() {
      bar();
    })cpp");
}

TEST_F(DefineInlineTest, UsingShadowDecls) {
  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_UNAVAILABLE(R"cpp(
  namespace ns1 { void foo(int); }
  namespace ns2 { void foo(int*); }
  template <typename T>
  void bar();

  using ns1::foo;
  using ns2::foo;

  template <typename T>
  void b^ar() {
    foo(T());
  })cpp");
}

TEST_F(DefineInlineTest, TransformNestedNamespaces) {
  auto Test = R"cpp(
    namespace a {
      void bar();
      namespace b {
        void baz();
        namespace c {
          void aux();
        }
      }
    }

    void foo();
    using namespace a;
    using namespace b;
    using namespace c;
    void f^oo() {
      bar();
      a::bar();

      baz();
      b::baz();
      a::b::baz();

      aux();
      c::aux();
      b::c::aux();
      a::b::c::aux();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      void bar();
      namespace b {
        void baz();
        namespace c {
          void aux();
        }
      }
    }

    void foo(){
      a::bar();
      a::bar();

      a::b::baz();
      a::b::baz();
      a::b::baz();

      a::b::c::aux();
      a::b::c::aux();
      a::b::c::aux();
      a::b::c::aux();
    }
    using namespace a;
    using namespace b;
    using namespace c;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformUsings) {
  auto Test = R"cpp(
    namespace a { namespace b { namespace c { void aux(); } } }

    void foo();
    void f^oo() {
      using namespace a;
      using namespace b;
      using namespace c;
      using c::aux;
      namespace d = c;
    })cpp";
  auto Expected = R"cpp(
    namespace a { namespace b { namespace c { void aux(); } } }

    void foo(){
      using namespace a;
      using namespace a::b;
      using namespace a::b::c;
      using a::b::c::aux;
      namespace d = a::b::c;
    }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformDecls) {
  auto Test = R"cpp(
    void foo();
    void f^oo() {
      class Foo {
      public:
        void foo();
        int x;
      };

      enum En { Zero, One };
      En x = Zero;

      enum class EnClass { Zero, One };
      EnClass y = EnClass::Zero;
    })cpp";
  auto Expected = R"cpp(
    void foo(){
      class Foo {
      public:
        void foo();
        int x;
      };

      enum En { Zero, One };
      En x = Zero;

      enum class EnClass { Zero, One };
      EnClass y = EnClass::Zero;
    }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformTemplDecls) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void bar();
      };
      template <typename T> T bar;
      template <typename T> void aux() {}
    }

    void foo();

    using namespace a;
    void f^oo() {
      bar<Bar<int>>.bar();
      aux<Bar<int>>();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void bar();
      };
      template <typename T> T bar;
      template <typename T> void aux() {}
    }

    void foo(){
      a::bar<a::Bar<int>>.bar();
      a::aux<a::Bar<int>>();
    }

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformMembers) {
  auto Test = R"cpp(
    class Foo {
      void foo();
    };

    void Foo::f^oo() {
      return;
    })cpp";
  auto Expected = R"cpp(
    class Foo {
      void foo(){
      return;
    }
    };

    )cpp";
  EXPECT_EQ(apply(Test), Expected);

  ExtraFiles["a.h"] = R"cpp(
    class Foo {
      void foo();
    };)cpp";

  llvm::StringMap<std::string> EditedFiles;
  Test = R"cpp(
    #include "a.h"
    void Foo::f^oo() {
      return;
    })cpp";
  Expected = R"cpp(
    #include "a.h"
    )cpp";
  EXPECT_EQ(apply(Test, &EditedFiles), Expected);

  Expected = R"cpp(
    class Foo {
      void foo(){
      return;
    }
    };)cpp";
  EXPECT_THAT(EditedFiles,
              ElementsAre(FileWithContents(testPath("a.h"), Expected)));
}

TEST_F(DefineInlineTest, TransformDependentTypes) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {};
    }

    template <typename T>
    void foo();

    using namespace a;
    template <typename T>
    void f^oo() {
      Bar<T> B;
      Bar<Bar<T>> q;
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {};
    }

    template <typename T>
    void foo(){
      a::Bar<T> B;
      a::Bar<a::Bar<T>> q;
    }

    using namespace a;
    )cpp";

  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformFunctionTempls) {
  // Check we select correct specialization decl.
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p);

          template <>
          void fo^o<int>(int p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p){
            return;
          }

          template <>
          void foo<char>(char p);

          )cpp"},
      {// Make sure we are not selecting the first specialization all the time.
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p);

          template <>
          void fo^o<char>(char p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <>
          void foo<char>(char p){
            return;
          }

          )cpp"},
      {R"cpp(
          template <typename T>
          void foo(T p);

          template <>
          void foo<int>(int p);

          template <typename T>
          void fo^o(T p) {
            return;
          })cpp",
       R"cpp(
          template <typename T>
          void foo(T p){
            return;
          }

          template <>
          void foo<int>(int p);

          )cpp"},
  };
  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, TransformTypeLocs) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        template <typename Q> class Baz {};
      };
      class Foo{};
    }

    void foo();

    using namespace a;
    void f^oo() {
      Bar<int> B;
      Foo foo;
      a::Bar<Bar<int>>::Baz<Bar<int>> q;
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        template <typename Q> class Baz {};
      };
      class Foo{};
    }

    void foo(){
      a::Bar<int> B;
      a::Foo foo;
      a::Bar<a::Bar<int>>::Baz<a::Bar<int>> q;
    }

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformDeclRefs) {
  auto Test = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void foo();
        static void bar();
        int x;
        static int y;
      };
      void bar();
      void test();
    }

    void foo();
    using namespace a;
    void f^oo() {
      a::Bar<int> B;
      B.foo();
      a::bar();
      Bar<Bar<int>>::bar();
      a::Bar<int>::bar();
      B.x = Bar<int>::y;
      Bar<int>::y = 3;
      bar();
      a::test();
    })cpp";
  auto Expected = R"cpp(
    namespace a {
      template <typename T> class Bar {
      public:
        void foo();
        static void bar();
        int x;
        static int y;
      };
      void bar();
      void test();
    }

    void foo(){
      a::Bar<int> B;
      B.foo();
      a::bar();
      a::Bar<a::Bar<int>>::bar();
      a::Bar<int>::bar();
      B.x = a::Bar<int>::y;
      a::Bar<int>::y = 3;
      a::bar();
      a::test();
    }
    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, StaticMembers) {
  auto Test = R"cpp(
    namespace ns { class X { static void foo(); void bar(); }; }
    void ns::X::b^ar() {
      foo();
    })cpp";
  auto Expected = R"cpp(
    namespace ns { class X { static void foo(); void bar(){
      foo();
    } }; }
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformParamNames) {
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
        void foo(int, bool b, int T\
est);
        void ^foo(int f, bool x, int z) {})cpp",
       R"cpp(
        void foo(int f, bool x, int z){}
        )cpp"},
      {R"cpp(
        #define PARAM int Z
        void foo(PARAM);

        void ^foo(int X) {})cpp",
       "fail: Cant rename parameter inside macro body."},
      {R"cpp(
        #define TYPE int
        #define PARAM TYPE Z
        #define BODY(x) 5 * (x) + 2
        template <int P>
        void foo(PARAM, TYPE Q, TYPE, TYPE W = BODY(P));
        template <int x>
        void ^foo(int Z, int b, int c, int d) {})cpp",
       R"cpp(
        #define TYPE int
        #define PARAM TYPE Z
        #define BODY(x) 5 * (x) + 2
        template <int x>
        void foo(PARAM, TYPE b, TYPE c, TYPE d = BODY(x)){}
        )cpp"},
  };
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, TransformTemplParamNames) {
  auto Test = R"cpp(
    struct Foo {
      struct Bar {
        template <class, class X,
                  template<typename> class, template<typename> class Y,
                  int, int Z>
        void foo(X, Y<X>, int W = 5 * Z + 2);
      };
    };

    template <class T, class U,
              template<typename> class V, template<typename> class W,
              int X, int Y>
    void Foo::Bar::f^oo(U, W<U>, int Q) {})cpp";
  auto Expected = R"cpp(
    struct Foo {
      struct Bar {
        template <class T, class U,
                  template<typename> class V, template<typename> class W,
                  int X, int Y>
        void foo(U, W<U>, int Q = 5 * Y + 2){}
      };
    };

    )cpp";
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TransformInlineNamespaces) {
  auto Test = R"cpp(
    namespace a { inline namespace b { namespace { struct Foo{}; } } }
    void foo();

    using namespace a;
    void ^foo() {Foo foo;})cpp";
  auto Expected = R"cpp(
    namespace a { inline namespace b { namespace { struct Foo{}; } } }
    void foo(){a::Foo foo;}

    using namespace a;
    )cpp";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineInlineTest, TokensBeforeSemicolon) {
  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      {R"cpp(
          void foo()    /*Comment -_-*/ /*Com 2*/ ;
          void fo^o() { return ; })cpp",
       R"cpp(
          void foo()    /*Comment -_-*/ /*Com 2*/ { return ; }
          )cpp"},

      {R"cpp(
          void foo();
          void fo^o() { return ; })cpp",
       R"cpp(
          void foo(){ return ; }
          )cpp"},

      {R"cpp(
          #define SEMI ;
          void foo() SEMI
          void fo^o() { return ; })cpp",
       "fail: Couldn't find semicolon for target declaration."},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, HandleMacros) {
  EXPECT_UNAVAILABLE(R"cpp(
    #define BODY { return; }
    void foo();
    void f^oo()BODY)cpp");

  EXPECT_UNAVAILABLE(R"cpp(
    #define BODY void foo(){ return; }
    void foo();
    [[BODY]])cpp");

  std::pair<llvm::StringRef, llvm::StringRef> Cases[] = {
      // We don't qualify declarations coming from macros.
      {R"cpp(
          #define BODY Foo
          namespace a { class Foo{}; }
          void foo();
          using namespace a;
          void f^oo(){BODY();})cpp",
       R"cpp(
          #define BODY Foo
          namespace a { class Foo{}; }
          void foo(){BODY();}
          using namespace a;
          )cpp"},

      // Macro is not visible at declaration location, but we proceed.
      {R"cpp(
          void foo();
          #define BODY return;
          void f^oo(){BODY})cpp",
       R"cpp(
          void foo(){BODY}
          #define BODY return;
          )cpp"},

      {R"cpp(
          #define TARGET void foo()
          TARGET;
          void f^oo(){ return; })cpp",
       R"cpp(
          #define TARGET void foo()
          TARGET{ return; }
          )cpp"},

      {R"cpp(
          #define TARGET foo
          void TARGET();
          void f^oo(){ return; })cpp",
       R"cpp(
          #define TARGET foo
          void TARGET(){ return; }
          )cpp"},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.first), Case.second) << Case.first;
}

TEST_F(DefineInlineTest, DropCommonNameSpecifiers) {
  struct {
    llvm::StringRef Test;
    llvm::StringRef Expected;
  } Cases[] = {
      {R"cpp(
        namespace a { namespace b { void aux(); } }
        namespace ns1 {
          void foo();
          namespace qq { void test(); }
          namespace ns2 {
            void bar();
            namespace ns3 { void baz(); }
          }
        }

        using namespace a;
        using namespace a::b;
        using namespace ns1::qq;
        void ns1::ns2::ns3::b^az() {
          foo();
          bar();
          baz();
          ns1::ns2::ns3::baz();
          aux();
          test();
        })cpp",
       R"cpp(
        namespace a { namespace b { void aux(); } }
        namespace ns1 {
          void foo();
          namespace qq { void test(); }
          namespace ns2 {
            void bar();
            namespace ns3 { void baz(){
          foo();
          bar();
          baz();
          ns1::ns2::ns3::baz();
          a::b::aux();
          qq::test();
        } }
          }
        }

        using namespace a;
        using namespace a::b;
        using namespace ns1::qq;
        )cpp"},
      {R"cpp(
        namespace ns1 {
          namespace qq { struct Foo { struct Bar {}; }; using B = Foo::Bar; }
          namespace ns2 { void baz(); }
        }

        using namespace ns1::qq;
        void ns1::ns2::b^az() { Foo f; B b; })cpp",
       R"cpp(
        namespace ns1 {
          namespace qq { struct Foo { struct Bar {}; }; using B = Foo::Bar; }
          namespace ns2 { void baz(){ qq::Foo f; qq::B b; } }
        }

        using namespace ns1::qq;
        )cpp"},
      {R"cpp(
        namespace ns1 {
          namespace qq {
            template<class T> struct Foo { template <class U> struct Bar {}; };
            template<class T, class U>
            using B = typename Foo<T>::template Bar<U>;
          }
          namespace ns2 { void baz(); }
        }

        using namespace ns1::qq;
        void ns1::ns2::b^az() { B<int, bool> b; })cpp",
       R"cpp(
        namespace ns1 {
          namespace qq {
            template<class T> struct Foo { template <class U> struct Bar {}; };
            template<class T, class U>
            using B = typename Foo<T>::template Bar<U>;
          }
          namespace ns2 { void baz(){ qq::B<int, bool> b; } }
        }

        using namespace ns1::qq;
        )cpp"},
  };
  for (const auto &Case : Cases)
    EXPECT_EQ(apply(Case.Test), Case.Expected) << Case.Test;
}

TEST_F(DefineInlineTest, QualifyWithUsingDirectives) {
  llvm::StringRef Test = R"cpp(
    namespace a {
      void bar();
      namespace b { struct Foo{}; void aux(); }
      namespace c { void cux(); }
    }
    using namespace a;
    using X = b::Foo;
    void foo();

    using namespace b;
    using namespace c;
    void ^foo() {
      cux();
      bar();
      X x;
      aux();
      using namespace c;
      // FIXME: The last reference to cux() in body of foo should not be
      // qualified, since there is a using directive inside the function body.
      cux();
    })cpp";
  llvm::StringRef Expected = R"cpp(
    namespace a {
      void bar();
      namespace b { struct Foo{}; void aux(); }
      namespace c { void cux(); }
    }
    using namespace a;
    using X = b::Foo;
    void foo(){
      c::cux();
      bar();
      X x;
      b::aux();
      using namespace c;
      // FIXME: The last reference to cux() in body of foo should not be
      // qualified, since there is a using directive inside the function body.
      c::cux();
    }

    using namespace b;
    using namespace c;
    )cpp";
  EXPECT_EQ(apply(Test), Expected) << Test;
}

TEST_F(DefineInlineTest, AddInline) {
  ExtraArgs.push_back("-fno-delayed-template-parsing");
  llvm::StringMap<std::string> EditedFiles;
  ExtraFiles["a.h"] = "void foo();";
  apply(R"cpp(#include "a.h"
              void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline void foo(){}")));

  // Check we put inline before cv-qualifiers.
  ExtraFiles["a.h"] = "const int foo();";
  apply(R"cpp(#include "a.h"
              const int fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline const int foo(){}")));

  // No double inline.
  ExtraFiles["a.h"] = "inline void foo();";
  apply(R"cpp(#include "a.h"
              inline void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "inline void foo(){}")));

  // Constexprs don't need "inline".
  ExtraFiles["a.h"] = "constexpr void foo();";
  apply(R"cpp(#include "a.h"
              constexpr void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                               testPath("a.h"), "constexpr void foo(){}")));

  // Class members don't need "inline".
  ExtraFiles["a.h"] = "struct Foo { void foo(); };";
  apply(R"cpp(#include "a.h"
              void Foo::fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(
                  testPath("a.h"), "struct Foo { void foo(){} };")));

  // Function template doesn't need to be "inline"d.
  ExtraFiles["a.h"] = "template <typename T> void foo();";
  apply(R"cpp(#include "a.h"
              template <typename T>
              void fo^o() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(
                  testPath("a.h"), "template <typename T> void foo(){}")));

  // Specializations needs to be marked "inline".
  ExtraFiles["a.h"] = R"cpp(
                            template <typename T> void foo();
                            template <> void foo<int>();)cpp";
  apply(R"cpp(#include "a.h"
              template <>
              void fo^o<int>() {})cpp",
        &EditedFiles);
  EXPECT_THAT(EditedFiles,
              testing::ElementsAre(FileWithContents(testPath("a.h"),
                                                    R"cpp(
                            template <typename T> void foo();
                            template <> inline void foo<int>(){})cpp")));
}

TWEAK_TEST(DefineOutline);
TEST_F(DefineOutlineTest, TriggersOnFunctionDecl) {
  FileName = "Test.cpp";
  // Not available unless in a header file.
  EXPECT_UNAVAILABLE(R"cpp(
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  FileName = "Test.hpp";
  // Not available unless function name or fully body is selected.
  EXPECT_UNAVAILABLE(R"cpp(
    // Not a definition
    vo^i[[d^ ^f]]^oo();

    [[vo^id ]]foo[[()]] {[[
      [[(void)(5+3);
      return;]]
    }]])cpp");

  // Available even if there are no implementation files.
  EXPECT_AVAILABLE(R"cpp(
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  // Not available for out-of-line methods.
  EXPECT_UNAVAILABLE(R"cpp(
    class Bar {
      void baz();
    };

    [[void [[Bar::[[b^a^z]]]]() [[{
      return;
    }]]]])cpp");

  // Basic check for function body and signature.
  EXPECT_AVAILABLE(R"cpp(
    class Bar {
      [[void [[f^o^o^]]() [[{ return; }]]]]
    };

    void foo();
    [[void [[f^o^o]]() [[{
      return;
    }]]]])cpp");

  // Not available on defaulted/deleted members.
  EXPECT_UNAVAILABLE(R"cpp(
    class Foo {
      Fo^o() = default;
      F^oo(const Foo&) = delete;
    };)cpp");

  // Not available within templated classes, as it is hard to spell class name
  // out-of-line in such cases.
  EXPECT_UNAVAILABLE(R"cpp(
    template <typename> struct Foo { void fo^o(){} };
    )cpp");
}

TEST_F(DefineOutlineTest, FailsWithoutSource) {
  FileName = "Test.hpp";
  llvm::StringRef Test = "void fo^o() { return; }";
  llvm::StringRef Expected =
      "fail: Couldn't find a suitable implementation file.";
  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DefineOutlineTest, ApplyTest) {
  llvm::StringMap<std::string> EditedFiles;
  ExtraFiles["Test.cpp"] = "";
  FileName = "Test.hpp";
  // Template body is not parsed until instantiation time on windows, which
  // results in arbitrary failures as function body becomes NULL.
  ExtraArgs.push_back("-fno-delayed-template-parsing");

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      // Simple check
      {
          "void fo^o() { return; }",
          "void foo() ;",
          "void foo() { return; }",
      },
      // Templated function.
      {
          "template <typename T> void fo^o(T, T x) { return; }",
          "template <typename T> void foo(T, T x) ;",
          "template <typename T> void foo(T, T x) { return; }",
      },
      {
          "template <typename> void fo^o() { return; }",
          "template <typename> void foo() ;",
          "template <typename> void foo() { return; }",
      },
      // Template specialization.
      {
          R"cpp(
            template <typename> void foo();
            template <> void fo^o<int>() { return; })cpp",
          R"cpp(
            template <typename> void foo();
            template <> void foo<int>() ;)cpp",
          "template <> void foo<int>() { return; }",
      },
      // Default args.
      {
          "void fo^o(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) {}",
          "void foo(int x, int y = 5, int = 2, int (*foo)(int) = nullptr) ;",
          "void foo(int x, int y , int , int (*foo)(int) ) {}",
      },
      // Constructors
      {
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Ba^r() {}
              Bar(int x) : f1(x) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() ;
              Bar(int x) : f1(x) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          "Bar::Bar() {}\n",
      },
      // Ctor with initializer.
      {
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() {}
              B^ar(int x) : f1(x), f2(3) {}
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          R"cpp(
            class Foo {public: Foo(); Foo(int);};
            class Bar {
              Bar() {}
              Bar(int x) ;
              Foo f1;
              Foo f2 = 2;
            };)cpp",
          "Bar::Bar(int x) : f1(x), f2(3) {}\n",
      },
      // Ctor initializer with attribute.
      {
          R"cpp(
              class Foo {
                F^oo(int z) __attribute__((weak)) : bar(2){}
                int bar;
              };)cpp",
          R"cpp(
              class Foo {
                Foo(int z) __attribute__((weak)) ;
                int bar;
              };)cpp",
          "Foo::Foo(int z) __attribute__((weak)) : bar(2){}\n",
      },
      // Virt specifiers.
      {
          R"cpp(
            struct A {
              virtual void f^oo() {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() ;
            };)cpp",
          " void A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual virtual void virtual f^oo() {}
            };)cpp",
          R"cpp(
            struct A {
              virtual virtual void virtual foo() ;
            };)cpp",
          "  void  A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() override {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() override ;
            };)cpp",
          "void B::foo()  {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() final {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() final ;
            };)cpp",
          "void B::foo()  {}\n",
      },
      {
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void fo^o() final override {}
            };)cpp",
          R"cpp(
            struct A {
              virtual void foo() = 0;
            };
            struct B : A {
              void foo() final override ;
            };)cpp",
          "void B::foo()   {}\n",
      },
      {
          R"cpp(
            struct A {
              static void fo^o() {}
            };)cpp",
          R"cpp(
            struct A {
              static void foo() ;
            };)cpp",
          " void A::foo() {}\n",
      },
      {
          R"cpp(
            struct A {
              static static void fo^o() {}
            };)cpp",
          R"cpp(
            struct A {
              static static void foo() ;
            };)cpp",
          "  void A::foo() {}\n",
      },
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
  }
}

TEST_F(DefineOutlineTest, HandleMacros) {
  llvm::StringMap<std::string> EditedFiles;
  ExtraFiles["Test.cpp"] = "";
  FileName = "Test.hpp";
  ExtraArgs.push_back("-DVIRTUAL=virtual");
  ExtraArgs.push_back("-DOVER=override");

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {R"cpp(
          #define BODY { return; }
          void f^oo()BODY)cpp",
       R"cpp(
          #define BODY { return; }
          void foo();)cpp",
       "void foo()BODY"},

      {R"cpp(
          #define BODY return;
          void f^oo(){BODY})cpp",
       R"cpp(
          #define BODY return;
          void foo();)cpp",
       "void foo(){BODY}"},

      {R"cpp(
          #define TARGET void foo()
          [[TARGET]]{ return; })cpp",
       R"cpp(
          #define TARGET void foo()
          TARGET;)cpp",
       "TARGET{ return; }"},

      {R"cpp(
          #define TARGET foo
          void [[TARGET]](){ return; })cpp",
       R"cpp(
          #define TARGET foo
          void TARGET();)cpp",
       "void TARGET(){ return; }"},
      {R"cpp(#define VIRT virtual
          struct A {
            VIRT void f^oo() {}
          };)cpp",
       R"cpp(#define VIRT virtual
          struct A {
            VIRT void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(
          struct A {
            VIRTUAL void f^oo() {}
          };)cpp",
       R"cpp(
          struct A {
            VIRTUAL void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(
          struct A {
            virtual void foo() = 0;
          };
          struct B : A {
            void fo^o() OVER {}
          };)cpp",
       R"cpp(
          struct A {
            virtual void foo() = 0;
          };
          struct B : A {
            void foo() OVER ;
          };)cpp",
       "void B::foo()  {}\n"},
      {R"cpp(#define STUPID_MACRO(X) virtual
          struct A {
            STUPID_MACRO(sizeof sizeof int) void f^oo() {}
          };)cpp",
       R"cpp(#define STUPID_MACRO(X) virtual
          struct A {
            STUPID_MACRO(sizeof sizeof int) void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(#define STAT static
          struct A {
            STAT void f^oo() {}
          };)cpp",
       R"cpp(#define STAT static
          struct A {
            STAT void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
      {R"cpp(#define STUPID_MACRO(X) static
          struct A {
            STUPID_MACRO(sizeof sizeof int) void f^oo() {}
          };)cpp",
       R"cpp(#define STUPID_MACRO(X) static
          struct A {
            STUPID_MACRO(sizeof sizeof int) void foo() ;
          };)cpp",
       " void A::foo() {}\n"},
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Test);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
  }
}

TEST_F(DefineOutlineTest, QualifyReturnValue) {
  FileName = "Test.hpp";
  ExtraFiles["Test.cpp"] = "";

  struct {
    llvm::StringRef Test;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {R"cpp(
        namespace a { class Foo{}; }
        using namespace a;
        Foo fo^o() { return {}; })cpp",
       R"cpp(
        namespace a { class Foo{}; }
        using namespace a;
        Foo foo() ;)cpp",
       "a::Foo foo() { return {}; }"},
      {R"cpp(
        namespace a {
          class Foo {
            class Bar {};
            Bar fo^o() { return {}; }
          };
        })cpp",
       R"cpp(
        namespace a {
          class Foo {
            class Bar {};
            Bar foo() ;
          };
        })cpp",
       "a::Foo::Bar a::Foo::foo() { return {}; }\n"},
      {R"cpp(
        class Foo {};
        Foo fo^o() { return {}; })cpp",
       R"cpp(
        class Foo {};
        Foo foo() ;)cpp",
       "Foo foo() { return {}; }"},
  };
  llvm::StringMap<std::string> EditedFiles;
  for (auto &Case : Cases) {
    apply(Case.Test, &EditedFiles);
    EXPECT_EQ(apply(Case.Test, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)));
  }
}

TEST_F(DefineOutlineTest, QualifyFunctionName) {
  FileName = "Test.hpp";
  struct {
    llvm::StringRef TestHeader;
    llvm::StringRef TestSource;
    llvm::StringRef ExpectedHeader;
    llvm::StringRef ExpectedSource;
  } Cases[] = {
      {
          R"cpp(
            namespace a {
              namespace b {
                class Foo {
                  void fo^o() {}
                };
              }
            })cpp",
          "",
          R"cpp(
            namespace a {
              namespace b {
                class Foo {
                  void foo() ;
                };
              }
            })cpp",
          "void a::b::Foo::foo() {}\n",
      },
      {
          "namespace a { namespace b { void f^oo() {} } }",
          "namespace a{}",
          "namespace a { namespace b { void foo() ; } }",
          "namespace a{void b::foo() {} }",
      },
      {
          "namespace a { namespace b { void f^oo() {} } }",
          "using namespace a;",
          "namespace a { namespace b { void foo() ; } }",
          // FIXME: Take using namespace directives in the source file into
          // account. This can be spelled as b::foo instead.
          "using namespace a;void a::b::foo() {} ",
      },
  };
  llvm::StringMap<std::string> EditedFiles;
  for (auto &Case : Cases) {
    ExtraFiles["Test.cpp"] = std::string(Case.TestSource);
    EXPECT_EQ(apply(Case.TestHeader, &EditedFiles), Case.ExpectedHeader);
    EXPECT_THAT(EditedFiles, testing::ElementsAre(FileWithContents(
                                 testPath("Test.cpp"), Case.ExpectedSource)))
        << Case.TestHeader;
  }
}

TEST_F(DefineOutlineTest, FailsMacroSpecifier) {
  FileName = "Test.hpp";
  ExtraFiles["Test.cpp"] = "";
  ExtraArgs.push_back("-DFINALOVER=final override");

  std::pair<StringRef, StringRef> Cases[] = {
      {
          R"cpp(
          #define VIRT virtual void
          struct A {
            VIRT fo^o() {}
          };)cpp",
          "fail: define outline: couldn't remove `virtual` keyword."},
      {
          R"cpp(
          #define OVERFINAL final override
          struct A {
            virtual void foo() {}
          };
          struct B : A {
            void fo^o() OVERFINAL {}
          };)cpp",
          "fail: define outline: Can't move out of line as function has a "
          "macro `override` specifier.\ndefine outline: Can't move out of line "
          "as function has a macro `final` specifier."},
      {
          R"cpp(
          struct A {
            virtual void foo() {}
          };
          struct B : A {
            void fo^o() FINALOVER {}
          };)cpp",
          "fail: define outline: Can't move out of line as function has a "
          "macro `override` specifier.\ndefine outline: Can't move out of line "
          "as function has a macro `final` specifier."},
  };
  for (const auto &Case : Cases) {
    EXPECT_EQ(apply(Case.first), Case.second);
  }
}

TWEAK_TEST(AddUsing);
TEST_F(AddUsingTest, Prepare) {
  const std::string Header = R"cpp(
#define NS(name) one::two::name
namespace one {
void oo() {}
template<typename TT> class tt {};
namespace two {
enum ee {};
void ff() {}
class cc {
public:
  struct st {};
  static void mm() {}
  cc operator|(const cc& x) const { return x; }
};
}
})cpp";

  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o^:^:^f^f(); }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^::^o^o(); }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o^:^:^e^e E; }");
  EXPECT_AVAILABLE(Header + "void fun() { o^n^e^:^:^t^w^o:^:^c^c C; }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^m^m(); }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^s^t inst; }");
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { o^n^e^:^:^t^w^o^:^:^c^c^:^:^s^t inst; }");
  EXPECT_UNAVAILABLE(Header + "void fun() { N^S(c^c) inst; }");
  // This used to crash. Ideally we would support this case, but for now we just
  // test that we don't crash.
  EXPECT_UNAVAILABLE(Header +
                     "template<typename TT> using foo = one::tt<T^T>;");
  // Test that we don't crash or misbehave on unnamed DeclRefExpr.
  EXPECT_UNAVAILABLE(Header +
                     "void fun() { one::two::cc() ^| one::two::cc(); }");

  // Check that we do not trigger in header files.
  FileName = "test.h";
  ExtraArgs.push_back("-xc++-header"); // .h file is treated a C by default.
  EXPECT_UNAVAILABLE(Header + "void fun() { one::two::f^f(); }");
  FileName = "test.hpp";
  EXPECT_UNAVAILABLE(Header + "void fun() { one::two::f^f(); }");
}

TEST_F(AddUsingTest, Apply) {
  FileName = "test.cpp";
  struct {
    llvm::StringRef TestSource;
    llvm::StringRef ExpectedSource;
  } Cases[]{{
                // Function, no other using, namespace.
                R"cpp(
#include "test.hpp"
namespace {
void fun() {
  ^o^n^e^:^:^t^w^o^:^:^f^f();
}
})cpp",
                R"cpp(
#include "test.hpp"
namespace {using one::two::ff;

void fun() {
  ff();
}
})cpp",
            },
            // Type, no other using, namespace.
            {
                R"cpp(
#include "test.hpp"
namespace {
void fun() {
  ::on^e::t^wo::c^c inst;
}
})cpp",
                R"cpp(
#include "test.hpp"
namespace {using ::one::two::cc;

void fun() {
  cc inst;
}
})cpp",
            },
            // Type, no other using, no namespace.
            {
                R"cpp(
#include "test.hpp"

void fun() {
  on^e::t^wo::e^e inst;
})cpp",
                R"cpp(
#include "test.hpp"

using one::two::ee;

void fun() {
  ee inst;
})cpp"},
            // Function, other usings.
            {
                R"cpp(
#include "test.hpp"

using one::two::cc;
using one::two::ee;

namespace {
void fun() {
  one::two::f^f();
}
})cpp",
                R"cpp(
#include "test.hpp"

using one::two::cc;
using one::two::ff;using one::two::ee;

namespace {
void fun() {
  ff();
}
})cpp",
            },
            // Function, other usings inside namespace.
            {
                R"cpp(
#include "test.hpp"

using one::two::cc;

namespace {

using one::two::ff;

void fun() {
  o^ne::o^o();
}
})cpp",
                R"cpp(
#include "test.hpp"

using one::two::cc;

namespace {

using one::oo;using one::two::ff;

void fun() {
  oo();
}
})cpp"},
            // Using comes after cursor.
            {
                R"cpp(
#include "test.hpp"

namespace {

void fun() {
  one::t^wo::ff();
}

using one::two::cc;

})cpp",
                R"cpp(
#include "test.hpp"

namespace {using one::two::ff;


void fun() {
  ff();
}

using one::two::cc;

})cpp"},
            // Pointer type.
            {R"cpp(
#include "test.hpp"

void fun() {
  one::two::c^c *p;
})cpp",
             R"cpp(
#include "test.hpp"

using one::two::cc;

void fun() {
  cc *p;
})cpp"},
            // Namespace declared via macro.
            {R"cpp(
#include "test.hpp"
#define NS_BEGIN(name) namespace name {

NS_BEGIN(foo)

void fun() {
  one::two::f^f();
}
})cpp",
             R"cpp(
#include "test.hpp"
#define NS_BEGIN(name) namespace name {

using one::two::ff;

NS_BEGIN(foo)

void fun() {
  ff();
}
})cpp"},
            // Inside macro argument.
            {R"cpp(
#include "test.hpp"
#define CALL(name) name()

void fun() {
  CALL(one::t^wo::ff);
})cpp",
             R"cpp(
#include "test.hpp"
#define CALL(name) name()

using one::two::ff;

void fun() {
  CALL(ff);
})cpp"},
            // Parent namespace != lexical parent namespace
            {R"cpp(
#include "test.hpp"
namespace foo { void fun(); }

void foo::fun() {
  one::two::f^f();
})cpp",
             R"cpp(
#include "test.hpp"
using one::two::ff;

namespace foo { void fun(); }

void foo::fun() {
  ff();
})cpp"}};
  llvm::StringMap<std::string> EditedFiles;
  for (const auto &Case : Cases) {
    for (const auto &SubCase : expandCases(Case.TestSource)) {
      ExtraFiles["test.hpp"] = R"cpp(
namespace one {
void oo() {}
namespace two {
enum ee {};
void ff() {}
class cc {
public:
  struct st { struct nested {}; };
  static void mm() {}
};
}
})cpp";
      EXPECT_EQ(apply(SubCase, &EditedFiles), Case.ExpectedSource);
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
