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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

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
  EXPECT_EQ(apply("^if (true) {return 100;} else {continue;}"),
            "if (true) {continue;} else {return 100;}");
  EXPECT_EQ(apply("^if () {return 100;} else {continue;}"),
            "if () {continue;} else {return 100;}") << "broken condition";
  EXPECT_AVAILABLE("^i^f^^(^t^r^u^e^) { return 100; } ^e^l^s^e^ { continue; }");
  EXPECT_UNAVAILABLE("if (true) {^return ^100;^ } else { ^continue^;^ }");
  // Available in subexpressions of the condition;
  EXPECT_THAT("if(2 + [[2]] + 2) { return 2 + 2 + 2; } else {continue;}",
              isAvailable());
  // But not as part of the branches.
  EXPECT_THAT("if(2 + 2 + 2) { return 2 + [[2]] + 2; } else { continue; }",
              Not(isAvailable()));
  // Range covers the "else" token, so available.
  EXPECT_THAT("if(2 + 2 + 2) { return 2 + [[2 + 2; } else {continue;]]}",
              isAvailable());
  // Not available in compound statements in condition.
  EXPECT_THAT(
      "if([]{return [[true]];}()) { return 2 + 2 + 2; } else { continue; }",
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
  EXPECT_UNAVAILABLE(R"cpp(^"a\r\n";)cpp"); // forbidden escape char

  const char *Input = R"cpp(R"(multi
token)" "\nst^ring\n" "literal")cpp";
  const char *Output = R"cpp(R"(multi
token
string
literal)")cpp";
  EXPECT_EQ(apply(Input), Output);
}

TWEAK_TEST(DumpAST);
TEST_F(DumpASTTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_UNAVAILABLE("/*c^omment*/ int foo() return 2 ^ + 2; }");
  EXPECT_THAT(apply("int x = 2 ^+ 2;"),
              AllOf(StartsWith("message:"), HasSubstr("BinaryOperator"),
                    HasSubstr("'+'"), HasSubstr("|-IntegerLiteral"),
                    HasSubstr("<col:9> 'int' 2\n`-IntegerLiteral"),
                    HasSubstr("<col:13> 'int' 2")));
}

TWEAK_TEST(ShowSelectionTree);
TEST_F(ShowSelectionTreeTest, Test) {
  EXPECT_AVAILABLE("^int f^oo() { re^turn 2 ^+ 2; }");
  EXPECT_AVAILABLE("/*c^omment*/ int foo() return 2 ^ + 2; }");

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
}

TWEAK_TEST(DumpRecordLayout);
TEST_F(DumpRecordLayoutTest, Test) {
  EXPECT_AVAILABLE("^s^truct ^X ^{ int x; ^};");
  EXPECT_THAT("struct X { int ^a; };", Not(isAvailable()));
  EXPECT_THAT("struct ^X;", Not(isAvailable()));
  EXPECT_THAT("template <typename T> struct ^X { T t; };", Not(isAvailable()));
  EXPECT_THAT("enum ^X {};", Not(isAvailable()));

  EXPECT_THAT(apply("struct ^X { int x; int y; }"),
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

  const char *NoCrashCases = R"cpp(
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
      auto lamb = [&[[a]], &[[b]]](int r = [[1]]) {return 1;}
      // assigment
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
  const std::vector<std::pair<llvm::StringLiteral, llvm::StringLiteral>>
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
          /*FIXME: It should be extracted like this. SelectionTree needs to be
            * fixed for macros.
           R"cpp(#define LOOP(x) while (1) {a = x;}
               void f(int a) {
                 auto dummy = 3; if(1)
                  LOOP(5 + dummy)
               })cpp"},*/
           R"cpp(#define LOOP(x) while (1) {a = x;}
                 void f(int a) {
                   auto dummy = LOOP(5 + 3); if(1)
                    dummy
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
  EXPECT_EQ("/* storage.type.primitive.cpp */void "
            "/* entity.name.function.cpp */f() {}",
            apply("void ^f() {}"));

  EXPECT_EQ(apply("[[void f1(); void f2();]]"),
            "/* storage.type.primitive.cpp */void "
            "/* entity.name.function.cpp */f1(); "
            "/* storage.type.primitive.cpp */void "
            "/* entity.name.function.cpp */f2();");

  EXPECT_EQ(apply("void f1(); void f2() {^}"),
            "void f1(); "
            "/* storage.type.primitive.cpp */void "
            "/* entity.name.function.cpp */f2() {}");
}

TWEAK_TEST(ExpandMacro);
TEST_F(ExpandMacroTest, Test) {
  Header = R"cpp(
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
      }
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
  // unknown types in a template should not be replaced
  EXPECT_THAT(apply("template <typename T> void x() { ^auto y = T::z(); }"),
              StartsWith("fail: Could not deduce type for 'auto' type"));
  // undefined functions should not be replaced
  EXPECT_THAT(apply("au^to x = doesnt_exist();"),
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
  EXPECT_EQ(apply(R"cpp(au^to x = "test")cpp"),
            R"cpp(const char * x = "test")cpp");

  EXPECT_UNAVAILABLE("dec^ltype(au^to) x = 10;");
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
  // initalizer/condition.
  EXPECT_THAT(apply(" for([[int i = 0;]];);"), HasSubstr("extracted"));
  // Don't extract because needs hoisting.
  EXPECT_THAT(apply(" [[int a = 5;]] a++; "), StartsWith("fail"));
  // Don't extract return
  EXPECT_THAT(apply(" if(true) [[return;]] "), StartsWith("fail"));
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

  // FIXME: This should be extractable after selectionTree works correctly for
  // macros (currently it doesn't select anything for the following case)
  std::string MacroFailInput = R"cpp(
    #define F(BODY) void f() { BODY }
    F ([[int x = 0;]])
  )cpp";
  EXPECT_EQ(apply(MacroFailInput), "unavailable");

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
} // namespace
} // namespace clangd
} // namespace clang
