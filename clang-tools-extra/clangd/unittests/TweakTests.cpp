//===-- TweakTests.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "TweakTesting.h"
#include "refactor/Tweak.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using llvm::Failed;
using llvm::Succeeded;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

// FIXME(sammccall): migrate the rest of the tests to use TweakTesting.h and
// remove these helpers.
std::string markRange(llvm::StringRef Code, Range R) {
  size_t Begin = llvm::cantFail(positionToOffset(Code, R.start));
  size_t End = llvm::cantFail(positionToOffset(Code, R.end));
  assert(Begin <= End);
  if (Begin == End) // Mark a single point.
    return (Code.substr(0, Begin) + "^" + Code.substr(Begin)).str();
  // Mark a range.
  return (Code.substr(0, Begin) + "[[" + Code.substr(Begin, End - Begin) +
          "]]" + Code.substr(End))
      .str();
}

void checkAvailable(StringRef ID, llvm::StringRef Input, bool Available) {
  Annotations Code(Input);
  ASSERT_TRUE(0 < Code.points().size() || 0 < Code.ranges().size())
      << "no points of interest specified";
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = Code.code();

  ParsedAST AST = TU.build();

  auto CheckOver = [&](Range Selection) {
    unsigned Begin = cantFail(positionToOffset(Code.code(), Selection.start));
    unsigned End = cantFail(positionToOffset(Code.code(), Selection.end));
    auto T = prepareTweak(ID, Tweak::Selection(AST, Begin, End));
    if (Available)
      EXPECT_THAT_EXPECTED(T, Succeeded())
          << "code is " << markRange(Code.code(), Selection);
    else
      EXPECT_THAT_EXPECTED(T, Failed())
          << "code is " << markRange(Code.code(), Selection);
  };
  for (auto P : Code.points())
    CheckOver(Range{P, P});
  for (auto R : Code.ranges())
    CheckOver(R);
}

/// Checks action is available at every point and range marked in \p Input.
void checkAvailable(StringRef ID, llvm::StringRef Input) {
  return checkAvailable(ID, Input, /*Available=*/true);
}

/// Same as checkAvailable, but checks the action is not available.
void checkNotAvailable(StringRef ID, llvm::StringRef Input) {
  return checkAvailable(ID, Input, /*Available=*/false);
}

llvm::Expected<Tweak::Effect> apply(StringRef ID, llvm::StringRef Input) {
  Annotations Code(Input);
  Range SelectionRng;
  if (Code.points().size() != 0) {
    assert(Code.ranges().size() == 0 &&
           "both a cursor point and a selection range were specified");
    SelectionRng = Range{Code.point(), Code.point()};
  } else {
    SelectionRng = Code.range();
  }
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = Code.code();

  ParsedAST AST = TU.build();
  unsigned Begin = cantFail(positionToOffset(Code.code(), SelectionRng.start));
  unsigned End = cantFail(positionToOffset(Code.code(), SelectionRng.end));
  Tweak::Selection S(AST, Begin, End);

  auto T = prepareTweak(ID, S);
  if (!T)
    return T.takeError();
  return (*T)->apply(S);
}

llvm::Expected<std::string> applyEdit(StringRef ID, llvm::StringRef Input) {
  auto Effect = apply(ID, Input);
  if (!Effect)
    return Effect.takeError();
  if (!Effect->ApplyEdit)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No replacements");
  Annotations Code(Input);
  return applyAllReplacements(Code.code(), *Effect->ApplyEdit);
}

void checkTransform(llvm::StringRef ID, llvm::StringRef Input,
                    std::string Output) {
  auto Result = applyEdit(ID, Input);
  ASSERT_TRUE(bool(Result)) << llvm::toString(Result.takeError()) << Input;
  EXPECT_EQ(Output, std::string(*Result)) << Input;
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

TEST(TweaksTest, ExtractVariable) {
  llvm::StringLiteral ID = "ExtractVariable";
  checkAvailable(ID, R"cpp(
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
        [[a++]];
      // do while
      do
        a = [[1]];
      while(a < [[3]]);
    }
  )cpp");
  // Should not crash.
  checkNotAvailable(ID, R"cpp(
    template<typename T, typename ...Args>
    struct Test<T, Args...> {
    Test(const T &v) :val[[(^]]) {}
      T val;
    };
  )cpp");
  checkNotAvailable(ID, R"cpp(
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
      [[a = 5]];
      [[a >>= 5]];
      [[a *= 5]];
      // Variable DeclRefExpr
      a = [[b]];
      // label statement
      goto label;
      label:
        a = [[1]];
    }
  )cpp");
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
                   PLUS([[1+a]]);
                 })cpp",
          /*FIXME: It should be extracted like this.
           R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   auto dummy = 1+a; int y = PLUS(dummy);
                 })cpp"},*/
           R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   auto dummy = PLUS(1+a); dummy;
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
    checkTransform(ID, IO.first, IO.second);
  }
}

TEST(TweaksTest, AnnotateHighlightings) {
  llvm::StringLiteral ID = "AnnotateHighlightings";
  checkAvailable(ID, "^vo^id^ ^f(^) {^}^"); // available everywhere.
  checkAvailable(ID, "[[int a; int b;]]");
  const char *Input = "void ^f() {}";
  const char *Output = "void /* entity.name.function.cpp */f() {}";
  checkTransform(ID, Input, Output);

  checkTransform(ID,
  R"cpp(
[[void f1();
void f2();]]
)cpp",
  R"cpp(
void /* entity.name.function.cpp */f1();
void /* entity.name.function.cpp */f2();
)cpp");

   checkTransform(ID,
  R"cpp(
void f1();
void f2() {^};
)cpp",

  R"cpp(
void f1();
void /* entity.name.function.cpp */f2() {};
)cpp");
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
}

} // namespace
} // namespace clangd
} // namespace clang
