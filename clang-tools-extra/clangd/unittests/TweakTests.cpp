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
#include "refactor/Tweak.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using llvm::Failed;
using llvm::Succeeded;

namespace clang {
namespace clangd {
namespace {

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

std::string getMessage(StringRef ID, llvm::StringRef Input) {
  auto Effect = apply(ID, Input);
  if (!Effect)
    return "error: " + llvm::toString(Effect.takeError());
  return Effect->ShowMessage.getValueOr("no message produced!");
}

void checkTransform(llvm::StringRef ID, llvm::StringRef Input,
                    std::string Output) {
  auto Result = applyEdit(ID, Input);
  ASSERT_TRUE(bool(Result)) << llvm::toString(Result.takeError()) << Input;
  EXPECT_EQ(Output, std::string(*Result)) << Input;
}

/// Check if apply returns an error and that the @ErrorMessage is contained
/// in that error
void checkApplyContainsError(llvm::StringRef ID, llvm::StringRef Input,
                             const std::string& ErrorMessage) {
  auto Result = apply(ID, Input);
  ASSERT_FALSE(Result) << "expected error message:\n   " << ErrorMessage <<
                       "\non input:" << Input;
  EXPECT_NE(std::string::npos,
            llvm::toString(Result.takeError()).find(ErrorMessage))
            << "Wrong error message:\n  " << llvm::toString(Result.takeError())
            << "\nexpected:\n  " << ErrorMessage;
}

TEST(TweakTest, SwapIfBranches) {
  llvm::StringLiteral ID = "SwapIfBranches";

  checkAvailable(ID, R"cpp(
    void test() {
      ^i^f^^(^t^r^u^e^) { return 100; } ^e^l^s^e^ { continue; }
    }
  )cpp");

  checkNotAvailable(ID, R"cpp(
    void test() {
      if (true) {^return ^100;^ } else { ^continue^;^ }
    }
  )cpp");

  llvm::StringLiteral Input = R"cpp(
    void test() {
      ^if (true) { return 100; } else { continue; }
    }
  )cpp";
  llvm::StringLiteral Output = R"cpp(
    void test() {
      if (true) { continue; } else { return 100; }
    }
  )cpp";
  checkTransform(ID, Input, Output);

  Input = R"cpp(
    void test() {
      ^if () { return 100; } else { continue; }
    }
  )cpp";
  Output = R"cpp(
    void test() {
      if () { continue; } else { return 100; }
    }
  )cpp";
  checkTransform(ID, Input, Output);

  // Available in subexpressions of the condition.
  checkAvailable(ID, R"cpp(
    void test() {
      if(2 + [[2]] + 2) { return 2 + 2 + 2; } else { continue; }
    }
  )cpp");
  // But not as part of the branches.
  checkNotAvailable(ID, R"cpp(
    void test() {
      if(2 + 2 + 2) { return 2 + [[2]] + 2; } else { continue; }
    }
  )cpp");
  // Range covers the "else" token, so available.
  checkAvailable(ID, R"cpp(
    void test() {
      if(2 + 2 + 2) { return 2 + [[2 + 2; } else { continue;]] }
    }
  )cpp");
  // Not available in compound statements in condition.
  checkNotAvailable(ID, R"cpp(
    void test() {
      if([]{return [[true]];}()) { return 2 + 2 + 2; } else { continue; }
    }
  )cpp");
  // Not available if both sides aren't braced.
  checkNotAvailable(ID, R"cpp(
    void test() {
      ^if (1) return; else { return; }
    }
  )cpp");
  // Only one if statement is supported!
  checkNotAvailable(ID, R"cpp(
    [[if(1){}else{}if(2){}else{}]]
  )cpp");
}

TEST(TweakTest, RawStringLiteral) {
  llvm::StringLiteral ID = "RawStringLiteral";

  checkAvailable(ID, R"cpp(
    const char *A = ^"^f^o^o^\^n^";
    const char *B = R"(multi )" ^"token " "str\ning";
  )cpp");

  checkNotAvailable(ID, R"cpp(
    const char *A = ^"f^o^o^o^"; // no chars need escaping
    const char *B = R"(multi )" ^"token " u8"str\ning"; // not all ascii
    const char *C = ^R^"^(^multi )" "token " "str\ning"; // chunk is raw
    const char *D = ^"token\n" __FILE__; // chunk is macro expansion
    const char *E = ^"a\r\n"; // contains forbidden escape character
  )cpp");

  const char *Input = R"cpp(
    const char *X = R"(multi
token)" "\nst^ring\n" "literal";
    }
  )cpp";
  const char *Output = R"cpp(
    const char *X = R"(multi
token
string
literal)";
    }
  )cpp";
  checkTransform(ID, Input, Output);
}

TEST(TweakTest, DumpAST) {
  llvm::StringLiteral ID = "DumpAST";

  checkAvailable(ID, "^int f^oo() { re^turn 2 ^+ 2; }");
  checkNotAvailable(ID, "/*c^omment*/ int foo() return 2 ^ + 2; }");

  const char *Input = "int x = 2 ^+ 2;";
  auto Result = getMessage(ID, Input);
  EXPECT_THAT(Result, ::testing::HasSubstr("BinaryOperator"));
  EXPECT_THAT(Result, ::testing::HasSubstr("'+'"));
  EXPECT_THAT(Result, ::testing::HasSubstr("|-IntegerLiteral"));
  EXPECT_THAT(Result,
              ::testing::HasSubstr("<col:9> 'int' 2\n`-IntegerLiteral"));
  EXPECT_THAT(Result, ::testing::HasSubstr("<col:13> 'int' 2"));
}

TEST(TweakTest, ShowSelectionTree) {
  llvm::StringLiteral ID = "ShowSelectionTree";

  checkAvailable(ID, "^int f^oo() { re^turn 2 ^+ 2; }");
  checkNotAvailable(ID, "/*c^omment*/ int foo() return 2 ^ + 2; }");

  const char *Input = "int fcall(int); int x = fca[[ll(2 +]]2);";
  const char *Output = R"(TranslationUnitDecl 
  VarDecl int x = fcall(2 + 2)
   .CallExpr fcall(2 + 2)
      ImplicitCastExpr fcall
       .DeclRefExpr fcall
     .BinaryOperator 2 + 2
       *IntegerLiteral 2
)";
  EXPECT_EQ(Output, getMessage(ID, Input));
}

TEST(TweakTest, DumpRecordLayout) {
  llvm::StringLiteral ID = "DumpRecordLayout";
  checkAvailable(ID, "^s^truct ^X ^{ int x; ^};");
  checkNotAvailable(ID, "struct X { int ^a; };");
  checkNotAvailable(ID, "struct ^X;");
  checkNotAvailable(ID, "template <typename T> struct ^X { T t; };");
  checkNotAvailable(ID, "enum ^X {};");

  const char *Input = "struct ^X { int x; int y; }";
  EXPECT_THAT(getMessage(ID, Input), ::testing::HasSubstr("0 |   int x"));
}
TEST(TweakTest, ExtractVariable) {
  llvm::StringLiteral ID = "ExtractVariable";
  checkAvailable(ID, R"cpp(
    int xyz() {
      // return statement
      return ^1;
    }
    void f() {
      int a = 5 + [[4 ^* ^xyz^()]];
      // multivariable initialization
      if(1)
        int x = ^1, y = ^a + 1, a = ^1, z = a + 1;
      // if without else
      if(^1) {}
      // if with else
      if(a < ^3)
        if(a == ^4)
          a = ^5;
        else
          a = ^6;
      else if (a < ^4)
        a = ^4;
      else
        a = ^5;
      // for loop 
      for(a = ^1; a > ^3^+^4; a++)
        a = ^2;
      // while 
      while(a < ^1)
        ^a++;
      // do while 
      do
        a = ^1;
      while(a < ^3);
    }
  )cpp");
  // Should not crash.
  checkNotAvailable(ID, R"cpp(
    template<typename T, typename ...Args>
    struct Test<T, Args...> {
    Test(const T &v) :val(^) {}
      T val;
    };
  )cpp");
  checkNotAvailable(ID, R"cpp(
    int xyz(int a = ^1) {
      return 1;
      class T {
        T(int a = ^1) {};
        int xyz = ^1;
      };
    }
    // function default argument
    void f(int b = ^1) {
      // void expressions
      auto i = new int, j = new int;
      de^lete i^, del^ete j;
      // if
      if(1)
        int x = 1, y = a + 1, a = 1, z = ^a + 1;
      if(int a = 1)
        if(^a == 4)
          a = ^a ^+ 1;
      // for loop 
      for(int a = 1, b = 2, c = 3; ^a > ^b ^+ ^c; ^a++)
        a = ^a ^+ 1;
      // lambda 
      auto lamb = [&^a, &^b](int r = ^1) {return 1;}
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
          // ensure InsertionPoint isn't inside a macro
          {R"cpp(#define LOOP(x) {int a = x + 1;}
                 void f(int a) {
                   if(1)
                    LOOP(5 + ^3)
                 })cpp",
           R"cpp(#define LOOP(x) {int a = x + 1;}
                 void f(int a) {
                   auto dummy = 3; if(1)
                    LOOP(5 + dummy)
                 })cpp"},
          // label and attribute testing
          {R"cpp(void f(int a) {
                    label: [ [gsl::suppress("type")] ] for (;;) a = ^1;
                 })cpp",
           R"cpp(void f(int a) {
                    auto dummy = 1; label: [ [gsl::suppress("type")] ] for (;;) a = dummy;
                 })cpp"},
          // FIXME: Doesn't work because bug in selection tree
          /*{R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   PLUS(^a);
                 })cpp",
           R"cpp(#define PLUS(x) x++
                 void f(int a) {
                   auto dummy = a; PLUS(dummy);
                 })cpp"},*/
          // FIXME: Doesn't work correctly for \[\[clang::uninitialized\]\] int b
          // = 1; since the attr is inside the DeclStmt and the bounds of
          // DeclStmt don't cover the attribute
      };
  for (const auto &IO : InputOutputs) {
    checkTransform(ID, IO.first, IO.second);
  }
}

TEST(TweakTest, AnnotateHighlightings) {
  llvm::StringLiteral ID = "AnnotateHighlightings";
  checkAvailable(ID, "^vo^id^ ^f(^) {^}^"); // available everywhere.
  const char *Input = "void ^f() {}";
  const char *Output = "void /* entity.name.function.cpp */f() {}";
  checkTransform(ID, Input, Output);
}

TEST(TweakTest, ExpandMacro) {
  llvm::StringLiteral ID = "ExpandMacro";

  // Available on macro names, not available anywhere else.
  checkAvailable(ID, R"cpp(
#define FOO 1 2 3
#define FUNC(X) X+X+X
^F^O^O^ BAR ^F^O^O^
^F^U^N^C^(1)
)cpp");
  checkNotAvailable(ID, R"cpp(
^#^d^efine^ ^FO^O 1 ^2 ^3^
FOO ^B^A^R^ FOO ^
FUNC(^1^)^
)cpp");

  // Works as expected on object-like macros.
  checkTransform(ID, R"cpp(
#define FOO 1 2 3
^FOO BAR FOO
)cpp",
                 R"cpp(
#define FOO 1 2 3
1 2 3 BAR FOO
)cpp");
  checkTransform(ID, R"cpp(
#define FOO 1 2 3
FOO BAR ^FOO
)cpp",
                 R"cpp(
#define FOO 1 2 3
FOO BAR 1 2 3
)cpp");

  // And function-like macros.
  checkTransform(ID, R"cpp(
#define FUNC(X) X+X+X
F^UNC(2)
)cpp",
                 R"cpp(
#define FUNC(X) X+X+X
2 + 2 + 2
)cpp");

  // Works on empty macros.
  checkTransform(ID, R"cpp(
#define EMPTY
int a ^EMPTY;
  )cpp",
                 R"cpp(
#define EMPTY
int a ;
  )cpp");
  checkTransform(ID, R"cpp(
#define EMPTY_FN(X)
int a ^EMPTY_FN(1 2 3);
  )cpp",
                 R"cpp(
#define EMPTY_FN(X)
int a ;
  )cpp");
  checkTransform(ID, R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123 ^EMPTY EMPTY_FN(1);
  )cpp",
                 R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123  EMPTY_FN(1);
  )cpp");
  checkTransform(ID, R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123 ^EMPTY_FN(1) EMPTY;
  )cpp",
                 R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123  EMPTY;
  )cpp");
  checkTransform(ID, R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123 EMPTY_FN(1) ^EMPTY;
  )cpp",
                 R"cpp(
#define EMPTY
#define EMPTY_FN(X)
int a = 123 EMPTY_FN(1) ;
  )cpp");
}

TEST(TweakTest, ExpandAutoType) {
  llvm::StringLiteral ID = "ExpandAutoType";

  checkAvailable(ID, R"cpp(
    ^a^u^t^o^ i = 0;
  )cpp");

  checkNotAvailable(ID, R"cpp(
    auto ^i^ ^=^ ^0^;^
  )cpp");

  llvm::StringLiteral Input = R"cpp(
    [[auto]] i = 0;
  )cpp";
  llvm::StringLiteral Output = R"cpp(
    int i = 0;
  )cpp";
  checkTransform(ID, Input, Output);

  // check primitive type
  Input = R"cpp(
    au^to i = 0;
  )cpp";
  Output = R"cpp(
    int i = 0;
  )cpp";
  checkTransform(ID, Input, Output);

  // check classes and namespaces
  Input = R"cpp(
    namespace testns {
      class TestClass {
        class SubClass {};
      };
    }
    ^auto C = testns::TestClass::SubClass();
  )cpp";
  Output = R"cpp(
    namespace testns {
      class TestClass {
        class SubClass {};
      };
    }
    testns::TestClass::SubClass C = testns::TestClass::SubClass();
  )cpp";
  checkTransform(ID, Input, Output);

  // check that namespaces are shortened
  Input = R"cpp(
    namespace testns {
    class TestClass {
    };
    void func() { ^auto C = TestClass(); }
    }
  )cpp";
  Output = R"cpp(
    namespace testns {
    class TestClass {
    };
    void func() { TestClass C = TestClass(); }
    }
  )cpp";
  checkTransform(ID, Input, Output);

  // unknown types in a template should not be replaced
  Input = R"cpp(
    template <typename T> void x() {
        ^auto y =  T::z();
        }
  )cpp";
  checkApplyContainsError(ID, Input, "Could not deduce type for 'auto' type");

  // undefined functions should not be replaced
  Input = R"cpp(
    a^uto x = doesnt_exist();
  )cpp";
  checkApplyContainsError(ID, Input, "Could not deduce type for 'auto' type");

  // function pointers should not be replaced
  Input = R"cpp(
    int foo();
    au^to x = &foo;
  )cpp";
  checkApplyContainsError(ID, Input,
      "Could not expand type of function pointer");

  // lambda types are not replaced
  Input = R"cpp(
    au^to x = []{};
  )cpp";
  checkApplyContainsError(ID, Input,
      "Could not expand type of lambda expression");

  // inline namespaces
  Input = R"cpp(
    inline namespace x {
      namespace { struct S; }
    }
    au^to y = S();
  )cpp";
  Output = R"cpp(
    inline namespace x {
      namespace { struct S; }
    }
    S y = S();
  )cpp";

  // local class
  Input = R"cpp(
  namespace x {
    void y() {
      struct S{};
      a^uto z = S();
  }}
  )cpp";
  Output = R"cpp(
  namespace x {
    void y() {
      struct S{};
      S z = S();
  }}
  )cpp";
  checkTransform(ID, Input, Output);

  // replace array types
  Input = R"cpp(
    au^to x = "test";
  )cpp";
  Output = R"cpp(
    const char * x = "test";
  )cpp";
  checkTransform(ID, Input, Output);
}

} // namespace
} // namespace clangd
} // namespace clang
