//===- BuildTreeTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the syntax tree generation from the ClangAST.
//
//===----------------------------------------------------------------------===//

#include "TreeTestBase.h"

using namespace clang;
using namespace clang::syntax;

namespace {

TEST_P(SyntaxTreeTest, Simple) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int main() {}
void foo() {}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-main
| | `-ParametersAndQualifiers
| |   |-(
| |   `-)
| `-CompoundStatement
|   |-{
|   `-}
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, SimpleVariable) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a;
int b = 42;
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | `-a
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-b
  | |-=
  | `-IntegerLiteralExpression
  |   `-42
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, SimpleFunction) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void foo(int a, int b) {}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-b
  |   `-)
  `-CompoundStatement
    |-{
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, If) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[if (1) {}]]
  [[if (1) {} else if (0) {}]]
}
)cpp",
      {R"txt(
IfStatement
|-if
|-(
|-IntegerLiteralExpression
| `-1
|-)
`-CompoundStatement
  |-{
  `-}
  )txt",
       R"txt(
IfStatement
|-if
|-(
|-IntegerLiteralExpression
| `-1
|-)
|-CompoundStatement
| |-{
| `-}
|-else
`-IfStatement
  |-if
  |-(
  |-IntegerLiteralExpression
  | `-0
  |-)
  `-CompoundStatement
    |-{
    `-}
)txt"}));
}

TEST_P(SyntaxTreeTest, For) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[for (;;)  {}]]
}
)cpp",
      {R"txt(
ForStatement
|-for
|-(
|-;
|-;
|-)
`-CompoundStatement
  |-{
  `-}
)txt"}));
}

TEST_P(SyntaxTreeTest, RangeBasedFor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  int a[3];
  [[for (int x : a)
    ;]]
}
)cpp",
      {R"txt(
RangeBasedForStatement
|-for
|-(
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | `-x
| `-:
|-IdExpression
| `-UnqualifiedId
|   `-a
|-)
`-EmptyStatement
  `-;
)txt"}));
}

TEST_P(SyntaxTreeTest, DeclarationStatement) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[int a = 10;]]
}
)cpp",
      {R"txt(
DeclarationStatement
|-SimpleDeclaration
| |-int
| `-SimpleDeclarator
|   |-a
|   |-=
|   `-IntegerLiteralExpression
|     `-10
`-;
)txt"}));
}

TEST_P(SyntaxTreeTest, Switch) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[switch (1) {
    case 0:
    default:;
  }]]
}
)cpp",
      {R"txt(
SwitchStatement
|-switch
|-(
|-IntegerLiteralExpression
| `-1
|-)
`-CompoundStatement
  |-{
  |-CaseStatement
  | |-case
  | |-IntegerLiteralExpression
  | | `-0
  | |-:
  | `-DefaultStatement
  |   |-default
  |   |-:
  |   `-EmptyStatement
  |     `-;
  `-}
)txt"}));
}

TEST_P(SyntaxTreeTest, While) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[while (1) { continue; break; }]]
}
)cpp",
      {R"txt(
WhileStatement
|-while
|-(
|-IntegerLiteralExpression
| `-1
|-)
`-CompoundStatement
  |-{
  |-ContinueStatement
  | |-continue
  | `-;
  |-BreakStatement
  | |-break
  | `-;
  `-}
)txt"}));
}

TEST_P(SyntaxTreeTest, UnhandledStatement) {
  // Unhandled statements should end up as 'unknown statement'.
  // This example uses a 'label statement', which does not yet have a syntax
  // counterpart.
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
int test() {
  [[foo: return 100;]]
}
)cpp",
      {R"txt(
UnknownStatement
|-foo
|-:
`-ReturnStatement
  |-return
  |-IntegerLiteralExpression
  | `-100
  `-;
)txt"}));
}

TEST_P(SyntaxTreeTest, Expressions) {
  // expressions should be wrapped in 'ExpressionStatement' when they appear
  // in a statement position.
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  test();
  if (1) test(); else test();
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-test
    | | |-(
    | | `-)
    | `-;
    |-IfStatement
    | |-if
    | |-(
    | |-IntegerLiteralExpression
    | | `-1
    | |-)
    | |-ExpressionStatement
    | | |-UnknownExpression
    | | | |-IdExpression
    | | | | `-UnqualifiedId
    | | | |   `-test
    | | | |-(
    | | | `-)
    | | `-;
    | |-else
    | `-ExpressionStatement
    |   |-UnknownExpression
    |   | |-IdExpression
    |   | | `-UnqualifiedId
    |   | |   `-test
    |   | |-(
    |   | `-)
    |   `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_Identifier) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a) {
  [[a]];
}
)cpp",
      {R"txt(
IdExpression
`-UnqualifiedId
  `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_OperatorFunctionId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  friend X operator+(const X&, const X&);
};
void test(X x) {
  [[operator+(x, x)]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| `-UnqualifiedId
|   |-operator
|   `-+
|-(
|-IdExpression
| `-UnqualifiedId
|   `-x
|-,
|-IdExpression
| `-UnqualifiedId
|   `-x
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_ConversionFunctionId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  operator int();
};
void test(X x) {
  [[x.operator int()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-x
| |-.
| `-IdExpression
|   `-UnqualifiedId
|     |-operator
|     `-int
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_LiteralOperatorId) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
unsigned operator "" _w(char);
void test() {
  [[operator "" _w('1')]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| `-UnqualifiedId
|   |-operator
|   |-""
|   `-_w
|-(
|-CharacterLiteralExpression
| `-'1'
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_Destructor) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X { };
void test(X x) {
  [[x.~X()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-x
| |-.
| `-IdExpression
|   `-UnqualifiedId
|     |-~
|     `-X
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_DecltypeDestructor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X { };
void test(X x) {
  // FIXME: Make `decltype(x)` a child of `MemberExpression`. It is currently
  // not because `Expr::getSourceRange()` returns the range of `x.~` for the
  // `MemberExpr` instead of the expected `x.~decltype(x)`, this is a bug in
  // clang.
  [[x.~decltype(x)()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-x
| |-.
| `-IdExpression
|   `-UnqualifiedId
|     `-~
|-decltype
|-(
|-x
|-)
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_TemplateId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T>
T f();
void test() {
  [[f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| `-UnqualifiedId
|   |-f
|   |-<
|   |-int
|   `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_NamespaceSpecifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
namespace n {
  struct S { };
}
void test() {
  // FIXME: Remove the `UnknownExpression` wrapping `s1` and `s2`. This
  // `UnknownExpression` comes from a leaf `CXXConstructExpr` in the
  // ClangAST. We need to ignore leaf implicit nodes.
  [[::n::S s1]];
  [[n::S s2]];
}
)cpp",
      {R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-::
| |-IdentifierNameSpecifier
| | `-n
| `-::
|-S
`-SimpleDeclarator
  `-UnknownExpression
    `-s1
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-IdentifierNameSpecifier
| | `-n
| `-::
|-S
`-SimpleDeclarator
  `-UnknownExpression
    `-s2
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_TemplateSpecifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T>
struct ST {
  struct S { };
};
void test() {
  [[::template ST<int>::S s1]];
  [[::ST<int>::S s2]];
}
)cpp",
      {R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-::
| |-SimpleTemplateNameSpecifier
| | |-template
| | |-ST
| | |-<
| | |-int
| | `->
| `-::
|-S
`-SimpleDeclarator
  `-UnknownExpression
    `-s1
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-::
| |-SimpleTemplateNameSpecifier
| | |-ST
| | |-<
| | |-int
| | `->
| `-::
|-S
`-SimpleDeclarator
  `-UnknownExpression
    `-s2
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_DecltypeSpecifier) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  static void f(){}
};
void test(S s) {
  [[decltype(s)::f()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-DecltypeNameSpecifier
| | | |-decltype
| | | |-(
| | | |-IdExpression
| | | | `-UnqualifiedId
| | | |   `-s
| | | `-)
| | `-::
| `-UnqualifiedId
|   `-f
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_OptionalTemplateKw) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  template<typename U>
  static U f();
};
void test() {
  [[S::f<int>()]];
  [[S::template f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier
| | | `-S
| | `-::
| `-UnqualifiedId
|   |-f
|   |-<
|   |-int
|   `->
|-(
`-)
)txt",
       R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier
| | | `-S
| | `-::
| |-template
| `-UnqualifiedId
|   |-f
|   |-<
|   |-int
|   `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_Complex) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
namespace n {
  template<typename T>
  struct ST {
    template<typename U>
    static U f();
  };
}
void test() {
  [[::n::template ST<int>::template f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-::
| | |-IdentifierNameSpecifier
| | | `-n
| | |-::
| | |-SimpleTemplateNameSpecifier
| | | |-template
| | | |-ST
| | | |-<
| | | |-int
| | | `->
| | `-::
| |-template
| `-UnqualifiedId
|   |-f
|   |-<
|   |-int
|   `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, QualifiedId_DependentType) {
  if (!GetParam().isCXX()) {
    return;
  }
  if (GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Make this test work on Windows by generating the expected syntax
    // tree when `-fdelayed-template-parsing` is active.
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <typename T>
void test() {
  [[T::template U<int>::f()]];
  [[T::U::f()]];
  [[T::template f<0>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier
| | | `-T
| | |-::
| | |-SimpleTemplateNameSpecifier
| | | |-template
| | | |-U
| | | |-<
| | | |-int
| | | `->
| | `-::
| `-UnqualifiedId
|   `-f
|-(
`-)
)txt",
       R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier
| | | `-T
| | |-::
| | |-IdentifierNameSpecifier
| | | `-U
| | `-::
| `-UnqualifiedId
|   `-f
|-(
`-)
)txt",
       R"txt(
UnknownExpression
|-IdExpression
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier
| | | `-T
| | `-::
| |-template
| `-UnqualifiedId
|   |-f
|   |-<
|   |-IntegerLiteralExpression
|   | `-0
|   `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, This_Simple) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  S* test(){
    return [[this]];
  }
};
)cpp",
      {R"txt(
ThisExpression
`-this
)txt"}));
}

TEST_P(SyntaxTreeTest, This_ExplicitMemberAccess) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  int a;
  void test(){
    [[this->a]];
  }
};
)cpp",
      {R"txt(
MemberExpression
|-ThisExpression
| `-this
|-->
`-IdExpression
  `-UnqualifiedId
    `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, This_ImplicitMemberAccess) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  int a;
  void test(){
    [[a]];
  }
};
)cpp",
      {R"txt(
IdExpression
`-UnqualifiedId
  `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, ParenExpr) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[(1)]];
  [[((1))]];
  [[(1 + (2))]];
}
)cpp",
      {R"txt(
ParenExpression
|-(
|-IntegerLiteralExpression
| `-1
`-)
)txt",
       R"txt(
ParenExpression
|-(
|-ParenExpression
| |-(
| |-IntegerLiteralExpression
| | `-1
| `-)
`-)
)txt",
       R"txt(
ParenExpression
|-(
|-BinaryOperatorExpression
| |-IntegerLiteralExpression
| | `-1
| |-+
| `-ParenExpression
|   |-(
|   |-IntegerLiteralExpression
|   | `-2
|   `-)
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Char) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
unsigned operator "" _c(char);
void test() {
  [['2'_c]];
}
    )cpp",
      {R"txt(
CharUserDefinedLiteralExpression
`-'2'_c
)txt"}));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_String) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
typedef decltype(sizeof(void *)) size_t;

unsigned operator "" _s(const char*, size_t);

void test() {
  [["12"_s]];
}
    )cpp",
      {R"txt(
StringUserDefinedLiteralExpression
`-"12"_s
)txt"}));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Integer) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
unsigned operator "" _i(unsigned long long);
unsigned operator "" _r(const char*);
template <char...>
unsigned operator "" _t();

void test() {
  [[12_i]];
  [[12_r]];
  [[12_t]];
}
    )cpp",
      {R"txt(
IntegerUserDefinedLiteralExpression
`-12_i
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression
`-12_r
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression
`-12_t
)txt"}));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Float) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
unsigned operator "" _f(long double);
unsigned operator "" _r(const char*);
template <char...>
unsigned operator "" _t();

void test() {
  [[1.2_f]];
  [[1.2_r]];
  [[1.2_t]];
}
    )cpp",
      {R"txt(
FloatUserDefinedLiteralExpression
`-1.2_f
)txt",
       R"txt(
FloatUserDefinedLiteralExpression
`-1.2_r
)txt",
       R"txt(
FloatUserDefinedLiteralExpression
`-1.2_t
)txt"}));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_LongLong) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[12ll]];
  [[12ull]];
}
)cpp",
      {R"txt(
IntegerLiteralExpression
`-12ll
)txt",
       R"txt(
IntegerLiteralExpression
`-12ull
)txt"}));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_Binary) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[0b1100]];
}
)cpp",
      {R"txt(
IntegerLiteralExpression
`-0b1100
)txt"}));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_WithDigitSeparators) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[1'2'0ull]];
}
)cpp",
      {R"txt(
IntegerLiteralExpression
`-1'2'0ull
)txt"}));
}

TEST_P(SyntaxTreeTest, CharacterLiteral) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [['a']];
  [['\n']];
  [['\x20']];
  [['\0']];
  [[L'a']];
  [[L'α']];
}
)cpp",
      {R"txt(
CharacterLiteralExpression
`-'a'
)txt",
       R"txt(
CharacterLiteralExpression
`-'\n'
)txt",
       R"txt(
CharacterLiteralExpression
`-'\x20'
)txt",
       R"txt(
CharacterLiteralExpression
`-'\0'
)txt",
       R"txt(
CharacterLiteralExpression
`-L'a'
)txt",
       R"txt(
CharacterLiteralExpression
`-L'α'
)txt"}));
}

TEST_P(SyntaxTreeTest, CharacterLiteral_Utf) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[u'a']];
  [[u'構']];
  [[U'a']];
  [[U'🌲']];
}
)cpp",
      {R"txt(
CharacterLiteralExpression
`-u'a'
)txt",
       R"txt(
CharacterLiteralExpression
`-u'構'
)txt",
       R"txt(
CharacterLiteralExpression
`-U'a'
)txt",
       R"txt(
CharacterLiteralExpression
`-U'🌲'
)txt"}));
}

TEST_P(SyntaxTreeTest, CharacterLiteral_Utf8) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[u8'a']];
  [[u8'\x7f']];
}
)cpp",
      {R"txt(
CharacterLiteralExpression
`-u8'a'
)txt",
       R"txt(
CharacterLiteralExpression
`-u8'\x7f'
)txt"}));
}

TEST_P(SyntaxTreeTest, FloatingLiteral) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[1e-2]];
  [[2.]];
  [[.2]];
  [[2.f]];
}
)cpp",
      {R"txt(
FloatingLiteralExpression
`-1e-2
)txt",
       R"txt(
FloatingLiteralExpression
`-2.
)txt",
       R"txt(
FloatingLiteralExpression
`-.2
)txt",
       R"txt(
FloatingLiteralExpression
`-2.f
)txt"}));
}

TEST_P(SyntaxTreeTest, FloatingLiteral_Hexadecimal) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[0xfp1]];
  [[0xf.p1]];
  [[0x.fp1]];
  [[0xf.fp1f]];
}
)cpp",
      {R"txt(
FloatingLiteralExpression
`-0xfp1
)txt",
       R"txt(
FloatingLiteralExpression
`-0xf.p1
)txt",
       R"txt(
FloatingLiteralExpression
`-0x.fp1
)txt",
       R"txt(
FloatingLiteralExpression
`-0xf.fp1f
)txt"}));
}

TEST_P(SyntaxTreeTest, StringLiteral) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [["a\n\0\x20"]];
  [[L"αβ"]];
}
)cpp",
      {R"txt(
StringLiteralExpression
`-"a\n\0\x20"
)txt",
       R"txt(
StringLiteralExpression
`-L"αβ"
)txt"}));
}

TEST_P(SyntaxTreeTest, StringLiteral_Utf) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[u8"a\x1f\x05"]];
  [[u"C++抽象構文木"]];
  [[U"📖🌲\n"]];
}
)cpp",
      {R"txt(
StringLiteralExpression
`-u8"a\x1f\x05"
)txt",
       R"txt(
StringLiteralExpression
`-u"C++抽象構文木"
)txt",
       R"txt(
StringLiteralExpression
`-U"📖🌲\n"
)txt"}));
}

TEST_P(SyntaxTreeTest, StringLiteral_Raw) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  // This test uses regular string literals instead of raw string literals to
  // hold source code and expected output because of a bug in MSVC up to MSVC
  // 2019 16.2:
  // https://developercommunity.visualstudio.com/content/problem/67300/stringifying-raw-string-literal.html
  EXPECT_TRUE(treeDumpEqual( //
      "void test() {\n"
      "  R\"SyntaxTree(\n"
      "  Hello \"Syntax\" \\\"\n"
      "  )SyntaxTree\";\n"
      "}\n",
      "*: TranslationUnit\n"
      "`-SimpleDeclaration\n"
      "  |-void\n"
      "  |-SimpleDeclarator\n"
      "  | |-test\n"
      "  | `-ParametersAndQualifiers\n"
      "  |   |-(\n"
      "  |   `-)\n"
      "  `-CompoundStatement\n"
      "    |-{\n"
      "    |-ExpressionStatement\n"
      "    | |-StringLiteralExpression\n"
      "    | | `-R\"SyntaxTree(\n"
      "  Hello \"Syntax\" \\\"\n"
      "  )SyntaxTree\"\n"
      "    | `-;\n"
      "    `-}\n"));
}

TEST_P(SyntaxTreeTest, BoolLiteral) {
  if (GetParam().isC()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[true]];
  [[false]];
}
)cpp",
      {R"txt(
BoolLiteralExpression
`-true
)txt",
       R"txt(
BoolLiteralExpression
`-false
)txt"}));
}

TEST_P(SyntaxTreeTest, CxxNullPtrLiteral) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[nullptr]];
}
)cpp",
      {R"txt(
CxxNullPtrExpression
`-nullptr
)txt"}));
}

TEST_P(SyntaxTreeTest, PostfixUnaryOperator) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a) {
  [[a++]];
  [[a--]];
}
)cpp",
      {R"txt(
PostfixUnaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
`-++
)txt",
       R"txt(
PostfixUnaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
`---
)txt"}));
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperator) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a, int *ap) {
  [[--a]]; [[++a]];
  [[~a]];
  [[-a]];
  [[+a]];
  [[&a]];
  [[*ap]];
  [[!a]];
  [[__real a]]; [[__imag a]];
}
)cpp",
      {R"txt(
PrefixUnaryOperatorExpression
|---
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-++
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-~
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|--
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-+
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-&
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-*
`-IdExpression
  `-UnqualifiedId
    `-ap
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-!
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-__real
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-__imag
`-IdExpression
  `-UnqualifiedId
    `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a, bool b) {
  [[compl a]];
  [[not b]];
}
)cpp",
      {R"txt(
PrefixUnaryOperatorExpression
|-compl
`-IdExpression
  `-UnqualifiedId
    `-a
)txt",
       R"txt(
PrefixUnaryOperatorExpression
|-not
`-IdExpression
  `-UnqualifiedId
    `-b
)txt"}));
}

TEST_P(SyntaxTreeTest, BinaryOperator) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a) {
  [[1 - 2]];
  [[1 == 2]];
  [[a = 1]];
  [[a <<= 1]];
  [[1 || 0]];
  [[1 & 2]];
  [[a != 3]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IntegerLiteralExpression
| `-1
|--
`-IntegerLiteralExpression
  `-2
)txt",
       R"txt(
BinaryOperatorExpression
|-IntegerLiteralExpression
| `-1
|-==
`-IntegerLiteralExpression
  `-2
)txt",
       R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
|-=
`-IntegerLiteralExpression
  `-1
)txt",
       R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
|-<<=
`-IntegerLiteralExpression
  `-1
)txt",
       R"txt(
BinaryOperatorExpression
|-IntegerLiteralExpression
| `-1
|-||
`-IntegerLiteralExpression
  `-0
)txt",
       R"txt(
BinaryOperatorExpression
|-IntegerLiteralExpression
| `-1
|-&
`-IntegerLiteralExpression
  `-2
)txt",
       R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
|-!=
`-IntegerLiteralExpression
  `-3
)txt"}));
}

TEST_P(SyntaxTreeTest, BinaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a) {
  [[true || false]];
  [[true or false]];
  [[1 bitand 2]];
  [[a xor_eq 3]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-BoolLiteralExpression
| `-true
|-||
`-BoolLiteralExpression
  `-false
)txt",
       R"txt(
BinaryOperatorExpression
|-BoolLiteralExpression
| `-true
|-or
`-BoolLiteralExpression
  `-false
)txt",
       R"txt(
BinaryOperatorExpression
|-IntegerLiteralExpression
| `-1
|-bitand
`-IntegerLiteralExpression
  `-2
)txt",
       R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
|-xor_eq
`-IntegerLiteralExpression
  `-3
)txt"}));
}

TEST_P(SyntaxTreeTest, BinaryOperator_NestedWithParenthesis) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[(1 + 2) * (4 / 2)]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-ParenExpression
| |-(
| |-BinaryOperatorExpression
| | |-IntegerLiteralExpression
| | | `-1
| | |-+
| | `-IntegerLiteralExpression
| |   `-2
| `-)
|-*
`-ParenExpression
  |-(
  |-BinaryOperatorExpression
  | |-IntegerLiteralExpression
  | | `-4
  | |-/
  | `-IntegerLiteralExpression
  |   `-2
  `-)
)txt"}));
}

TEST_P(SyntaxTreeTest, BinaryOperator_Associativity) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test(int a, int b) {
  [[a + b + 42]];
  [[a = b = 42]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-BinaryOperatorExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-a
| |-+
| `-IdExpression
|   `-UnqualifiedId
|     `-b
|-+
`-IntegerLiteralExpression
  `-42
)txt",
       R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-a
|-=
`-BinaryOperatorExpression
  |-IdExpression
  | `-UnqualifiedId
  |   `-b
  |-=
  `-IntegerLiteralExpression
    `-42
)txt"}));
}

TEST_P(SyntaxTreeTest, BinaryOperator_Precedence) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void test() {
  [[1 + 2 * 3 + 4]];
  [[1 % 2 + 3 * 4]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-BinaryOperatorExpression
| |-IntegerLiteralExpression
| | `-1
| |-+
| `-BinaryOperatorExpression
|   |-IntegerLiteralExpression
|   | `-2
|   |-*
|   `-IntegerLiteralExpression
|     `-3
|-+
`-IntegerLiteralExpression
  `-4
)txt",
       R"txt(
BinaryOperatorExpression
|-BinaryOperatorExpression
| |-IntegerLiteralExpression
| | `-1
| |-%
| `-IntegerLiteralExpression
|   `-2
|-+
`-BinaryOperatorExpression
  |-IntegerLiteralExpression
  | `-3
  |-*
  `-IntegerLiteralExpression
    `-4
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Assignment) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X& operator=(const X&);
};
void test(X x, X y) {
  [[x = y]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-x
|-=
`-IdExpression
  `-UnqualifiedId
    `-y
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Plus) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  friend X operator+(X, const X&);
};
// FIXME: Remove additional `UnknownExpression` wrapping `x`. For that, ignore
// implicit copy constructor called on `x`. This should've been ignored already,
// as we `IgnoreImplicit` when traversing an `Stmt`.
void test(X x, X y) {
  [[x + y]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-UnknownExpression
| `-IdExpression
|   `-UnqualifiedId
|     `-x
|-+
`-IdExpression
  `-UnqualifiedId
    `-y
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Less) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  friend bool operator<(const X&, const X&);
};
void test(X x, X y) {
  [[x < y]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-x
|-<
`-IdExpression
  `-UnqualifiedId
    `-y
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_LeftShift) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  friend X operator<<(X&, const X&);
};
void test(X x, X y) {
  [[x << y]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-x
|-<<
`-IdExpression
  `-UnqualifiedId
    `-y
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Comma) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X operator,(X&);
};
void test(X x, X y) {
  [[x, y]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-x
|-,
`-IdExpression
  `-UnqualifiedId
    `-y
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PointerToMember) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X operator->*(int);
};
void test(X* xp, int X::* pmi) {
  [[xp->*pmi]];
}
)cpp",
      {R"txt(
BinaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-xp
|-->*
`-IdExpression
  `-UnqualifiedId
    `-pmi
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Negation) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  bool operator!();
};
void test(X x) {
  [[!x]];
}
)cpp",
      {R"txt(
PrefixUnaryOperatorExpression
|-!
`-IdExpression
  `-UnqualifiedId
    `-x
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_AddressOf) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X* operator&();
};
void test(X x) {
  [[&x]];
}
)cpp",
      {R"txt(
PrefixUnaryOperatorExpression
|-&
`-IdExpression
  `-UnqualifiedId
    `-x
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PrefixIncrement) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X operator++();
};
void test(X x) {
  [[++x]];
}
)cpp",
      {R"txt(
PrefixUnaryOperatorExpression
|-++
`-IdExpression
  `-UnqualifiedId
    `-x
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PostfixIncrement) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  X operator++(int);
};
void test(X x) {
  [[x++]];
}
)cpp",
      {R"txt(
PostfixUnaryOperatorExpression
|-IdExpression
| `-UnqualifiedId
|   `-x
`-++
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_SimpleWithDot) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  int a;
};
void test(struct S s) {
  [[s.a]];
}
)cpp",
      {R"txt(
MemberExpression
|-IdExpression
| `-UnqualifiedId
|   `-s
|-.
`-IdExpression
  `-UnqualifiedId
    `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_StaticDataMember) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  static int a;
};
void test(S s) {
  [[s.a]];
}
)cpp",
      {R"txt(
MemberExpression
|-IdExpression
| `-UnqualifiedId
|   `-s
|-.
`-IdExpression
  `-UnqualifiedId
    `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_SimpleWithArrow) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  int a;
};
void test(struct S* sp) {
  [[sp->a]];
}
)cpp",
      {R"txt(
MemberExpression
|-IdExpression
| `-UnqualifiedId
|   `-sp
|-->
`-IdExpression
  `-UnqualifiedId
    `-a
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_Chaining) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  struct S* next;
};
void test(struct S s){
  [[s.next->next]];
}
)cpp",
      {R"txt(
MemberExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-s
| |-.
| `-IdExpression
|   `-UnqualifiedId
|     `-next
|-->
`-IdExpression
  `-UnqualifiedId
    `-next
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_OperatorFunction) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  bool operator!();
};
void test(S s) {
  [[s.operator!()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-s
| |-.
| `-IdExpression
|   `-UnqualifiedId
|     |-operator
|     `-!
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_VariableTemplate) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  template<typename T>
  static constexpr T x = 42;
};
// FIXME: `<int>` should be a child of `MemberExpression` and `;` of
// `ExpressionStatement`. This is a bug in clang, in `getSourceRange` methods.
void test(S s) [[{
  s.x<int>;
}]]
)cpp",
      {R"txt(
CompoundStatement
|-{
|-ExpressionStatement
| `-MemberExpression
|   |-IdExpression
|   | `-UnqualifiedId
|   |   `-s
|   |-.
|   `-IdExpression
|     `-UnqualifiedId
|       `-x
|-<
|-int
|->
|-;
`-}
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_FunctionTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  template<typename T>
  T f();
};
void test(S* sp){
  [[sp->f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-sp
| |-->
| `-IdExpression
|   `-UnqualifiedId
|     |-f
|     |-<
|     |-int
|     `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_FunctionTemplateWithTemplateKeyword) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  template<typename T>
  T f();
};
void test(S s){
  [[s.template f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-s
| |-.
| |-template
| `-IdExpression
|   `-UnqualifiedId
|     |-f
|     |-<
|     |-int
|     `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_WithQualifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct Base {
  void f();
};
struct S : public Base {};
void test(S s){
  [[s.Base::f()]];
  [[s.::S::~S()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-s
| |-.
| `-IdExpression
|   |-NestedNameSpecifier
|   | |-IdentifierNameSpecifier
|   | | `-Base
|   | `-::
|   `-UnqualifiedId
|     `-f
|-(
`-)
      )txt",
       R"txt(
UnknownExpression
|-MemberExpression
| |-IdExpression
| | `-UnqualifiedId
| |   `-s
| |-.
| `-IdExpression
|   |-NestedNameSpecifier
|   | |-::
|   | |-IdentifierNameSpecifier
|   | | `-S
|   | `-::
|   `-UnqualifiedId
|     |-~
|     `-S
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberExpression_Complex) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T>
struct U {
  template<typename U>
  U f();
};
struct S {
  U<int> getU();
};
void test(S* sp) {
  // FIXME: The first 'template' keyword is a child of `NestedNameSpecifier`,
  // but it should be a child of `MemberExpression` according to the grammar.
  // However one might argue that the 'template' keyword fits better inside
  // `NestedNameSpecifier` because if we change `U<int>` to `UI` we would like
  // equally to change the `NameSpecifier` `template U<int>` to just `UI`.
  [[sp->getU().template U<int>::template f<int>()]];
}
)cpp",
      {R"txt(
UnknownExpression
|-MemberExpression
| |-UnknownExpression
| | |-MemberExpression
| | | |-IdExpression
| | | | `-UnqualifiedId
| | | |   `-sp
| | | |-->
| | | `-IdExpression
| | |   `-UnqualifiedId
| | |     `-getU
| | |-(
| | `-)
| |-.
| `-IdExpression
|   |-NestedNameSpecifier
|   | |-SimpleTemplateNameSpecifier
|   | | |-template
|   | | |-U
|   | | |-<
|   | | |-int
|   | | `->
|   | `-::
|   |-template
|   `-UnqualifiedId
|     |-f
|     |-<
|     |-int
|     `->
|-(
`-)
)txt"}));
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGrouping) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int *a, b;
int *c, d;
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-*
| | `-a
| |-,
| |-SimpleDeclarator
| | `-b
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-*
  | `-c
  |-,
  |-SimpleDeclarator
  | `-d
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGroupingTypedef) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
typedef int *a, b;
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-typedef
  |-int
  |-SimpleDeclarator
  | |-*
  | `-a
  |-,
  |-SimpleDeclarator
  | `-b
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsInsideStatement) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void foo() {
  int *a, b;
  typedef int *ta, tb;
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | |-SimpleDeclarator
    | | | |-*
    | | | `-a
    | | |-,
    | | `-SimpleDeclarator
    | |   `-b
    | `-;
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-typedef
    | | |-int
    | | |-SimpleDeclarator
    | | | |-*
    | | | `-ta
    | | |-,
    | | `-SimpleDeclarator
    | |   `-tb
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, SizeTTypedef) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
typedef decltype(sizeof(void *)) size_t;
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-typedef
  |-decltype
  |-(
  |-UnknownExpression
  | |-sizeof
  | |-(
  | |-void
  | |-*
  | `-)
  |-)
  |-SimpleDeclarator
  | `-size_t
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, Namespace_Nested) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace a { namespace b {} }
namespace a::b {}
)cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-a
| |-{
| |-NamespaceDefinition
| | |-namespace
| | |-b
| | |-{
| | `-}
| `-}
`-NamespaceDefinition
  |-namespace
  |-a
  |-::
  |-b
  |-{
  `-}
)txt"));
}

TEST_P(SyntaxTreeTest, Namespace_Unnamed) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace {}
)cpp",
      R"txt(
*: TranslationUnit
`-NamespaceDefinition
  |-namespace
  |-{
  `-}
)txt"));
}

TEST_P(SyntaxTreeTest, Namespace_Alias) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
namespace a {}
[[namespace foo = a;]]
)cpp",
      {R"txt(
NamespaceAliasDefinition
|-namespace
|-foo
|-=
|-a
`-;
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingDirective) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
namespace ns {}
[[using namespace ::ns;]]
)cpp",
      {R"txt(
UsingNamespaceDirective
|-using
|-namespace
|-NestedNameSpecifier
| `-::
|-ns
`-;
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
namespace ns { int a; }
[[using ns::a;]]
)cpp",
      {R"txt(
UsingDeclaration
|-using
|-NestedNameSpecifier
| |-IdentifierNameSpecifier
| | `-ns
| `-::
|-a
`-;
)txt"}));
}

TEST_P(SyntaxTreeTest, FreeStandingClasses) {
  // Free-standing classes, must live inside a SimpleDeclaration.
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X;
struct X {};

struct Y *y1;
struct Y {} *y2;

struct {} *a1;
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| `-;
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-struct
| |-Y
| |-SimpleDeclarator
| | |-*
| | `-y1
| `-;
|-SimpleDeclaration
| |-struct
| |-Y
| |-{
| |-}
| |-SimpleDeclarator
| | |-*
| | `-y2
| `-;
`-SimpleDeclaration
  |-struct
  |-{
  |-}
  |-SimpleDeclarator
  | |-*
  | `-a1
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, StaticMemberFunction) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct S {
  static void f(){}
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-S
  |-{
  |-SimpleDeclaration
  | |-static
  | |-void
  | |-SimpleDeclarator
  | | |-f
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   `-)
  | `-CompoundStatement
  |   |-{
  |   `-}
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ConversionMemberFunction) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  operator int();
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-X
  |-{
  |-SimpleDeclaration
  | |-SimpleDeclarator
  | | |-operator
  | | |-int
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   `-)
  | `-;
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, LiteralOperatorDeclaration) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
unsigned operator "" _c(char);
    )cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-unsigned
  |-SimpleDeclarator
  | |-operator
  | |-""
  | |-_c
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | `-char
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, NumericLiteralOperatorTemplateDeclaration) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <char...>
unsigned operator "" _t();
    )cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-SimpleDeclaration
  | `-char
  |-...
  |->
  `-SimpleDeclaration
    |-unsigned
    |-SimpleDeclarator
    | |-operator
    | |-""
    | |-_t
    | `-ParametersAndQualifiers
    |   |-(
    |   `-)
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperatorDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X& operator=(const X&);
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-X
  |-{
  |-SimpleDeclaration
  | |-X
  | |-SimpleDeclarator
  | | |-&
  | | |-operator
  | | |-=
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-SimpleDeclaration
  | |   | |-const
  | |   | |-X
  | |   | `-SimpleDeclarator
  | |   |   `-&
  | |   `-)
  | `-;
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperatorFriendDeclarataion) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  friend X operator+(X, const X&);
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-X
  |-{
  |-UnknownDeclaration
  | `-SimpleDeclaration
  |   |-friend
  |   |-X
  |   |-SimpleDeclarator
  |   | |-operator
  |   | |-+
  |   | `-ParametersAndQualifiers
  |   |   |-(
  |   |   |-SimpleDeclaration
  |   |   | `-X
  |   |   |-,
  |   |   |-SimpleDeclaration
  |   |   | |-const
  |   |   | |-X
  |   |   | `-SimpleDeclarator
  |   |   |   `-&
  |   |   `-)
  |   `-;
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ClassTemplateDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template<typename T>
struct ST {};
)cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-typename
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-ST
    |-{
    |-}
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, FunctionTemplateDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template<typename T>
T f();
)cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-typename
  | `-T
  |->
  `-SimpleDeclaration
    |-T
    |-SimpleDeclarator
    | |-f
    | `-ParametersAndQualifiers
    |   |-(
    |   `-)
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, VariableTemplateDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> T var = 10;
)cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-T
    |-SimpleDeclarator
    | |-var
    | |-=
    | `-IntegerLiteralExpression
    |   `-10
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, StaticMemberFunctionTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct S {
  template<typename U>
  static U f();
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-S
  |-{
  |-TemplateDeclaration
  | |-template
  | |-<
  | |-UnknownDeclaration
  | | |-typename
  | | `-U
  | |->
  | `-SimpleDeclaration
  |   |-static
  |   |-U
  |   |-SimpleDeclarator
  |   | |-f
  |   | `-ParametersAndQualifiers
  |   |   |-(
  |   |   `-)
  |   `-;
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, NestedTemplates) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T>
struct X {
  template <class U>
  U foo();
};
)cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-X
    |-{
    |-TemplateDeclaration
    | |-template
    | |-<
    | |-UnknownDeclaration
    | | |-class
    | | `-U
    | |->
    | `-SimpleDeclaration
    |   |-U
    |   |-SimpleDeclarator
    |   | |-foo
    |   | `-ParametersAndQualifiers
    |   |   |-(
    |   |   `-)
    |   `-;
    |-}
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, NestedTemplatesInNamespace) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace n {
  template<typename T>
  struct ST {
    template<typename U>
    static U f();
  };
}
)cpp",
      R"txt(
*: TranslationUnit
`-NamespaceDefinition
  |-namespace
  |-n
  |-{
  |-TemplateDeclaration
  | |-template
  | |-<
  | |-UnknownDeclaration
  | | |-typename
  | | `-T
  | |->
  | `-SimpleDeclaration
  |   |-struct
  |   |-ST
  |   |-{
  |   |-TemplateDeclaration
  |   | |-template
  |   | |-<
  |   | |-UnknownDeclaration
  |   | | |-typename
  |   | | `-U
  |   | |->
  |   | `-SimpleDeclaration
  |   |   |-static
  |   |   |-U
  |   |   |-SimpleDeclarator
  |   |   | |-f
  |   |   | `-ParametersAndQualifiers
  |   |   |   |-(
  |   |   |   `-)
  |   |   `-;
  |   |-}
  |   `-;
  `-}
)txt"));
}

TEST_P(SyntaxTreeTest, Templates2) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> struct X { struct Y; };
template <class T> struct X<T>::Y {};
)cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-{
|   |-SimpleDeclaration
|   | |-struct
|   | |-Y
|   | `-;
|   |-}
|   `-;
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-NestedNameSpecifier
    | |-SimpleTemplateNameSpecifier
    | | |-X
    | | |-<
    | | |-T
    | | `->
    | `-::
    |-Y
    |-{
    |-}
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, TemplatesUsingUsing) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> struct X {
  using T::foo;
  using typename T::bar;
};
)cpp",
      R"txt(
*: TranslationUnit
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-struct
    |-X
    |-{
    |-UsingDeclaration
    | |-using
    | |-NestedNameSpecifier
    | | |-IdentifierNameSpecifier
    | | | `-T
    | | `-::
    | |-foo
    | `-;
    |-UsingDeclaration
    | |-using
    | |-typename
    | |-NestedNameSpecifier
    | | |-IdentifierNameSpecifier
    | | | `-T
    | | `-::
    | |-bar
    | `-;
    |-}
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ExplicitTemplateInstantations) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> struct X {};
template <class T> struct X<T*> {};
template <> struct X<int> {};

template struct X<double>;
extern template struct X<float>;
)cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-{
|   |-}
|   `-;
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-class
| | `-T
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-T
|   |-*
|   |->
|   |-{
|   |-}
|   `-;
|-TemplateDeclaration
| |-template
| |-<
| |->
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-int
|   |->
|   |-{
|   |-}
|   `-;
|-ExplicitTemplateInstantiation
| |-template
| `-SimpleDeclaration
|   |-struct
|   |-X
|   |-<
|   |-double
|   |->
|   `-;
`-ExplicitTemplateInstantiation
  |-extern
  |-template
  `-SimpleDeclaration
    |-struct
    |-X
    |-<
    |-float
    |->
    `-;
)txt"));
}

TEST_P(SyntaxTreeTest, UsingType) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
using type = int;
)cpp",
      R"txt(
*: TranslationUnit
`-TypeAliasDeclaration
  |-using
  |-type
  |-=
  |-int
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, EmptyDeclaration) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
;
)cpp",
      R"txt(
*: TranslationUnit
`-EmptyDeclaration
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, StaticAssert) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
static_assert(true, "message");
static_assert(true);
)cpp",
      R"txt(
*: TranslationUnit
|-StaticAssertDeclaration
| |-static_assert
| |-(
| |-BoolLiteralExpression
| | `-true
| |-,
| |-StringLiteralExpression
| | `-"message"
| |-)
| `-;
`-StaticAssertDeclaration
  |-static_assert
  |-(
  |-BoolLiteralExpression
  | `-true
  |-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ExternC) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
extern "C" int a;
extern "C" { int b; int c; }
)cpp",
      R"txt(
*: TranslationUnit
|-LinkageSpecificationDeclaration
| |-extern
| |-"C"
| `-SimpleDeclaration
|   |-int
|   |-SimpleDeclarator
|   | `-a
|   `-;
`-LinkageSpecificationDeclaration
  |-extern
  |-"C"
  |-{
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | `-b
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | `-c
  | `-;
  `-}
)txt"));
}

TEST_P(SyntaxTreeTest, NonModifiableNodes) {
  // Some nodes are non-modifiable, they are marked with 'I:'.
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
#define HALF_IF if (1+
#define HALF_IF_2 1) {}
void test() {
  HALF_IF HALF_IF_2 else {}
})cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-IfStatement
    | |-I: if
    | |-I: (
    | |-I: BinaryOperatorExpression
    | | |-I: IntegerLiteralExpression
    | | | `-I: 1
    | | |-I: +
    | | `-I: IntegerLiteralExpression
    | |   `-I: 1
    | |-I: )
    | |-I: CompoundStatement
    | | |-I: {
    | | `-I: }
    | |-else
    | `-CompoundStatement
    |   |-{
    |   `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, ModifiableNodes) {
  // All nodes can be mutated.
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
#define OPEN {
#define CLOSE }

void test() {
  OPEN
    1;
  CLOSE

  OPEN
    2;
  }
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-CompoundStatement
    | |-{
    | |-ExpressionStatement
    | | |-IntegerLiteralExpression
    | | | `-1
    | | `-;
    | `-}
    |-CompoundStatement
    | |-{
    | |-ExpressionStatement
    | | |-IntegerLiteralExpression
    | | | `-2
    | | `-;
    | `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, ArraySubscriptsInDeclarators) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a[10];
int b[1][2][3];
int c[] = {1,2,3};
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ArraySubscript
| |   |-[
| |   |-IntegerLiteralExpression
| |   | `-10
| |   `-]
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-b
| | |-ArraySubscript
| | | |-[
| | | |-IntegerLiteralExpression
| | | | `-1
| | | `-]
| | |-ArraySubscript
| | | |-[
| | | |-IntegerLiteralExpression
| | | | `-2
| | | `-]
| | `-ArraySubscript
| |   |-[
| |   |-IntegerLiteralExpression
| |   | `-3
| |   `-]
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-c
  | |-ArraySubscript
  | | |-[
  | | `-]
  | |-=
  | `-UnknownExpression
  |   `-UnknownExpression
  |     |-{
  |     |-IntegerLiteralExpression
  |     | `-1
  |     |-,
  |     |-IntegerLiteralExpression
  |     | `-2
  |     |-,
  |     |-IntegerLiteralExpression
  |     | `-3
  |     `-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, StaticArraySubscriptsInDeclarators) {
  if (!GetParam().isC99OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void f(int xs[static 10]);
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-f
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-xs
  |   |   `-ArraySubscript
  |   |     |-[
  |   |     |-static
  |   |     |-IntegerLiteralExpression
  |   |     | `-10
  |   |     `-]
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctions) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func1();
int func2a(int a);
int func2b(int);
int func3a(int *ap);
int func3b(int *);
int func4a(int a, float b);
int func4b(int, float);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func1
| | `-ParametersAndQualifiers
| |   |-(
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func2a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func2b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-int
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func3a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   |-*
| |   |   `-ap
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func3b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-*
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func4a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   |-,
| |   |-SimpleDeclaration
| |   | |-float
| |   | `-SimpleDeclarator
| |   |   `-b
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func4b
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | `-int
  |   |-,
  |   |-SimpleDeclaration
  |   | `-float
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctionsCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func1(const int a, volatile int b, const volatile int c);
int func2(int& a);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-func1
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-a
| |   |-,
| |   |-SimpleDeclaration
| |   | |-volatile
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-b
| |   |-,
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-volatile
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-c
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func2
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-&
  |   |   `-a
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInFreeFunctionsCxx11) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func1(int&& a);
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-func1
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-&&
  |   |   `-a
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiersInMemberFunctions) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct Test {
  int a();
  int b() const;
  int c() volatile;
  int d() const volatile;
  int e() &;
  int f() &&;
};
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-struct
  |-Test
  |-{
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-a
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   `-)
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-b
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-const
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-c
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-volatile
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-d
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   |-const
  | |   `-volatile
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-e
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-&
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-SimpleDeclarator
  | | |-f
  | | `-ParametersAndQualifiers
  | |   |-(
  | |   |-)
  | |   `-&&
  | `-;
  |-}
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, TrailingReturn) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
auto foo() -> int;
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-auto
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   `-TrailingReturnType
  |     |-->
  |     `-int
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, DynamicExceptionSpecification) {
  if (!GetParam().supportsCXXDynamicExceptionSpecification()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct MyException1 {};
struct MyException2 {};
int a() throw();
int b() throw(...);
int c() throw(MyException1);
int d() throw(MyException1, MyException2);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-MyException1
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-struct
| |-MyException2
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-b
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   |-...
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-c
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   |-throw
| |   |-(
| |   |-MyException1
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-d
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   |-throw
  |   |-(
  |   |-MyException1
  |   |-,
  |   |-MyException2
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, NoexceptExceptionSpecification) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a() noexcept;
int b() noexcept(true);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-a
| | `-ParametersAndQualifiers
| |   |-(
| |   |-)
| |   `-noexcept
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-b
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   |-noexcept
  |   |-(
  |   |-BoolLiteralExpression
  |   | `-true
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, DeclaratorsInParentheses) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int (a);
int *(b);
int (*c)(int);
int *(d)(int);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | `-ParenDeclarator
| |   |-(
| |   |-a
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-*
| | `-ParenDeclarator
| |   |-(
| |   |-b
| |   `-)
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-ParenDeclarator
| | | |-(
| | | |-*
| | | |-c
| | | `-)
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-int
| |   `-)
| `-;
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-*
  | |-ParenDeclarator
  | | |-(
  | | |-d
  | | `-)
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | `-int
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ConstVolatileQualifiers) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
const int west = -1;
int const east = 1;
const int const universal = 0;
const int const *const *volatile b;
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-const
| |-int
| |-SimpleDeclarator
| | |-west
| | |-=
| | `-PrefixUnaryOperatorExpression
| |   |--
| |   `-IntegerLiteralExpression
| |     `-1
| `-;
|-SimpleDeclaration
| |-int
| |-const
| |-SimpleDeclarator
| | |-east
| | |-=
| | `-IntegerLiteralExpression
| |   `-1
| `-;
|-SimpleDeclaration
| |-const
| |-int
| |-const
| |-SimpleDeclarator
| | |-universal
| | |-=
| | `-IntegerLiteralExpression
| |   `-0
| `-;
`-SimpleDeclaration
  |-const
  |-int
  |-const
  |-SimpleDeclarator
  | |-*
  | |-const
  | |-*
  | |-volatile
  | `-b
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, RangesOfDeclaratorsWithTrailingReturnTypes) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
auto foo() -> auto(*)(int) -> double*;
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-auto
  |-SimpleDeclarator
  | |-foo
  | `-ParametersAndQualifiers
  |   |-(
  |   |-)
  |   `-TrailingReturnType
  |     |-->
  |     |-auto
  |     `-SimpleDeclarator
  |       |-ParenDeclarator
  |       | |-(
  |       | |-*
  |       | `-)
  |       `-ParametersAndQualifiers
  |         |-(
  |         |-SimpleDeclaration
  |         | `-int
  |         |-)
  |         `-TrailingReturnType
  |           |-->
  |           |-double
  |           `-SimpleDeclarator
  |             `-*
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, MemberPointers) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {};
int X::* a;
const int X::* b;
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
|-SimpleDeclaration
| |-int
| |-SimpleDeclarator
| | |-MemberPointer
| | | |-X
| | | |-::
| | | `-*
| | `-a
| `-;
`-SimpleDeclaration
  |-const
  |-int
  |-SimpleDeclarator
  | |-MemberPointer
  | | |-X
  | | |-::
  | | `-*
  | `-b
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, MemberFunctionPointer) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  struct Y {};
};
void (X::*xp)();
void (X::**xpp)(const int*);
// FIXME: Generate the right syntax tree for this type,
// i.e. create a syntax node for the outer member pointer
void (X::Y::*xyp)(const int*, char);
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-struct
| | |-Y
| | |-{
| | |-}
| | `-;
| |-}
| `-;
|-SimpleDeclaration
| |-void
| |-SimpleDeclarator
| | |-ParenDeclarator
| | | |-(
| | | |-MemberPointer
| | | | |-X
| | | | |-::
| | | | `-*
| | | |-xp
| | | `-)
| | `-ParametersAndQualifiers
| |   |-(
| |   `-)
| `-;
|-SimpleDeclaration
| |-void
| |-SimpleDeclarator
| | |-ParenDeclarator
| | | |-(
| | | |-MemberPointer
| | | | |-X
| | | | |-::
| | | | `-*
| | | |-*
| | | |-xpp
| | | `-)
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-int
| |   | `-SimpleDeclarator
| |   |   `-*
| |   `-)
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-ParenDeclarator
  | | |-(
  | | |-X
  | | |-::
  | | |-MemberPointer
  | | | |-Y
  | | | |-::
  | | | `-*
  | | |-xyp
  | | `-)
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-const
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-*
  |   |-,
  |   |-SimpleDeclaration
  |   | `-char
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ComplexDeclarator) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int));
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-x
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-char
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-short
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-b
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | `-int
  |   |     `-)
  |   `-)
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, ComplexDeclarator2) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int), long (**c)(long long));
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-x
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-char
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-short
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-b
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | `-int
  |   |     `-)
  |   |-,
  |   |-SimpleDeclaration
  |   | |-long
  |   | `-SimpleDeclarator
  |   |   |-ParenDeclarator
  |   |   | |-(
  |   |   | |-*
  |   |   | |-*
  |   |   | |-c
  |   |   | `-)
  |   |   `-ParametersAndQualifiers
  |   |     |-(
  |   |     |-SimpleDeclaration
  |   |     | |-long
  |   |     | `-long
  |   |     `-)
  |   `-)
  `-;
)txt"));
}

} // namespace
