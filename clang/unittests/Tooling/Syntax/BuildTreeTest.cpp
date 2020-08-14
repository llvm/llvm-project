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
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int main() {
  if (1) {}
  if (1) {} else if (0) {}
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-main
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-IfStatement
    | |-if
    | |-(
    | |-IntegerLiteralExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    |-IfStatement
    | |-if
    | |-(
    | |-IntegerLiteralExpression
    | | `-1
    | |-)
    | |-CompoundStatement
    | | |-{
    | | `-}
    | |-else
    | `-IfStatement
    |   |-if
    |   |-(
    |   |-IntegerLiteralExpression
    |   | `-0
    |   |-)
    |   `-CompoundStatement
    |     |-{
    |     `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, For) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  for (;;)  {}
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
    |-ForStatement
    | |-for
    | |-(
    | |-;
    | |-;
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, RangeBasedFor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  int a[3];
  for (int x : a)
    ;
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
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | `-SimpleDeclarator
    | |   |-a
    | |   `-ArraySubscript
    | |     |-[
    | |     |-IntegerLiteralExpression
    | |     | `-3
    | |     `-]
    | `-;
    |-RangeBasedForStatement
    | |-for
    | |-(
    | |-SimpleDeclaration
    | | |-int
    | | |-SimpleDeclarator
    | | | `-x
    | | `-:
    | |-IdExpression
    | | `-UnqualifiedId
    | |   `-a
    | |-)
    | `-EmptyStatement
    |   `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, DeclarationStatement) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  int a = 10;
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
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | `-SimpleDeclarator
    | |   |-a
    | |   |-=
    | |   `-IntegerLiteralExpression
    | |     `-10
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, Switch) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  switch (1) {
    case 0:
    default:;
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
    |-SwitchStatement
    | |-switch
    | |-(
    | |-IntegerLiteralExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-CaseStatement
    |   | |-case
    |   | |-IntegerLiteralExpression
    |   | | `-0
    |   | |-:
    |   | `-DefaultStatement
    |   |   |-default
    |   |   |-:
    |   |   `-EmptyStatement
    |   |     `-;
    |   `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, While) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  while (1) { continue; break; }
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
    |-WhileStatement
    | |-while
    | |-(
    | |-IntegerLiteralExpression
    | | `-1
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-ContinueStatement
    |   | |-continue
    |   | `-;
    |   |-BreakStatement
    |   | |-break
    |   | `-;
    |   `-}
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnhandledStatement) {
  // Unhandled statements should end up as 'unknown statement'.
  // This example uses a 'label statement', which does not yet have a syntax
  // counterpart.
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int main() {
  foo: return 100;
}
)cpp",
      R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-SimpleDeclarator
  | |-main
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-UnknownStatement
    | |-foo
    | |-:
    | `-ReturnStatement
    |   |-return
    |   |-IntegerLiteralExpression
    |   | `-100
    |   `-;
    `-}
)txt"));
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
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a) {
  a;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-IdExpression
    | | `-UnqualifiedId
    | |   `-a
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_OperatorFunctionId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  friend X operator+(const X&, const X&);
};
void test(X x) {
  operator+(x, x);
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-X
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-+
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   |-operator
    | | |   `-+
    | | |-(
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | |-,
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_ConversionFunctionId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  operator int();
};
void test(X x) {
  // TODO: Expose `id-expression` from `MemberExpr`
  x.operator int();
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-SimpleDeclarator
| | | |-operator
| | | |-int
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-UnknownExpression
    | | | |-IdExpression
    | | | | `-UnqualifiedId
    | | | |   `-x
    | | | |-.
    | | | |-operator
    | | | `-int
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_LiteralOperatorId) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
unsigned operator "" _w(char);
void test() {
  operator "" _w('1');
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_w
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-char
| |   `-)
| `-;
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
    | | |   |-operator
    | | |   |-""
    | | |   `-_w
    | | |-(
    | | |-CharacterLiteralExpression
    | | | `-'1'
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_Destructor) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X { };
void test(X x) {
  // TODO: Expose `id-expression` from `MemberExpr`
  x.~X();
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-UnknownExpression
    | | | |-IdExpression
    | | | | `-UnqualifiedId
    | | | |   `-x
    | | | |-.
    | | | |-~
    | | | `-X
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_DecltypeDestructor) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X { };
void test(X x) {
  // TODO: Expose `id-expression` from `MemberExpr`
  x.~decltype(x)();
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-UnknownExpression
    | | | |-IdExpression
    | | | | `-UnqualifiedId
    | | | |   `-x
    | | | |-.
    | | | `-~
    | | |-decltype
    | | |-(
    | | |-x
    | | |-)
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UnqualifiedId_TemplateId) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template<typename T>
T f();
void test() {
  f<int>();
}
)cpp",
      R"txt(
*: TranslationUnit
|-TemplateDeclaration
| |-template
| |-<
| |-UnknownDeclaration
| | |-typename
| | `-T
| |->
| `-SimpleDeclaration
|   |-T
|   |-SimpleDeclarator
|   | |-f
|   | `-ParametersAndQualifiers
|   |   |-(
|   |   `-)
|   `-;
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
    | | |   |-f
    | | |   |-<
    | | |   |-int
    | | |   `->
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, QualifiedId_NamespaceSpecifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace n {
  struct S { };
}
void test() {
  ::n::S s1;
  n::S s2;
}
)cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-n
| |-{
| |-SimpleDeclaration
| | |-struct
| | |-S
| | |-{
| | |-}
| | `-;
| `-}
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-NestedNameSpecifier
    | | | |-::
    | | | |-IdentifierNameSpecifier
    | | | | `-n
    | | | `-::
    | | |-S
    | | `-SimpleDeclarator
    | |   `-UnknownExpression
    | |     `-s1
    | `-;
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-NestedNameSpecifier
    | | | |-IdentifierNameSpecifier
    | | | | `-n
    | | | `-::
    | | |-S
    | | `-SimpleDeclarator
    | |   `-UnknownExpression
    | |     `-s2
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, QualifiedId_TemplateSpecifier) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template<typename T>
struct ST {
  struct S { };
};
void test() {
  ::template ST<int>::S s1;
  ::ST<int>::S s2;
}
)cpp",
      R"txt(
*: TranslationUnit
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
|   |-SimpleDeclaration
|   | |-struct
|   | |-S
|   | |-{
|   | |-}
|   | `-;
|   |-}
|   `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   `-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-NestedNameSpecifier
    | | | |-::
    | | | |-SimpleTemplateNameSpecifier
    | | | | |-template
    | | | | |-ST
    | | | | |-<
    | | | | |-int
    | | | | `->
    | | | `-::
    | | |-S
    | | `-SimpleDeclarator
    | |   `-UnknownExpression
    | |     `-s1
    | `-;
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-NestedNameSpecifier
    | | | |-::
    | | | |-SimpleTemplateNameSpecifier
    | | | | |-ST
    | | | | |-<
    | | | | |-int
    | | | | `->
    | | | `-::
    | | |-S
    | | `-SimpleDeclarator
    | |   `-UnknownExpression
    | |     `-s2
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, QualifiedId_DecltypeSpecifier) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct S {
  static void f(){}
};
void test(S s) {
  decltype(s)::f();
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-S
| |-{
| |-SimpleDeclaration
| | |-static
| | |-void
| | |-SimpleDeclarator
| | | |-f
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   `-)
| | `-CompoundStatement
| |   |-{
| |   `-}
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-S
  |   | `-SimpleDeclarator
  |   |   `-s
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-IdExpression
    | | | |-NestedNameSpecifier
    | | | | |-DecltypeNameSpecifier
    | | | | | |-decltype
    | | | | | |-(
    | | | | | |-IdExpression
    | | | | | | `-UnqualifiedId
    | | | | | |   `-s
    | | | | | `-)
    | | | | `-::
    | | | `-UnqualifiedId
    | | |   `-f
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, QualifiedId_OptionalTemplateKw) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct S {
  template<typename U>
  static U f();
};
void test() {
  S::f<int>();
  S::template f<int>();
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-S
| |-{
| |-TemplateDeclaration
| | |-template
| | |-<
| | |-UnknownDeclaration
| | | |-typename
| | | `-U
| | |->
| | `-SimpleDeclaration
| |   |-static
| |   |-U
| |   |-SimpleDeclarator
| |   | |-f
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   `-)
| |   `-;
| |-}
| `-;
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
    | | | |-NestedNameSpecifier
    | | | | |-IdentifierNameSpecifier
    | | | | | `-S
    | | | | `-::
    | | | `-UnqualifiedId
    | | |   |-f
    | | |   |-<
    | | |   |-int
    | | |   `->
    | | |-(
    | | `-)
    | `-;
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-IdExpression
    | | | |-NestedNameSpecifier
    | | | | |-IdentifierNameSpecifier
    | | | | | `-S
    | | | | `-::
    | | | |-template
    | | | `-UnqualifiedId
    | | |   |-f
    | | |   |-<
    | | |   |-int
    | | |   `->
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, QualifiedId_Complex) {
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
void test() {
  ::n::template ST<int>::template f<int>();
}
)cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-n
| |-{
| |-TemplateDeclaration
| | |-template
| | |-<
| | |-UnknownDeclaration
| | | |-typename
| | | `-T
| | |->
| | `-SimpleDeclaration
| |   |-struct
| |   |-ST
| |   |-{
| |   |-TemplateDeclaration
| |   | |-template
| |   | |-<
| |   | |-UnknownDeclaration
| |   | | |-typename
| |   | | `-U
| |   | |->
| |   | `-SimpleDeclaration
| |   |   |-static
| |   |   |-U
| |   |   |-SimpleDeclarator
| |   |   | |-f
| |   |   | `-ParametersAndQualifiers
| |   |   |   |-(
| |   |   |   `-)
| |   |   `-;
| |   |-}
| |   `-;
| `-}
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
    | | | |-NestedNameSpecifier
    | | | | |-::
    | | | | |-IdentifierNameSpecifier
    | | | | | `-n
    | | | | |-::
    | | | | |-SimpleTemplateNameSpecifier
    | | | | | |-template
    | | | | | |-ST
    | | | | | |-<
    | | | | | |-int
    | | | | | `->
    | | | | `-::
    | | | |-template
    | | | `-UnqualifiedId
    | | |   |-f
    | | |   |-<
    | | |   |-int
    | | |   `->
    | | |-(
    | | `-)
    | `-;
    `-}
)txt"));
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
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <typename T>
void test() {
  T::template U<int>::f();
  T::U::f();
  T::template f<0>();
}
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
      | | | |-NestedNameSpecifier
      | | | | |-IdentifierNameSpecifier
      | | | | | `-T
      | | | | |-::
      | | | | |-SimpleTemplateNameSpecifier
      | | | | | |-template
      | | | | | |-U
      | | | | | |-<
      | | | | | |-int
      | | | | | `->
      | | | | `-::
      | | | `-UnqualifiedId
      | | |   `-f
      | | |-(
      | | `-)
      | `-;
      |-ExpressionStatement
      | |-UnknownExpression
      | | |-IdExpression
      | | | |-NestedNameSpecifier
      | | | | |-IdentifierNameSpecifier
      | | | | | `-T
      | | | | |-::
      | | | | |-IdentifierNameSpecifier
      | | | | | `-U
      | | | | `-::
      | | | `-UnqualifiedId
      | | |   `-f
      | | |-(
      | | `-)
      | `-;
      |-ExpressionStatement
      | |-UnknownExpression
      | | |-IdExpression
      | | | |-NestedNameSpecifier
      | | | | |-IdentifierNameSpecifier
      | | | | | `-T
      | | | | `-::
      | | | |-template
      | | | `-UnqualifiedId
      | | |   |-f
      | | |   |-<
      | | |   |-IntegerLiteralExpression
      | | |   | `-0
      | | |   `->
      | | |-(
      | | `-)
      | `-;
      `-}
)txt"));
}

TEST_P(SyntaxTreeTest, ParenExpr) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  (1);
  ((1));
  (1 + (2));
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
    | |-ParenExpression
    | | |-(
    | | |-IntegerLiteralExpression
    | | | `-1
    | | `-)
    | `-;
    |-ExpressionStatement
    | |-ParenExpression
    | | |-(
    | | |-ParenExpression
    | | | |-(
    | | | |-IntegerLiteralExpression
    | | | | `-1
    | | | `-)
    | | `-)
    | `-;
    |-ExpressionStatement
    | |-ParenExpression
    | | |-(
    | | |-BinaryOperatorExpression
    | | | |-IntegerLiteralExpression
    | | | | `-1
    | | | |-+
    | | | `-ParenExpression
    | | |   |-(
    | | |   |-IntegerLiteralExpression
    | | |   | `-2
    | | |   `-)
    | | `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Char) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
unsigned operator "" _c(char);
void test() {
  '2'_c;
}
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_c
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | `-char
| |   `-)
| `-;
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
    | |-CharUserDefinedLiteralExpression
    | | `-'2'_c
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_String) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
typedef decltype(sizeof(void *)) size_t;

unsigned operator "" _s(const char*, size_t);

void test() {
  "12"_s;
}
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-typedef
| |-decltype
| |-(
| |-UnknownExpression
| | |-sizeof
| | |-(
| | |-void
| | |-*
| | `-)
| |-)
| |-SimpleDeclarator
| | `-size_t
| `-;
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_s
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-char
| |   | `-SimpleDeclarator
| |   |   `-*
| |   |-,
| |   |-SimpleDeclaration
| |   | `-size_t
| |   `-)
| `-;
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
    | |-StringUserDefinedLiteralExpression
    | | `-"12"_s
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Integer) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
unsigned operator "" _i(unsigned long long);
unsigned operator "" _r(const char*);
template <char...>
unsigned operator "" _t();

void test() {
  12_i;
  12_r;
  12_t;
}
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_i
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-unsigned
| |   | |-long
| |   | `-long
| |   `-)
| `-;
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_r
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-char
| |   | `-SimpleDeclarator
| |   |   `-*
| |   `-)
| `-;
|-TemplateDeclaration
| |-template
| |-<
| |-SimpleDeclaration
| | `-char
| |-...
| |->
| `-SimpleDeclaration
|   |-unsigned
|   |-SimpleDeclarator
|   | |-operator
|   | |-""
|   | |-_t
|   | `-ParametersAndQualifiers
|   |   |-(
|   |   `-)
|   `-;
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
    | |-IntegerUserDefinedLiteralExpression
    | | `-12_i
    | `-;
    |-ExpressionStatement
    | |-IntegerUserDefinedLiteralExpression
    | | `-12_r
    | `-;
    |-ExpressionStatement
    | |-IntegerUserDefinedLiteralExpression
    | | `-12_t
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, UserDefinedLiteral_Float) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
unsigned operator "" _f(long double);
unsigned operator "" _r(const char*);
template <char...>
unsigned operator "" _t();

void test() {
  1.2_f;  // call: operator "" _f(1.2L)       | kind: float
  1.2_r;  // call: operator "" _i("1.2")      | kind: float
  1.2_t;  // call: operator<'1', '2'> "" _x() | kind: float
}
    )cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_f
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-long
| |   | `-double
| |   `-)
| `-;
|-SimpleDeclaration
| |-unsigned
| |-SimpleDeclarator
| | |-operator
| | |-""
| | |-_r
| | `-ParametersAndQualifiers
| |   |-(
| |   |-SimpleDeclaration
| |   | |-const
| |   | |-char
| |   | `-SimpleDeclarator
| |   |   `-*
| |   `-)
| `-;
|-TemplateDeclaration
| |-template
| |-<
| |-SimpleDeclaration
| | `-char
| |-...
| |->
| `-SimpleDeclaration
|   |-unsigned
|   |-SimpleDeclarator
|   | |-operator
|   | |-""
|   | |-_t
|   | `-ParametersAndQualifiers
|   |   |-(
|   |   `-)
|   `-;
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
    | |-FloatUserDefinedLiteralExpression
    | | `-1.2_f
    | `-;
    |-ExpressionStatement
    | |-FloatUserDefinedLiteralExpression
    | | `-1.2_r
    | `-;
    |-ExpressionStatement
    | |-FloatUserDefinedLiteralExpression
    | | `-1.2_t
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_LongLong) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  12ll;
  12ull;
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
    | |-IntegerLiteralExpression
    | | `-12ll
    | `-;
    |-ExpressionStatement
    | |-IntegerLiteralExpression
    | | `-12ull
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_Binary) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  0b1100;
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
    | |-IntegerLiteralExpression
    | | `-0b1100
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, IntegerLiteral_WithDigitSeparators) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  1'2'0ull;
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
    | |-IntegerLiteralExpression
    | | `-1'2'0ull
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, CharacterLiteral) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  'a';
  '\n';
  '\x20';
  '\0';
  L'a';
  L'α';
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
    | |-CharacterLiteralExpression
    | | `-'a'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-'\n'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-'\x20'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-'\0'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-L'a'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-L'α'
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, CharacterLiteral_Utf) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  u'a';
  u'構';
  U'a';
  U'🌲';
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
    | |-CharacterLiteralExpression
    | | `-u'a'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-u'構'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-U'a'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-U'🌲'
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, CharacterLiteral_Utf8) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  u8'a';
  u8'\x7f';
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
    | |-CharacterLiteralExpression
    | | `-u8'a'
    | `-;
    |-ExpressionStatement
    | |-CharacterLiteralExpression
    | | `-u8'\x7f'
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, FloatingLiteral) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  1e-2;
  2.;
  .2;
  2.f;
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
    | |-FloatingLiteralExpression
    | | `-1e-2
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-2.
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-.2
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-2.f
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, FloatingLiteral_Hexadecimal) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  0xfp1;
  0xf.p1;
  0x.fp1;
  0xf.fp1f;
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
    | |-FloatingLiteralExpression
    | | `-0xfp1
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-0xf.p1
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-0x.fp1
    | `-;
    |-ExpressionStatement
    | |-FloatingLiteralExpression
    | | `-0xf.fp1f
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, StringLiteral) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  "a\n\0\x20";
  L"αβ";
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
    | |-StringLiteralExpression
    | | `-"a\n\0\x20"
    | `-;
    |-ExpressionStatement
    | |-StringLiteralExpression
    | | `-L"αβ"
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, StringLiteral_Utf) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  u8"a\x1f\x05";
  u"C++抽象構文木";
  U"📖🌲\n";
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
    | |-StringLiteralExpression
    | | `-u8"a\x1f\x05"
    | `-;
    |-ExpressionStatement
    | |-StringLiteralExpression
    | | `-u"C++抽象構文木"
    | `-;
    |-ExpressionStatement
    | |-StringLiteralExpression
    | | `-U"📖🌲\n"
    | `-;
    `-}
)txt"));
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
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  true;
  false;
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
    | |-BoolLiteralExpression
    | | `-true
    | `-;
    |-ExpressionStatement
    | |-BoolLiteralExpression
    | | `-false
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, CxxNullPtrLiteral) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  nullptr;
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
    | |-CxxNullPtrExpression
    | | `-nullptr
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, PostfixUnaryOperator) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a) {
  a++;
  a--;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PostfixUnaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | `-++
    | `-;
    |-ExpressionStatement
    | |-PostfixUnaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | `---
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperator) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a, int *ap) {
  --a; ++a;
  ~a;
  -a;
  +a;
  &a;
  *ap;
  !a;
  __real a; __imag a;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-*
  |   |   `-ap
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |---
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-++
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-~
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |--
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-+
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-&
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-*
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-ap
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-!
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-__real
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-__imag
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, PrefixUnaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a, bool b) {
  compl a;
  not b;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   |-,
  |   |-SimpleDeclaration
  |   | |-bool
  |   | `-SimpleDeclarator
  |   |   `-b
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-compl
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-a
    | `-;
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-not
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-b
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, BinaryOperator) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a) {
  1 - 2;
  1 == 2;
  a = 1;
  a <<= 1;
  1 || 0;
  1 & 2;
  a ^= 3;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IntegerLiteralExpression
    | | | `-1
    | | |--
    | | `-IntegerLiteralExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IntegerLiteralExpression
    | | | `-1
    | | |-==
    | | `-IntegerLiteralExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | |-=
    | | `-IntegerLiteralExpression
    | |   `-1
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | |-<<=
    | | `-IntegerLiteralExpression
    | |   `-1
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IntegerLiteralExpression
    | | | `-1
    | | |-||
    | | `-IntegerLiteralExpression
    | |   `-0
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IntegerLiteralExpression
    | | | `-1
    | | |-&
    | | `-IntegerLiteralExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | |-^=
    | | `-IntegerLiteralExpression
    | |   `-3
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, BinaryOperatorCxx) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a) {
  true || false;
  true or false;
  1 bitand 2;
  a xor_eq 3;
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
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   `-a
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BoolLiteralExpression
    | | | `-true
    | | |-||
    | | `-BoolLiteralExpression
    | |   `-false
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BoolLiteralExpression
    | | | `-true
    | | |-or
    | | `-BoolLiteralExpression
    | |   `-false
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IntegerLiteralExpression
    | | | `-1
    | | |-bitand
    | | `-IntegerLiteralExpression
    | |   `-2
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | |-xor_eq
    | | `-IntegerLiteralExpression
    | |   `-3
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, BinaryOperator_NestedWithParenthesis) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  (1 + 2) * (4 / 2);
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
    | |-BinaryOperatorExpression
    | | |-ParenExpression
    | | | |-(
    | | | |-BinaryOperatorExpression
    | | | | |-IntegerLiteralExpression
    | | | | | `-1
    | | | | |-+
    | | | | `-IntegerLiteralExpression
    | | | |   `-2
    | | | `-)
    | | |-*
    | | `-ParenExpression
    | |   |-(
    | |   |-BinaryOperatorExpression
    | |   | |-IntegerLiteralExpression
    | |   | | `-4
    | |   | |-/
    | |   | `-IntegerLiteralExpression
    | |   |   `-2
    | |   `-)
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, BinaryOperator_Associativity) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int a, int b) {
  a + b + 42;
  a = b = 42;
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
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-IdExpression
    | | | | `-UnqualifiedId
    | | | |   `-a
    | | | |-+
    | | | `-IdExpression
    | | |   `-UnqualifiedId
    | | |     `-b
    | | |-+
    | | `-IntegerLiteralExpression
    | |   `-42
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-a
    | | |-=
    | | `-BinaryOperatorExpression
    | |   |-IdExpression
    | |   | `-UnqualifiedId
    | |   |   `-b
    | |   |-=
    | |   `-IntegerLiteralExpression
    | |     `-42
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, BinaryOperator_Precedence) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test() {
  1 + 2 * 3 + 4;
  1 % 2 + 3 * 4;
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
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-IntegerLiteralExpression
    | | | | `-1
    | | | |-+
    | | | `-BinaryOperatorExpression
    | | |   |-IntegerLiteralExpression
    | | |   | `-2
    | | |   |-*
    | | |   `-IntegerLiteralExpression
    | | |     `-3
    | | |-+
    | | `-IntegerLiteralExpression
    | |   `-4
    | `-;
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-BinaryOperatorExpression
    | | | |-IntegerLiteralExpression
    | | | | `-1
    | | | |-%
    | | | `-IntegerLiteralExpression
    | | |   `-2
    | | |-+
    | | `-BinaryOperatorExpression
    | |   |-IntegerLiteralExpression
    | |   | `-3
    | |   |-*
    | |   `-IntegerLiteralExpression
    | |     `-4
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Assignment) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X& operator=(const X&);
};
void test(X x, X y) {
  x = y;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-&
| | | |-operator
| | | |-=
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   |-SimpleDeclaration
| | |   | |-const
| | |   | |-X
| | |   | `-SimpleDeclarator
| | |   |   `-&
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | |-=
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-y
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Plus) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  friend X operator+(X, const X&);
};
void test(X x, X y) {
  x + y;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-X
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-+
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | `-X
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-UnknownExpression
    | | | `-IdExpression
    | | |   `-UnqualifiedId
    | | |     `-x
    | | |-+
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-y
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Less) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  friend bool operator<(const X&, const X&);
};
void test(X x, X y) {
  x < y;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-bool
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-<
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | |-<
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-y
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_LeftShift) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  friend X operator<<(X&, const X&);
};
void test(X x, X y) {
  x << y;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-UnknownDeclaration
| | `-SimpleDeclaration
| |   |-friend
| |   |-X
| |   |-SimpleDeclarator
| |   | |-operator
| |   | |-<<
| |   | `-ParametersAndQualifiers
| |   |   |-(
| |   |   |-SimpleDeclaration
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   |-,
| |   |   |-SimpleDeclaration
| |   |   | |-const
| |   |   | |-X
| |   |   | `-SimpleDeclarator
| |   |   |   `-&
| |   |   `-)
| |   `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | |-<<
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-y
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Comma) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X operator,(X&);
};
void test(X x, X y) {
  x, y;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-operator
| | | |-,
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   |-SimpleDeclaration
| | |   | |-X
| | |   | `-SimpleDeclarator
| | |   |   `-&
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   |-,
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-y
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | |-,
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-y
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PointerToMember) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X operator->*(int);
};
void test(X* xp, int X::* pmi) {
  xp->*pmi;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-operator
| | | |-->*
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   |-SimpleDeclaration
| | |   | `-int
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   |-*
  |   |   `-xp
  |   |-,
  |   |-SimpleDeclaration
  |   | |-int
  |   | `-SimpleDeclarator
  |   |   |-MemberPointer
  |   |   | |-X
  |   |   | |-::
  |   |   | `-*
  |   |   `-pmi
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-BinaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-xp
    | | |-->*
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-pmi
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_Negation) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  bool operator!();
};
void test(X x) {
  !x;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-bool
| | |-SimpleDeclarator
| | | |-operator
| | | |-!
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-!
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-x
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_AddressOf) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X* operator&();
};
void test(X x) {
  &x;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-*
| | | |-operator
| | | |-&
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-&
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-x
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PrefixIncrement) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X operator++();
};
void test(X x) {
  ++x;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-operator
| | | |-++
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PrefixUnaryOperatorExpression
    | | |-++
    | | `-IdExpression
    | |   `-UnqualifiedId
    | |     `-x
    | `-;
    `-}
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperator_PostfixIncrement) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct X {
  X operator++(int);
};
void test(X x) {
  x++;
}
)cpp",
      R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-struct
| |-X
| |-{
| |-SimpleDeclaration
| | |-X
| | |-SimpleDeclarator
| | | |-operator
| | | |-++
| | | `-ParametersAndQualifiers
| | |   |-(
| | |   |-SimpleDeclaration
| | |   | `-int
| | |   `-)
| | `-;
| |-}
| `-;
`-SimpleDeclaration
  |-void
  |-SimpleDeclarator
  | |-test
  | `-ParametersAndQualifiers
  |   |-(
  |   |-SimpleDeclaration
  |   | |-X
  |   | `-SimpleDeclarator
  |   |   `-x
  |   `-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-PostfixUnaryOperatorExpression
    | | |-IdExpression
    | | | `-UnqualifiedId
    | | |   `-x
    | | `-++
    | `-;
    `-}
)txt"));
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

TEST_P(SyntaxTreeTest, Namespaces) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace a { namespace b {} }
namespace a::b {}
namespace {}

namespace foo = a;
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
|-NamespaceDefinition
| |-namespace
| |-a
| |-::
| |-b
| |-{
| `-}
|-NamespaceDefinition
| |-namespace
| |-{
| `-}
`-NamespaceAliasDefinition
  |-namespace
  |-foo
  |-=
  |-a
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, UsingDirective) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace ns {}
using namespace ::ns;
)cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| `-}
`-UsingNamespaceDirective
  |-using
  |-namespace
  |-NestedNameSpecifier
  | `-::
  |-ns
  `-;
)txt"));
}

TEST_P(SyntaxTreeTest, UsingDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace ns { int a; }
using ns::a;
)cpp",
      R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| |-SimpleDeclaration
| | |-int
| | |-SimpleDeclarator
| | | `-a
| | `-;
| `-}
`-UsingDeclaration
  |-using
  |-NestedNameSpecifier
  | |-IdentifierNameSpecifier
  | | `-ns
  | `-::
  |-a
  `-;
)txt"));
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

TEST_P(SyntaxTreeTest, Templates) {
  if (!GetParam().isCXX()) {
    return;
  }
  if (GetParam().hasDelayedTemplateParsing()) {
    // FIXME: Make this test work on Windows by generating the expected syntax
    // tree when `-fdelayed-template-parsing` is active.
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> struct cls {};
template <class T> int var = 10;
template <class T> int fun() {}
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
|   |-cls
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
|   |-int
|   |-SimpleDeclarator
|   | |-var
|   | |-=
|   | `-IntegerLiteralExpression
|   |   `-10
|   `-;
`-TemplateDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  `-SimpleDeclaration
    |-int
    |-SimpleDeclarator
    | |-fun
    | `-ParametersAndQualifiers
    |   |-(
    |   `-)
    `-CompoundStatement
      |-{
      `-}
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
