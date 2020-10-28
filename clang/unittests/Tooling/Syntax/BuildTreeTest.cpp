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
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'main'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   `-')' CloseParen
| `-CompoundStatement
|   |-'{' OpenParen
|   `-'}' CloseParen
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    `-'}' CloseParen
)txt"));
}

TEST_P(SyntaxTreeTest, SimpleVariable) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a;
int b = 42;
)cpp",
      R"txt(
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | `-'a'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'b'
  | |-'='
  | `-IntegerLiteralExpression
  |   `-'42' LiteralToken
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, SimpleFunction) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void foo(int a, int b) {}
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | |-'int'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'a'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'int'
  |   |   `-SimpleDeclarator Declarator
  |   |     `-'b'
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    `-'}' CloseParen
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
IfStatement Statement
|-'if' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement ThenStatement
  |-'{' OpenParen
  `-'}' CloseParen
  )txt",
       R"txt(
IfStatement Statement
|-'if' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
|-CompoundStatement ThenStatement
| |-'{' OpenParen
| `-'}' CloseParen
|-'else' ElseKeyword
`-IfStatement ElseStatement
  |-'if' IntroducerKeyword
  |-'('
  |-IntegerLiteralExpression
  | `-'0' LiteralToken
  |-')'
  `-CompoundStatement ThenStatement
    |-'{' OpenParen
    `-'}' CloseParen
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
ForStatement Statement
|-'for' IntroducerKeyword
|-'('
|-';'
|-';'
|-')'
`-CompoundStatement BodyStatement
  |-'{' OpenParen
  `-'}' CloseParen
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
RangeBasedForStatement Statement
|-'for' IntroducerKeyword
|-'('
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | `-'x'
| `-':'
|-IdExpression
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-')'
`-EmptyStatement BodyStatement
  `-';'
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
DeclarationStatement Statement
|-SimpleDeclaration
| |-'int'
| `-SimpleDeclarator Declarator
|   |-'a'
|   |-'='
|   `-IntegerLiteralExpression
|     `-'10' LiteralToken
`-';'
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
SwitchStatement Statement
|-'switch' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement BodyStatement
  |-'{' OpenParen
  |-CaseStatement Statement
  | |-'case' IntroducerKeyword
  | |-IntegerLiteralExpression CaseValue
  | | `-'0' LiteralToken
  | |-':'
  | `-DefaultStatement BodyStatement
  |   |-'default' IntroducerKeyword
  |   |-':'
  |   `-EmptyStatement BodyStatement
  |     `-';'
  `-'}' CloseParen
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
WhileStatement Statement
|-'while' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement BodyStatement
  |-'{' OpenParen
  |-ContinueStatement Statement
  | |-'continue' IntroducerKeyword
  | `-';'
  |-BreakStatement Statement
  | |-'break' IntroducerKeyword
  | `-';'
  `-'}' CloseParen
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
UnknownStatement Statement
|-'foo'
|-':'
`-ReturnStatement
  |-'return' IntroducerKeyword
  |-IntegerLiteralExpression ReturnValue
  | `-'100' LiteralToken
  `-';'
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-ExpressionStatement Statement
    | |-CallExpression Expression
    | | |-IdExpression Callee
    | | | `-UnqualifiedId UnqualifiedId
    | | |   `-'test'
    | | |-'(' OpenParen
    | | `-')' CloseParen
    | `-';'
    |-IfStatement Statement
    | |-'if' IntroducerKeyword
    | |-'('
    | |-IntegerLiteralExpression
    | | `-'1' LiteralToken
    | |-')'
    | |-ExpressionStatement ThenStatement
    | | |-CallExpression Expression
    | | | |-IdExpression Callee
    | | | | `-UnqualifiedId UnqualifiedId
    | | | |   `-'test'
    | | | |-'(' OpenParen
    | | | `-')' CloseParen
    | | `-';'
    | |-'else' ElseKeyword
    | `-ExpressionStatement ElseStatement
    |   |-CallExpression Expression
    |   | |-IdExpression Callee
    |   | | `-UnqualifiedId UnqualifiedId
    |   | |   `-'test'
    |   | |-'(' OpenParen
    |   | `-')' CloseParen
    |   `-';'
    `-'}' CloseParen
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
IdExpression Expression
`-UnqualifiedId UnqualifiedId
  `-'a'
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
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   |-'operator'
|   `-'+'
|-'(' OpenParen
|-CallArguments Arguments
| |-IdExpression ListElement
| | `-UnqualifiedId UnqualifiedId
| |   `-'x'
| |-',' ListDelimiter
| `-IdExpression ListElement
|   `-UnqualifiedId UnqualifiedId
|     `-'x'
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'x'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     |-'operator'
|     `-'int'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   |-'operator'
|   |-'""'
|   `-'_w'
|-'(' OpenParen
|-CallArguments Arguments
| `-CharacterLiteralExpression ListElement
|   `-''1'' LiteralToken
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'x'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     |-'~'
|     `-'X'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'x'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     `-'~'
|-'decltype'
|-'('
|-'x'
|-')'
|-'('
`-')' CloseParen
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
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'(' OpenParen
`-')' CloseParen
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
| |-'::' ListDelimiter
| |-IdentifierNameSpecifier ListElement
| | `-'n'
| `-'::' ListDelimiter
|-'S'
`-SimpleDeclarator Declarator
  `-UnknownExpression
    `-'s1'
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-IdentifierNameSpecifier ListElement
| | `-'n'
| `-'::' ListDelimiter
|-'S'
`-SimpleDeclarator Declarator
  `-UnknownExpression
    `-'s2'
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
| |-'::' ListDelimiter
| |-SimpleTemplateNameSpecifier ListElement
| | |-'template'
| | |-'ST'
| | |-'<'
| | |-'int'
| | `-'>'
| `-'::' ListDelimiter
|-'S'
`-SimpleDeclarator Declarator
  `-UnknownExpression
    `-'s1'
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-'::' ListDelimiter
| |-SimpleTemplateNameSpecifier ListElement
| | |-'ST'
| | |-'<'
| | |-'int'
| | `-'>'
| `-'::' ListDelimiter
|-'S'
`-SimpleDeclarator Declarator
  `-UnknownExpression
    `-'s2'
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
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-DecltypeNameSpecifier ListElement
| | | |-'decltype'
| | | |-'('
| | | |-IdExpression
| | | | `-UnqualifiedId UnqualifiedId
| | | |   `-'s'
| | | `-')'
| | `-'::' ListDelimiter
| `-UnqualifiedId UnqualifiedId
|   `-'f'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'S'
| | `-'::' ListDelimiter
| `-UnqualifiedId UnqualifiedId
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'(' OpenParen
`-')' CloseParen
)txt",
       R"txt(
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'S'
| | `-'::' ListDelimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId UnqualifiedId
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-'::' ListDelimiter
| | |-IdentifierNameSpecifier ListElement
| | | `-'n'
| | |-'::' ListDelimiter
| | |-SimpleTemplateNameSpecifier ListElement
| | | |-'template'
| | | |-'ST'
| | | |-'<'
| | | |-'int'
| | | `-'>'
| | `-'::' ListDelimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId UnqualifiedId
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'T'
| | |-'::' ListDelimiter
| | |-SimpleTemplateNameSpecifier ListElement
| | | |-'template'
| | | |-'U'
| | | |-'<'
| | | |-'int'
| | | `-'>'
| | `-'::' ListDelimiter
| `-UnqualifiedId UnqualifiedId
|   `-'f'
|-'(' OpenParen
`-')' CloseParen
)txt",
       R"txt(
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'T'
| | |-'::' ListDelimiter
| | |-IdentifierNameSpecifier ListElement
| | | `-'U'
| | `-'::' ListDelimiter
| `-UnqualifiedId UnqualifiedId
|   `-'f'
|-'(' OpenParen
`-')' CloseParen
)txt",
       R"txt(
CallExpression Expression
|-IdExpression Callee
| |-NestedNameSpecifier Qualifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'T'
| | `-'::' ListDelimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId UnqualifiedId
|   |-'f'
|   |-'<'
|   |-IntegerLiteralExpression
|   | `-'0' LiteralToken
|   `-'>'
|-'(' OpenParen
`-')' CloseParen
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
ThisExpression ReturnValue
`-'this' IntroducerKeyword
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
MemberExpression Expression
|-ThisExpression Object
| `-'this' IntroducerKeyword
|-'->' AccessToken
`-IdExpression Member
  `-UnqualifiedId UnqualifiedId
    `-'a'
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
IdExpression Expression
`-UnqualifiedId UnqualifiedId
  `-'a'
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
ParenExpression Expression
|-'(' OpenParen
|-IntegerLiteralExpression SubExpression
| `-'1' LiteralToken
`-')' CloseParen
)txt",
       R"txt(
ParenExpression Expression
|-'(' OpenParen
|-ParenExpression SubExpression
| |-'(' OpenParen
| |-IntegerLiteralExpression SubExpression
| | `-'1' LiteralToken
| `-')' CloseParen
`-')' CloseParen
)txt",
       R"txt(
ParenExpression Expression
|-'(' OpenParen
|-BinaryOperatorExpression SubExpression
| |-IntegerLiteralExpression LeftHandSide
| | `-'1' LiteralToken
| |-'+' OperatorToken
| `-ParenExpression RightHandSide
|   |-'(' OpenParen
|   |-IntegerLiteralExpression SubExpression
|   | `-'2' LiteralToken
|   `-')' CloseParen
`-')' CloseParen
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
CharUserDefinedLiteralExpression Expression
`-''2'_c' LiteralToken
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
StringUserDefinedLiteralExpression Expression
`-'"12"_s' LiteralToken
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
IntegerUserDefinedLiteralExpression Expression
`-'12_i' LiteralToken
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression Expression
`-'12_r' LiteralToken
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression Expression
`-'12_t' LiteralToken
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
FloatUserDefinedLiteralExpression Expression
`-'1.2_f' LiteralToken
)txt",
       R"txt(
FloatUserDefinedLiteralExpression Expression
`-'1.2_r' LiteralToken
)txt",
       R"txt(
FloatUserDefinedLiteralExpression Expression
`-'1.2_t' LiteralToken
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
IntegerLiteralExpression Expression
`-'12ll' LiteralToken
)txt",
       R"txt(
IntegerLiteralExpression Expression
`-'12ull' LiteralToken
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
IntegerLiteralExpression Expression
`-'0b1100' LiteralToken
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
IntegerLiteralExpression Expression
`-'1'2'0ull' LiteralToken
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
CharacterLiteralExpression Expression
`-''a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-''\n'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-''\x20'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-''\0'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'L'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'L'α'' LiteralToken
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
CharacterLiteralExpression Expression
`-'u'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'u'構'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'U'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'U'🌲'' LiteralToken
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
CharacterLiteralExpression Expression
`-'u8'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression Expression
`-'u8'\x7f'' LiteralToken
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
FloatingLiteralExpression Expression
`-'1e-2' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'2.' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'.2' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'2.f' LiteralToken
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
FloatingLiteralExpression Expression
`-'0xfp1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'0xf.p1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'0x.fp1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression Expression
`-'0xf.fp1f' LiteralToken
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
StringLiteralExpression Expression
`-'"a\n\0\x20"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression Expression
`-'L"αβ"' LiteralToken
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
StringLiteralExpression Expression
`-'u8"a\x1f\x05"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression Expression
`-'u"C++抽象構文木"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression Expression
`-'U"📖🌲\n"' LiteralToken
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
      "TranslationUnit Detached\n"
      "`-SimpleDeclaration\n"
      "  |-'void'\n"
      "  |-SimpleDeclarator Declarator\n"
      "  | |-'test'\n"
      "  | `-ParametersAndQualifiers\n"
      "  |   |-'(' OpenParen\n"
      "  |   `-')' CloseParen\n"
      "  `-CompoundStatement\n"
      "    |-'{' OpenParen\n"
      "    |-ExpressionStatement Statement\n"
      "    | |-StringLiteralExpression Expression\n"
      "    | | `-'R\"SyntaxTree(\n"
      "  Hello \"Syntax\" \\\"\n"
      "  )SyntaxTree\"' LiteralToken\n"
      "    | `-';'\n"
      "    `-'}' CloseParen\n"));
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
BoolLiteralExpression Expression
`-'true' LiteralToken
)txt",
       R"txt(
BoolLiteralExpression Expression
`-'false' LiteralToken
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
CxxNullPtrExpression Expression
`-'nullptr' LiteralToken
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
PostfixUnaryOperatorExpression Expression
|-IdExpression Operand
| `-UnqualifiedId UnqualifiedId
|   `-'a'
`-'++' OperatorToken
)txt",
       R"txt(
PostfixUnaryOperatorExpression Expression
|-IdExpression Operand
| `-UnqualifiedId UnqualifiedId
|   `-'a'
`-'--' OperatorToken
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
PrefixUnaryOperatorExpression Expression
|-'--' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'++' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'~' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'-' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'+' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'&' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'*' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'ap'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'!' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'__real' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'__imag' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
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
PrefixUnaryOperatorExpression Expression
|-'compl' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression Expression
|-'not' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'b'
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
BinaryOperatorExpression Expression
|-IntegerLiteralExpression LeftHandSide
| `-'1' LiteralToken
|-'-' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IntegerLiteralExpression LeftHandSide
| `-'1' LiteralToken
|-'==' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-'=' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'1' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-'<<=' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'1' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IntegerLiteralExpression LeftHandSide
| `-'1' LiteralToken
|-'||' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'0' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IntegerLiteralExpression LeftHandSide
| `-'1' LiteralToken
|-'&' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-'!=' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'3' LiteralToken
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
BinaryOperatorExpression Expression
|-BoolLiteralExpression LeftHandSide
| `-'true' LiteralToken
|-'||' OperatorToken
`-BoolLiteralExpression RightHandSide
  `-'false' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-BoolLiteralExpression LeftHandSide
| `-'true' LiteralToken
|-'or' OperatorToken
`-BoolLiteralExpression RightHandSide
  `-'false' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IntegerLiteralExpression LeftHandSide
| `-'1' LiteralToken
|-'bitand' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-'xor_eq' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'3' LiteralToken
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
BinaryOperatorExpression Expression
|-ParenExpression LeftHandSide
| |-'(' OpenParen
| |-BinaryOperatorExpression SubExpression
| | |-IntegerLiteralExpression LeftHandSide
| | | `-'1' LiteralToken
| | |-'+' OperatorToken
| | `-IntegerLiteralExpression RightHandSide
| |   `-'2' LiteralToken
| `-')' CloseParen
|-'*' OperatorToken
`-ParenExpression RightHandSide
  |-'(' OpenParen
  |-BinaryOperatorExpression SubExpression
  | |-IntegerLiteralExpression LeftHandSide
  | | `-'4' LiteralToken
  | |-'/' OperatorToken
  | `-IntegerLiteralExpression RightHandSide
  |   `-'2' LiteralToken
  `-')' CloseParen
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
BinaryOperatorExpression Expression
|-BinaryOperatorExpression LeftHandSide
| |-IdExpression LeftHandSide
| | `-UnqualifiedId UnqualifiedId
| |   `-'a'
| |-'+' OperatorToken
| `-IdExpression RightHandSide
|   `-UnqualifiedId UnqualifiedId
|     `-'b'
|-'+' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'42' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'a'
|-'=' OperatorToken
`-BinaryOperatorExpression RightHandSide
  |-IdExpression LeftHandSide
  | `-UnqualifiedId UnqualifiedId
  |   `-'b'
  |-'=' OperatorToken
  `-IntegerLiteralExpression RightHandSide
    `-'42' LiteralToken
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
BinaryOperatorExpression Expression
|-BinaryOperatorExpression LeftHandSide
| |-IntegerLiteralExpression LeftHandSide
| | `-'1' LiteralToken
| |-'+' OperatorToken
| `-BinaryOperatorExpression RightHandSide
|   |-IntegerLiteralExpression LeftHandSide
|   | `-'2' LiteralToken
|   |-'*' OperatorToken
|   `-IntegerLiteralExpression RightHandSide
|     `-'3' LiteralToken
|-'+' OperatorToken
`-IntegerLiteralExpression RightHandSide
  `-'4' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression Expression
|-BinaryOperatorExpression LeftHandSide
| |-IntegerLiteralExpression LeftHandSide
| | `-'1' LiteralToken
| |-'%' OperatorToken
| `-IntegerLiteralExpression RightHandSide
|   `-'2' LiteralToken
|-'+' OperatorToken
`-BinaryOperatorExpression RightHandSide
  |-IntegerLiteralExpression LeftHandSide
  | `-'3' LiteralToken
  |-'*' OperatorToken
  `-IntegerLiteralExpression RightHandSide
    `-'4' LiteralToken
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
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'x'
|-'=' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'y'
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
BinaryOperatorExpression Expression
|-UnknownExpression LeftHandSide
| `-IdExpression
|   `-UnqualifiedId UnqualifiedId
|     `-'x'
|-'+' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'y'
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
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'x'
|-'<' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'y'
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
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'x'
|-'<<' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'y'
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
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'x'
|-',' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'y'
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
BinaryOperatorExpression Expression
|-IdExpression LeftHandSide
| `-UnqualifiedId UnqualifiedId
|   `-'xp'
|-'->*' OperatorToken
`-IdExpression RightHandSide
  `-UnqualifiedId UnqualifiedId
    `-'pmi'
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
PrefixUnaryOperatorExpression Expression
|-'!' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'x'
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
PrefixUnaryOperatorExpression Expression
|-'&' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'x'
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
PrefixUnaryOperatorExpression Expression
|-'++' OperatorToken
`-IdExpression Operand
  `-UnqualifiedId UnqualifiedId
    `-'x'
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
PostfixUnaryOperatorExpression Expression
|-IdExpression Operand
| `-UnqualifiedId UnqualifiedId
|   `-'x'
`-'++' OperatorToken
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
MemberExpression Expression
|-IdExpression Object
| `-UnqualifiedId UnqualifiedId
|   `-'s'
|-'.' AccessToken
`-IdExpression Member
  `-UnqualifiedId UnqualifiedId
    `-'a'
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
MemberExpression Expression
|-IdExpression Object
| `-UnqualifiedId UnqualifiedId
|   `-'s'
|-'.' AccessToken
`-IdExpression Member
  `-UnqualifiedId UnqualifiedId
    `-'a'
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
MemberExpression Expression
|-IdExpression Object
| `-UnqualifiedId UnqualifiedId
|   `-'sp'
|-'->' AccessToken
`-IdExpression Member
  `-UnqualifiedId UnqualifiedId
    `-'a'
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
MemberExpression Expression
|-MemberExpression Object
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     `-'next'
|-'->' AccessToken
`-IdExpression Member
  `-UnqualifiedId UnqualifiedId
    `-'next'
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     |-'operator'
|     `-'!'
|-'(' OpenParen
`-')' CloseParen
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
|-'{' OpenParen
|-ExpressionStatement Statement
| `-MemberExpression Expression
|   |-IdExpression Object
|   | `-UnqualifiedId UnqualifiedId
|   |   `-'s'
|   |-'.' AccessToken
|   `-IdExpression Member
|     `-UnqualifiedId UnqualifiedId
|       `-'x'
|-'<'
|-'int'
|-'>'
|-';'
`-'}' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'sp'
| |-'->' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| |-'template'
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| `-IdExpression Member
|   |-NestedNameSpecifier Qualifier
|   | |-IdentifierNameSpecifier ListElement
|   | | `-'Base'
|   | `-'::' ListDelimiter
|   `-UnqualifiedId UnqualifiedId
|     `-'f'
|-'(' OpenParen
`-')' CloseParen
      )txt",
       R"txt(
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| `-IdExpression Member
|   |-NestedNameSpecifier Qualifier
|   | |-'::' ListDelimiter
|   | |-IdentifierNameSpecifier ListElement
|   | | `-'S'
|   | `-'::' ListDelimiter
|   `-UnqualifiedId UnqualifiedId
|     |-'~'
|     `-'S'
|-'(' OpenParen
`-')' CloseParen
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
CallExpression Expression
|-MemberExpression Callee
| |-CallExpression Object
| | |-MemberExpression Callee
| | | |-IdExpression Object
| | | | `-UnqualifiedId UnqualifiedId
| | | |   `-'sp'
| | | |-'->' AccessToken
| | | `-IdExpression Member
| | |   `-UnqualifiedId UnqualifiedId
| | |     `-'getU'
| | |-'(' OpenParen
| | `-')' CloseParen
| |-'.' AccessToken
| `-IdExpression Member
|   |-NestedNameSpecifier Qualifier
|   | |-SimpleTemplateNameSpecifier ListElement
|   | | |-'template'
|   | | |-'U'
|   | | |-'<'
|   | | |-'int'
|   | | `-'>'
|   | `-'::' ListDelimiter
|   |-'template' TemplateKeyword
|   `-UnqualifiedId UnqualifiedId
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_Member) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S{
  void f();
};
void test(S s) {
  [[s.f()]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-MemberExpression Callee
| |-IdExpression Object
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'.' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     `-'f'
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_OperatorParens) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  void operator()();
};
void test(S s) {
  [[s()]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   `-'s'
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_OperatorParensChaining) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  S operator()();
};
void test(S s) {
  [[s()()]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-CallExpression Callee
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'s'
| |-'(' OpenParen
| `-')' CloseParen
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_MemberWithThis) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct Base {
  void f();
};
struct S: public Base {
  void f();
  void test() {
    [[this->f()]];
    [[f()]];
    [[this->Base::f()]];
  }
};
)cpp",
      {R"txt(
CallExpression Expression
|-MemberExpression Callee
| |-ThisExpression Object
| | `-'this' IntroducerKeyword
| |-'->' AccessToken
| `-IdExpression Member
|   `-UnqualifiedId UnqualifiedId
|     `-'f'
|-'(' OpenParen
`-')' CloseParen
      )txt",
       R"txt(
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   `-'f'
|-'(' OpenParen
`-')' CloseParen
      )txt",
       R"txt(
CallExpression Expression
|-MemberExpression Callee
| |-ThisExpression Object
| | `-'this' IntroducerKeyword
| |-'->' AccessToken
| `-IdExpression Member
|   |-NestedNameSpecifier Qualifier
|   | |-IdentifierNameSpecifier ListElement
|   | | `-'Base'
|   | `-'::' ListDelimiter
|   `-UnqualifiedId UnqualifiedId
|     `-'f'
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_FunctionPointer) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void (*pf)();
void test() {
  [[pf()]];
  [[(*pf)()]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-IdExpression Callee
| `-UnqualifiedId UnqualifiedId
|   `-'pf'
|-'(' OpenParen
`-')' CloseParen
)txt",
       R"txt(
CallExpression Expression
|-ParenExpression Callee
| |-'(' OpenParen
| |-PrefixUnaryOperatorExpression SubExpression
| | |-'*' OperatorToken
| | `-IdExpression Operand
| |   `-UnqualifiedId UnqualifiedId
| |     `-'pf'
| `-')' CloseParen
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Callee_MemberFunctionPointer) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  void f();
};
void test(S s) {
  void (S::*pmf)();
  pmf = &S::f;
  [[(s.*pmf)()]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-ParenExpression Callee
| |-'(' OpenParen
| |-BinaryOperatorExpression SubExpression
| | |-IdExpression LeftHandSide
| | | `-UnqualifiedId UnqualifiedId
| | |   `-'s'
| | |-'.*' OperatorToken
| | `-IdExpression RightHandSide
| |   `-UnqualifiedId UnqualifiedId
| |     `-'pmf'
| `-')' CloseParen
|-'(' OpenParen
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_Zero) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void f();
void test() {
  [[f();]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_One) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void f(int);
void test() {
  [[f(1);]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | `-IntegerLiteralExpression ListElement
| |   `-'1' LiteralToken
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_Multiple) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void f(int, char, float);
void test() {
  [[f(1, '2', 3.);]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | |-IntegerLiteralExpression ListElement
| | | `-'1' LiteralToken
| | |-',' ListDelimiter
| | |-CharacterLiteralExpression ListElement
| | | `-''2'' LiteralToken
| | |-',' ListDelimiter
| | `-FloatingLiteralExpression ListElement
| |   `-'3.' LiteralToken
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_Assignment) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void f(int);
void test(int a) {
  [[f(a = 1);]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | `-BinaryOperatorExpression ListElement
| |   |-IdExpression LeftHandSide
| |   | `-UnqualifiedId UnqualifiedId
| |   |   `-'a'
| |   |-'=' OperatorToken
| |   `-IntegerLiteralExpression RightHandSide
| |     `-'1' LiteralToken
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_BracedInitList_Empty) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
void f(int[]);
void test() {
  [[f({});]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | `-UnknownExpression ListElement
| |   `-UnknownExpression
| |     |-'{'
| |     `-'}'
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_BracedInitList_Simple) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct TT {};
struct T{
  int a;
  TT b;
};
void f(T);
void test() {
  [[f({1, {}});]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | `-UnknownExpression ListElement
| |   `-UnknownExpression
| |     |-'{'
| |     |-IntegerLiteralExpression
| |     | `-'1' LiteralToken
| |     |-','
| |     |-UnknownExpression
| |     | `-UnknownExpression
| |     |   |-'{'
| |     |   `-'}'
| |     `-'}'
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_BracedInitList_Designated) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct TT {};
struct T{
  int a;
  TT b;
};
void f(T);
void test() {
  [[f({.a = 1, .b {}});]]
}
)cpp",
      {R"txt(
ExpressionStatement Statement
|-CallExpression Expression
| |-IdExpression Callee
| | `-UnqualifiedId UnqualifiedId
| |   `-'f'
| |-'(' OpenParen
| |-CallArguments Arguments
| | `-UnknownExpression ListElement
| |   `-UnknownExpression
| |     |-'{'
| |     |-UnknownExpression
| |     | |-'.'
| |     | |-'a'
| |     | |-'='
| |     | `-IntegerLiteralExpression
| |     |   `-'1' LiteralToken
| |     |-','
| |     |-UnknownExpression
| |     | |-'.'
| |     | |-'b'
| |     | `-UnknownExpression
| |     |   `-UnknownExpression
| |     |     |-'{'
| |     |     `-'}'
| |     `-'}'
| `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, CallExpression_Arguments_ParameterPack) {
  if (!GetParam().isCXX11OrLater() || GetParam().hasDelayedTemplateParsing()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T, typename... Args>
void test(T t, Args... args) {
  [[test(args...)]];
}
)cpp",
      {R"txt(
CallExpression Expression
|-UnknownExpression Callee
| `-'test'
|-'(' OpenParen
|-CallArguments Arguments
| `-UnknownExpression ListElement
|   |-IdExpression
|   | `-UnqualifiedId UnqualifiedId
|   |   `-'args'
|   `-'...'
`-')' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGrouping) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int *a, b;
int *c, d;
)cpp",
      R"txt(
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'*'
| | `-'a'
| |-','
| |-SimpleDeclarator Declarator
| | `-'b'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'*'
  | `-'c'
  |-','
  |-SimpleDeclarator Declarator
  | `-'d'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, MultipleDeclaratorsGroupingTypedef) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
typedef int *a, b;
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'typedef'
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'*'
  | `-'a'
  |-','
  |-SimpleDeclarator Declarator
  | `-'b'
  `-';'
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-DeclarationStatement Statement
    | |-SimpleDeclaration
    | | |-'int'
    | | |-SimpleDeclarator Declarator
    | | | |-'*'
    | | | `-'a'
    | | |-','
    | | `-SimpleDeclarator Declarator
    | |   `-'b'
    | `-';'
    |-DeclarationStatement Statement
    | |-SimpleDeclaration
    | | |-'typedef'
    | | |-'int'
    | | |-SimpleDeclarator Declarator
    | | | |-'*'
    | | | `-'ta'
    | | |-','
    | | `-SimpleDeclarator Declarator
    | |   `-'tb'
    | `-';'
    `-'}' CloseParen
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'typedef'
  |-'decltype'
  |-'('
  |-UnknownExpression
  | |-'sizeof'
  | |-'('
  | |-'void'
  | |-'*'
  | `-')'
  |-')'
  |-SimpleDeclarator Declarator
  | `-'size_t'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, Namespace_Nested) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace a { namespace b {} }
)cpp",
      R"txt(
TranslationUnit Detached
`-NamespaceDefinition
  |-'namespace'
  |-'a'
  |-'{'
  |-NamespaceDefinition
  | |-'namespace'
  | |-'b'
  | |-'{'
  | `-'}'
  `-'}'
)txt"));
}

TEST_P(SyntaxTreeTest, Namespace_NestedDefinition) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
namespace a::b {}
)cpp",
      R"txt(
TranslationUnit Detached
`-NamespaceDefinition
  |-'namespace'
  |-'a'
  |-'::'
  |-'b'
  |-'{'
  `-'}'
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
TranslationUnit Detached
`-NamespaceDefinition
  |-'namespace'
  |-'{'
  `-'}'
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
|-'namespace'
|-'foo'
|-'='
|-'a'
`-';'
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
|-'using'
|-'namespace'
|-NestedNameSpecifier
| `-'::' ListDelimiter
|-'ns'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingDeclaration_Namespace) {
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
|-'using'
|-NestedNameSpecifier
| |-IdentifierNameSpecifier ListElement
| | `-'ns'
| `-'::' ListDelimiter
|-'a'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingDeclaration_ClassMember) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X {
  [[using T::foo;]]
  [[using typename T::bar;]]
};
)cpp",
      {R"txt(
UsingDeclaration
|-'using'
|-NestedNameSpecifier
| |-IdentifierNameSpecifier ListElement
| | `-'T'
| `-'::' ListDelimiter
|-'foo'
`-';'
)txt",
       R"txt(
UsingDeclaration
|-'using'
|-'typename'
|-NestedNameSpecifier
| |-IdentifierNameSpecifier ListElement
| | `-'T'
| `-'::' ListDelimiter
|-'bar'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingTypeAlias) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
using type = int;
)cpp",
      R"txt(
TranslationUnit Detached
`-TypeAliasDeclaration
  |-'using'
  |-'type'
  |-'='
  |-'int'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, FreeStandingClass_ForwardDeclaration) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
[[struct X;]]
[[struct Y *y1;]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'struct'
|-'X'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'struct'
|-'Y'
|-SimpleDeclarator Declarator
| |-'*'
| `-'y1'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, FreeStandingClasses_Definition) {
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
[[struct X {};]]
[[struct Y {} *y2;]]
[[struct {} *a1;]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'struct'
|-'X'
|-'{'
|-'}'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'struct'
|-'Y'
|-'{'
|-'}'
|-SimpleDeclarator Declarator
| |-'*'
| `-'y2'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'struct'
|-'{'
|-'}'
|-SimpleDeclarator Declarator
| |-'*'
| `-'a1'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, StaticMemberFunction) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  [[static void f(){}]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-'static'
|-'void'
|-SimpleDeclarator Declarator
| |-'f'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   `-')' CloseParen
`-CompoundStatement
  |-'{' OpenParen
  `-'}' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, OutOfLineMemberFunctionDefinition) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  void f();
};
[[void S::f(){}]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-NestedNameSpecifier
| | |-IdentifierNameSpecifier ListElement
| | | `-'S'
| | `-'::' ListDelimiter
| |-'f'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   `-')' CloseParen
`-CompoundStatement
  |-'{' OpenParen
  `-'}' CloseParen
)txt"}));
}

TEST_P(SyntaxTreeTest, ConversionMemberFunction) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  [[operator int();]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-SimpleDeclarator Declarator
| |-'operator'
| |-'int'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   `-')' CloseParen
`-';'
)txt"}));
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'unsigned'
  |-SimpleDeclarator Declarator
  | |-'operator'
  | |-'""'
  | |-'_c'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | `-SimpleDeclaration ListElement
  |   |   `-'char'
  |   `-')' CloseParen
  `-';'
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
TranslationUnit Detached
`-TemplateDeclaration Declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-SimpleDeclaration
  | `-'char'
  |-'...'
  |-'>'
  `-SimpleDeclaration
    |-'unsigned'
    |-SimpleDeclarator Declarator
    | |-'operator'
    | |-'""'
    | |-'_t'
    | `-ParametersAndQualifiers
    |   |-'(' OpenParen
    |   `-')' CloseParen
    `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, OverloadedOperatorDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  [[X& operator=(const X&);]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-'X'
|-SimpleDeclarator Declarator
| |-'&'
| |-'operator'
| |-'='
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-ParameterDeclarationList Parameters
|   | `-SimpleDeclaration ListElement
|   |   |-'const'
|   |   |-'X'
|   |   `-SimpleDeclarator Declarator
|   |     `-'&'
|   `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, OverloadedOperatorFriendDeclaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  [[friend X operator+(X, const X&);]]
};
)cpp",
      {R"txt(
UnknownDeclaration
`-SimpleDeclaration
  |-'friend'
  |-'X'
  |-SimpleDeclarator Declarator
  | |-'operator'
  | |-'+'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | `-'X'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'const'
  |   |   |-'X'
  |   |   `-SimpleDeclarator Declarator
  |   |     `-'&'
  |   `-')' CloseParen
  `-';'
)txt"}));
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
TranslationUnit Detached
`-TemplateDeclaration Declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'typename'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'struct'
    |-'ST'
    |-'{'
    |-'}'
    `-';'
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
TranslationUnit Detached
`-TemplateDeclaration Declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'typename'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'T'
    |-SimpleDeclarator Declarator
    | |-'f'
    | `-ParametersAndQualifiers
    |   |-'(' OpenParen
    |   `-')' CloseParen
    `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, VariableTemplateDeclaration) {
  if (!GetParam().isCXX14OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
template <class T> T var = 10;
)cpp",
      R"txt(
TranslationUnit Detached
`-TemplateDeclaration Declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'class'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'T'
    |-SimpleDeclarator Declarator
    | |-'var'
    | |-'='
    | `-IntegerLiteralExpression
    |   `-'10' LiteralToken
    `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, StaticMemberFunctionTemplate) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  [[template<typename U>
  static U f();]]
};
)cpp",
      {R"txt(
TemplateDeclaration Declaration
|-'template' IntroducerKeyword
|-'<'
|-UnknownDeclaration
| |-'typename'
| `-'U'
|-'>'
`-SimpleDeclaration
  |-'static'
  |-'U'
  |-SimpleDeclarator Declarator
  | |-'f'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-';'
)txt"}));
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
TranslationUnit Detached
`-TemplateDeclaration Declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'class'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'struct'
    |-'X'
    |-'{'
    |-TemplateDeclaration Declaration
    | |-'template' IntroducerKeyword
    | |-'<'
    | |-UnknownDeclaration
    | | |-'class'
    | | `-'U'
    | |-'>'
    | `-SimpleDeclaration
    |   |-'U'
    |   |-SimpleDeclarator Declarator
    |   | |-'foo'
    |   | `-ParametersAndQualifiers
    |   |   |-'(' OpenParen
    |   |   `-')' CloseParen
    |   `-';'
    |-'}'
    `-';'
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
TranslationUnit Detached
`-NamespaceDefinition
  |-'namespace'
  |-'n'
  |-'{'
  |-TemplateDeclaration Declaration
  | |-'template' IntroducerKeyword
  | |-'<'
  | |-UnknownDeclaration
  | | |-'typename'
  | | `-'T'
  | |-'>'
  | `-SimpleDeclaration
  |   |-'struct'
  |   |-'ST'
  |   |-'{'
  |   |-TemplateDeclaration Declaration
  |   | |-'template' IntroducerKeyword
  |   | |-'<'
  |   | |-UnknownDeclaration
  |   | | |-'typename'
  |   | | `-'U'
  |   | |-'>'
  |   | `-SimpleDeclaration
  |   |   |-'static'
  |   |   |-'U'
  |   |   |-SimpleDeclarator Declarator
  |   |   | |-'f'
  |   |   | `-ParametersAndQualifiers
  |   |   |   |-'(' OpenParen
  |   |   |   `-')' CloseParen
  |   |   `-';'
  |   |-'}'
  |   `-';'
  `-'}'
)txt"));
}

TEST_P(SyntaxTreeTest, ClassTemplate_MemberClassDefinition) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X { struct Y; };
[[template <class T> struct X<T>::Y {};]]
)cpp",
      {R"txt(
TemplateDeclaration Declaration
|-'template' IntroducerKeyword
|-'<'
|-UnknownDeclaration
| |-'class'
| `-'T'
|-'>'
`-SimpleDeclaration
  |-'struct'
  |-NestedNameSpecifier
  | |-SimpleTemplateNameSpecifier ListElement
  | | |-'X'
  | | |-'<'
  | | |-'T'
  | | `-'>'
  | `-'::' ListDelimiter
  |-'Y'
  |-'{'
  |-'}'
  `-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ExplicitClassTemplateInstantation_Definition) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X {};
[[template struct X<double>;]]
)cpp",
      {R"txt(
ExplicitTemplateInstantiation
|-'template' IntroducerKeyword
`-SimpleDeclaration Declaration
  |-'struct'
  |-'X'
  |-'<'
  |-'double'
  |-'>'
  `-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ExplicitClassTemplateInstantation_Declaration) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X {};
[[extern template struct X<float>;]]
)cpp",
      {R"txt(
ExplicitTemplateInstantiation
|-'extern' ExternKeyword
|-'template' IntroducerKeyword
`-SimpleDeclaration Declaration
  |-'struct'
  |-'X'
  |-'<'
  |-'float'
  |-'>'
  `-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ClassTemplateSpecialization_Partial) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X {};
[[template <class T> struct X<T*> {};]]
)cpp",
      {R"txt(
TemplateDeclaration Declaration
|-'template' IntroducerKeyword
|-'<'
|-UnknownDeclaration
| |-'class'
| `-'T'
|-'>'
`-SimpleDeclaration
  |-'struct'
  |-'X'
  |-'<'
  |-'T'
  |-'*'
  |-'>'
  |-'{'
  |-'}'
  `-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ClassTemplateSpecialization_Full) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template <class T> struct X {};
[[template <> struct X<int> {};]]
)cpp",
      {R"txt(
TemplateDeclaration Declaration
|-'template' IntroducerKeyword
|-'<'
|-'>'
`-SimpleDeclaration
  |-'struct'
  |-'X'
  |-'<'
  |-'int'
  |-'>'
  |-'{'
  |-'}'
  `-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, EmptyDeclaration) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
;
)cpp",
      R"txt(
TranslationUnit Detached
`-EmptyDeclaration
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, StaticAssert) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
static_assert(true, "message");
)cpp",
      R"txt(
TranslationUnit Detached
`-StaticAssertDeclaration
  |-'static_assert'
  |-'('
  |-BoolLiteralExpression Condition
  | `-'true' LiteralToken
  |-','
  |-StringLiteralExpression Message
  | `-'"message"' LiteralToken
  |-')'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, StaticAssert_WithoutMessage) {
  if (!GetParam().isCXX17OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
static_assert(true);
)cpp",
      R"txt(
TranslationUnit Detached
`-StaticAssertDeclaration
  |-'static_assert'
  |-'('
  |-BoolLiteralExpression Condition
  | `-'true' LiteralToken
  |-')'
  `-';'
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
TranslationUnit Detached
|-LinkageSpecificationDeclaration
| |-'extern'
| |-'"C"'
| `-SimpleDeclaration
|   |-'int'
|   |-SimpleDeclarator Declarator
|   | `-'a'
|   `-';'
`-LinkageSpecificationDeclaration
  |-'extern'
  |-'"C"'
  |-'{'
  |-SimpleDeclaration
  | |-'int'
  | |-SimpleDeclarator Declarator
  | | `-'b'
  | `-';'
  |-SimpleDeclaration
  | |-'int'
  | |-SimpleDeclarator Declarator
  | | `-'c'
  | `-';'
  `-'}'
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-IfStatement Statement
    | |-'if' IntroducerKeyword unmodifiable
    | |-'(' unmodifiable
    | |-BinaryOperatorExpression unmodifiable
    | | |-IntegerLiteralExpression LeftHandSide unmodifiable
    | | | `-'1' LiteralToken unmodifiable
    | | |-'+' OperatorToken unmodifiable
    | | `-IntegerLiteralExpression RightHandSide unmodifiable
    | |   `-'1' LiteralToken unmodifiable
    | |-')' unmodifiable
    | |-CompoundStatement ThenStatement unmodifiable
    | | |-'{' OpenParen unmodifiable
    | | `-'}' CloseParen unmodifiable
    | |-'else' ElseKeyword
    | `-CompoundStatement ElseStatement
    |   |-'{' OpenParen
    |   `-'}' CloseParen
    `-'}' CloseParen
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-CompoundStatement Statement
    | |-'{' OpenParen
    | |-ExpressionStatement Statement
    | | |-IntegerLiteralExpression Expression
    | | | `-'1' LiteralToken
    | | `-';'
    | `-'}' CloseParen
    |-CompoundStatement Statement
    | |-'{' OpenParen
    | |-ExpressionStatement Statement
    | | |-IntegerLiteralExpression Expression
    | | | `-'2' LiteralToken
    | | `-';'
    | `-'}' CloseParen
    `-'}' CloseParen
)txt"));
}

TEST_P(SyntaxTreeTest, InitDeclarator_Brace) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a {};
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'a'
  | `-UnknownExpression
  |   `-UnknownExpression
  |     |-'{'
  |     `-'}'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, InitDeclarator_Paren) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct S {
  S(int);
};
[[S s(1);]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'S'
|-SimpleDeclarator Declarator
| `-UnknownExpression
|   |-'s'
|   |-'('
|   |-IntegerLiteralExpression
|   | `-'1' LiteralToken
|   `-')'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ArrayDeclarator_Simple) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int a[10];
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'a'
  | `-ArraySubscript
  |   |-'[' OpenParen
  |   |-IntegerLiteralExpression Size
  |   | `-'10' LiteralToken
  |   `-']' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ArrayDeclarator_Multidimensional) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int b[1][2][3];
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'b'
  | |-ArraySubscript
  | | |-'[' OpenParen
  | | |-IntegerLiteralExpression Size
  | | | `-'1' LiteralToken
  | | `-']' CloseParen
  | |-ArraySubscript
  | | |-'[' OpenParen
  | | |-IntegerLiteralExpression Size
  | | | `-'2' LiteralToken
  | | `-']' CloseParen
  | `-ArraySubscript
  |   |-'[' OpenParen
  |   |-IntegerLiteralExpression Size
  |   | `-'3' LiteralToken
  |   `-']' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ArrayDeclarator_UnknownBound) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int c[] = {1,2,3};
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'c'
  | |-ArraySubscript
  | | |-'[' OpenParen
  | | `-']' CloseParen
  | |-'='
  | `-UnknownExpression
  |   `-UnknownExpression
  |     |-'{'
  |     |-IntegerLiteralExpression
  |     | `-'1' LiteralToken
  |     |-','
  |     |-IntegerLiteralExpression
  |     | `-'2' LiteralToken
  |     |-','
  |     |-IntegerLiteralExpression
  |     | `-'3' LiteralToken
  |     `-'}'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ArrayDeclarator_Static) {
  if (!GetParam().isC99OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void f(int xs[static 10]);
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'f'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | `-SimpleDeclaration ListElement
  |   |   |-'int'
  |   |   `-SimpleDeclarator Declarator
  |   |     |-'xs'
  |   |     `-ArraySubscript
  |   |       |-'[' OpenParen
  |   |       |-'static'
  |   |       |-IntegerLiteralExpression Size
  |   |       | `-'10' LiteralToken
  |   |       `-']' CloseParen
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InFreeFunctions_Empty) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func();
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InFreeFunctions_Named) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func1(int a);
int func2(int *ap);
int func3(int a, float b);
)cpp",
      R"txt(
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'func1'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-ParameterDeclarationList Parameters
| |   | `-SimpleDeclaration ListElement
| |   |   |-'int'
| |   |   `-SimpleDeclarator Declarator
| |   |     `-'a'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'func2'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-ParameterDeclarationList Parameters
| |   | `-SimpleDeclaration ListElement
| |   |   |-'int'
| |   |   `-SimpleDeclarator Declarator
| |   |     |-'*'
| |   |     `-'ap'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func3'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | |-'int'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'a'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'float'
  |   |   `-SimpleDeclarator Declarator
  |   |     `-'b'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InFreeFunctions_Unnamed) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func1(int);
int func2(int *);
int func3(int, float);
)cpp",
      R"txt(
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'func1'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-ParameterDeclarationList Parameters
| |   | `-SimpleDeclaration ListElement
| |   |   `-'int'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'func2'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-ParameterDeclarationList Parameters
| |   | `-SimpleDeclaration ListElement
| |   |   |-'int'
| |   |   `-SimpleDeclarator Declarator
| |   |     `-'*'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func3'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | `-'int'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   `-'float'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest,
       ParametersAndQualifiers_InVariadicFunctionTemplate_ParameterPack) {
  if (!GetParam().isCXX11OrLater() || GetParam().hasDelayedTemplateParsing()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T, typename... Args>
[[void test(T , Args... );]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-'test'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-ParameterDeclarationList Parameters
|   | |-SimpleDeclaration ListElement
|   | | `-'T'
|   | |-',' ListDelimiter
|   | `-SimpleDeclaration ListElement
|   |   |-'Args'
|   |   `-'...'
|   `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest,
       ParametersAndQualifiers_InVariadicFunctionTemplate_NamedParameterPack) {
  if (!GetParam().isCXX11OrLater() || GetParam().hasDelayedTemplateParsing()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
template<typename T, typename... Args>
[[void test(T t, Args... args);]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-'test'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-ParameterDeclarationList Parameters
|   | |-SimpleDeclaration ListElement
|   | | |-'T'
|   | | `-SimpleDeclarator Declarator
|   | |   `-'t'
|   | |-',' ListDelimiter
|   | `-SimpleDeclaration ListElement
|   |   |-'Args'
|   |   |-'...'
|   |   `-SimpleDeclarator Declarator
|   |     `-'args'
|   `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest,
       ParametersAndQualifiers_InFreeFunctions_VariadicArguments) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void test(int , char ...);
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | `-'int'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   `-'char'
  |   |-'...'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest,
       ParametersAndQualifiers_InFreeFunctions_Cxx_CvQualifiers) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func(const int a, volatile int b, const volatile int c);
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | |-'const'
  |   | | |-'int'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'a'
  |   | |-',' ListDelimiter
  |   | |-SimpleDeclaration ListElement
  |   | | |-'volatile'
  |   | | |-'int'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'b'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'const'
  |   |   |-'volatile'
  |   |   |-'int'
  |   |   `-SimpleDeclarator Declarator
  |   |     `-'c'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InFreeFunctions_Cxx_Ref) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func(int& a);
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | `-SimpleDeclaration ListElement
  |   |   |-'int'
  |   |   `-SimpleDeclarator Declarator
  |   |     |-'&'
  |   |     `-'a'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InFreeFunctions_Cxx11_RefRef) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
int func(int&& a);
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | `-SimpleDeclaration ListElement
  |   |   |-'int'
  |   |   `-SimpleDeclarator Declarator
  |   |     |-'&&'
  |   |     `-'a'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InMemberFunctions_Simple) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
struct Test {
  int a();
};
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'struct'
  |-'Test'
  |-'{'
  |-SimpleDeclaration
  | |-'int'
  | |-SimpleDeclarator Declarator
  | | |-'a'
  | | `-ParametersAndQualifiers
  | |   |-'(' OpenParen
  | |   `-')' CloseParen
  | `-';'
  |-'}'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InMemberFunctions_CvQualifiers) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct Test {
  [[int b() const;]]
  [[int c() volatile;]]
  [[int d() const volatile;]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'b'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   `-'const'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'c'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   `-'volatile'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'d'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   |-'const'
|   `-'volatile'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InMemberFunctions_Ref) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct Test {
  [[int e() &;]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'e'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   `-'&'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ParametersAndQualifiers_InMemberFunctions_RefRef) {
  if (!GetParam().isCXX11OrLater()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct Test {
  [[int f() &&;]]
};
)cpp",
      {R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'f'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   `-'&&'
`-';'
)txt"}));
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'auto'
  |-SimpleDeclarator Declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-')' CloseParen
  |   `-TrailingReturnType TrailingReturn
  |     |-'->' ArrowToken
  |     `-'int'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, DynamicExceptionSpecification) {
  if (!GetParam().supportsCXXDynamicExceptionSpecification()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct MyException1 {};
struct MyException2 {};
[[int a() throw();]]
[[int b() throw(...);]]
[[int c() throw(MyException1);]]
[[int d() throw(MyException1, MyException2);]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'a'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   |-'throw'
|   |-'('
|   `-')'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'b'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   |-'throw'
|   |-'('
|   |-'...'
|   `-')'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'c'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   |-'throw'
|   |-'('
|   |-'MyException1'
|   `-')'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-'d'
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-')' CloseParen
|   |-'throw'
|   |-'('
|   |-'MyException1'
|   |-','
|   |-'MyException2'
|   `-')'
`-';'
)txt"}));
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
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'a'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-')' CloseParen
| |   `-'noexcept'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'b'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-')' CloseParen
  |   |-'noexcept'
  |   |-'('
  |   |-BoolLiteralExpression
  |   | `-'true' LiteralToken
  |   `-')'
  `-';'
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
TranslationUnit Detached
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | `-ParenDeclarator
| |   |-'(' OpenParen
| |   |-'a'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'*'
| | `-ParenDeclarator
| |   |-'(' OpenParen
| |   |-'b'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator Declarator
| | |-ParenDeclarator
| | | |-'(' OpenParen
| | | |-'*'
| | | |-'c'
| | | `-')' CloseParen
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-ParameterDeclarationList Parameters
| |   | `-SimpleDeclaration ListElement
| |   |   `-'int'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator Declarator
  | |-'*'
  | |-ParenDeclarator
  | | |-'(' OpenParen
  | | |-'d'
  | | `-')' CloseParen
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | `-SimpleDeclaration ListElement
  |   |   `-'int'
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, Declaration_ConstVolatileQualifiers_SimpleConst) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
const int west = -1;
int const east = 1;
)cpp",
      R"txt(
TranslationUnit Detached
|-SimpleDeclaration
| |-'const'
| |-'int'
| |-SimpleDeclarator Declarator
| | |-'west'
| | |-'='
| | `-PrefixUnaryOperatorExpression
| |   |-'-' OperatorToken
| |   `-IntegerLiteralExpression Operand
| |     `-'1' LiteralToken
| `-';'
`-SimpleDeclaration
  |-'int'
  |-'const'
  |-SimpleDeclarator Declarator
  | |-'east'
  | |-'='
  | `-IntegerLiteralExpression
  |   `-'1' LiteralToken
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, Declaration_ConstVolatileQualifiers_MultipleConst) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
const int const universal = 0;
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'const'
  |-'int'
  |-'const'
  |-SimpleDeclarator Declarator
  | |-'universal'
  | |-'='
  | `-IntegerLiteralExpression
  |   `-'0' LiteralToken
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, Declaration_ConstVolatileQualifiers_ConstAndVolatile) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
const int const *const *volatile b;
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'const'
  |-'int'
  |-'const'
  |-SimpleDeclarator Declarator
  | |-'*'
  | |-'const'
  | |-'*'
  | |-'volatile'
  | `-'b'
  `-';'
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
TranslationUnit Detached
`-SimpleDeclaration
  |-'auto'
  |-SimpleDeclarator Declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-')' CloseParen
  |   `-TrailingReturnType TrailingReturn
  |     |-'->' ArrowToken
  |     |-'auto'
  |     `-SimpleDeclarator Declarator
  |       |-ParenDeclarator
  |       | |-'(' OpenParen
  |       | |-'*'
  |       | `-')' CloseParen
  |       `-ParametersAndQualifiers
  |         |-'(' OpenParen
  |         |-ParameterDeclarationList Parameters
  |         | `-SimpleDeclaration ListElement
  |         |   `-'int'
  |         |-')' CloseParen
  |         `-TrailingReturnType TrailingReturn
  |           |-'->' ArrowToken
  |           |-'double'
  |           `-SimpleDeclarator Declarator
  |             `-'*'
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, MemberPointers) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {};
[[int X::* a;]]
[[const int X::* b;]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'int'
|-SimpleDeclarator Declarator
| |-MemberPointer
| | |-'X'
| | |-'::'
| | `-'*'
| `-'a'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'const'
|-'int'
|-SimpleDeclarator Declarator
| |-MemberPointer
| | |-'X'
| | |-'::'
| | `-'*'
| `-'b'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, MemberFunctionPointer) {
  if (!GetParam().isCXX()) {
    return;
  }
  EXPECT_TRUE(treeDumpEqualOnAnnotations(
      R"cpp(
struct X {
  struct Y {};
};
[[void (X::*xp)();]]
[[void (X::**xpp)(const int*);]]
// FIXME: Generate the right syntax tree for this type,
// i.e. create a syntax node for the outer member pointer
[[void (X::Y::*xyp)(const int*, char);]]
)cpp",
      {R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-ParenDeclarator
| | |-'(' OpenParen
| | |-MemberPointer
| | | |-'X'
| | | |-'::'
| | | `-'*'
| | |-'xp'
| | `-')' CloseParen
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   `-')' CloseParen
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-ParenDeclarator
| | |-'(' OpenParen
| | |-MemberPointer
| | | |-'X'
| | | |-'::'
| | | `-'*'
| | |-'*'
| | |-'xpp'
| | `-')' CloseParen
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-ParameterDeclarationList Parameters
|   | `-SimpleDeclaration ListElement
|   |   |-'const'
|   |   |-'int'
|   |   `-SimpleDeclarator Declarator
|   |     `-'*'
|   `-')' CloseParen
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator Declarator
| |-ParenDeclarator
| | |-'(' OpenParen
| | |-'X'
| | |-'::'
| | |-MemberPointer
| | | |-'Y'
| | | |-'::'
| | | `-'*'
| | |-'xyp'
| | `-')' CloseParen
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-ParameterDeclarationList Parameters
|   | |-SimpleDeclaration ListElement
|   | | |-'const'
|   | | |-'int'
|   | | `-SimpleDeclarator Declarator
|   | |   `-'*'
|   | |-',' ListDelimiter
|   | `-SimpleDeclaration ListElement
|   |   `-'char'
|   `-')' CloseParen
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, ComplexDeclarator) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int));
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'x'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | |-'char'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'a'
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'short'
  |   |   `-SimpleDeclarator Declarator
  |   |     |-ParenDeclarator
  |   |     | |-'(' OpenParen
  |   |     | |-'*'
  |   |     | |-'b'
  |   |     | `-')' CloseParen
  |   |     `-ParametersAndQualifiers
  |   |       |-'(' OpenParen
  |   |       |-ParameterDeclarationList Parameters
  |   |       | `-SimpleDeclaration ListElement
  |   |       |   `-'int'
  |   |       `-')' CloseParen
  |   `-')' CloseParen
  `-';'
)txt"));
}

TEST_P(SyntaxTreeTest, ComplexDeclarator2) {
  EXPECT_TRUE(treeDumpEqual(
      R"cpp(
void x(char a, short (*b)(int), long (**c)(long long));
)cpp",
      R"txt(
TranslationUnit Detached
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator Declarator
  | |-'x'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-ParameterDeclarationList Parameters
  |   | |-SimpleDeclaration ListElement
  |   | | |-'char'
  |   | | `-SimpleDeclarator Declarator
  |   | |   `-'a'
  |   | |-',' ListDelimiter
  |   | |-SimpleDeclaration ListElement
  |   | | |-'short'
  |   | | `-SimpleDeclarator Declarator
  |   | |   |-ParenDeclarator
  |   | |   | |-'(' OpenParen
  |   | |   | |-'*'
  |   | |   | |-'b'
  |   | |   | `-')' CloseParen
  |   | |   `-ParametersAndQualifiers
  |   | |     |-'(' OpenParen
  |   | |     |-ParameterDeclarationList Parameters
  |   | |     | `-SimpleDeclaration ListElement
  |   | |     |   `-'int'
  |   | |     `-')' CloseParen
  |   | |-',' ListDelimiter
  |   | `-SimpleDeclaration ListElement
  |   |   |-'long'
  |   |   `-SimpleDeclarator Declarator
  |   |     |-ParenDeclarator
  |   |     | |-'(' OpenParen
  |   |     | |-'*'
  |   |     | |-'*'
  |   |     | |-'c'
  |   |     | `-')' CloseParen
  |   |     `-ParametersAndQualifiers
  |   |       |-'(' OpenParen
  |   |       |-ParameterDeclarationList Parameters
  |   |       | `-SimpleDeclaration ListElement
  |   |       |   |-'long'
  |   |       |   `-'long'
  |   |       `-')' CloseParen
  |   `-')' CloseParen
  `-';'
)txt"));
}

} // namespace
