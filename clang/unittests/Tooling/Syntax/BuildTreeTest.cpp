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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'main'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   `-')' CloseParen
| `-CompoundStatement
|   |-'{' OpenParen
|   `-'}' CloseParen
`-SimpleDeclaration
  |-'void'
  |-SimpleDeclarator SimpleDeclaration_declarator
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | `-'a'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'a'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'b'
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
IfStatement CompoundStatement_statement
|-'if' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement IfStatement_thenStatement
  |-'{' OpenParen
  `-'}' CloseParen
  )txt",
       R"txt(
IfStatement CompoundStatement_statement
|-'if' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
|-CompoundStatement IfStatement_thenStatement
| |-'{' OpenParen
| `-'}' CloseParen
|-'else' IfStatement_elseKeyword
`-IfStatement IfStatement_elseStatement
  |-'if' IntroducerKeyword
  |-'('
  |-IntegerLiteralExpression
  | `-'0' LiteralToken
  |-')'
  `-CompoundStatement IfStatement_thenStatement
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
ForStatement CompoundStatement_statement
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
RangeBasedForStatement CompoundStatement_statement
|-'for' IntroducerKeyword
|-'('
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator SimpleDeclaration_declarator
| | `-'x'
| `-':'
|-IdExpression
| `-UnqualifiedId IdExpression_id
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
DeclarationStatement CompoundStatement_statement
|-SimpleDeclaration
| |-'int'
| `-SimpleDeclarator SimpleDeclaration_declarator
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
SwitchStatement CompoundStatement_statement
|-'switch' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement BodyStatement
  |-'{' OpenParen
  |-CaseStatement CompoundStatement_statement
  | |-'case' IntroducerKeyword
  | |-IntegerLiteralExpression CaseStatement_value
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
WhileStatement CompoundStatement_statement
|-'while' IntroducerKeyword
|-'('
|-IntegerLiteralExpression
| `-'1' LiteralToken
|-')'
`-CompoundStatement BodyStatement
  |-'{' OpenParen
  |-ContinueStatement CompoundStatement_statement
  | |-'continue' IntroducerKeyword
  | `-';'
  |-BreakStatement CompoundStatement_statement
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
UnknownStatement CompoundStatement_statement
|-'foo'
|-':'
`-ReturnStatement
  |-'return' IntroducerKeyword
  |-IntegerLiteralExpression ReturnStatement_value
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-ExpressionStatement CompoundStatement_statement
    | |-UnknownExpression ExpressionStatement_expression
    | | |-IdExpression
    | | | `-UnqualifiedId IdExpression_id
    | | |   `-'test'
    | | |-'('
    | | `-')'
    | `-';'
    |-IfStatement CompoundStatement_statement
    | |-'if' IntroducerKeyword
    | |-'('
    | |-IntegerLiteralExpression
    | | `-'1' LiteralToken
    | |-')'
    | |-ExpressionStatement IfStatement_thenStatement
    | | |-UnknownExpression ExpressionStatement_expression
    | | | |-IdExpression
    | | | | `-UnqualifiedId IdExpression_id
    | | | |   `-'test'
    | | | |-'('
    | | | `-')'
    | | `-';'
    | |-'else' IfStatement_elseKeyword
    | `-ExpressionStatement IfStatement_elseStatement
    |   |-UnknownExpression ExpressionStatement_expression
    |   | |-IdExpression
    |   | | `-UnqualifiedId IdExpression_id
    |   | |   `-'test'
    |   | |-'('
    |   | `-')'
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
IdExpression ExpressionStatement_expression
`-UnqualifiedId IdExpression_id
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| `-UnqualifiedId IdExpression_id
|   |-'operator'
|   `-'+'
|-'('
|-IdExpression
| `-UnqualifiedId IdExpression_id
|   `-'x'
|-','
|-IdExpression
| `-UnqualifiedId IdExpression_id
|   `-'x'
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'x'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     |-'operator'
|     `-'int'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| `-UnqualifiedId IdExpression_id
|   |-'operator'
|   |-'""'
|   `-'_w'
|-'('
|-CharacterLiteralExpression
| `-''1'' LiteralToken
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'x'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     |-'~'
|     `-'X'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'x'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     `-'~'
|-'decltype'
|-'('
|-'x'
|-')'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| `-UnqualifiedId IdExpression_id
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'('
`-')'
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
| |-'::' List_delimiter
| |-IdentifierNameSpecifier List_element
| | `-'n'
| `-'::' List_delimiter
|-'S'
`-SimpleDeclarator SimpleDeclaration_declarator
  `-UnknownExpression
    `-'s1'
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-IdentifierNameSpecifier List_element
| | `-'n'
| `-'::' List_delimiter
|-'S'
`-SimpleDeclarator SimpleDeclaration_declarator
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
| |-'::' List_delimiter
| |-SimpleTemplateNameSpecifier List_element
| | |-'template'
| | |-'ST'
| | |-'<'
| | |-'int'
| | `-'>'
| `-'::' List_delimiter
|-'S'
`-SimpleDeclarator SimpleDeclaration_declarator
  `-UnknownExpression
    `-'s1'
)txt",
       R"txt(
SimpleDeclaration
|-NestedNameSpecifier
| |-'::' List_delimiter
| |-SimpleTemplateNameSpecifier List_element
| | |-'ST'
| | |-'<'
| | |-'int'
| | `-'>'
| `-'::' List_delimiter
|-'S'
`-SimpleDeclarator SimpleDeclaration_declarator
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-DecltypeNameSpecifier List_element
| | | |-'decltype'
| | | |-'('
| | | |-IdExpression
| | | | `-UnqualifiedId IdExpression_id
| | | |   `-'s'
| | | `-')'
| | `-'::' List_delimiter
| `-UnqualifiedId IdExpression_id
|   `-'f'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-IdentifierNameSpecifier List_element
| | | `-'S'
| | `-'::' List_delimiter
| `-UnqualifiedId IdExpression_id
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'('
`-')'
)txt",
       R"txt(
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-IdentifierNameSpecifier List_element
| | | `-'S'
| | `-'::' List_delimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId IdExpression_id
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-'::' List_delimiter
| | |-IdentifierNameSpecifier List_element
| | | `-'n'
| | |-'::' List_delimiter
| | |-SimpleTemplateNameSpecifier List_element
| | | |-'template'
| | | |-'ST'
| | | |-'<'
| | | |-'int'
| | | `-'>'
| | `-'::' List_delimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId IdExpression_id
|   |-'f'
|   |-'<'
|   |-'int'
|   `-'>'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-IdentifierNameSpecifier List_element
| | | `-'T'
| | |-'::' List_delimiter
| | |-SimpleTemplateNameSpecifier List_element
| | | |-'template'
| | | |-'U'
| | | |-'<'
| | | |-'int'
| | | `-'>'
| | `-'::' List_delimiter
| `-UnqualifiedId IdExpression_id
|   `-'f'
|-'('
`-')'
)txt",
       R"txt(
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-IdentifierNameSpecifier List_element
| | | `-'T'
| | |-'::' List_delimiter
| | |-IdentifierNameSpecifier List_element
| | | `-'U'
| | `-'::' List_delimiter
| `-UnqualifiedId IdExpression_id
|   `-'f'
|-'('
`-')'
)txt",
       R"txt(
UnknownExpression ExpressionStatement_expression
|-IdExpression
| |-NestedNameSpecifier IdExpression_qualifier
| | |-IdentifierNameSpecifier List_element
| | | `-'T'
| | `-'::' List_delimiter
| |-'template' TemplateKeyword
| `-UnqualifiedId IdExpression_id
|   |-'f'
|   |-'<'
|   |-IntegerLiteralExpression
|   | `-'0' LiteralToken
|   `-'>'
|-'('
`-')'
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
ThisExpression ReturnStatement_value
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
MemberExpression ExpressionStatement_expression
|-ThisExpression MemberExpression_object
| `-'this' IntroducerKeyword
|-'->' MemberExpression_accessToken
`-IdExpression MemberExpression_member
  `-UnqualifiedId IdExpression_id
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
IdExpression ExpressionStatement_expression
`-UnqualifiedId IdExpression_id
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
ParenExpression ExpressionStatement_expression
|-'(' OpenParen
|-IntegerLiteralExpression ParenExpression_subExpression
| `-'1' LiteralToken
`-')' CloseParen
)txt",
       R"txt(
ParenExpression ExpressionStatement_expression
|-'(' OpenParen
|-ParenExpression ParenExpression_subExpression
| |-'(' OpenParen
| |-IntegerLiteralExpression ParenExpression_subExpression
| | `-'1' LiteralToken
| `-')' CloseParen
`-')' CloseParen
)txt",
       R"txt(
ParenExpression ExpressionStatement_expression
|-'(' OpenParen
|-BinaryOperatorExpression ParenExpression_subExpression
| |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| | `-'1' LiteralToken
| |-'+' OperatorExpression_operatorToken
| `-ParenExpression BinaryOperatorExpression_rightHandSide
|   |-'(' OpenParen
|   |-IntegerLiteralExpression ParenExpression_subExpression
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
CharUserDefinedLiteralExpression ExpressionStatement_expression
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
StringUserDefinedLiteralExpression ExpressionStatement_expression
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
IntegerUserDefinedLiteralExpression ExpressionStatement_expression
`-'12_i' LiteralToken
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression ExpressionStatement_expression
`-'12_r' LiteralToken
)txt",
       R"txt(
IntegerUserDefinedLiteralExpression ExpressionStatement_expression
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
FloatUserDefinedLiteralExpression ExpressionStatement_expression
`-'1.2_f' LiteralToken
)txt",
       R"txt(
FloatUserDefinedLiteralExpression ExpressionStatement_expression
`-'1.2_r' LiteralToken
)txt",
       R"txt(
FloatUserDefinedLiteralExpression ExpressionStatement_expression
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
IntegerLiteralExpression ExpressionStatement_expression
`-'12ll' LiteralToken
)txt",
       R"txt(
IntegerLiteralExpression ExpressionStatement_expression
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
IntegerLiteralExpression ExpressionStatement_expression
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
IntegerLiteralExpression ExpressionStatement_expression
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
CharacterLiteralExpression ExpressionStatement_expression
`-''a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-''\n'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-''\x20'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-''\0'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-'L'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
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
CharacterLiteralExpression ExpressionStatement_expression
`-'u'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-'u'構'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
`-'U'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
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
CharacterLiteralExpression ExpressionStatement_expression
`-'u8'a'' LiteralToken
)txt",
       R"txt(
CharacterLiteralExpression ExpressionStatement_expression
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
FloatingLiteralExpression ExpressionStatement_expression
`-'1e-2' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
`-'2.' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
`-'.2' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
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
FloatingLiteralExpression ExpressionStatement_expression
`-'0xfp1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
`-'0xf.p1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
`-'0x.fp1' LiteralToken
)txt",
       R"txt(
FloatingLiteralExpression ExpressionStatement_expression
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
StringLiteralExpression ExpressionStatement_expression
`-'"a\n\0\x20"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression ExpressionStatement_expression
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
StringLiteralExpression ExpressionStatement_expression
`-'u8"a\x1f\x05"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression ExpressionStatement_expression
`-'u"C++抽象構文木"' LiteralToken
)txt",
       R"txt(
StringLiteralExpression ExpressionStatement_expression
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
      "  |-SimpleDeclarator SimpleDeclaration_declarator\n"
      "  | |-'test'\n"
      "  | `-ParametersAndQualifiers\n"
      "  |   |-'(' OpenParen\n"
      "  |   `-')' CloseParen\n"
      "  `-CompoundStatement\n"
      "    |-'{' OpenParen\n"
      "    |-ExpressionStatement CompoundStatement_statement\n"
      "    | |-StringLiteralExpression ExpressionStatement_expression\n"
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
BoolLiteralExpression ExpressionStatement_expression
`-'true' LiteralToken
)txt",
       R"txt(
BoolLiteralExpression ExpressionStatement_expression
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
CxxNullPtrExpression ExpressionStatement_expression
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
PostfixUnaryOperatorExpression ExpressionStatement_expression
|-IdExpression UnaryOperatorExpression_operand
| `-UnqualifiedId IdExpression_id
|   `-'a'
`-'++' OperatorExpression_operatorToken
)txt",
       R"txt(
PostfixUnaryOperatorExpression ExpressionStatement_expression
|-IdExpression UnaryOperatorExpression_operand
| `-UnqualifiedId IdExpression_id
|   `-'a'
`-'--' OperatorExpression_operatorToken
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
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'--' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'++' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'~' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'-' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'+' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'&' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'*' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'ap'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'!' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'__real' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'__imag' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
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
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'compl' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
    `-'a'
)txt",
       R"txt(
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'not' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'1' LiteralToken
|-'-' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'1' LiteralToken
|-'==' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'a'
|-'=' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'1' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'a'
|-'<<=' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'1' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'1' LiteralToken
|-'||' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'0' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'1' LiteralToken
|-'&' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'a'
|-'!=' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
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
BinaryOperatorExpression ExpressionStatement_expression
|-BoolLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'true' LiteralToken
|-'||' OperatorExpression_operatorToken
`-BoolLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'false' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-BoolLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'true' LiteralToken
|-'or' OperatorExpression_operatorToken
`-BoolLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'false' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| `-'1' LiteralToken
|-'bitand' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'2' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'a'
|-'xor_eq' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
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
BinaryOperatorExpression ExpressionStatement_expression
|-ParenExpression BinaryOperatorExpression_leftHandSide
| |-'(' OpenParen
| |-BinaryOperatorExpression ParenExpression_subExpression
| | |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| | | `-'1' LiteralToken
| | |-'+' OperatorExpression_operatorToken
| | `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
| |   `-'2' LiteralToken
| `-')' CloseParen
|-'*' OperatorExpression_operatorToken
`-ParenExpression BinaryOperatorExpression_rightHandSide
  |-'(' OpenParen
  |-BinaryOperatorExpression ParenExpression_subExpression
  | |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
  | | `-'4' LiteralToken
  | |-'/' OperatorExpression_operatorToken
  | `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
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
BinaryOperatorExpression ExpressionStatement_expression
|-BinaryOperatorExpression BinaryOperatorExpression_leftHandSide
| |-IdExpression BinaryOperatorExpression_leftHandSide
| | `-UnqualifiedId IdExpression_id
| |   `-'a'
| |-'+' OperatorExpression_operatorToken
| `-IdExpression BinaryOperatorExpression_rightHandSide
|   `-UnqualifiedId IdExpression_id
|     `-'b'
|-'+' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'42' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'a'
|-'=' OperatorExpression_operatorToken
`-BinaryOperatorExpression BinaryOperatorExpression_rightHandSide
  |-IdExpression BinaryOperatorExpression_leftHandSide
  | `-UnqualifiedId IdExpression_id
  |   `-'b'
  |-'=' OperatorExpression_operatorToken
  `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
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
BinaryOperatorExpression ExpressionStatement_expression
|-BinaryOperatorExpression BinaryOperatorExpression_leftHandSide
| |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| | `-'1' LiteralToken
| |-'+' OperatorExpression_operatorToken
| `-BinaryOperatorExpression BinaryOperatorExpression_rightHandSide
|   |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
|   | `-'2' LiteralToken
|   |-'*' OperatorExpression_operatorToken
|   `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
|     `-'3' LiteralToken
|-'+' OperatorExpression_operatorToken
`-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
  `-'4' LiteralToken
)txt",
       R"txt(
BinaryOperatorExpression ExpressionStatement_expression
|-BinaryOperatorExpression BinaryOperatorExpression_leftHandSide
| |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
| | `-'1' LiteralToken
| |-'%' OperatorExpression_operatorToken
| `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
|   `-'2' LiteralToken
|-'+' OperatorExpression_operatorToken
`-BinaryOperatorExpression BinaryOperatorExpression_rightHandSide
  |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide
  | `-'3' LiteralToken
  |-'*' OperatorExpression_operatorToken
  `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide
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
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'x'
|-'=' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-UnknownExpression BinaryOperatorExpression_leftHandSide
| `-IdExpression
|   `-UnqualifiedId IdExpression_id
|     `-'x'
|-'+' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'x'
|-'<' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'x'
|-'<<' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'x'
|-',' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
BinaryOperatorExpression ExpressionStatement_expression
|-IdExpression BinaryOperatorExpression_leftHandSide
| `-UnqualifiedId IdExpression_id
|   `-'xp'
|-'->*' OperatorExpression_operatorToken
`-IdExpression BinaryOperatorExpression_rightHandSide
  `-UnqualifiedId IdExpression_id
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
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'!' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
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
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'&' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
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
PrefixUnaryOperatorExpression ExpressionStatement_expression
|-'++' OperatorExpression_operatorToken
`-IdExpression UnaryOperatorExpression_operand
  `-UnqualifiedId IdExpression_id
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
PostfixUnaryOperatorExpression ExpressionStatement_expression
|-IdExpression UnaryOperatorExpression_operand
| `-UnqualifiedId IdExpression_id
|   `-'x'
`-'++' OperatorExpression_operatorToken
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
MemberExpression ExpressionStatement_expression
|-IdExpression MemberExpression_object
| `-UnqualifiedId IdExpression_id
|   `-'s'
|-'.' MemberExpression_accessToken
`-IdExpression MemberExpression_member
  `-UnqualifiedId IdExpression_id
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
MemberExpression ExpressionStatement_expression
|-IdExpression MemberExpression_object
| `-UnqualifiedId IdExpression_id
|   `-'s'
|-'.' MemberExpression_accessToken
`-IdExpression MemberExpression_member
  `-UnqualifiedId IdExpression_id
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
MemberExpression ExpressionStatement_expression
|-IdExpression MemberExpression_object
| `-UnqualifiedId IdExpression_id
|   `-'sp'
|-'->' MemberExpression_accessToken
`-IdExpression MemberExpression_member
  `-UnqualifiedId IdExpression_id
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
MemberExpression ExpressionStatement_expression
|-MemberExpression MemberExpression_object
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'s'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     `-'next'
|-'->' MemberExpression_accessToken
`-IdExpression MemberExpression_member
  `-UnqualifiedId IdExpression_id
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'s'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     |-'operator'
|     `-'!'
|-'('
`-')'
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
|-ExpressionStatement CompoundStatement_statement
| `-MemberExpression ExpressionStatement_expression
|   |-IdExpression MemberExpression_object
|   | `-UnqualifiedId IdExpression_id
|   |   `-'s'
|   |-'.' MemberExpression_accessToken
|   `-IdExpression MemberExpression_member
|     `-UnqualifiedId IdExpression_id
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'sp'
| |-'->' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'s'
| |-'.' MemberExpression_accessToken
| |-'template'
| `-IdExpression MemberExpression_member
|   `-UnqualifiedId IdExpression_id
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'s'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   |-NestedNameSpecifier IdExpression_qualifier
|   | |-IdentifierNameSpecifier List_element
|   | | `-'Base'
|   | `-'::' List_delimiter
|   `-UnqualifiedId IdExpression_id
|     `-'f'
|-'('
`-')'
      )txt",
       R"txt(
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-IdExpression MemberExpression_object
| | `-UnqualifiedId IdExpression_id
| |   `-'s'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   |-NestedNameSpecifier IdExpression_qualifier
|   | |-'::' List_delimiter
|   | |-IdentifierNameSpecifier List_element
|   | | `-'S'
|   | `-'::' List_delimiter
|   `-UnqualifiedId IdExpression_id
|     |-'~'
|     `-'S'
|-'('
`-')'
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
UnknownExpression ExpressionStatement_expression
|-MemberExpression
| |-UnknownExpression MemberExpression_object
| | |-MemberExpression
| | | |-IdExpression MemberExpression_object
| | | | `-UnqualifiedId IdExpression_id
| | | |   `-'sp'
| | | |-'->' MemberExpression_accessToken
| | | `-IdExpression MemberExpression_member
| | |   `-UnqualifiedId IdExpression_id
| | |     `-'getU'
| | |-'('
| | `-')'
| |-'.' MemberExpression_accessToken
| `-IdExpression MemberExpression_member
|   |-NestedNameSpecifier IdExpression_qualifier
|   | |-SimpleTemplateNameSpecifier List_element
|   | | |-'template'
|   | | |-'U'
|   | | |-'<'
|   | | |-'int'
|   | | `-'>'
|   | `-'::' List_delimiter
|   |-'template' TemplateKeyword
|   `-UnqualifiedId IdExpression_id
|     |-'f'
|     |-'<'
|     |-'int'
|     `-'>'
|-'('
`-')'
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'*'
| | `-'a'
| |-','
| |-SimpleDeclarator SimpleDeclaration_declarator
| | `-'b'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'*'
  | `-'c'
  |-','
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'*'
  | `-'a'
  |-','
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-DeclarationStatement CompoundStatement_statement
    | |-SimpleDeclaration
    | | |-'int'
    | | |-SimpleDeclarator SimpleDeclaration_declarator
    | | | |-'*'
    | | | `-'a'
    | | |-','
    | | `-SimpleDeclarator SimpleDeclaration_declarator
    | |   `-'b'
    | `-';'
    |-DeclarationStatement CompoundStatement_statement
    | |-SimpleDeclaration
    | | |-'typedef'
    | | |-'int'
    | | |-SimpleDeclarator SimpleDeclaration_declarator
    | | | |-'*'
    | | | `-'ta'
    | | |-','
    | | `-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
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
namespace a::b {}
)cpp",
      R"txt(
TranslationUnit Detached
|-NamespaceDefinition
| |-'namespace'
| |-'a'
| |-'{'
| |-NamespaceDefinition
| | |-'namespace'
| | |-'b'
| | |-'{'
| | `-'}'
| `-'}'
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
| `-'::' List_delimiter
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
| |-IdentifierNameSpecifier List_element
| | `-'ns'
| `-'::' List_delimiter
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
| |-IdentifierNameSpecifier List_element
| | `-'T'
| `-'::' List_delimiter
|-'foo'
`-';'
)txt",
       R"txt(
UsingDeclaration
|-'using'
|-'typename'
|-NestedNameSpecifier
| |-IdentifierNameSpecifier List_element
| | `-'T'
| `-'::' List_delimiter
|-'bar'
`-';'
)txt"}));
}

TEST_P(SyntaxTreeTest, UsingTypeAlias) {
  if (!GetParam().isCXX()) {
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
| |-'*'
| `-'y2'
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'struct'
|-'{'
|-'}'
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'operator'
  | |-'""'
  | |-'_c'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | `-'char'
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
`-TemplateDeclaration TemplateDeclaration_declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-SimpleDeclaration
  | `-'char'
  |-'...'
  |-'>'
  `-SimpleDeclaration
    |-'unsigned'
    |-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
| |-'&'
| |-'operator'
| |-'='
| `-ParametersAndQualifiers
|   |-'(' OpenParen
|   |-SimpleDeclaration ParametersAndQualifiers_parameter
|   | |-'const'
|   | |-'X'
|   | `-SimpleDeclarator SimpleDeclaration_declarator
|   |   `-'&'
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'operator'
  | |-'+'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | `-'X'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'const'
  |   | |-'X'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'&'
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
`-TemplateDeclaration TemplateDeclaration_declaration
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
`-TemplateDeclaration TemplateDeclaration_declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'typename'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'T'
    |-SimpleDeclarator SimpleDeclaration_declarator
    | |-'f'
    | `-ParametersAndQualifiers
    |   |-'(' OpenParen
    |   `-')' CloseParen
    `-';'
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
TranslationUnit Detached
`-TemplateDeclaration TemplateDeclaration_declaration
  |-'template' IntroducerKeyword
  |-'<'
  |-UnknownDeclaration
  | |-'class'
  | `-'T'
  |-'>'
  `-SimpleDeclaration
    |-'T'
    |-SimpleDeclarator SimpleDeclaration_declarator
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
TemplateDeclaration TemplateDeclaration_declaration
|-'template' IntroducerKeyword
|-'<'
|-UnknownDeclaration
| |-'typename'
| `-'U'
|-'>'
`-SimpleDeclaration
  |-'static'
  |-'U'
  |-SimpleDeclarator SimpleDeclaration_declarator
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
`-TemplateDeclaration TemplateDeclaration_declaration
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
    |-TemplateDeclaration TemplateDeclaration_declaration
    | |-'template' IntroducerKeyword
    | |-'<'
    | |-UnknownDeclaration
    | | |-'class'
    | | `-'U'
    | |-'>'
    | `-SimpleDeclaration
    |   |-'U'
    |   |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-TemplateDeclaration TemplateDeclaration_declaration
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
  |   |-TemplateDeclaration TemplateDeclaration_declaration
  |   | |-'template' IntroducerKeyword
  |   | |-'<'
  |   | |-UnknownDeclaration
  |   | | |-'typename'
  |   | | `-'U'
  |   | |-'>'
  |   | `-SimpleDeclaration
  |   |   |-'static'
  |   |   |-'U'
  |   |   |-SimpleDeclarator SimpleDeclaration_declarator
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
TemplateDeclaration TemplateDeclaration_declaration
|-'template' IntroducerKeyword
|-'<'
|-UnknownDeclaration
| |-'class'
| `-'T'
|-'>'
`-SimpleDeclaration
  |-'struct'
  |-NestedNameSpecifier
  | |-SimpleTemplateNameSpecifier List_element
  | | |-'X'
  | | |-'<'
  | | |-'T'
  | | `-'>'
  | `-'::' List_delimiter
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
`-SimpleDeclaration ExplicitTemplateInstantiation_declaration
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
`-SimpleDeclaration ExplicitTemplateInstantiation_declaration
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
TemplateDeclaration TemplateDeclaration_declaration
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
TemplateDeclaration TemplateDeclaration_declaration
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
static_assert(true);
)cpp",
      R"txt(
TranslationUnit Detached
|-StaticAssertDeclaration
| |-'static_assert'
| |-'('
| |-BoolLiteralExpression StaticAssertDeclaration_condition
| | `-'true' LiteralToken
| |-','
| |-StringLiteralExpression StaticAssertDeclaration_message
| | `-'"message"' LiteralToken
| |-')'
| `-';'
`-StaticAssertDeclaration
  |-'static_assert'
  |-'('
  |-BoolLiteralExpression StaticAssertDeclaration_condition
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
|   |-SimpleDeclarator SimpleDeclaration_declarator
|   | `-'a'
|   `-';'
`-LinkageSpecificationDeclaration
  |-'extern'
  |-'"C"'
  |-'{'
  |-SimpleDeclaration
  | |-'int'
  | |-SimpleDeclarator SimpleDeclaration_declarator
  | | `-'b'
  | `-';'
  |-SimpleDeclaration
  | |-'int'
  | |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-IfStatement CompoundStatement_statement
    | |-'if' IntroducerKeyword unmodifiable
    | |-'(' unmodifiable
    | |-BinaryOperatorExpression unmodifiable
    | | |-IntegerLiteralExpression BinaryOperatorExpression_leftHandSide unmodifiable
    | | | `-'1' LiteralToken unmodifiable
    | | |-'+' OperatorExpression_operatorToken unmodifiable
    | | `-IntegerLiteralExpression BinaryOperatorExpression_rightHandSide unmodifiable
    | |   `-'1' LiteralToken unmodifiable
    | |-')' unmodifiable
    | |-CompoundStatement IfStatement_thenStatement unmodifiable
    | | |-'{' OpenParen unmodifiable
    | | `-'}' CloseParen unmodifiable
    | |-'else' IfStatement_elseKeyword
    | `-CompoundStatement IfStatement_elseStatement
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'test'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   `-')' CloseParen
  `-CompoundStatement
    |-'{' OpenParen
    |-CompoundStatement CompoundStatement_statement
    | |-'{' OpenParen
    | |-ExpressionStatement CompoundStatement_statement
    | | |-IntegerLiteralExpression ExpressionStatement_expression
    | | | `-'1' LiteralToken
    | | `-';'
    | `-'}' CloseParen
    |-CompoundStatement CompoundStatement_statement
    | |-'{' OpenParen
    | |-ExpressionStatement CompoundStatement_statement
    | | |-IntegerLiteralExpression ExpressionStatement_expression
    | | | `-'2' LiteralToken
    | | `-';'
    | `-'}' CloseParen
    `-'}' CloseParen
)txt"));
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'a'
  | `-ArraySubscript
  |   |-'[' OpenParen
  |   |-IntegerLiteralExpression ArraySubscript_sizeExpression
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'b'
  | |-ArraySubscript
  | | |-'[' OpenParen
  | | |-IntegerLiteralExpression ArraySubscript_sizeExpression
  | | | `-'1' LiteralToken
  | | `-']' CloseParen
  | |-ArraySubscript
  | | |-'[' OpenParen
  | | |-IntegerLiteralExpression ArraySubscript_sizeExpression
  | | | `-'2' LiteralToken
  | | `-']' CloseParen
  | `-ArraySubscript
  |   |-'[' OpenParen
  |   |-IntegerLiteralExpression ArraySubscript_sizeExpression
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
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'f'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-'xs'
  |   |   `-ArraySubscript
  |   |     |-'[' OpenParen
  |   |     |-'static'
  |   |     |-IntegerLiteralExpression ArraySubscript_sizeExpression
  |   |     | `-'10' LiteralToken
  |   |     `-']' CloseParen
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
  |-SimpleDeclarator SimpleDeclaration_declarator
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'func1'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-SimpleDeclaration ParametersAndQualifiers_parameter
| |   | |-'int'
| |   | `-SimpleDeclarator SimpleDeclaration_declarator
| |   |   `-'a'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'func2'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-SimpleDeclaration ParametersAndQualifiers_parameter
| |   | |-'int'
| |   | `-SimpleDeclarator SimpleDeclaration_declarator
| |   |   |-'*'
| |   |   `-'ap'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'func3'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'a'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'float'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'b'
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'func1'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-SimpleDeclaration ParametersAndQualifiers_parameter
| |   | `-'int'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'func2'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-SimpleDeclaration ParametersAndQualifiers_parameter
| |   | |-'int'
| |   | `-SimpleDeclarator SimpleDeclaration_declarator
| |   |   `-'*'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'func3'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | `-'int'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | `-'float'
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'const'
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'a'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'volatile'
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'b'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'const'
  |   | |-'volatile'
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'c'
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-'&'
  |   |   `-'a'
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'func'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'int'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-'&&'
  |   |   `-'a'
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
  | |-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
  if (!GetParam().isCXX()) {
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-')' CloseParen
  |   `-TrailingReturnType ParametersAndQualifiers_trailingReturn
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'a'
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-')' CloseParen
| |   `-'noexcept'
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | `-ParenDeclarator
| |   |-'(' OpenParen
| |   |-'a'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'*'
| | `-ParenDeclarator
| |   |-'(' OpenParen
| |   |-'b'
| |   `-')' CloseParen
| `-';'
|-SimpleDeclaration
| |-'int'
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-ParenDeclarator
| | | |-'(' OpenParen
| | | |-'*'
| | | |-'c'
| | | `-')' CloseParen
| | `-ParametersAndQualifiers
| |   |-'(' OpenParen
| |   |-SimpleDeclaration ParametersAndQualifiers_parameter
| |   | `-'int'
| |   `-')' CloseParen
| `-';'
`-SimpleDeclaration
  |-'int'
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'*'
  | |-ParenDeclarator
  | | |-'(' OpenParen
  | | |-'d'
  | | `-')' CloseParen
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | `-'int'
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
| |-SimpleDeclarator SimpleDeclaration_declarator
| | |-'west'
| | |-'='
| | `-PrefixUnaryOperatorExpression
| |   |-'-' OperatorExpression_operatorToken
| |   `-IntegerLiteralExpression UnaryOperatorExpression_operand
| |     `-'1' LiteralToken
| `-';'
`-SimpleDeclaration
  |-'int'
  |-'const'
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'foo'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-')' CloseParen
  |   `-TrailingReturnType ParametersAndQualifiers_trailingReturn
  |     |-'->' ArrowToken
  |     |-'auto'
  |     `-SimpleDeclarator TrailingReturnType_declarator
  |       |-ParenDeclarator
  |       | |-'(' OpenParen
  |       | |-'*'
  |       | `-')' CloseParen
  |       `-ParametersAndQualifiers
  |         |-'(' OpenParen
  |         |-SimpleDeclaration ParametersAndQualifiers_parameter
  |         | `-'int'
  |         |-')' CloseParen
  |         `-TrailingReturnType ParametersAndQualifiers_trailingReturn
  |           |-'->' ArrowToken
  |           |-'double'
  |           `-SimpleDeclarator TrailingReturnType_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|-SimpleDeclarator SimpleDeclaration_declarator
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
|   |-SimpleDeclaration ParametersAndQualifiers_parameter
|   | |-'const'
|   | |-'int'
|   | `-SimpleDeclarator SimpleDeclaration_declarator
|   |   `-'*'
|   `-')' CloseParen
`-';'
)txt",
       R"txt(
SimpleDeclaration
|-'void'
|-SimpleDeclarator SimpleDeclaration_declarator
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
|   |-SimpleDeclaration ParametersAndQualifiers_parameter
|   | |-'const'
|   | |-'int'
|   | `-SimpleDeclarator SimpleDeclaration_declarator
|   |   `-'*'
|   |-','
|   |-SimpleDeclaration ParametersAndQualifiers_parameter
|   | `-'char'
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'x'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'char'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'a'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'short'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-ParenDeclarator
  |   |   | |-'(' OpenParen
  |   |   | |-'*'
  |   |   | |-'b'
  |   |   | `-')' CloseParen
  |   |   `-ParametersAndQualifiers
  |   |     |-'(' OpenParen
  |   |     |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   |     | `-'int'
  |   |     `-')' CloseParen
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
  |-SimpleDeclarator SimpleDeclaration_declarator
  | |-'x'
  | `-ParametersAndQualifiers
  |   |-'(' OpenParen
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'char'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   `-'a'
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'short'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-ParenDeclarator
  |   |   | |-'(' OpenParen
  |   |   | |-'*'
  |   |   | |-'b'
  |   |   | `-')' CloseParen
  |   |   `-ParametersAndQualifiers
  |   |     |-'(' OpenParen
  |   |     |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   |     | `-'int'
  |   |     `-')' CloseParen
  |   |-','
  |   |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   | |-'long'
  |   | `-SimpleDeclarator SimpleDeclaration_declarator
  |   |   |-ParenDeclarator
  |   |   | |-'(' OpenParen
  |   |   | |-'*'
  |   |   | |-'*'
  |   |   | |-'c'
  |   |   | `-')' CloseParen
  |   |   `-ParametersAndQualifiers
  |   |     |-'(' OpenParen
  |   |     |-SimpleDeclaration ParametersAndQualifiers_parameter
  |   |     | |-'long'
  |   |     | `-'long'
  |   |     `-')' CloseParen
  |   `-')' CloseParen
  `-';'
)txt"));
}

} // namespace
