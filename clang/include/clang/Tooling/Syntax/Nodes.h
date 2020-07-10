//===- Nodes.h - syntax nodes for C/C++ grammar constructs ----*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Syntax tree nodes for C, C++ and Objective-C grammar constructs.
//
// Nodes provide access to their syntactic components, e.g. IfStatement provides
// a way to get its condition, then and else branches, tokens for 'if' and
// 'else' keywords.
// When using the accessors, please assume they can return null. This happens
// because:
//   - the corresponding subnode is optional in the C++ grammar, e.g. an else
//     branch of an if statement,
//   - syntactic errors occurred while parsing the corresponding subnode.
// One notable exception is "introducer" keywords, e.g. the accessor for the
// 'if' keyword of an if statement will never return null.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_SYNTAX_NODES_H
#define LLVM_CLANG_TOOLING_SYNTAX_NODES_H

#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
namespace clang {
namespace syntax {

/// A kind of a syntax node, used for implementing casts. The ordering and
/// blocks of enumerator constants must correspond to the inheritance hierarchy
/// of syntax::Node.
enum class NodeKind : uint16_t {
  Leaf,
  TranslationUnit,

  // Expressions.
  UnknownExpression,
  PrefixUnaryOperatorExpression,
  PostfixUnaryOperatorExpression,
  BinaryOperatorExpression,
  ParenExpression,
  IntegerLiteralExpression,
  CharacterLiteralExpression,
  FloatingLiteralExpression,
  StringLiteralExpression,
  BoolLiteralExpression,
  CxxNullPtrExpression,
  IntegerUserDefinedLiteralExpression,
  FloatUserDefinedLiteralExpression,
  CharUserDefinedLiteralExpression,
  StringUserDefinedLiteralExpression,
  IdExpression,

  // Statements.
  UnknownStatement,
  DeclarationStatement,
  EmptyStatement,
  SwitchStatement,
  CaseStatement,
  DefaultStatement,
  IfStatement,
  ForStatement,
  WhileStatement,
  ContinueStatement,
  BreakStatement,
  ReturnStatement,
  RangeBasedForStatement,
  ExpressionStatement,
  CompoundStatement,

  // Declarations.
  UnknownDeclaration,
  EmptyDeclaration,
  StaticAssertDeclaration,
  LinkageSpecificationDeclaration,
  SimpleDeclaration,
  TemplateDeclaration,
  ExplicitTemplateInstantiation,
  NamespaceDefinition,
  NamespaceAliasDefinition,
  UsingNamespaceDirective,
  UsingDeclaration,
  TypeAliasDeclaration,

  // Declarators.
  SimpleDeclarator,
  ParenDeclarator,

  ArraySubscript,
  TrailingReturnType,
  ParametersAndQualifiers,
  MemberPointer,
  NestedNameSpecifier,
  NameSpecifier,
  UnqualifiedId
};
/// For debugging purposes.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeKind K);

/// A relation between a parent and child node, e.g. 'left-hand-side of
/// a binary expression'. Used for implementing accessors.
///
/// Some roles describe parent/child relations that occur multiple times in
/// language grammar. We define only one role to describe all instances of such
/// recurring relations. For example, grammar for both "if" and "while"
/// statements requires an opening paren and a closing paren. The opening
/// paren token is assigned the OpenParen role regardless of whether it appears
/// as a child of IfStatement or WhileStatement node. More generally, when
/// grammar requires a certain fixed token (like a specific keyword, or an
/// opening paren), we define a role for this token and use it across all
/// grammar rules with the same requirement. Names of such reusable roles end
/// with a ~Token or a ~Keyword suffix.
///
/// Some roles are assigned only to child nodes of one specific parent syntax
/// node type. Names of such roles start with the name of the parent syntax tree
/// node type. For example, a syntax node with a role
/// BinaryOperatorExpression_leftHandSide can only appear as a child of a
/// BinaryOperatorExpression node.
enum class NodeRole : uint8_t {
  // Roles common to multiple node kinds.
  /// A node without a parent
  Detached,
  /// Children of an unknown semantic nature, e.g. skipped tokens, comments.
  Unknown,
  /// An opening parenthesis in argument lists and blocks, e.g. '{', '(', etc.
  OpenParen,
  /// A closing parenthesis in argument lists and blocks, e.g. '}', ')', etc.
  CloseParen,
  /// A keywords that introduces some grammar construct, e.g. 'if', 'try', etc.
  IntroducerKeyword,
  /// A token that represents a literal, e.g. 'nullptr', '1', 'true', etc.
  LiteralToken,
  /// Tokens or Keywords
  ArrowToken,
  ExternKeyword,
  /// An inner statement for those that have only a single child of kind
  /// statement, e.g. loop body for while, for, etc; inner statement for case,
  /// default, etc.
  BodyStatement,

  // Roles specific to particular node kinds.
  OperatorExpression_operatorToken,
  UnaryOperatorExpression_operand,
  BinaryOperatorExpression_leftHandSide,
  BinaryOperatorExpression_rightHandSide,
  CaseStatement_value,
  IfStatement_thenStatement,
  IfStatement_elseKeyword,
  IfStatement_elseStatement,
  ReturnStatement_value,
  ExpressionStatement_expression,
  CompoundStatement_statement,
  StaticAssertDeclaration_condition,
  StaticAssertDeclaration_message,
  SimpleDeclaration_declarator,
  TemplateDeclaration_declaration,
  ExplicitTemplateInstantiation_declaration,
  ArraySubscript_sizeExpression,
  TrailingReturnType_declarator,
  ParametersAndQualifiers_parameter,
  ParametersAndQualifiers_trailingReturn,
  IdExpression_id,
  IdExpression_qualifier,
  NestedNameSpecifier_specifier,
  ParenExpression_subExpression
};
/// For debugging purposes.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeRole R);

class SimpleDeclarator;

/// A root node for a translation unit. Parent is always null.
class TranslationUnit final : public Tree {
public:
  TranslationUnit() : Tree(NodeKind::TranslationUnit) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TranslationUnit;
  }
};

/// A base class for all expressions. Note that expressions are not statements,
/// even though they are in clang.
class Expression : public Tree {
public:
  Expression(NodeKind K) : Tree(K) {}
  static bool classof(const Node *N) {
    return NodeKind::UnknownExpression <= N->kind() &&
           N->kind() <= NodeKind::UnknownExpression;
  }
};

/// A sequence of these specifiers make a `nested-name-specifier`.
/// e.g. the `std::` or `vector<int>::` in `std::vector<int>::size`.
class NameSpecifier final : public Tree {
public:
  NameSpecifier() : Tree(NodeKind::NameSpecifier) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::NameSpecifier;
  }
};

/// Models a `nested-name-specifier`. C++ [expr.prim.id.qual]
/// e.g. the `std::vector<int>::` in `std::vector<int>::size`.
class NestedNameSpecifier final : public Tree {
public:
  NestedNameSpecifier() : Tree(NodeKind::NestedNameSpecifier) {}
  static bool classof(const Node *N) {
    return N->kind() <= NodeKind::NestedNameSpecifier;
  }
  std::vector<syntax::NameSpecifier *> specifiers();
};

/// Models an `unqualified-id`. C++ [expr.prim.id.unqual]
/// e.g. the `size` in `std::vector<int>::size`.
class UnqualifiedId final : public Tree {
public:
  UnqualifiedId() : Tree(NodeKind::UnqualifiedId) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UnqualifiedId;
  }
};

/// Models an `id-expression`, e.g. `std::vector<int>::size`.
/// C++ [expr.prim.id]
/// id-expression:
///   unqualified-id
///   qualified-id
/// qualified-id:
///   nested-name-specifier template_opt unqualified-id
class IdExpression final : public Expression {
public:
  IdExpression() : Expression(NodeKind::IdExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::IdExpression;
  }
  syntax::NestedNameSpecifier *qualifier();
  // TODO after expose `id-expression` from `DependentScopeDeclRefExpr`:
  // Add accessor for `template_opt`.
  syntax::UnqualifiedId *unqualifiedId();
};

/// An expression of an unknown kind, i.e. one not currently handled by the
/// syntax tree.
class UnknownExpression final : public Expression {
public:
  UnknownExpression() : Expression(NodeKind::UnknownExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UnknownExpression;
  }
};

/// Models a parenthesized expression `(E)`. C++ [expr.prim.paren]
/// e.g. `(3 + 2)` in `a = 1 + (3 + 2);`
class ParenExpression final : public Expression {
public:
  ParenExpression() : Expression(NodeKind::ParenExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ParenExpression;
  }
  syntax::Leaf *openParen();
  syntax::Expression *subExpression();
  syntax::Leaf *closeParen();
};

/// Expression for integer literals. C++ [lex.icon]
class IntegerLiteralExpression final : public Expression {
public:
  IntegerLiteralExpression() : Expression(NodeKind::IntegerLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::IntegerLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for character literals. C++ [lex.ccon]
class CharacterLiteralExpression final : public Expression {
public:
  CharacterLiteralExpression()
      : Expression(NodeKind::CharacterLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CharacterLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for floating-point literals. C++ [lex.fcon]
class FloatingLiteralExpression final : public Expression {
public:
  FloatingLiteralExpression()
      : Expression(NodeKind::FloatingLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::FloatingLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for string-literals. C++ [lex.string]
class StringLiteralExpression final : public Expression {
public:
  StringLiteralExpression() : Expression(NodeKind::StringLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::StringLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for boolean literals. C++ [lex.bool]
class BoolLiteralExpression final : public Expression {
public:
  BoolLiteralExpression() : Expression(NodeKind::BoolLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::BoolLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for the `nullptr` literal. C++ [lex.nullptr]
class CxxNullPtrExpression final : public Expression {
public:
  CxxNullPtrExpression() : Expression(NodeKind::CxxNullPtrExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CxxNullPtrExpression;
  }
  syntax::Leaf *nullPtrKeyword();
};

/// Expression for user-defined literal. C++ [lex.ext]
/// user-defined-literal:
///   user-defined-integer-literal
///   user-defined-floating-point-literal
///   user-defined-string-literal
///   user-defined-character-literal
class UserDefinedLiteralExpression : public Expression {
public:
  UserDefinedLiteralExpression(NodeKind K) : Expression(K) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::IntegerUserDefinedLiteralExpression ||
           N->kind() == NodeKind::FloatUserDefinedLiteralExpression ||
           N->kind() == NodeKind::CharUserDefinedLiteralExpression ||
           N->kind() == NodeKind::StringUserDefinedLiteralExpression;
  }
  syntax::Leaf *literalToken();
};

/// Expression for user-defined-integer-literal. C++ [lex.ext]
class IntegerUserDefinedLiteralExpression final
    : public UserDefinedLiteralExpression {
public:
  IntegerUserDefinedLiteralExpression()
      : UserDefinedLiteralExpression(
            NodeKind::IntegerUserDefinedLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::IntegerUserDefinedLiteralExpression;
  }
};

/// Expression for user-defined-floating-point-literal. C++ [lex.ext]
class FloatUserDefinedLiteralExpression final
    : public UserDefinedLiteralExpression {
public:
  FloatUserDefinedLiteralExpression()
      : UserDefinedLiteralExpression(
            NodeKind::FloatUserDefinedLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::FloatUserDefinedLiteralExpression;
  }
};

/// Expression for user-defined-character-literal. C++ [lex.ext]
class CharUserDefinedLiteralExpression final
    : public UserDefinedLiteralExpression {
public:
  CharUserDefinedLiteralExpression()
      : UserDefinedLiteralExpression(
            NodeKind::CharUserDefinedLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CharUserDefinedLiteralExpression;
  }
};

/// Expression for user-defined-string-literal. C++ [lex.ext]
class StringUserDefinedLiteralExpression final
    : public UserDefinedLiteralExpression {
public:
  StringUserDefinedLiteralExpression()
      : UserDefinedLiteralExpression(
            NodeKind::StringUserDefinedLiteralExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::StringUserDefinedLiteralExpression;
  }
};

/// An abstract class for prefix and postfix unary operators.
class UnaryOperatorExpression : public Expression {
public:
  UnaryOperatorExpression(NodeKind K) : Expression(K) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::PrefixUnaryOperatorExpression ||
           N->kind() == NodeKind::PostfixUnaryOperatorExpression;
  }
  syntax::Leaf *operatorToken();
  syntax::Expression *operand();
};

/// <operator> <operand>
///
/// For example:
///   +a          -b
///   !c          not c
///   ~d          compl d
///   *e          &f
///   ++h         --h
///   __real i    __imag i
class PrefixUnaryOperatorExpression final : public UnaryOperatorExpression {
public:
  PrefixUnaryOperatorExpression()
      : UnaryOperatorExpression(NodeKind::PrefixUnaryOperatorExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::PrefixUnaryOperatorExpression;
  }
};

/// <operand> <operator>
///
/// For example:
///   a++
///   b--
class PostfixUnaryOperatorExpression final : public UnaryOperatorExpression {
public:
  PostfixUnaryOperatorExpression()
      : UnaryOperatorExpression(NodeKind::PostfixUnaryOperatorExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::PostfixUnaryOperatorExpression;
  }
};

/// <lhs> <operator> <rhs>
///
/// For example:
///   a + b
///   a bitor 1
///   a |= b
///   a and_eq b
class BinaryOperatorExpression final : public Expression {
public:
  BinaryOperatorExpression() : Expression(NodeKind::BinaryOperatorExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::BinaryOperatorExpression;
  }
  syntax::Expression *lhs();
  syntax::Leaf *operatorToken();
  syntax::Expression *rhs();
};

/// An abstract node for C++ statements, e.g. 'while', 'if', etc.
/// FIXME: add accessors for semicolon of statements that have it.
class Statement : public Tree {
public:
  Statement(NodeKind K) : Tree(K) {}
  static bool classof(const Node *N) {
    return NodeKind::UnknownStatement <= N->kind() &&
           N->kind() <= NodeKind::CompoundStatement;
  }
};

/// A statement of an unknown kind, i.e. one not currently handled by the syntax
/// tree.
class UnknownStatement final : public Statement {
public:
  UnknownStatement() : Statement(NodeKind::UnknownStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UnknownStatement;
  }
};

/// E.g. 'int a, b = 10;'
class DeclarationStatement final : public Statement {
public:
  DeclarationStatement() : Statement(NodeKind::DeclarationStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::DeclarationStatement;
  }
};

/// The no-op statement, i.e. ';'.
class EmptyStatement final : public Statement {
public:
  EmptyStatement() : Statement(NodeKind::EmptyStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::EmptyStatement;
  }
};

/// switch (<cond>) <body>
class SwitchStatement final : public Statement {
public:
  SwitchStatement() : Statement(NodeKind::SwitchStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::SwitchStatement;
  }
  syntax::Leaf *switchKeyword();
  syntax::Statement *body();
};

/// case <value>: <body>
class CaseStatement final : public Statement {
public:
  CaseStatement() : Statement(NodeKind::CaseStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CaseStatement;
  }
  syntax::Leaf *caseKeyword();
  syntax::Expression *value();
  syntax::Statement *body();
};

/// default: <body>
class DefaultStatement final : public Statement {
public:
  DefaultStatement() : Statement(NodeKind::DefaultStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::DefaultStatement;
  }
  syntax::Leaf *defaultKeyword();
  syntax::Statement *body();
};

/// if (cond) <then-statement> else <else-statement>
/// FIXME: add condition that models 'expression  or variable declaration'
class IfStatement final : public Statement {
public:
  IfStatement() : Statement(NodeKind::IfStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::IfStatement;
  }
  syntax::Leaf *ifKeyword();
  syntax::Statement *thenStatement();
  syntax::Leaf *elseKeyword();
  syntax::Statement *elseStatement();
};

/// for (<init>; <cond>; <increment>) <body>
class ForStatement final : public Statement {
public:
  ForStatement() : Statement(NodeKind::ForStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ForStatement;
  }
  syntax::Leaf *forKeyword();
  syntax::Statement *body();
};

/// while (<cond>) <body>
class WhileStatement final : public Statement {
public:
  WhileStatement() : Statement(NodeKind::WhileStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::WhileStatement;
  }
  syntax::Leaf *whileKeyword();
  syntax::Statement *body();
};

/// continue;
class ContinueStatement final : public Statement {
public:
  ContinueStatement() : Statement(NodeKind::ContinueStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ContinueStatement;
  }
  syntax::Leaf *continueKeyword();
};

/// break;
class BreakStatement final : public Statement {
public:
  BreakStatement() : Statement(NodeKind::BreakStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::BreakStatement;
  }
  syntax::Leaf *breakKeyword();
};

/// return <expr>;
/// return;
class ReturnStatement final : public Statement {
public:
  ReturnStatement() : Statement(NodeKind::ReturnStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ReturnStatement;
  }
  syntax::Leaf *returnKeyword();
  syntax::Expression *value();
};

/// for (<decl> : <init>) <body>
class RangeBasedForStatement final : public Statement {
public:
  RangeBasedForStatement() : Statement(NodeKind::RangeBasedForStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::RangeBasedForStatement;
  }
  syntax::Leaf *forKeyword();
  syntax::Statement *body();
};

/// Expression in a statement position, e.g. functions calls inside compound
/// statements or inside a loop body.
class ExpressionStatement final : public Statement {
public:
  ExpressionStatement() : Statement(NodeKind::ExpressionStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ExpressionStatement;
  }
  syntax::Expression *expression();
};

/// { statement1; statement2; … }
class CompoundStatement final : public Statement {
public:
  CompoundStatement() : Statement(NodeKind::CompoundStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CompoundStatement;
  }
  syntax::Leaf *lbrace();
  /// FIXME: use custom iterator instead of 'vector'.
  std::vector<syntax::Statement *> statements();
  syntax::Leaf *rbrace();
};

/// A declaration that can appear at the top-level. Note that this does *not*
/// correspond 1-to-1 to clang::Decl. Syntax trees distinguish between top-level
/// declarations (e.g. namespace definitions) and declarators (e.g. variables,
/// typedefs, etc.). Declarators are stored inside SimpleDeclaration.
class Declaration : public Tree {
public:
  Declaration(NodeKind K) : Tree(K) {}
  static bool classof(const Node *N) {
    return NodeKind::UnknownDeclaration <= N->kind() &&
           N->kind() <= NodeKind::TypeAliasDeclaration;
  }
};

/// Declaration of an unknown kind, e.g. not yet supported in syntax trees.
class UnknownDeclaration final : public Declaration {
public:
  UnknownDeclaration() : Declaration(NodeKind::UnknownDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UnknownDeclaration;
  }
};

/// A semicolon in the top-level context. Does not declare anything.
class EmptyDeclaration final : public Declaration {
public:
  EmptyDeclaration() : Declaration(NodeKind::EmptyDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::EmptyDeclaration;
  }
};

/// static_assert(<condition>, <message>)
/// static_assert(<condition>)
class StaticAssertDeclaration final : public Declaration {
public:
  StaticAssertDeclaration() : Declaration(NodeKind::StaticAssertDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::StaticAssertDeclaration;
  }
  syntax::Expression *condition();
  syntax::Expression *message();
};

/// extern <string-literal> declaration
/// extern <string-literal> { <decls>  }
class LinkageSpecificationDeclaration final : public Declaration {
public:
  LinkageSpecificationDeclaration()
      : Declaration(NodeKind::LinkageSpecificationDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::LinkageSpecificationDeclaration;
  }
};

/// Groups multiple declarators (e.g. variables, typedefs, etc.) together. All
/// grouped declarators share the same declaration specifiers (e.g. 'int' or
/// 'typedef').
class SimpleDeclaration final : public Declaration {
public:
  SimpleDeclaration() : Declaration(NodeKind::SimpleDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::SimpleDeclaration;
  }
  /// FIXME: use custom iterator instead of 'vector'.
  std::vector<syntax::SimpleDeclarator *> declarators();
};

/// template <template-parameters> <declaration>
class TemplateDeclaration final : public Declaration {
public:
  TemplateDeclaration() : Declaration(NodeKind::TemplateDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TemplateDeclaration;
  }
  syntax::Leaf *templateKeyword();
  syntax::Declaration *declaration();
};

/// template <declaration>
/// Examples:
///     template struct X<int>
///     template void foo<int>()
///     template int var<double>
class ExplicitTemplateInstantiation final : public Declaration {
public:
  ExplicitTemplateInstantiation()
      : Declaration(NodeKind::ExplicitTemplateInstantiation) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ExplicitTemplateInstantiation;
  }
  syntax::Leaf *templateKeyword();
  syntax::Leaf *externKeyword();
  syntax::Declaration *declaration();
};

/// namespace <name> { <decls> }
class NamespaceDefinition final : public Declaration {
public:
  NamespaceDefinition() : Declaration(NodeKind::NamespaceDefinition) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::NamespaceDefinition;
  }
};

/// namespace <name> = <namespace-reference>
class NamespaceAliasDefinition final : public Declaration {
public:
  NamespaceAliasDefinition()
      : Declaration(NodeKind::NamespaceAliasDefinition) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::NamespaceAliasDefinition;
  }
};

/// using namespace <name>
class UsingNamespaceDirective final : public Declaration {
public:
  UsingNamespaceDirective() : Declaration(NodeKind::UsingNamespaceDirective) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UsingNamespaceDirective;
  }
};

/// using <scope>::<name>
/// using typename <scope>::<name>
class UsingDeclaration final : public Declaration {
public:
  UsingDeclaration() : Declaration(NodeKind::UsingDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UsingDeclaration;
  }
};

/// using <name> = <type>
class TypeAliasDeclaration final : public Declaration {
public:
  TypeAliasDeclaration() : Declaration(NodeKind::TypeAliasDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TypeAliasDeclaration;
  }
};

/// Covers a name, an initializer and a part of the type outside declaration
/// specifiers. Examples are:
///     `*a` in `int *a`
///     `a[10]` in `int a[10]`
///     `*a = nullptr` in `int *a = nullptr`
/// Declarators can be unnamed too:
///     `**` in `new int**`
///     `* = nullptr` in `void foo(int* = nullptr)`
/// Most declarators you encounter are instances of SimpleDeclarator. They may
/// contain an inner declarator inside parentheses, we represent it as
/// ParenDeclarator. E.g.
///     `(*a)` in `int (*a) = 10`
class Declarator : public Tree {
public:
  Declarator(NodeKind K) : Tree(K) {}
  static bool classof(const Node *N) {
    return NodeKind::SimpleDeclarator <= N->kind() &&
           N->kind() <= NodeKind::ParenDeclarator;
  }
};

/// A top-level declarator without parentheses. See comment of Declarator for
/// more details.
class SimpleDeclarator final : public Declarator {
public:
  SimpleDeclarator() : Declarator(NodeKind::SimpleDeclarator) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::SimpleDeclarator;
  }
};

/// Declarator inside parentheses.
/// E.g. `(***a)` from `int (***a) = nullptr;`
/// See comment of Declarator for more details.
class ParenDeclarator final : public Declarator {
public:
  ParenDeclarator() : Declarator(NodeKind::ParenDeclarator) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ParenDeclarator;
  }
  syntax::Leaf *lparen();
  syntax::Leaf *rparen();
};

/// Array size specified inside a declarator.
/// E.g:
///   `[10]` in `int a[10];`
///   `[static 10]` in `void f(int xs[static 10]);`
class ArraySubscript final : public Tree {
public:
  ArraySubscript() : Tree(NodeKind::ArraySubscript) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ArraySubscript;
  }
  // TODO: add an accessor for the "static" keyword.
  syntax::Leaf *lbracket();
  syntax::Expression *sizeExpression();
  syntax::Leaf *rbracket();
};

/// Trailing return type after the parameter list, including the arrow token.
/// E.g. `-> int***`.
class TrailingReturnType final : public Tree {
public:
  TrailingReturnType() : Tree(NodeKind::TrailingReturnType) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TrailingReturnType;
  }
  // TODO: add accessors for specifiers.
  syntax::Leaf *arrowToken();
  syntax::SimpleDeclarator *declarator();
};

/// Parameter list for a function type and a trailing return type, if the
/// function has one.
/// E.g.:
///  `(int a) volatile ` in `int foo(int a) volatile;`
///  `(int a) &&` in `int foo(int a) &&;`
///  `() -> int` in `auto foo() -> int;`
///  `() const` in `int foo() const;`
///  `() noexcept` in `int foo() noexcept;`
///  `() throw()` in `int foo() throw();`
///
/// (!) override doesn't belong here.
class ParametersAndQualifiers final : public Tree {
public:
  ParametersAndQualifiers() : Tree(NodeKind::ParametersAndQualifiers) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::ParametersAndQualifiers;
  }
  syntax::Leaf *lparen();
  /// FIXME: use custom iterator instead of 'vector'.
  std::vector<syntax::SimpleDeclaration *> parameters();
  syntax::Leaf *rparen();
  syntax::TrailingReturnType *trailingReturn();
};

/// Member pointer inside a declarator
/// E.g. `X::*` in `int X::* a = 0;`
class MemberPointer final : public Tree {
public:
  MemberPointer() : Tree(NodeKind::MemberPointer) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::MemberPointer;
  }
};

} // namespace syntax
} // namespace clang
#endif
