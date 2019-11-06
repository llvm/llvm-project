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
  TopLevelDeclaration,

  // Expressions
  UnknownExpression,

  // Statements
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
  CompoundStatement
};
/// For debugging purposes.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeKind K);

/// A relation between a parent and child node, e.g. 'left-hand-side of
/// a binary expression'. Used for implementing accessors.
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
  /// An inner statement for those that have only a single child of kind
  /// statement, e.g. loop body for while, for, etc; inner statement for case,
  /// default, etc.
  BodyStatement,

  // Roles specific to particular node kinds.
  CaseStatement_value,
  IfStatement_thenStatement,
  IfStatement_elseKeyword,
  IfStatement_elseStatement,
  ReturnStatement_value,
  ExpressionStatement_expression,
  CompoundStatement_statement
};
/// For debugging purposes.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeRole R);

/// A root node for a translation unit. Parent is always null.
class TranslationUnit final : public Tree {
public:
  TranslationUnit() : Tree(NodeKind::TranslationUnit) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TranslationUnit;
  }
};

/// FIXME: this node is temporary and will be replaced with nodes for various
///        'declarations' and 'declarators' from the C/C++ grammar
///
/// Represents any top-level declaration. Only there to give the syntax tree a
/// bit of structure until we implement syntax nodes for declarations and
/// declarators.
class TopLevelDeclaration final : public Tree {
public:
  TopLevelDeclaration() : Tree(NodeKind::TopLevelDeclaration) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::TopLevelDeclaration;
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

/// An expression of an unknown kind, i.e. one not currently handled by the
/// syntax tree.
class UnknownExpression final : public Expression {
public:
  UnknownExpression() : Expression(NodeKind::UnknownExpression) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::UnknownExpression;
  }
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

} // namespace syntax
} // namespace clang
#endif
