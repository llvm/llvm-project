//===- Nodes.h - syntax nodes for C/C++ grammar constructs ----*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Syntax tree nodes for C, C++ and Objective-C grammar constructs.
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

/// A kind of a syntax node, used for implementing casts.
enum class NodeKind : uint16_t {
  Leaf,
  TranslationUnit,
  TopLevelDeclaration,
  CompoundStatement
};
/// For debugging purposes.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeKind K);

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

/// An abstract node for C++ statements, e.g. 'while', 'if', etc.
class Statement : public Tree {
public:
  Statement(NodeKind K) : Tree(K) {}
  static bool classof(const Node *N) {
    return NodeKind::CompoundStatement <= N->kind() &&
           N->kind() <= NodeKind::CompoundStatement;
  }
};

/// { statement1; statement2; … }
class CompoundStatement final : public Statement {
public:
  CompoundStatement() : Statement(NodeKind::CompoundStatement) {}
  static bool classof(const Node *N) {
    return N->kind() == NodeKind::CompoundStatement;
  }
  syntax::Leaf *lbrace();
  syntax::Leaf *rbrace();

  struct Roles {
    static constexpr NodeRole lbrace = 1;
    static constexpr NodeRole rbrace = 2;
  };
};

} // namespace syntax
} // namespace clang
#endif
