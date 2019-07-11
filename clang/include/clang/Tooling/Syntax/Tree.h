//===- Tree.h - structure of the syntax tree ------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines the basic structure of the syntax tree. There are two kinds of nodes:
//   - leaf nodes correspond to a token in the expanded token stream,
//   - tree nodes correspond to language grammar constructs.
//
// The tree is initially built from an AST. Each node of a newly built tree
// covers a continous subrange of expanded tokens (i.e. tokens after
// preprocessing), the specific tokens coverered are stored in the leaf nodes of
// a tree. A post-order traversal of a tree will visit leaf nodes in an order
// corresponding the original order of expanded tokens.
//
// This is still work in progress and highly experimental, we leave room for
// ourselves to completely change the design and/or implementation.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_SYNTAX_TREE_CASCADE_H
#define LLVM_CLANG_TOOLING_SYNTAX_TREE_CASCADE_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"
#include <cstdint>

namespace clang {
namespace syntax {

/// A memory arena for syntax trees. Also tracks the underlying token buffers,
/// source manager, etc.
class Arena {
public:
  Arena(SourceManager &SourceMgr, const LangOptions &LangOpts,
        TokenBuffer Tokens);

  const SourceManager &sourceManager() const { return SourceMgr; }
  const LangOptions &langOptions() const { return LangOpts; }

  const TokenBuffer &tokenBuffer() const;
  llvm::BumpPtrAllocator &allocator() { return Allocator; }

  /// Add \p Buffer to the underlying source manager, tokenize it and store the
  /// resulting tokens. Useful when there is a need to materialize tokens that
  /// were not written in user code.
  std::pair<FileID, llvm::ArrayRef<syntax::Token>>
  lexBuffer(std::unique_ptr<llvm::MemoryBuffer> Buffer);

private:
  SourceManager &SourceMgr;
  const LangOptions &LangOpts;
  TokenBuffer Tokens;
  /// IDs and storage for additional tokenized files.
  llvm::DenseMap<FileID, std::vector<syntax::Token>> ExtraTokens;
  /// Keeps all the allocated nodes and their intermediate data structures.
  llvm::BumpPtrAllocator Allocator;
};

class Tree;
class TreeBuilder;
enum class NodeKind : uint16_t;
enum class NodeRole : uint8_t;

/// A node in a syntax tree. Each node is either a Leaf (representing tokens) or
/// a Tree (representing language constructrs).
class Node {
public:
  /// Newly created nodes are detached from a tree, parent and sibling links are
  /// set when the node is added as a child to another one.
  Node(NodeKind Kind);

  NodeKind kind() const { return static_cast<NodeKind>(Kind); }
  NodeRole role() const { return static_cast<NodeRole>(Role); }

  const Tree *parent() const { return Parent; }
  Tree *parent() { return Parent; }

  const Node *nextSibling() const { return NextSibling; }
  Node *nextSibling() { return NextSibling; }

  /// Dumps the structure of a subtree. For debugging and testing purposes.
  std::string dump(const Arena &A) const;
  /// Dumps the tokens forming this subtree.
  std::string dumpTokens(const Arena &A) const;

private:
  // Tree is allowed to change the Parent link and Role.
  friend class Tree;

  Tree *Parent;
  Node *NextSibling;
  unsigned Kind : 16;
  unsigned Role : 8;
};

/// A leaf node points to a single token inside the expanded token stream.
class Leaf final : public Node {
public:
  Leaf(const syntax::Token *T);
  static bool classof(const Node *N);

  const syntax::Token *token() const { return Tok; }

private:
  const syntax::Token *Tok;
};

/// A node that has children and represents a syntactic language construct.
class Tree : public Node {
public:
  using Node::Node;
  static bool classof(const Node *N);

  Node *firstChild() { return FirstChild; }
  const Node *firstChild() const { return FirstChild; }

protected:
  /// Find the first node with a corresponding role.
  syntax::Node *findChild(NodeRole R);

private:
  /// Prepend \p Child to the list of children and and sets the parent pointer.
  /// A very low-level operation that does not check any invariants, only used
  /// by TreeBuilder.
  /// EXPECTS: Role != NodeRoleDetached.
  void prependChildLowLevel(Node *Child, NodeRole Role);
  friend class TreeBuilder;

  Node *FirstChild = nullptr;
};

} // namespace syntax
} // namespace clang

#endif
