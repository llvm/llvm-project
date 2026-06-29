//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the Markdown AST node hierarchy for the clang-doc Markdown parser.
///
/// Block nodes represent structural constructs (paragraphs, headings, lists,
/// fenced code blocks, etc). Inline nodes represent span-level content (text,
/// emphasis, inline code) that appears inside block nodes.
///
/// All nodes are arena-allocated via ASTContext, which owns their lifetime.
/// The parser builds the tree by calling push_back() on simple_ilist members.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::doc::markdown {

enum class NodeKind {
  // Inline nodes
  NK_Text,
  NK_InlineCode,
  NK_Emphasis,
  NK_Strong,
  // Block nodes
  NK_Paragraph,
  NK_Heading,
  NK_FencedCode,
  NK_Table, // TODO: add TableNode
  NK_UnorderedList,
  NK_OrderedList,
  NK_BlockQuote,
  NK_ThematicBreak,
  NK_Document,
};

struct InlineNode;
struct BlockNode;

//===----------------------------------------------------------------------===//
// Inline nodes -- span-level content inside block nodes
//===----------------------------------------------------------------------===//

/// Base class for all inline nodes. Inline nodes represent span-level content
/// such as text, emphasis, and inline code. They live in InlineList members
/// of block nodes.
struct InlineNode : llvm::ilist_node<InlineNode> {
  NodeKind Kind;
  explicit InlineNode(NodeKind K) : Kind(K) {}
  virtual ~InlineNode() = default;
  virtual void print(llvm::raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD void dump() const;
};

using InlineList = llvm::simple_ilist<InlineNode>;

/// A plain text run.
struct TextNode : InlineNode {
private:
  llvm::StringRef Text;

public:
  explicit TextNode(llvm::StringRef T)
      : InlineNode(NodeKind::NK_Text), Text(T) {}
  llvm::StringRef getText() const { return Text; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Text;
  }
};

/// A backtick-delimited inline code span.
struct InlineCodeNode : InlineNode {
private:
  llvm::StringRef Code;

public:
  explicit InlineCodeNode(llvm::StringRef C)
      : InlineNode(NodeKind::NK_InlineCode), Code(C) {}
  llvm::StringRef getCode() const { return Code; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};

/// An emphasis span (* or _).
struct EmphasisNode : InlineNode {
  InlineList Children;
  EmphasisNode() : InlineNode(NodeKind::NK_Emphasis) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};

/// A strong emphasis span (** or __).
struct StrongNode : InlineNode {
  InlineList Children;
  StrongNode() : InlineNode(NodeKind::NK_Strong) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Strong;
  }
};

//===----------------------------------------------------------------------===//
// Block nodes -- structural constructs
//===----------------------------------------------------------------------===//

/// Base class for all block nodes. Block nodes represent structural constructs
/// such as paragraphs, headings, and lists. They live in BlockList members of
/// container block nodes.
struct BlockNode : llvm::ilist_node<BlockNode> {
  NodeKind Kind;
  explicit BlockNode(NodeKind K) : Kind(K) {}
  virtual ~BlockNode() = default;
  virtual void print(llvm::raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD void dump() const;
};

using BlockList = llvm::simple_ilist<BlockNode>;

/// A paragraph of inline content.
struct ParagraphNode : BlockNode {
  InlineList Children;
  ParagraphNode() : BlockNode(NodeKind::NK_Paragraph) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};

/// An ATX heading (# through ######).
struct HeadingNode : BlockNode {
private:
  unsigned Level;

public:
  InlineList Children;
  explicit HeadingNode(unsigned L)
      : BlockNode(NodeKind::NK_Heading), Level(L) {}
  unsigned getLevel() const { return Level; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Heading;
  }
};

/// A fenced code block (``` or ~~~). Lang holds the info string.
struct FencedCodeNode : BlockNode {
private:
  llvm::StringRef Lang;
  llvm::StringRef Code;

public:
  FencedCodeNode(llvm::StringRef L, llvm::StringRef C)
      : BlockNode(NodeKind::NK_FencedCode), Lang(L), Code(C) {}
  llvm::StringRef getLang() const { return Lang; }
  llvm::StringRef getCode() const { return Code; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};

/// A single item in an unordered or ordered list.
/// ListItemNode is not a BlockNode -- it only lives inside list nodes.
struct ListItemNode : llvm::ilist_node<ListItemNode> {
  InlineList Children;
  ListItemNode() = default;
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// An unordered list (-, *, or + markers).
struct UnorderedListNode : BlockNode {
  llvm::simple_ilist<ListItemNode> Items;
  UnorderedListNode() : BlockNode(NodeKind::NK_UnorderedList) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};

/// An ordered list (1. 2. 3. markers). Start holds the first item number.
struct OrderedListNode : BlockNode {
private:
  unsigned Start;

public:
  llvm::simple_ilist<ListItemNode> Items;
  explicit OrderedListNode(unsigned S = 1)
      : BlockNode(NodeKind::NK_OrderedList), Start(S) {}
  unsigned getStart() const { return Start; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};

/// A block quote (> marker).
struct BlockQuoteNode : BlockNode {
  BlockList Children;
  BlockQuoteNode() : BlockNode(NodeKind::NK_BlockQuote) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};

/// A thematic break (---, ***, or ___).
struct ThematicBreakNode : BlockNode {
  ThematicBreakNode() : BlockNode(NodeKind::NK_ThematicBreak) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};

/// The root document node. Contains all top-level block nodes.
/// Children are added by the parser via push_back on the Children ilist.
struct DocumentNode : BlockNode {
  BlockList Children;
  DocumentNode() : BlockNode(NodeKind::NK_Document) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Document;
  }
};

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
