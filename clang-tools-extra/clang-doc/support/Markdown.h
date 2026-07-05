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

/// Base class for all inline nodes. Inline nodes represent span-level content
/// such as text, emphasis, and inline code.
class InlineNode : public llvm::ilist_node<InlineNode> {
public:
  explicit InlineNode(NodeKind K) : Kind(K) {}
  virtual ~InlineNode() = default;
  NodeKind getKind() const { return Kind; }

  /// Recursively prints the node and its children to OS.
  virtual void print(llvm::raw_ostream &OS) const = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Prints to llvm::errs(). Only available in assert builds.
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  NodeKind Kind;
};

using InlineList = llvm::simple_ilist<InlineNode>;

/// A plain text run.
class TextNode : public InlineNode {
public:
  explicit TextNode(llvm::StringRef T)
      : InlineNode(NodeKind::NK_Text), Text(T) {}
  llvm::StringRef getText() const { return Text; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->getKind() == NodeKind::NK_Text;
  }

private:
  llvm::StringRef Text;
};

/// A backtick-delimited inline code span.
class InlineCodeNode : public InlineNode {
public:
  explicit InlineCodeNode(llvm::StringRef C)
      : InlineNode(NodeKind::NK_InlineCode), Code(C) {}
  llvm::StringRef getCode() const { return Code; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->getKind() == NodeKind::NK_InlineCode;
  }

private:
  llvm::StringRef Code;
};

/// An emphasis span (* or _).
class EmphasisNode : public InlineNode {
public:
  EmphasisNode() : InlineNode(NodeKind::NK_Emphasis) {}
  void addChild(InlineNode &N) { Children.push_back(N); }
  void removeChild(InlineNode &N) { Children.remove(N); }
  InlineList &children() { return Children; }
  const InlineList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->getKind() == NodeKind::NK_Emphasis;
  }

private:
  InlineList Children;
};

/// A strong emphasis span (** or __).
class StrongNode : public InlineNode {
public:
  StrongNode() : InlineNode(NodeKind::NK_Strong) {}
  void addChild(InlineNode &N) { Children.push_back(N); }
  void removeChild(InlineNode &N) { Children.remove(N); }
  InlineList &children() { return Children; }
  const InlineList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const InlineNode *N) {
    return N->getKind() == NodeKind::NK_Strong;
  }

private:
  InlineList Children;
};

/// Base class for all block nodes. Block nodes represent structural constructs
/// such as paragraphs, headings, and lists.
class BlockNode : public llvm::ilist_node<BlockNode> {
public:
  explicit BlockNode(NodeKind K) : Kind(K) {}
  virtual ~BlockNode() = default;
  NodeKind getKind() const { return Kind; }

  /// Recursively prints the node and its children to OS.
  virtual void print(llvm::raw_ostream &OS) const = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Prints to llvm::errs(). Only available in assert builds.
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  NodeKind Kind;
};

using BlockList = llvm::simple_ilist<BlockNode>;

/// A paragraph of inline content.
class ParagraphNode : public BlockNode {
public:
  ParagraphNode() : BlockNode(NodeKind::NK_Paragraph) {}
  void addChild(InlineNode &N) { Children.push_back(N); }
  void removeChild(InlineNode &N) { Children.remove(N); }
  InlineList &children() { return Children; }
  const InlineList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_Paragraph;
  }

private:
  InlineList Children;
};

/// An ATX heading (# through ######).
class HeadingNode : public BlockNode {
public:
  explicit HeadingNode(unsigned L)
      : BlockNode(NodeKind::NK_Heading), Level(L) {}
  unsigned getLevel() const { return Level; }
  void addChild(InlineNode &N) { Children.push_back(N); }
  void removeChild(InlineNode &N) { Children.remove(N); }
  InlineList &children() { return Children; }
  const InlineList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_Heading;
  }

private:
  unsigned Level;
  InlineList Children;
};

/// A fenced code block (``` or ~~~). Lang holds the info string.
class FencedCodeNode : public BlockNode {
public:
  FencedCodeNode(llvm::StringRef L, llvm::StringRef C)
      : BlockNode(NodeKind::NK_FencedCode), Lang(L), Code(C) {}
  llvm::StringRef getLang() const { return Lang; }
  llvm::StringRef getCode() const { return Code; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_FencedCode;
  }

private:
  llvm::StringRef Lang;
  llvm::StringRef Code;
};

/// A single item in an unordered or ordered list.
/// ListItemNode is not a BlockNode -- it only lives inside list nodes.
class ListItemNode : public llvm::ilist_node<ListItemNode> {
public:
  ListItemNode() = default;
  void addChild(InlineNode &N) { Children.push_back(N); }
  void removeChild(InlineNode &N) { Children.remove(N); }
  InlineList &children() { return Children; }
  const InlineList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Prints to llvm::errs(). Only available in assert builds.
  /// ListItemNode provides its own dump() since it does not inherit BlockNode.
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  InlineList Children;
};

using ItemList = llvm::simple_ilist<ListItemNode>;

/// An unordered list (-, *, or + markers).
class UnorderedListNode : public BlockNode {
public:
  UnorderedListNode() : BlockNode(NodeKind::NK_UnorderedList) {}
  void addItem(ListItemNode &N) { Items.push_back(N); }
  void removeItem(ListItemNode &N) { Items.remove(N); }
  ItemList &items() { return Items; }
  const ItemList &items() const { return Items; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_UnorderedList;
  }

private:
  ItemList Items;
};

/// An ordered list (1. 2. 3. markers). Start holds the first item number.
class OrderedListNode : public BlockNode {
public:
  explicit OrderedListNode(unsigned S = 1)
      : BlockNode(NodeKind::NK_OrderedList), Start(S) {}
  unsigned getStart() const { return Start; }
  void addItem(ListItemNode &N) { Items.push_back(N); }
  void removeItem(ListItemNode &N) { Items.remove(N); }
  ItemList &items() { return Items; }
  const ItemList &items() const { return Items; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_OrderedList;
  }

private:
  unsigned Start;
  ItemList Items;
};

/// A block quote (> marker).
class BlockQuoteNode : public BlockNode {
public:
  BlockQuoteNode() : BlockNode(NodeKind::NK_BlockQuote) {}
  void addChild(BlockNode &N) { Children.push_back(N); }
  void removeChild(BlockNode &N) { Children.remove(N); }
  BlockList &children() { return Children; }
  const BlockList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_BlockQuote;
  }

private:
  BlockList Children;
};

/// A thematic break (---, ***, or ___).
class ThematicBreakNode : public BlockNode {
public:
  ThematicBreakNode() : BlockNode(NodeKind::NK_ThematicBreak) {}
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_ThematicBreak;
  }
};

/// The root document node. Contains all top-level block nodes.
class DocumentNode : public BlockNode {
public:
  DocumentNode() : BlockNode(NodeKind::NK_Document) {}
  void addChild(BlockNode &N) { Children.push_back(N); }
  void removeChild(BlockNode &N) { Children.remove(N); }
  BlockList &children() { return Children; }
  const BlockList &children() const { return Children; }
  void print(llvm::raw_ostream &OS) const override;
  static bool classof(const BlockNode *N) {
    return N->getKind() == NodeKind::NK_Document;
  }

private:
  BlockList Children;
};

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
