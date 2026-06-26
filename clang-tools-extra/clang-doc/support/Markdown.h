//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

namespace clang::doc::markdown {

enum class NodeKind {
  NK_Text,
  NK_InlineCode,
  NK_Emphasis,
  NK_Strong,
  NK_Paragraph,
  NK_Heading,
  NK_FencedCode,
  NK_Table,
  NK_UnorderedList,
  NK_OrderedList,
  NK_ListItem,
  NK_BlockQuote,
  NK_ThematicBreak,
};

class Node {
public:
  NodeKind Kind;
  explicit Node(NodeKind K) : Kind(K) {}
  void dump() const { llvm::errs() << "Node\n"; }
  static bool classof(const Node *) { return true; }
};

class TextNode : public Node {
  llvm::StringRef Text;

public:
  explicit TextNode(llvm::StringRef T) : Node(NodeKind::NK_Text), Text(T) {}
  llvm::StringRef getText() const { return Text; }
  void dump() const { llvm::errs() << "TextNode: " << Text << "\n"; }
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Text; }
};
static_assert(std::is_trivially_destructible_v<TextNode>);

class InlineCodeNode : public Node {
  llvm::StringRef Code;

public:
  explicit InlineCodeNode(llvm::StringRef C)
      : Node(NodeKind::NK_InlineCode), Code(C) {}
  llvm::StringRef getCode() const { return Code; }
  void dump() const { llvm::errs() << "InlineCodeNode: " << Code << "\n"; }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};
static_assert(std::is_trivially_destructible_v<InlineCodeNode>);

class EmphasisNode : public Node {
  llvm::ArrayRef<Node *> Children;

public:
  explicit EmphasisNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Emphasis), Children(C) {}
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "EmphasisNode (" << Children.size() << " children)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};
static_assert(std::is_trivially_destructible_v<EmphasisNode>);

class StrongNode : public Node {
  llvm::ArrayRef<Node *> Children;

public:
  explicit StrongNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Strong), Children(C) {}
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "StrongNode (" << Children.size() << " children)\n";
  }
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Strong; }
};
static_assert(std::is_trivially_destructible_v<StrongNode>);

class ParagraphNode : public Node {
  llvm::ArrayRef<Node *> Children;

public:
  explicit ParagraphNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Paragraph), Children(C) {}
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "ParagraphNode (" << Children.size() << " children)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};
static_assert(std::is_trivially_destructible_v<ParagraphNode>);

class HeadingNode : public Node {
  unsigned Level;
  llvm::ArrayRef<Node *> Children;

public:
  HeadingNode(unsigned L, llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Heading), Level(L), Children(C) {}
  unsigned getLevel() const { return Level; }
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "HeadingNode: level=" << Level << " (" << Children.size()
                 << " children)\n";
  }
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Heading; }
};
static_assert(std::is_trivially_destructible_v<HeadingNode>);

class FencedCodeNode : public Node {
  llvm::StringRef Lang;
  llvm::ArrayRef<llvm::StringRef> Lines;

public:
  FencedCodeNode(llvm::StringRef L, llvm::ArrayRef<llvm::StringRef> Ls)
      : Node(NodeKind::NK_FencedCode), Lang(L), Lines(Ls) {}
  llvm::StringRef getLang() const { return Lang; }
  llvm::ArrayRef<llvm::StringRef> getLines() const { return Lines; }
  void dump() const {
    llvm::errs() << "FencedCodeNode: lang=" << Lang << " (" << Lines.size()
                 << " lines)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};
static_assert(std::is_trivially_destructible_v<FencedCodeNode>);

struct TableCell {
  llvm::ArrayRef<Node *> Children;
};
static_assert(std::is_trivially_destructible_v<TableCell>);

struct TableRow {
  llvm::ArrayRef<TableCell> Cells;
};
static_assert(std::is_trivially_destructible_v<TableRow>);

class TableNode : public Node {
  TableRow Header;
  llvm::ArrayRef<TableRow> Body;

public:
  TableNode(TableRow H, llvm::ArrayRef<TableRow> B)
      : Node(NodeKind::NK_Table), Header(H), Body(B) {}
  const TableRow &getHeader() const { return Header; }
  llvm::ArrayRef<TableRow> getBody() const { return Body; }
  void dump() const {
    llvm::errs() << "TableNode: " << Header.Cells.size() << " header cells, "
                 << Body.size() << " rows\n";
  }
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Table; }
};
static_assert(std::is_trivially_destructible_v<TableNode>);

class ListItemNode : public Node {
  llvm::ArrayRef<Node *> Children;

public:
  explicit ListItemNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_ListItem), Children(C) {}
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "ListItemNode (" << Children.size() << " children)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};
static_assert(std::is_trivially_destructible_v<ListItemNode>);

class UnorderedListNode : public Node {
  llvm::ArrayRef<ListItemNode *> Items;

public:
  UnorderedListNode() : Node(NodeKind::NK_UnorderedList), Items({}) {}
  explicit UnorderedListNode(llvm::ArrayRef<ListItemNode *> I)
      : Node(NodeKind::NK_UnorderedList), Items(I) {}
  llvm::ArrayRef<ListItemNode *> getItems() const { return Items; }
  void dump() const {
    llvm::errs() << "UnorderedListNode (" << Items.size() << " items)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};
static_assert(std::is_trivially_destructible_v<UnorderedListNode>);

class OrderedListNode : public Node {
  unsigned Start;
  llvm::ArrayRef<ListItemNode *> Items;

public:
  OrderedListNode(unsigned S, llvm::ArrayRef<ListItemNode *> I)
      : Node(NodeKind::NK_OrderedList), Start(S), Items(I) {}
  unsigned getStart() const { return Start; }
  llvm::ArrayRef<ListItemNode *> getItems() const { return Items; }
  void dump() const {
    llvm::errs() << "OrderedListNode: start=" << Start << " (" << Items.size()
                 << " items)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};
static_assert(std::is_trivially_destructible_v<OrderedListNode>);

class BlockQuoteNode : public Node {
  llvm::ArrayRef<Node *> Children;

public:
  explicit BlockQuoteNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_BlockQuote), Children(C) {}
  llvm::ArrayRef<Node *> getChildren() const { return Children; }
  void dump() const {
    llvm::errs() << "BlockQuoteNode (" << Children.size() << " children)\n";
  }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};
static_assert(std::is_trivially_destructible_v<BlockQuoteNode>);

class ThematicBreakNode : public Node {
public:
  ThematicBreakNode() : Node(NodeKind::NK_ThematicBreak) {}
  void dump() const { llvm::errs() << "ThematicBreakNode\n"; }
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};
static_assert(std::is_trivially_destructible_v<ThematicBreakNode>);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
