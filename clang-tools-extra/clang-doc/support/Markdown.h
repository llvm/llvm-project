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

struct Node {
  NodeKind Kind;
  explicit Node(NodeKind K) : Kind(K) {}
};

struct TextNode : Node {
  llvm::StringRef Text;
  explicit TextNode(llvm::StringRef T) : Node(NodeKind::NK_Text), Text(T) {}
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Text; }
};
static_assert(std::is_trivially_destructible_v<TextNode>);

struct InlineCodeNode : Node {
  llvm::StringRef Code;
  explicit InlineCodeNode(llvm::StringRef C)
      : Node(NodeKind::NK_InlineCode), Code(C) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};
static_assert(std::is_trivially_destructible_v<InlineCodeNode>);

struct EmphasisNode : Node {
  llvm::ArrayRef<Node *> Children;
  explicit EmphasisNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Emphasis), Children(C) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};
static_assert(std::is_trivially_destructible_v<EmphasisNode>);

struct StrongNode : Node {
  llvm::ArrayRef<Node *> Children;
  explicit StrongNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Strong), Children(C) {}
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Strong; }
};
static_assert(std::is_trivially_destructible_v<StrongNode>);

struct ParagraphNode : Node {
  llvm::ArrayRef<Node *> Children;
  explicit ParagraphNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Paragraph), Children(C) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};
static_assert(std::is_trivially_destructible_v<ParagraphNode>);

struct HeadingNode : Node {
  unsigned Level;
  llvm::ArrayRef<Node *> Children;
  HeadingNode(unsigned L, llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_Heading), Level(L), Children(C) {}
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Heading; }
};
static_assert(std::is_trivially_destructible_v<HeadingNode>);

struct FencedCodeNode : Node {
  llvm::StringRef Lang;
  llvm::ArrayRef<llvm::StringRef> Lines;
  FencedCodeNode(llvm::StringRef L, llvm::ArrayRef<llvm::StringRef> Ls)
      : Node(NodeKind::NK_FencedCode), Lang(L), Lines(Ls) {}
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

struct TableNode : Node {
  TableRow Header;
  llvm::ArrayRef<TableRow> Body;
  TableNode(TableRow H, llvm::ArrayRef<TableRow> B)
      : Node(NodeKind::NK_Table), Header(H), Body(B) {}
  static bool classof(const Node *N) { return N->Kind == NodeKind::NK_Table; }
};
static_assert(std::is_trivially_destructible_v<TableNode>);

struct ListItemNode : Node {
  llvm::ArrayRef<Node *> Children;
  explicit ListItemNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_ListItem), Children(C) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};
static_assert(std::is_trivially_destructible_v<ListItemNode>);

struct UnorderedListNode : Node {
  llvm::ArrayRef<ListItemNode *> Items;
  explicit UnorderedListNode(llvm::ArrayRef<ListItemNode *> I)
      : Node(NodeKind::NK_UnorderedList), Items(I) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};
static_assert(std::is_trivially_destructible_v<UnorderedListNode>);

struct OrderedListNode : Node {
  unsigned Start;
  llvm::ArrayRef<ListItemNode *> Items;
  OrderedListNode(unsigned S, llvm::ArrayRef<ListItemNode *> I)
      : Node(NodeKind::NK_OrderedList), Start(S), Items(I) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};
static_assert(std::is_trivially_destructible_v<OrderedListNode>);

struct BlockQuoteNode : Node {
  llvm::ArrayRef<Node *> Children;
  explicit BlockQuoteNode(llvm::ArrayRef<Node *> C)
      : Node(NodeKind::NK_BlockQuote), Children(C) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};
static_assert(std::is_trivially_destructible_v<BlockQuoteNode>);

struct ThematicBreakNode : Node {
  ThematicBreakNode() : Node(NodeKind::NK_ThematicBreak) {}
  static bool classof(const Node *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};
static_assert(std::is_trivially_destructible_v<ThematicBreakNode>);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H