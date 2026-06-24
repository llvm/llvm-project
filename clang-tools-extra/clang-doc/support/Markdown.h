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
#include "llvm/Support/Allocator.h"
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

struct MDNode {
  NodeKind Kind;
  explicit MDNode(NodeKind K) : Kind(K) {}
};

struct TextNode : MDNode {
  llvm::StringRef Text;
  explicit TextNode(llvm::StringRef T) : MDNode(NodeKind::NK_Text), Text(T) {}
  static bool classof(const MDNode *N) { return N->Kind == NodeKind::NK_Text; }
};
static_assert(std::is_trivially_destructible_v<TextNode>);

struct InlineCodeNode : MDNode {
  llvm::StringRef Code;
  explicit InlineCodeNode(llvm::StringRef C)
      : MDNode(NodeKind::NK_InlineCode), Code(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};
static_assert(std::is_trivially_destructible_v<InlineCodeNode>);

struct EmphasisNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit EmphasisNode(llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_Emphasis), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};
static_assert(std::is_trivially_destructible_v<EmphasisNode>);

struct StrongNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit StrongNode(llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_Strong), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Strong;
  }
};
static_assert(std::is_trivially_destructible_v<StrongNode>);

struct ParagraphNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit ParagraphNode(llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_Paragraph), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};
static_assert(std::is_trivially_destructible_v<ParagraphNode>);

struct HeadingNode : MDNode {
  unsigned Level;
  llvm::ArrayRef<MDNode *> Children;
  HeadingNode(unsigned L, llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_Heading), Level(L), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Heading;
  }
};
static_assert(std::is_trivially_destructible_v<HeadingNode>);

struct FencedCodeNode : MDNode {
  llvm::StringRef Lang;
  llvm::ArrayRef<llvm::StringRef> Lines;
  FencedCodeNode(llvm::StringRef L, llvm::ArrayRef<llvm::StringRef> Ls)
      : MDNode(NodeKind::NK_FencedCode), Lang(L), Lines(Ls) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};
static_assert(std::is_trivially_destructible_v<FencedCodeNode>);

struct TableCell {
  llvm::ArrayRef<MDNode *> Children;
};
static_assert(std::is_trivially_destructible_v<TableCell>);

struct TableRow {
  llvm::ArrayRef<TableCell> Cells;
};
static_assert(std::is_trivially_destructible_v<TableRow>);

struct TableNode : MDNode {
  TableRow Header;
  llvm::ArrayRef<TableRow> Body;
  TableNode(TableRow H, llvm::ArrayRef<TableRow> B)
      : MDNode(NodeKind::NK_Table), Header(H), Body(B) {}
  static bool classof(const MDNode *N) { return N->Kind == NodeKind::NK_Table; }
};
static_assert(std::is_trivially_destructible_v<TableNode>);

struct ListItemNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit ListItemNode(llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_ListItem), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};
static_assert(std::is_trivially_destructible_v<ListItemNode>);

struct UnorderedListNode : MDNode {
  llvm::ArrayRef<ListItemNode *> Items;
  explicit UnorderedListNode(llvm::ArrayRef<ListItemNode *> I)
      : MDNode(NodeKind::NK_UnorderedList), Items(I) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};
static_assert(std::is_trivially_destructible_v<UnorderedListNode>);

struct OrderedListNode : MDNode {
  unsigned Start;
  llvm::ArrayRef<ListItemNode *> Items;
  OrderedListNode(unsigned S, llvm::ArrayRef<ListItemNode *> I)
      : MDNode(NodeKind::NK_OrderedList), Start(S), Items(I) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};
static_assert(std::is_trivially_destructible_v<OrderedListNode>);

struct BlockQuoteNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit BlockQuoteNode(llvm::ArrayRef<MDNode *> C)
      : MDNode(NodeKind::NK_BlockQuote), Children(C) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};
static_assert(std::is_trivially_destructible_v<BlockQuoteNode>);

struct ThematicBreakNode : MDNode {
  ThematicBreakNode() : MDNode(NodeKind::NK_ThematicBreak) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};
static_assert(std::is_trivially_destructible_v<ThematicBreakNode>);

llvm::ArrayRef<MDNode *> parseMarkdown(llvm::StringRef Text,
                                       llvm::BumpPtrAllocator &Arena);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H