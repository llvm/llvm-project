//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

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
  NK_Table,
  NK_UnorderedList,
  NK_OrderedList,
  NK_ListItem,
  NK_BlockQuote,
  NK_ThematicBreak,
  NK_Document,
};

// Forward declarations
struct InlineNode;
struct BlockNode;

//===----------------------------------------------------------------------===//
// Inline nodes
//===----------------------------------------------------------------------===//

struct InlineNode
    : llvm::ilist_node<InlineNode, llvm::ilist_sentinel_tracking<true>> {
  NodeKind Kind;
  explicit InlineNode(NodeKind K) : Kind(K) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const;
};

struct TextNode : InlineNode {
private:
  llvm::StringRef Text;

public:
  explicit TextNode(llvm::StringRef T)
      : InlineNode(NodeKind::NK_Text), Text(T) {}
  llvm::StringRef getText() const { return Text; }
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "TextNode: " << Text << "\n";
  }
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Text;
  }
};
static_assert(std::is_trivially_destructible_v<TextNode>);

struct InlineCodeNode : InlineNode {
private:
  llvm::StringRef Code;

public:
  explicit InlineCodeNode(llvm::StringRef C)
      : InlineNode(NodeKind::NK_InlineCode), Code(C) {}
  llvm::StringRef getCode() const { return Code; }
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "InlineCodeNode: " << Code << "\n";
  }
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};
static_assert(std::is_trivially_destructible_v<InlineCodeNode>);

struct EmphasisNode : InlineNode {
  llvm::simple_ilist<InlineNode, llvm::ilist_sentinel_tracking<true>> Children;
  EmphasisNode() : InlineNode(NodeKind::NK_Emphasis) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "EmphasisNode\n";
  }
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};

struct StrongNode : InlineNode {
  llvm::simple_ilist<InlineNode, llvm::ilist_sentinel_tracking<true>> Children;
  StrongNode() : InlineNode(NodeKind::NK_Strong) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "StrongNode\n";
  }
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Strong;
  }
};

//===----------------------------------------------------------------------===//
// Block nodes
//===----------------------------------------------------------------------===//

struct BlockNode
    : llvm::ilist_node<BlockNode, llvm::ilist_sentinel_tracking<true>> {
  NodeKind Kind;
  explicit BlockNode(NodeKind K) : Kind(K) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const;
};

using InlineList =
    llvm::simple_ilist<InlineNode, llvm::ilist_sentinel_tracking<true>>;
using BlockList =
    llvm::simple_ilist<BlockNode, llvm::ilist_sentinel_tracking<true>>;

struct ParagraphNode : BlockNode {
  InlineList Children;
  ParagraphNode() : BlockNode(NodeKind::NK_Paragraph) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "ParagraphNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};

struct HeadingNode : BlockNode {
private:
  unsigned Level;

public:
  InlineList Children;
  explicit HeadingNode(unsigned L)
      : BlockNode(NodeKind::NK_Heading), Level(L) {}
  unsigned getLevel() const { return Level; }
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "HeadingNode: level=" << Level << "\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Heading;
  }
};

struct FencedCodeNode : BlockNode {
private:
  llvm::StringRef Lang;
  llvm::StringRef Code;

public:
  FencedCodeNode(llvm::StringRef L, llvm::StringRef C)
      : BlockNode(NodeKind::NK_FencedCode), Lang(L), Code(C) {}
  llvm::StringRef getLang() const { return Lang; }
  llvm::StringRef getCode() const { return Code; }
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "FencedCodeNode: lang=" << Lang << "\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};
static_assert(std::is_trivially_destructible_v<FencedCodeNode>);

struct ListItemNode : BlockNode {
  InlineList Children;
  ListItemNode() : BlockNode(NodeKind::NK_ListItem) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "ListItemNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};

struct UnorderedListNode : BlockNode {
  llvm::simple_ilist<ListItemNode, llvm::ilist_sentinel_tracking<true>> Items;
  UnorderedListNode() : BlockNode(NodeKind::NK_UnorderedList) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "UnorderedListNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};

struct OrderedListNode : BlockNode {
private:
  unsigned Start;

public:
  llvm::simple_ilist<ListItemNode, llvm::ilist_sentinel_tracking<true>> Items;
  explicit OrderedListNode(unsigned S = 1)
      : BlockNode(NodeKind::NK_OrderedList), Start(S) {}
  unsigned getStart() const { return Start; }
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "OrderedListNode: start=" << Start << "\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};

struct BlockQuoteNode : BlockNode {
  BlockList Children;
  BlockQuoteNode() : BlockNode(NodeKind::NK_BlockQuote) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "BlockQuoteNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};

struct ThematicBreakNode : BlockNode {
  ThematicBreakNode() : BlockNode(NodeKind::NK_ThematicBreak) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "ThematicBreakNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};

struct DocumentNode : BlockNode {
  BlockList Children;
  DocumentNode() : BlockNode(NodeKind::NK_Document) {}
  void dump(llvm::raw_ostream &OS = llvm::errs()) const {
    OS << "DocumentNode\n";
  }
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Document;
  }
};

//===----------------------------------------------------------------------===//
// ASTContext - owns the arena and string pool
//===----------------------------------------------------------------------===//

class ASTContext {
  llvm::BumpPtrAllocator Arena;
  llvm::StringSaver SSaver;
  DocumentNode *Root = nullptr;

public:
  ASTContext() : SSaver(Arena) {}

  template <typename T, typename... Args> T *allocate(Args &&...args) {
    return new (Arena.Allocate<T>()) T(std::forward<Args>(args)...);
  }

  llvm::StringRef intern(llvm::StringRef S) { return SSaver.save(S); }
  DocumentNode *getRoot() { return Root; }
  void setRoot(DocumentNode *R) { Root = R; }
};

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
