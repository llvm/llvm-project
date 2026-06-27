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
#include "llvm/Support/Compiler.h"
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

struct InlineNode;
struct BlockNode;

//===----------------------------------------------------------------------===//
// Inline nodes
//===----------------------------------------------------------------------===//

struct InlineNode : llvm::ilist_node<InlineNode> {
  NodeKind Kind;
  explicit InlineNode(NodeKind K) : Kind(K) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

using InlineList = llvm::simple_ilist<InlineNode>;

struct TextNode : InlineNode {
private:
  llvm::StringRef Text;

public:
  explicit TextNode(llvm::StringRef T)
      : InlineNode(NodeKind::NK_Text), Text(T) {}
  llvm::StringRef getText() const { return Text; }
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
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
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};
static_assert(std::is_trivially_destructible_v<InlineCodeNode>);

struct EmphasisNode : InlineNode {
  InlineList Children;
  EmphasisNode() : InlineNode(NodeKind::NK_Emphasis) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};

struct StrongNode : InlineNode {
  InlineList Children;
  StrongNode() : InlineNode(NodeKind::NK_Strong) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const InlineNode *N) {
    return N->Kind == NodeKind::NK_Strong;
  }
};

//===----------------------------------------------------------------------===//
// Block nodes
//===----------------------------------------------------------------------===//

struct BlockNode : llvm::ilist_node<BlockNode> {
  NodeKind Kind;
  explicit BlockNode(NodeKind K) : Kind(K) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

using BlockList = llvm::simple_ilist<BlockNode>;

struct ParagraphNode : BlockNode {
  InlineList Children;
  ParagraphNode() : BlockNode(NodeKind::NK_Paragraph) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
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
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
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
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};
static_assert(std::is_trivially_destructible_v<FencedCodeNode>);

struct ListItemNode : BlockNode, llvm::ilist_node<ListItemNode> {
  InlineList Children;
  ListItemNode() : BlockNode(NodeKind::NK_ListItem) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};

struct UnorderedListNode : BlockNode {
  llvm::simple_ilist<ListItemNode> Items;
  UnorderedListNode() : BlockNode(NodeKind::NK_UnorderedList) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};

struct OrderedListNode : BlockNode {
private:
  unsigned Start;

public:
  llvm::simple_ilist<ListItemNode> Items;
  explicit OrderedListNode(unsigned S = 1)
      : BlockNode(NodeKind::NK_OrderedList), Start(S) {}
  unsigned getStart() const { return Start; }
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};

struct BlockQuoteNode : BlockNode {
  BlockList Children;
  BlockQuoteNode() : BlockNode(NodeKind::NK_BlockQuote) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};

struct ThematicBreakNode : BlockNode {
  ThematicBreakNode() : BlockNode(NodeKind::NK_ThematicBreak) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};

struct DocumentNode : BlockNode {
  // FIXME: add constructor that accepts children once parser is in place
  BlockList Children;
  DocumentNode() : BlockNode(NodeKind::NK_Document) {}
  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  static bool classof(const BlockNode *N) {
    return N->Kind == NodeKind::NK_Document;
  }
};

//===----------------------------------------------------------------------===//
// ASTContext - owns the arena
//===----------------------------------------------------------------------===//

template <typename T>
using IsMarkdownNode = std::enable_if_t<std::is_base_of_v<InlineNode, T> ||
                                        std::is_base_of_v<BlockNode, T>>;

class ASTContext {
  llvm::BumpPtrAllocator Arena;
  DocumentNode *Root = nullptr;

public:
  ASTContext() = default;

  template <typename T, typename... Args, typename = IsMarkdownNode<T>>
  T *allocate(Args &&...args) {
    return new (Arena.Allocate<T>()) T(std::forward<Args>(args)...);
  }

  DocumentNode *getRoot() { return Root; }
  void setRoot(DocumentNode *R) { Root = R; }
};

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
