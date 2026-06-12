//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Standalone Markdown parsing library for the LLVM ecosystem.
///
/// The parser takes a single paragraph of plain text and returns a list of
/// nodes describing the Markdown it found. Each kind of construct has its own
/// node type, and every node shares a common MDNode base, so you can use
/// llvm::isa<>/cast<>/dyn_cast<> to check what a node is.
///
/// Inline nodes (appear inside ParagraphNode, HeadingNode, etc.):
///   TextNode       -- plain text run
///   SoftBreakNode  -- soft line break
///   HardBreakNode  -- hard line break (trailing spaces or backslash)
///   InlineCodeNode -- inline code span (`code`)
///   EmphasisNode   -- emphasis (*text* or _text_)
///   StrongNode     -- strong emphasis (**text** or __text__)
///
/// Block nodes:
///   ParagraphNode     -- sequence of inline nodes
///   HeadingNode       -- ATX heading (# through ######), level 1-6
///   FencedCodeNode    -- fenced code block (``` or ~~~)
///   TableNode         -- pipe table (raw row text; TODO: structured cells)
///   UnorderedListNode -- bullet list (-, *, +)
///   OrderedListNode   -- numbered list with explicit start number
///   ListItemNode      -- single item inside a list
///   BlockQuoteNode    -- block quote (>)
///   ThematicBreakNode -- horizontal rule (---, ***, ___)
///
/// All nodes are arena-allocated. The caller owns the arena and must keep it
/// alive for the lifetime of any returned nodes. The parser never crashes on
/// malformed input; unrecognized text falls back to TextNode.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace clang::doc::markdown {

/// Discriminator for all Markdown AST nodes. Inline kinds are grouped before
/// block kinds so that the sentinels NK_LastInline and NK_FirstBlock enable
/// cheap range-based checks in classof() implementations.
enum class NodeKind {
  // Inline nodes
  NK_Text,
  NK_SoftBreak,
  NK_HardBreak,
  NK_InlineCode,
  NK_Emphasis,
  NK_Strong,
  NK_LastInline = NK_Strong, // sentinel -- all inline kinds are <= this

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
  NK_FirstBlock = NK_Paragraph, // sentinel -- all block kinds are >= this
};

/// Base type for all Markdown AST nodes. Carries only the kind discriminator.
/// Nodes are arena-allocated and have no virtual destructor; use
/// llvm::isa<>/cast<>/dyn_cast<> for type-safe downcasting.
struct MDNode {
  NodeKind Kind;
  explicit MDNode(NodeKind K) : Kind(K) {}
};

//===----------------------------------------------------------------------===//
// Inline nodes
//===----------------------------------------------------------------------===//

/// Plain text run.
struct TextNode : MDNode {
  llvm::StringRef Text;
  explicit TextNode(llvm::StringRef Text)
      : MDNode(NodeKind::NK_Text), Text(Text) {}
  static bool classof(const MDNode *N) { return N->Kind == NodeKind::NK_Text; }
};

/// Soft line break -- a newline that does not end the paragraph.
struct SoftBreakNode : MDNode {
  SoftBreakNode() : MDNode(NodeKind::NK_SoftBreak) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_SoftBreak;
  }
};

/// Hard line break -- two trailing spaces or a backslash before a newline.
struct HardBreakNode : MDNode {
  HardBreakNode() : MDNode(NodeKind::NK_HardBreak) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_HardBreak;
  }
};

/// Inline code span: `code`. Code does not include the surrounding backticks.
struct InlineCodeNode : MDNode {
  llvm::StringRef Code;
  explicit InlineCodeNode(llvm::StringRef Code)
      : MDNode(NodeKind::NK_InlineCode), Code(Code) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_InlineCode;
  }
};

/// Emphasized text: *text* or _text_.
struct EmphasisNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit EmphasisNode(llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_Emphasis), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Emphasis;
  }
};

/// Strongly emphasized text: **text** or __text__.
struct StrongNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit StrongNode(llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_Strong), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Strong;
  }
};

//===----------------------------------------------------------------------===//
// Block nodes
//===----------------------------------------------------------------------===//

/// A paragraph -- sequence of inline nodes separated from other blocks by
/// blank lines.
struct ParagraphNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit ParagraphNode(llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_Paragraph), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Paragraph;
  }
};

/// ATX heading: one to six leading # characters.
struct HeadingNode : MDNode {
  unsigned Level;                    // 1-6
  llvm::ArrayRef<MDNode *> Children; // inline content
  HeadingNode(unsigned Level, llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_Heading), Level(Level), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_Heading;
  }
};

/// Fenced code block opened with ``` or ~~~. Lang is the info string (e.g.
/// "cpp"); empty when no language was specified. Lines contains the raw text
/// of each interior line, without the opening or closing fence.
///
/// TODO: Follow CommonMark spec §4.5 -- the opening fence may be indented up
/// to 3 spaces; the closing fence must use the same character and be at least
/// as long as the opening fence; only spaces may follow the closing fence.
struct FencedCodeNode : MDNode {
  llvm::StringRef Lang;
  llvm::ArrayRef<llvm::StringRef> Lines;
  FencedCodeNode(llvm::StringRef Lang, llvm::ArrayRef<llvm::StringRef> Lines)
      : MDNode(NodeKind::NK_FencedCode), Lang(Lang), Lines(Lines) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_FencedCode;
  }
};

/// Pipe table. Rows contains the raw text of each row line including the
/// header and separator rows.
/// TODO: replace with a structured header/body/cell representation.
struct TableNode : MDNode {
  llvm::ArrayRef<llvm::StringRef> Rows;
  explicit TableNode(llvm::ArrayRef<llvm::StringRef> Rows)
      : MDNode(NodeKind::NK_Table), Rows(Rows) {}
  static bool classof(const MDNode *N) { return N->Kind == NodeKind::NK_Table; }
};

/// A single list item. Children may contain block-level nodes for loose
/// lists, or a single inline sequence for tight lists.
struct ListItemNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit ListItemNode(llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_ListItem), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_ListItem;
  }
};

/// Unordered (bullet) list. Markers are -, *, or +.
struct UnorderedListNode : MDNode {
  llvm::ArrayRef<ListItemNode *> Items;
  explicit UnorderedListNode(llvm::ArrayRef<ListItemNode *> Items)
      : MDNode(NodeKind::NK_UnorderedList), Items(Items) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_UnorderedList;
  }
};

/// Ordered (numbered) list. Start is the number on the first item.
struct OrderedListNode : MDNode {
  unsigned Start;
  llvm::ArrayRef<ListItemNode *> Items;
  OrderedListNode(unsigned Start, llvm::ArrayRef<ListItemNode *> Items)
      : MDNode(NodeKind::NK_OrderedList), Start(Start), Items(Items) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_OrderedList;
  }
};

/// Block quote (> ...). Children are block-level nodes inside the quote.
struct BlockQuoteNode : MDNode {
  llvm::ArrayRef<MDNode *> Children;
  explicit BlockQuoteNode(llvm::ArrayRef<MDNode *> Children)
      : MDNode(NodeKind::NK_BlockQuote), Children(Children) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_BlockQuote;
  }
};

/// Thematic break: a line of three or more ---, ***, or ___ characters.
struct ThematicBreakNode : MDNode {
  ThematicBreakNode() : MDNode(NodeKind::NK_ThematicBreak) {}
  static bool classof(const MDNode *N) {
    return N->Kind == NodeKind::NK_ThematicBreak;
  }
};

//===----------------------------------------------------------------------===//
// Parser entry point
//===----------------------------------------------------------------------===//

/// Parse Markdown from a single paragraph of plain text. Returns a list of
/// top-level block nodes allocated in Arena. Returns an empty ArrayRef if no
/// Markdown constructs are found, letting callers fall back to plain-text
/// rendering at zero cost. The parser never crashes on malformed input.
///
/// The caller must keep Arena alive for the lifetime of any returned nodes.
llvm::ArrayRef<MDNode *> parseMarkdown(llvm::StringRef ParagraphText,
                                       llvm::BumpPtrAllocator &Arena);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SUPPORT_MARKDOWN_H
