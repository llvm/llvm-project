//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a standalone Markdown parsing library for the LLVM
/// ecosystem. The parser takes plain text and returns a tree of typed nodes
/// with no knowledge of comments, Doxygen, or Clang-Doc internals.
///
/// This is a simple Markdown parser for use inside Clang-Doc's comment
/// pipeline. You give it a paragraph of text and an arena allocator, and it
/// gives back a list of typed nodes describing the Markdown structure it found.
///
/// The main entry point is parseMarkdown(). If the text has no Markdown in it,
/// you get back an empty list and can fall back to plain-text output. If it
/// does, you get a tree of MDNode structs where each node has a kind, optional
/// content (like the language tag on a code fence), and optional children.
///
/// All nodes are allocated in the arena you pass in. You own the arena and are
/// responsible for keeping it alive as long as you use the nodes.
///
/// The parser handles fenced code blocks, pipe tables, and unordered lists.
/// Anything it does not recognize comes back as a plain text node. It will
/// never crash on bad input.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace clang::doc::markdown {

enum class NodeKind {
  // Block nodes
  NK_Paragraph,
  NK_FencedCode,
  NK_Table,
  NK_UnorderedList,
  NK_OrderedList,
  NK_ListItem,
  NK_ThematicBreak,
  // Inline nodes
  NK_Text,
  NK_InlineCode,
  NK_Emphasis,
  NK_Strong,
  NK_SoftBreak,
};

struct MDNode {
  NodeKind Kind;
  llvm::StringRef Content; // lang tag for FencedCode, leaf text for Text
  llvm::ArrayRef<MDNode> Children; // arena allocated
};

/// Parses Markdown from a single comment paragraph's text.
/// Returns an empty ArrayRef if no Markdown constructs are found,
/// so generators can fall back to plain-text rendering at zero cost.
llvm::ArrayRef<MDNode> parseMarkdown(llvm::StringRef ParagraphText,
                                     llvm::BumpPtrAllocator &Arena);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H