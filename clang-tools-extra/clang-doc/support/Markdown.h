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
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace clang {
namespace doc {
namespace markdown {

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

} // namespace markdown
} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_H