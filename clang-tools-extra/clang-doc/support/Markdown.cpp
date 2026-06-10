//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markdown.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "clang-doc"

using namespace llvm;

namespace clang::doc::markdown {

static MDNode makeText(StringRef S) {
  return {NodeKind::NK_Text, S, {}};
}

// A line is a table separator if it only contains |, -, :, and spaces,
// and has at least one -.
static bool isSepRow(StringRef Line) {
  return Line.contains('-') &&
         Line.find_first_not_of("|-: ") == StringRef::npos;
}

// Returns true if Line begins with a bullet list marker (-, *, or +)
// followed by a space.
static bool isListItem(StringRef Line) {
  return Line.starts_with("- ") || Line.starts_with("* ") ||
         Line.starts_with("+ ");
}

static ArrayRef<MDNode> allocateNodes(const SmallVectorImpl<MDNode> &Nodes,
                                      BumpPtrAllocator &Arena) {
  if (Nodes.empty())
    return {};
  MDNode *Allocated = Arena.Allocate<MDNode>(Nodes.size());
  std::uninitialized_copy(Nodes.begin(), Nodes.end(), Allocated);
  return ArrayRef<MDNode>(Allocated, Nodes.size());
}

ArrayRef<MDNode> parseMarkdown(StringRef ParagraphText,
                               BumpPtrAllocator &Arena) {
  if (ParagraphText.trim().empty())
    return {};

  SmallVector<StringRef, 16> Lines;
  ParagraphText.split(Lines, '\n');

  SmallVector<MDNode> Nodes;
  size_t I = 0, E = Lines.size();

  while (I < E) {
    StringRef Line = Lines[I].trim();

    if (Line.empty()) {
      ++I;
      continue;
    }

    // TODO: Follow CommonMark spec §4.5 more closely -- opening fences may be
    // indented up to 3 spaces, the closing fence must use the same character
    // and be at least as long as the opening fence, and the closing fence may
    // only be followed by spaces. Doxygen specifics should be handled on a
    // case-by-case basis.
    if (Line.starts_with("```") || Line.starts_with("~~~")) {
      char Fence = Line[0];
      StringRef Lang = Line.drop_front(3).trim();
      SmallVector<MDNode> CodeLines;
      ++I;
      while (I < E) {
        StringRef CodeLine = Lines[I].trim();
        if (CodeLine.size() >= 3 &&
            all_of(CodeLine.take_front(3),
                   [Fence](char C) { return C == Fence; }))
          break;
        CodeLines.push_back(makeText(Lines[I]));
        ++I;
      }
      ++I; // skip closing fence
      MDNode Code;
      Code.Kind = NodeKind::NK_FencedCode;
      Code.Content = Lang;
      Code.Children = allocateNodes(CodeLines, Arena);
      LDBG() << "emitting NK_FencedCode lang='" << Lang
             << "' lines=" << CodeLines.size();
      Nodes.push_back(Code);
      continue;
    }

    // Pipe table: current line has | and next line is a separator row.
    if (Line.contains('|') && I + 1 < E && isSepRow(Lines[I + 1].trim())) {
      SmallVector<MDNode> Rows;
      while (I < E && Lines[I].trim().contains('|')) {
        Rows.push_back(makeText(Lines[I].trim()));
        ++I;
      }
      MDNode Table;
      Table.Kind = NodeKind::NK_Table;
      Table.Content = {};
      Table.Children = allocateNodes(Rows, Arena);
      LDBG() << "emitting NK_Table rows=" << Rows.size();
      Nodes.push_back(Table);
      continue;
    }

    // Unordered list item.
    if (isListItem(Line)) {
      SmallVector<MDNode> Items;
      while (I < E) {
        StringRef L = Lines[I].trim();
        if (!isListItem(L))
          break;
        MDNode Item;
        Item.Kind = NodeKind::NK_ListItem;
        Item.Content = L.drop_front(2).trim();
        Item.Children = {};
        Items.push_back(Item);
        ++I;
      }
      MDNode List;
      List.Kind = NodeKind::NK_UnorderedList;
      List.Content = {};
      List.Children = allocateNodes(Items, Arena);
      LDBG() << "emitting NK_UnorderedList items=" << Items.size();
      Nodes.push_back(List);
      continue;
    }

    // Plain text fallback.
    Nodes.push_back(makeText(Line));
    ++I;
  }

  LDBG() << "parseMarkdown done nodes=" << Nodes.size();
  return allocateNodes(Nodes, Arena);
}

} // namespace clang::doc::markdown
