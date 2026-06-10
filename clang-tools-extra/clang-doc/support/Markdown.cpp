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

// Allocates a contiguous array of T in the arena and returns an ArrayRef.
template <typename T>
static ArrayRef<T> allocateArray(SmallVectorImpl<T> &Vec,
                                 BumpPtrAllocator &Arena) {
  if (Vec.empty())
    return {};
  T *Allocated = Arena.Allocate<T>(Vec.size());
  std::uninitialized_copy(Vec.begin(), Vec.end(), Allocated);
  return ArrayRef<T>(Allocated, Vec.size());
}

// Interns a StringRef into the arena so it outlives the parse loop.
static StringRef internString(StringRef S, BumpPtrAllocator &Arena) {
  if (S.empty())
    return {};
  char *Buf = Arena.Allocate<char>(S.size());
  std::copy(S.begin(), S.end(), Buf);
  return StringRef(Buf, S.size());
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

ArrayRef<MDNode *> parseMarkdown(StringRef ParagraphText,
                                 BumpPtrAllocator &Arena) {
  if (ParagraphText.trim().empty())
    return {};

  SmallVector<StringRef, 16> Lines;
  ParagraphText.split(Lines, '\n');

  SmallVector<MDNode *> Nodes;
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
      StringRef Lang = internString(Line.drop_front(3).trim(), Arena);
      SmallVector<StringRef> CodeLines;
      ++I;
      while (I < E) {
        StringRef CodeLine = Lines[I].trim();
        if (CodeLine.size() >= 3 &&
            all_of(CodeLine.take_front(3),
                   [Fence](char C) { return C == Fence; }))
          break;
        CodeLines.push_back(internString(Lines[I], Arena));
        ++I;
      }
      ++I; // skip closing fence
      auto *Code =
          new (Arena) FencedCodeNode(Lang, allocateArray(CodeLines, Arena));
      LDBG() << "emitting FencedCodeNode lang='" << Lang
             << "' lines=" << CodeLines.size();
      Nodes.push_back(Code);
      continue;
    }

    // Pipe table: current line has | and next line is a separator row.
    if (Line.contains('|') && I + 1 < E && isSepRow(Lines[I + 1].trim())) {
      SmallVector<StringRef> Rows;
      while (I < E && Lines[I].trim().contains('|')) {
        Rows.push_back(internString(Lines[I].trim(), Arena));
        ++I;
      }
      auto *Table = new (Arena) TableNode(allocateArray(Rows, Arena));
      LDBG() << "emitting TableNode rows=" << Rows.size();
      Nodes.push_back(Table);
      continue;
    }

    // Unordered list item.
    if (isListItem(Line)) {
      SmallVector<ListItemNode *> Items;
      while (I < E) {
        StringRef L = Lines[I].trim();
        if (!isListItem(L))
          break;
        StringRef ItemText = internString(L.drop_front(2).trim(), Arena);
        SmallVector<MDNode *> ItemChildren;
        ItemChildren.push_back(new (Arena) TextNode(ItemText));
        auto *Item =
            new (Arena) ListItemNode(allocateArray(ItemChildren, Arena));
        Items.push_back(Item);
        ++I;
      }
      auto *List = new (Arena) UnorderedListNode(allocateArray(Items, Arena));
      LDBG() << "emitting UnorderedListNode items=" << Items.size();
      Nodes.push_back(List);
      continue;
    }

    // Plain text fallback.
    Nodes.push_back(new (Arena) TextNode(internString(Line, Arena)));
    ++I;
  }

  LDBG() << "parseMarkdown done nodes=" << Nodes.size();
  return allocateArray(Nodes, Arena);
}

} // namespace clang::doc::markdown