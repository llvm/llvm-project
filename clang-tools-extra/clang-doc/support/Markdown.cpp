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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "clang-doc-markdown"

namespace clang {
namespace doc {
namespace markdown {

static MDNode makeText(llvm::StringRef S) {
  return {NodeKind::NK_Text, S, {}};
}

// A line is a table separator if it only contains |, -, :, and spaces,
// and has at least one -.
static bool isSepRow(llvm::StringRef Line) {
  return Line.contains('-') &&
         Line.find_first_not_of("|-: ") == llvm::StringRef::npos;
}

static llvm::ArrayRef<MDNode>
allocateNodes(llvm::SmallVectorImpl<MDNode> &Nodes,
              llvm::BumpPtrAllocator &Arena) {
  if (Nodes.empty())
    return {};
  MDNode *Allocated = Arena.Allocate<MDNode>(Nodes.size());
  std::uninitialized_copy(Nodes.begin(), Nodes.end(), Allocated);
  return llvm::ArrayRef<MDNode>(Allocated, Nodes.size());
}

llvm::ArrayRef<MDNode> parseMarkdown(llvm::StringRef ParagraphText,
                                     llvm::BumpPtrAllocator &Arena) {
  if (ParagraphText.trim().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[md] empty input, returning nothing\n");
    return {};
  }

  llvm::SmallVector<llvm::StringRef, 16> Lines;
  ParagraphText.split(Lines, '\n');

  LLVM_DEBUG(llvm::dbgs() << "[md] parsing " << Lines.size() << " line(s)\n");

  llvm::SmallVector<MDNode, 8> Nodes;
  unsigned I = 0;

  while (I < Lines.size()) {
    llvm::StringRef Line = Lines[I].trim();

    if (Line.empty()) {
      ++I;
      continue;
    }

    // Fenced code block: ``` or ~~~
    if (Line.starts_with("```") || Line.starts_with("~~~")) {
      char Fence = Line[0];
      llvm::StringRef Lang = Line.drop_front(3).trim();
      LLVM_DEBUG(llvm::dbgs()
                 << "[md] fenced code block, lang='" << Lang << "'\n");
      llvm::SmallVector<MDNode, 4> CodeLines;
      ++I;
      while (I < Lines.size()) {
        llvm::StringRef CodeLine = Lines[I].trim();
        if (CodeLine.size() >= 3 &&
            llvm::all_of(CodeLine.take_front(3),
                         [Fence](char C) { return C == Fence; })) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[md] closing fence found at line " << I << "\n");
          break;
        }
        CodeLines.push_back(makeText(Lines[I]));
        ++I;
      }
      ++I; // skip closing fence
      MDNode Code;
      Code.Kind = NodeKind::NK_FencedCode;
      Code.Content = Lang;
      Code.Children = allocateNodes(CodeLines, Arena);
      LLVM_DEBUG(llvm::dbgs() << "[md] emitting NK_FencedCode with "
                              << CodeLines.size() << " line(s)\n");
      Nodes.push_back(Code);
      continue;
    }

    // Pipe table: current line has | and next line is a separator row
    if (Line.contains('|') && I + 1 < Lines.size() &&
        isSepRow(Lines[I + 1].trim())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[md] pipe table detected at line " << I << "\n");
      llvm::SmallVector<MDNode, 4> Rows;
      while (I < Lines.size() && Lines[I].trim().contains('|')) {
        Rows.push_back(makeText(Lines[I].trim()));
        ++I;
      }
      MDNode Table;
      Table.Kind = NodeKind::NK_Table;
      Table.Content = {};
      Table.Children = allocateNodes(Rows, Arena);
      LLVM_DEBUG(llvm::dbgs() << "[md] emitting NK_Table with " << Rows.size()
                              << " row(s)\n");
      Nodes.push_back(Table);
      continue;
    }

    // Unordered list item
    if (Line.starts_with("- ") || Line.starts_with("* ") ||
        Line.starts_with("+ ")) {
      LLVM_DEBUG(llvm::dbgs() << "[md] unordered list at line " << I << "\n");
      llvm::SmallVector<MDNode, 4> Items;
      while (I < Lines.size()) {
        llvm::StringRef L = Lines[I].trim();
        if (!L.starts_with("- ") && !L.starts_with("* ") &&
            !L.starts_with("+ "))
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
      LLVM_DEBUG(llvm::dbgs() << "[md] emitting NK_UnorderedList with "
                              << Items.size() << " item(s)\n");
      Nodes.push_back(List);
      continue;
    }

    // Plain text fallback
    LLVM_DEBUG(llvm::dbgs() << "[md] plain text: '" << Line << "'\n");
    Nodes.push_back(makeText(Line));
    ++I;
  }

  LLVM_DEBUG(llvm::dbgs() << "[md] done, " << Nodes.size()
                          << " top-level node(s)\n");
  return allocateNodes(Nodes, Arena);
}

} // namespace markdown
} // namespace doc
} // namespace clang