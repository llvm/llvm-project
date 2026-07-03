//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a libFuzzer harness for parseMarkdown(). It feeds
/// arbitrary bytes to the parser and recursively walks the returned node tree,
/// so every node and its children are exercised, not just the top-level nodes.
///
//===----------------------------------------------------------------------===//

#include "support/Markdown.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <cstdint>

using namespace clang::doc::markdown;

static void visit(const MDNode *Node);

static void visitAll(llvm::ArrayRef<MDNode *> Nodes) {
  for (const MDNode *N : Nodes)
    visit(N);
}

// Recurses into a node's children so the whole tree is walked.
static void visit(const MDNode *Node) {
  if (!Node)
    return;
  if (const auto *P = llvm::dyn_cast<ParagraphNode>(Node))
    visitAll(P->Children);
  else if (const auto *H = llvm::dyn_cast<HeadingNode>(Node))
    visitAll(H->Children);
  else if (const auto *E = llvm::dyn_cast<EmphasisNode>(Node))
    visitAll(E->Children);
  else if (const auto *S = llvm::dyn_cast<StrongNode>(Node))
    visitAll(S->Children);
  else if (const auto *Q = llvm::dyn_cast<BlockQuoteNode>(Node))
    visitAll(Q->Children);
  else if (const auto *LI = llvm::dyn_cast<ListItemNode>(Node))
    visitAll(LI->Children);
  else if (const auto *UL = llvm::dyn_cast<UnorderedListNode>(Node))
    for (const ListItemNode *Item : UL->Items)
      visit(Item);
  else if (const auto *OL = llvm::dyn_cast<OrderedListNode>(Node))
    for (const ListItemNode *Item : OL->Items)
      visit(Item);
  else if (const auto *T = llvm::dyn_cast<TableNode>(Node)) {
    for (const TableCell &Cell : T->Header.Cells)
      visitAll(Cell.Children);
    for (const TableRow &Row : T->Body)
      for (const TableCell &Cell : Row.Cells)
        visitAll(Cell.Children);
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  llvm::BumpPtrAllocator Arena;
  llvm::StringRef Input(reinterpret_cast<const char *>(Data), Size);
  for (const MDNode *Node : parseMarkdown(Input, Arena))
    visit(Node);
  return 0;
}
