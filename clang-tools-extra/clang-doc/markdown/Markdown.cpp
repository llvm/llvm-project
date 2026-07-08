//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markdown.h"

namespace clang::doc::markdown {

// TODO: print() currently outputs nodes flat with no indentation. Add
// S-expression style formatting (as used by the Swift AST printer) to make
// dumped trees easier to read.

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void InlineNode::dump() const { print(llvm::errs()); }
#endif

void TextNode::print(llvm::raw_ostream &OS) const {
  OS << "TextNode: " << getText() << "\n";
}

void InlineCodeNode::print(llvm::raw_ostream &OS) const {
  OS << "InlineCodeNode: " << getCode() << "\n";
}

void EmphasisNode::print(llvm::raw_ostream &OS) const {
  OS << "EmphasisNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

void StrongNode::print(llvm::raw_ostream &OS) const {
  OS << "StrongNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void BlockNode::dump() const { print(llvm::errs()); }
#endif

void ParagraphNode::print(llvm::raw_ostream &OS) const {
  OS << "ParagraphNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

void HeadingNode::print(llvm::raw_ostream &OS) const {
  OS << "HeadingNode: level=" << getLevel() << "\n";
  for (const auto &Child : children())
    Child.print(OS);
}

void FencedCodeNode::print(llvm::raw_ostream &OS) const {
  OS << "FencedCodeNode: lang=" << getLang() << "\n" << getCode() << "\n";
}

void ListItemNode::print(llvm::raw_ostream &OS) const {
  OS << "ListItemNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ListItemNode::dump() const { print(llvm::errs()); }
#endif

void UnorderedListNode::print(llvm::raw_ostream &OS) const {
  OS << "UnorderedListNode\n";
  for (const auto &Item : items())
    Item.print(OS);
}

void OrderedListNode::print(llvm::raw_ostream &OS) const {
  OS << "OrderedListNode: start=" << getStart() << "\n";
  for (const auto &Item : items())
    Item.print(OS);
}

void BlockQuoteNode::print(llvm::raw_ostream &OS) const {
  OS << "BlockQuoteNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

void ThematicBreakNode::print(llvm::raw_ostream &OS) const {
  OS << "ThematicBreakNode\n";
}

void DocumentNode::print(llvm::raw_ostream &OS) const {
  OS << "DocumentNode\n";
  for (const auto &Child : children())
    Child.print(OS);
}

} // namespace clang::doc::markdown
