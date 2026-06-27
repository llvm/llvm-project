//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markdown.h"
#include "llvm/Support/Casting.h"

namespace clang::doc::markdown {

//===----------------------------------------------------------------------===//
// Inline node print/dump
//===----------------------------------------------------------------------===//

void InlineNode::print(llvm::raw_ostream &OS) const {
  switch (Kind) {
  case NodeKind::NK_Text:
    llvm::cast<TextNode>(this)->print(OS);
    break;
  case NodeKind::NK_InlineCode:
    llvm::cast<InlineCodeNode>(this)->print(OS);
    break;
  case NodeKind::NK_Emphasis:
    llvm::cast<EmphasisNode>(this)->print(OS);
    break;
  case NodeKind::NK_Strong:
    llvm::cast<StrongNode>(this)->print(OS);
    break;
  default:
    OS << "UnknownInlineNode\n";
    break;
  }
}

LLVM_DUMP_METHOD void InlineNode::dump() const { print(llvm::errs()); }

void TextNode::print(llvm::raw_ostream &OS) const {
  OS << "TextNode: " << getText() << "\n";
}

LLVM_DUMP_METHOD void TextNode::dump() const { print(llvm::errs()); }

void InlineCodeNode::print(llvm::raw_ostream &OS) const {
  OS << "InlineCodeNode: " << getCode() << "\n";
}

LLVM_DUMP_METHOD void InlineCodeNode::dump() const { print(llvm::errs()); }

void EmphasisNode::print(llvm::raw_ostream &OS) const {
  OS << "EmphasisNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void EmphasisNode::dump() const { print(llvm::errs()); }

void StrongNode::print(llvm::raw_ostream &OS) const {
  OS << "StrongNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void StrongNode::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// Block node print/dump
//===----------------------------------------------------------------------===//

void BlockNode::print(llvm::raw_ostream &OS) const {
  switch (Kind) {
  case NodeKind::NK_Paragraph:
    llvm::cast<ParagraphNode>(this)->print(OS);
    break;
  case NodeKind::NK_Heading:
    llvm::cast<HeadingNode>(this)->print(OS);
    break;
  case NodeKind::NK_FencedCode:
    llvm::cast<FencedCodeNode>(this)->print(OS);
    break;
  case NodeKind::NK_UnorderedList:
    llvm::cast<UnorderedListNode>(this)->print(OS);
    break;
  case NodeKind::NK_OrderedList:
    llvm::cast<OrderedListNode>(this)->print(OS);
    break;
  case NodeKind::NK_ListItem:
    llvm::cast<ListItemNode>(this)->print(OS);
    break;
  case NodeKind::NK_BlockQuote:
    llvm::cast<BlockQuoteNode>(this)->print(OS);
    break;
  case NodeKind::NK_ThematicBreak:
    llvm::cast<ThematicBreakNode>(this)->print(OS);
    break;
  case NodeKind::NK_Document:
    llvm::cast<DocumentNode>(this)->print(OS);
    break;
  default:
    OS << "UnknownBlockNode\n";
    break;
  }
}

LLVM_DUMP_METHOD void BlockNode::dump() const { print(llvm::errs()); }

void ParagraphNode::print(llvm::raw_ostream &OS) const {
  OS << "ParagraphNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void ParagraphNode::dump() const { print(llvm::errs()); }

void HeadingNode::print(llvm::raw_ostream &OS) const {
  OS << "HeadingNode: level=" << getLevel() << "\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void HeadingNode::dump() const { print(llvm::errs()); }

void FencedCodeNode::print(llvm::raw_ostream &OS) const {
  OS << "FencedCodeNode: lang=" << getLang() << "\n" << getCode() << "\n";
}

LLVM_DUMP_METHOD void FencedCodeNode::dump() const { print(llvm::errs()); }

void ListItemNode::print(llvm::raw_ostream &OS) const {
  OS << "ListItemNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void ListItemNode::dump() const { print(llvm::errs()); }

void UnorderedListNode::print(llvm::raw_ostream &OS) const {
  OS << "UnorderedListNode\n";
  for (const auto &Item : Items)
    Item.print(OS);
}

LLVM_DUMP_METHOD void UnorderedListNode::dump() const { print(llvm::errs()); }

void OrderedListNode::print(llvm::raw_ostream &OS) const {
  OS << "OrderedListNode: start=" << getStart() << "\n";
  for (const auto &Item : Items)
    Item.print(OS);
}

LLVM_DUMP_METHOD void OrderedListNode::dump() const { print(llvm::errs()); }

void BlockQuoteNode::print(llvm::raw_ostream &OS) const {
  OS << "BlockQuoteNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void BlockQuoteNode::dump() const { print(llvm::errs()); }

void ThematicBreakNode::print(llvm::raw_ostream &OS) const {
  OS << "ThematicBreakNode\n";
}

LLVM_DUMP_METHOD void ThematicBreakNode::dump() const { print(llvm::errs()); }

void DocumentNode::print(llvm::raw_ostream &OS) const {
  OS << "DocumentNode\n";
  for (const auto &Child : Children)
    Child.print(OS);
}

LLVM_DUMP_METHOD void DocumentNode::dump() const { print(llvm::errs()); }

} // namespace clang::doc::markdown
