//===--- SymbolDocumentation.cpp ==-------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolDocumentation.h"

#include "support/Markup.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {
namespace {

std::string commandMarkerAsString(comments::CommandMarkerKind CommandMarker) {
  switch (CommandMarker) {
  case comments::CommandMarkerKind::CMK_At:
    return "@";
  case comments::CommandMarkerKind::CMK_Backslash:
    return "\\";
  }
  llvm_unreachable("Unknown command marker kind");
}

void commandToMarkup(markup::Paragraph &Out, StringRef Command,
                     comments::CommandMarkerKind CommandMarker,
                     StringRef Args) {
  Out.appendBoldText(commandMarkerAsString(CommandMarker) + Command.str());
  if (!Args.empty()) {
    Out.appendSpace();
    Out.appendEmphasizedText(Args.str());
  }
}
} // namespace

class ParagraphToMarkupDocument
    : public comments::ConstCommentVisitor<ParagraphToMarkupDocument> {
public:
  ParagraphToMarkupDocument(markup::Paragraph &Out,
                            const comments::CommandTraits &Traits)
      : Out(Out), Traits(Traits) {}

  void visitParagraphComment(const comments::ParagraphComment *C) {
    if (!C)
      return;

    for (const auto *Child = C->child_begin(); Child != C->child_end();
         ++Child) {
      visit(*Child);
    }
  }

  void visitTextComment(const comments::TextComment *C) {
    // Always trim leading space after a newline.
    StringRef Text = C->getText();
    if (LastChunkEndsWithNewline && C->getText().starts_with(' '))
      Text = Text.drop_front();

    LastChunkEndsWithNewline = C->hasTrailingNewline();
    Out.appendText(Text.str() + (LastChunkEndsWithNewline ? "\n" : ""));
  }

  void visitInlineCommandComment(const comments::InlineCommandComment *C) {

    if (C->getNumArgs() > 0) {
      std::string ArgText;
      for (unsigned I = 0; I < C->getNumArgs(); ++I) {
        if (!ArgText.empty())
          ArgText += " ";
        ArgText += C->getArgText(I);
      }

      switch (C->getRenderKind()) {
      case comments::InlineCommandRenderKind::Monospaced:
        Out.appendCode(ArgText);
        break;
      case comments::InlineCommandRenderKind::Bold:
        Out.appendBoldText(ArgText);
        break;
      case comments::InlineCommandRenderKind::Emphasized:
        Out.appendEmphasizedText(ArgText);
        break;
      default:
        commandToMarkup(Out, C->getCommandName(Traits), C->getCommandMarker(),
                        ArgText);
        break;
      }
    } else {
      if (C->getCommandName(Traits) == "n") {
        // \n is a special case, it is used to create a new line.
        Out.appendText("  \n");
        LastChunkEndsWithNewline = true;
        return;
      }

      commandToMarkup(Out, C->getCommandName(Traits), C->getCommandMarker(),
                      "");
    }
  }

  void visitHTMLStartTagComment(const comments::HTMLStartTagComment *STC) {
    std::string TagText = "<" + STC->getTagName().str();

    for (unsigned I = 0; I < STC->getNumAttrs(); ++I) {
      const comments::HTMLStartTagComment::Attribute &Attr = STC->getAttr(I);
      TagText += " " + Attr.Name.str() + "=\"" + Attr.Value.str() + "\"";
    }

    if (STC->isSelfClosing())
      TagText += " /";
    TagText += ">";

    LastChunkEndsWithNewline = STC->hasTrailingNewline();
    Out.appendText(TagText + (LastChunkEndsWithNewline ? "\n" : ""));
  }

  void visitHTMLEndTagComment(const comments::HTMLEndTagComment *ETC) {
    LastChunkEndsWithNewline = ETC->hasTrailingNewline();
    Out.appendText("</" + ETC->getTagName().str() + ">" +
                   (LastChunkEndsWithNewline ? "\n" : ""));
  }

private:
  markup::Paragraph &Out;
  const comments::CommandTraits &Traits;

  /// If true, the next leading space after a new line is trimmed.
  bool LastChunkEndsWithNewline = false;
};

class ParagraphToString
    : public comments::ConstCommentVisitor<ParagraphToString> {
public:
  ParagraphToString(llvm::raw_string_ostream &Out,
                    const comments::CommandTraits &Traits)
      : Out(Out), Traits(Traits) {}

  void visitParagraphComment(const comments::ParagraphComment *C) {
    if (!C)
      return;

    for (const auto *Child = C->child_begin(); Child != C->child_end();
         ++Child) {
      visit(*Child);
    }
  }

  void visitTextComment(const comments::TextComment *C) { Out << C->getText(); }

  void visitInlineCommandComment(const comments::InlineCommandComment *C) {
    Out << commandMarkerAsString(C->getCommandMarker());
    Out << C->getCommandName(Traits);
    if (C->getNumArgs() > 0) {
      for (unsigned I = 0; I < C->getNumArgs(); ++I)
        Out << " " << C->getArgText(I);
    }
    Out << " ";
  }

  void visitHTMLStartTagComment(const comments::HTMLStartTagComment *STC) {
    Out << "<" << STC->getTagName().str();

    for (unsigned I = 0; I < STC->getNumAttrs(); ++I) {
      const comments::HTMLStartTagComment::Attribute &Attr = STC->getAttr(I);
      Out << " " << Attr.Name.str();
      if (!Attr.Value.str().empty())
        Out << "=\"" << Attr.Value.str() << "\"";
    }

    if (STC->isSelfClosing())
      Out << " /";
    Out << ">";

    Out << (STC->hasTrailingNewline() ? "\n" : "");
  }

  void visitHTMLEndTagComment(const comments::HTMLEndTagComment *ETC) {
    Out << "</" << ETC->getTagName().str() << ">"
        << (ETC->hasTrailingNewline() ? "\n" : "");
  }

private:
  llvm::raw_string_ostream &Out;
  const comments::CommandTraits &Traits;
};

class BlockCommentToMarkupDocument
    : public comments::ConstCommentVisitor<BlockCommentToMarkupDocument> {
public:
  BlockCommentToMarkupDocument(markup::Document &Out,
                               const comments::CommandTraits &Traits)
      : Out(Out), Traits(Traits) {}

  void visitBlockCommandComment(const comments::BlockCommandComment *B) {

    switch (B->getCommandID()) {
    case comments::CommandTraits::KCI_arg:
    case comments::CommandTraits::KCI_li:
      // \li and \arg are special cases, they are used to create a list item.
      // In markdown it is a bullet list.
      ParagraphToMarkupDocument(Out.addBulletList().addItem().addParagraph(),
                                Traits)
          .visit(B->getParagraph());
      break;
    default: {
      // Some commands have arguments, like \throws.
      // The arguments are not part of the paragraph.
      // We need reconstruct them here.
      std::string ArgText;
      for (unsigned I = 0; I < B->getNumArgs(); ++I) {
        if (!ArgText.empty())
          ArgText += " ";
        ArgText += B->getArgText(I);
      }
      auto &P = Out.addParagraph();
      commandToMarkup(P, B->getCommandName(Traits), B->getCommandMarker(),
                      ArgText);
      if (B->getParagraph() && !B->getParagraph()->isWhitespace()) {
        // For commands with arguments, the paragraph starts after the first
        // space. Therefore we need to append a space manually in this case.
        if (!ArgText.empty())
          P.appendSpace();
        ParagraphToMarkupDocument(P, Traits).visit(B->getParagraph());
      }
    }
    }
  }

  void visitVerbatimBlockComment(const comments::VerbatimBlockComment *VB) {
    commandToMarkup(Out.addParagraph(), VB->getCommandName(Traits),
                    VB->getCommandMarker(), "");

    std::string VerbatimText;

    for (const auto *LI = VB->child_begin(); LI != VB->child_end(); ++LI) {
      if (const auto *Line = cast<comments::VerbatimBlockLineComment>(*LI)) {
        VerbatimText += Line->getText().str() + "\n";
      }
    }

    Out.addCodeBlock(VerbatimText, "");

    commandToMarkup(Out.addParagraph(), VB->getCloseName(),
                    VB->getCommandMarker(), "");
  }

  void visitVerbatimLineComment(const comments::VerbatimLineComment *VL) {
    auto &P = Out.addParagraph();
    commandToMarkup(P, VL->getCommandName(Traits), VL->getCommandMarker(), "");
    P.appendSpace().appendCode(VL->getText().str(), true).appendSpace();
  }

private:
  markup::Document &Out;
  const comments::CommandTraits &Traits;
  StringRef CommentEscapeMarker;
};

void SymbolDocCommentVisitor::parameterDocToMarkup(StringRef ParamName,
                                                   markup::Paragraph &Out) {
  if (ParamName.empty())
    return;

  if (const auto *P = Parameters.lookup(ParamName)) {
    ParagraphToMarkupDocument(Out, Traits).visit(P->getParagraph());
  }
}

void SymbolDocCommentVisitor::parameterDocToString(
    StringRef ParamName, llvm::raw_string_ostream &Out) {
  if (ParamName.empty())
    return;

  if (const auto *P = Parameters.lookup(ParamName)) {
    ParagraphToString(Out, Traits).visit(P->getParagraph());
  }
}

void SymbolDocCommentVisitor::docToMarkup(markup::Document &Out) {
  for (unsigned I = 0; I < CommentPartIndex; ++I) {
    if (const auto *BC = BlockCommands.lookup(I)) {
      BlockCommentToMarkupDocument(Out, Traits).visit(BC);
    } else if (const auto *P = FreeParagraphs.lookup(I)) {
      ParagraphToMarkupDocument(Out.addParagraph(), Traits).visit(P);
    }
  }
}

} // namespace clangd
} // namespace clang
