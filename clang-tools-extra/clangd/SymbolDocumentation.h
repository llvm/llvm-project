//===--- SymbolDocumentation.h ==---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class to parse doxygen comments into a flat structure for consumption
// in e.g. Hover and Code Completion
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H

#include "support/Markup.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/CommentParser.h"
#include "clang/AST/CommentSema.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace clang {
namespace clangd {

class SymbolDocCommentVisitor
    : public comments::ConstCommentVisitor<SymbolDocCommentVisitor> {
public:
  SymbolDocCommentVisitor(comments::FullComment *FC,
                          const CommentOptions &CommentOpts)
      : Traits(Allocator, CommentOpts), Allocator() {
    if (!FC)
      return;

    for (auto *Block : FC->getBlocks()) {
      visit(Block);
    }
  }

  SymbolDocCommentVisitor(llvm::StringRef Documentation,
                          const CommentOptions &CommentOpts)
      : Traits(Allocator, CommentOpts), Allocator() {

    if (Documentation.empty())
      return;

    CommentWithMarkers.reserve(Documentation.size() +
                               Documentation.count('\n') * 3);

    // The comment lexer expects doxygen markers, so add them back.
    // We need to use the /// style doxygen markers because the comment could
    // contain the closing the closing tag "*/" of a C Style "/** */" comment
    // which would break the parsing if we would just enclose the comment text
    // with "/** */".
    CommentWithMarkers = "///";
    bool NewLine = true;
    for (char C : Documentation) {
      if (C == '\n') {
        CommentWithMarkers += "\n///";
        NewLine = true;
      } else {
        if (NewLine && (C == '<')) {
          // A comment line starting with '///<' is treated as a doxygen
          // comment. Therefore add a space to separate the '<' from the comment
          // marker. This allows to parse html tags at the beginning of a line
          // and the escape marker prevents adding the artificial space in the
          // markup documentation. The extra space will not be rendered, since
          // we render it as markdown.
          CommentWithMarkers += ' ';
        }
        CommentWithMarkers += C;
        NewLine = false;
      }
    }
    SourceManagerForFile SourceMgrForFile("mock_file.cpp", CommentWithMarkers);

    SourceManager &SourceMgr = SourceMgrForFile.get();
    // The doxygen Sema requires a Diagostics consumer, since it reports
    // warnings e.g. when parameters are not documented correctly. These
    // warnings are not relevant for us, so we can ignore them.
    SourceMgr.getDiagnostics().setClient(new IgnoringDiagConsumer);

    comments::Sema S(Allocator, SourceMgr, SourceMgr.getDiagnostics(), Traits,
                     /*PP=*/nullptr);
    comments::Lexer L(Allocator, SourceMgr.getDiagnostics(), Traits,
                      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID()),
                      CommentWithMarkers.data(),
                      CommentWithMarkers.data() + CommentWithMarkers.size());
    comments::Parser P(L, S, Allocator, SourceMgr, SourceMgr.getDiagnostics(),
                       Traits);
    comments::FullComment *FC = P.parseFullComment();

    if (!FC)
      return;

    for (auto *Block : FC->getBlocks()) {
      visit(Block);
    }
  }

  bool isParameterDocumented(StringRef ParamName) const {
    return Parameters.contains(ParamName);
  }

  bool isTemplateTypeParmDocumented(StringRef ParamName) const {
    return TemplateParameters.contains(ParamName);
  }

  bool hasBriefCommand() const { return BriefParagraph; }

  bool hasReturnCommand() const { return ReturnParagraph; }

  bool hasRetvalCommands() const { return !RetvalParagraphs.empty(); }

  bool hasNoteCommands() const { return !NoteParagraphs.empty(); }

  bool hasWarningCommands() const { return !WarningParagraphs.empty(); }

  /// Converts all unhandled comment commands to a markup document.
  void docToMarkup(markup::Document &Out) const;
  /// Converts the "brief" command(s) to a markup document.
  void briefToMarkup(markup::Paragraph &Out) const;
  /// Converts the "return" command(s) to a markup document.
  void returnToMarkup(markup::Paragraph &Out) const;
  /// Converts the "note" command(s) to a markup document.
  void notesToMarkup(markup::Document &Out) const;
  /// Converts the "warning" command(s) to a markup document.
  void warningsToMarkup(markup::Document &Out) const;

  void visitBlockCommandComment(const comments::BlockCommandComment *B);

  void templateTypeParmDocToMarkup(StringRef TemplateParamName,
                                   markup::Paragraph &Out) const;

  void templateTypeParmDocToString(StringRef TemplateParamName,
                                   llvm::raw_string_ostream &Out) const;

  void parameterDocToMarkup(StringRef ParamName, markup::Paragraph &Out) const;

  void parameterDocToString(StringRef ParamName,
                            llvm::raw_string_ostream &Out) const;

  void visitParagraphComment(const comments::ParagraphComment *P) {
    FreeParagraphs[CommentPartIndex] = P;
    CommentPartIndex++;
  }

  void visitParamCommandComment(const comments::ParamCommandComment *P) {
    Parameters[P->getParamNameAsWritten()] = P;
  }

  void visitTParamCommandComment(const comments::TParamCommandComment *TP) {
    TemplateParameters[TP->getParamNameAsWritten()] = std::move(TP);
  }

private:
  comments::CommandTraits Traits;
  llvm::BumpPtrAllocator Allocator;
  std::string CommentWithMarkers;

  /// Index to keep track of the order of the comments.
  /// We want to rearange some commands like \\param.
  /// This index allows us to keep the order of the other comment parts.
  unsigned CommentPartIndex = 0;

  /// Paragraph of the "brief" command.
  const comments::ParagraphComment *BriefParagraph = nullptr;

  /// Paragraph of the "return" command.
  const comments::ParagraphComment *ReturnParagraph = nullptr;

  /// Paragraph(s) of the "note" command(s)
  llvm::SmallVector<const comments::ParagraphComment *> RetvalParagraphs;

  /// Paragraph(s) of the "note" command(s)
  llvm::SmallVector<const comments::ParagraphComment *> NoteParagraphs;

  /// Paragraph(s) of the "warning" command(s)
  llvm::SmallVector<const comments::ParagraphComment *> WarningParagraphs;

  /// All the paragraphs we don't have any special handling for,
  /// e.g. "details".
  llvm::SmallDenseMap<unsigned, const comments::BlockCommandComment *>
      UnhandledCommands;

  /// Parsed paragaph(s) of the "param" comamnd(s)
  llvm::SmallDenseMap<StringRef, const comments::ParamCommandComment *>
      Parameters;

  /// Parsed paragaph(s) of the "tparam" comamnd(s)
  llvm::SmallDenseMap<StringRef, const comments::TParamCommandComment *>
      TemplateParameters;

  /// All "free" text paragraphs.
  llvm::SmallDenseMap<unsigned, const comments::ParagraphComment *>
      FreeParagraphs;

  void paragraphsToMarkup(
      markup::Document &Out,
      const llvm::SmallVectorImpl<const comments::ParagraphComment *>
          &Paragraphs) const;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
