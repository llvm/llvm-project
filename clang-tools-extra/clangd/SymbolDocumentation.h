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
#include "llvm/ADT/StringRef.h"
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

    preprocessDocumentation(Documentation);

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

    // If we have not seen a brief command, use the very first free paragraph as
    // the brief.
    if (!BriefParagraph && !FreeParagraphs.empty() &&
        FreeParagraphs.contains(0)) {
      BriefParagraph = FreeParagraphs.lookup(0);
      FreeParagraphs.erase(0);
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

  bool hasDetailedDoc() const {
    return !FreeParagraphs.empty() || !BlockCommands.empty();
  }

  /// Converts all unhandled comment commands to a markup document.
  void detailedDocToMarkup(markup::Document &Out) const;
  /// Converts the "brief" command(s) to a markup document.
  void briefToMarkup(markup::Paragraph &Out) const;
  /// Converts the "return" command(s) to a markup document.
  void returnToMarkup(markup::Paragraph &Out) const;
  /// Converts the "retval" command(s) to a markup document.
  void retvalsToMarkup(markup::Document &Out) const;

  void visitBlockCommandComment(const comments::BlockCommandComment *B);

  void templateTypeParmDocToMarkup(StringRef TemplateParamName,
                                   markup::Paragraph &Out) const;

  void templateTypeParmDocToString(StringRef TemplateParamName,
                                   llvm::raw_string_ostream &Out) const;

  void parameterDocToMarkup(StringRef ParamName, markup::Paragraph &Out) const;

  void parameterDocToString(StringRef ParamName,
                            llvm::raw_string_ostream &Out) const;

  void visitParagraphComment(const comments::ParagraphComment *P) {
    if (!P->isWhitespace()) {
      FreeParagraphs[CommentPartIndex] = P;
      CommentPartIndex++;
    }
  }

  void visitParamCommandComment(const comments::ParamCommandComment *P) {
    Parameters[P->getParamNameAsWritten()] = P;
  }

  void visitTParamCommandComment(const comments::TParamCommandComment *TP) {
    TemplateParameters[TP->getParamNameAsWritten()] = std::move(TP);
  }

  /// \brief Preprocesses the raw documentation string to prepare it for doxygen
  /// parsing.
  ///
  /// This is a workaround to provide better support for markdown in
  /// doxygen. Clang's doxygen parser e.g. does not handle markdown code blocks.
  ///
  /// The documentation string is preprocessed to replace some markdown
  /// constructs with parsable doxygen commands. E.g. markdown code blocks are
  /// replaced with doxygen \\code{.lang} ...
  /// \\endcode blocks.
  ///
  /// Additionally, potential doxygen commands inside markdown
  /// inline code spans are escaped to avoid that doxygen tries to interpret
  /// them as commands.
  ///
  /// \note Although this is a workaround, it is very similar to what
  /// doxygen itself does for markdown. In doxygen, the first parsing step is
  /// also a markdown preprocessing step.
  /// See https://www.doxygen.nl/manual/markdown.html
  void preprocessDocumentation(StringRef Doc);

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

  /// All the "retval" command(s)
  llvm::SmallVector<const comments::BlockCommandComment *> RetvalCommands;

  /// All the parsed doxygen block commands.
  /// They might have special handling internally like \\note or \\warning
  llvm::SmallDenseMap<unsigned, const comments::BlockCommandComment *>
      BlockCommands;

  /// Parsed paragaph(s) of the "param" comamnd(s)
  llvm::SmallDenseMap<StringRef, const comments::ParamCommandComment *>
      Parameters;

  /// Parsed paragaph(s) of the "tparam" comamnd(s)
  llvm::SmallDenseMap<StringRef, const comments::TParamCommandComment *>
      TemplateParameters;

  /// All "free" text paragraphs.
  llvm::SmallDenseMap<unsigned, const comments::ParagraphComment *>
      FreeParagraphs;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
