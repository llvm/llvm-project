//===--- WhitespaceManager.h - Format C++ code ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WhitespaceManager class manages whitespace around tokens and their
/// replacements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_WHITESPACEMANAGER_H
#define LLVM_CLANG_LIB_FORMAT_WHITESPACEMANAGER_H

#include "TokenAnnotator.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include <string>

namespace clang {
namespace format {

/// \brief Manages the whitespaces around tokens and their replacements.
///
/// This includes special handling for certain constructs, e.g. the alignment of
/// trailing line comments.
///
/// To guarantee correctness of alignment operations, the \c WhitespaceManager
/// must be informed about every token in the source file; for each token, there
/// must be exactly one call to either \c replaceWhitespace or
/// \c addUntouchableToken.
///
/// There may be multiple calls to \c breakToken for a given token.
class WhitespaceManager {
public:
  WhitespaceManager(const SourceManager &SourceMgr, const FormatStyle &Style,
                    bool UseCRLF)
      : SourceMgr(SourceMgr), Style(Style), UseCRLF(UseCRLF) {}

  /// \brief Prepares the \c WhitespaceManager for another run.
  void reset();

  /// \brief Replaces the whitespace in front of \p Tok. Only call once for
  /// each \c AnnotatedToken.
  void replaceWhitespace(FormatToken &Tok, unsigned Newlines,
                         unsigned IndentLevel, unsigned Spaces,
                         unsigned StartOfTokenColumn,
                         bool InPPDirective = false);

  /// \brief Adds information about an unchangeable token's whitespace.
  ///
  /// Needs to be called for every token for which \c replaceWhitespace
  /// was not called.
  void addUntouchableToken(const FormatToken &Tok, bool InPPDirective);

  /// \brief Inserts or replaces whitespace in the middle of a token.
  ///
  /// Inserts \p PreviousPostfix, \p Newlines, \p Spaces and \p CurrentPrefix
  /// (in this order) at \p Offset inside \p Tok, replacing \p ReplaceChars
  /// characters.
  ///
  /// Note: \p Spaces can be negative to retain information about initial
  /// relative column offset between a line of a block comment and the start of
  /// the comment. This negative offset may be compensated by trailing comment
  /// alignment here. In all other cases negative \p Spaces will be truncated to
  /// 0.
  ///
  /// When \p InPPDirective is true, escaped newlines are inserted. \p Spaces is
  /// used to align backslashes correctly.
  void replaceWhitespaceInToken(const FormatToken &Tok, unsigned Offset,
                                unsigned ReplaceChars,
                                StringRef PreviousPostfix,
                                StringRef CurrentPrefix, bool InPPDirective,
                                unsigned Newlines, unsigned IndentLevel,
                                int Spaces);

  /// \brief Returns all the \c Replacements created during formatting.
  const tooling::Replacements &generateReplacements();

  /// \brief Represents a change before a token, a break inside a token,
  /// or the layout of an unchanged token (or whitespace within).
  struct Change {
    /// \brief Functor to sort changes in original source order.
    class IsBeforeInFile {
    public:
      IsBeforeInFile(const SourceManager &SourceMgr) : SourceMgr(SourceMgr) {}
      bool operator()(const Change &C1, const Change &C2) const;

    private:
      const SourceManager &SourceMgr;
    };

    Change() {}

    /// \brief Creates a \c Change.
    ///
    /// The generated \c Change will replace the characters at
    /// \p OriginalWhitespaceRange with a concatenation of
    /// \p PreviousLinePostfix, \p NewlinesBefore line breaks, \p Spaces spaces
    /// and \p CurrentLinePrefix.
    ///
    /// \p StartOfTokenColumn and \p InPPDirective will be used to lay out
    /// trailing comments and escaped newlines.
    Change(bool CreateReplacement, SourceRange OriginalWhitespaceRange,
           unsigned IndentLevel, int Spaces, unsigned StartOfTokenColumn,
           unsigned NewlinesBefore, StringRef PreviousLinePostfix,
           StringRef CurrentLinePrefix, tok::TokenKind Kind,
           bool ContinuesPPDirective, bool IsStartOfDeclName,
           bool IsInsideToken);

    bool CreateReplacement;
    // Changes might be in the middle of a token, so we cannot just keep the
    // FormatToken around to query its information.
    SourceRange OriginalWhitespaceRange;
    unsigned StartOfTokenColumn;
    unsigned NewlinesBefore;
    std::string PreviousLinePostfix;
    std::string CurrentLinePrefix;
    // The kind of the token whose whitespace this change replaces, or in which
    // this change inserts whitespace.
    // FIXME: Currently this is not set correctly for breaks inside comments, as
    // the \c BreakableToken is still doing its own alignment.
    tok::TokenKind Kind;
    bool ContinuesPPDirective;
    bool IsStartOfDeclName;

    // The number of nested blocks the token is in. This is used to add tabs
    // only for the indentation, and not for alignment, when
    // UseTab = US_ForIndentation.
    unsigned IndentLevel;

    // The number of spaces in front of the token or broken part of the token.
    // This will be adapted when aligning tokens.
    // Can be negative to retain information about the initial relative offset
    // of the lines in a block comment. This is used when aligning trailing
    // comments. Uncompensated negative offset is truncated to 0.
    int Spaces;

    // If this change is inside of a token but not at the start of the token or
    // directly after a newline.
    bool IsInsideToken;

    // \c IsTrailingComment, \c TokenLength, \c PreviousEndOfTokenColumn and
    // \c EscapedNewlineColumn will be calculated in
    // \c calculateLineBreakInformation.
    bool IsTrailingComment;
    unsigned TokenLength;
    unsigned PreviousEndOfTokenColumn;
    unsigned EscapedNewlineColumn;

    // These fields are used to retain correct relative line indentation in a
    // block comment when aligning trailing comments.
    //
    // If this Change represents a continuation of a block comment,
    // \c StartOfBlockComment is pointer to the first Change in the block
    // comment. \c IndentationOffset is a relative column offset to this
    // change, so that the correct column can be reconstructed at the end of
    // the alignment process.
    const Change *StartOfBlockComment;
    int IndentationOffset;
  };

private:
  /// \brief Calculate \c IsTrailingComment, \c TokenLength for the last tokens
  /// or token parts in a line and \c PreviousEndOfTokenColumn and
  /// \c EscapedNewlineColumn for the first tokens or token parts in a line.
  void calculateLineBreakInformation();

  /// \brief Align consecutive assignments over all \c Changes.
  void alignConsecutiveAssignments();

  /// \brief Align consecutive declarations over all \c Changes.
  void alignConsecutiveDeclarations();

  /// \brief Align trailing comments over all \c Changes.
  void alignTrailingComments();

  /// \brief Align trailing comments from change \p Start to change \p End at
  /// the specified \p Column.
  void alignTrailingComments(unsigned Start, unsigned End, unsigned Column);

  /// \brief Align escaped newlines over all \c Changes.
  void alignEscapedNewlines();

  /// \brief Align escaped newlines from change \p Start to change \p End at
  /// the specified \p Column.
  void alignEscapedNewlines(unsigned Start, unsigned End, unsigned Column);

  /// \brief Fill \c Replaces with the replacements for all effective changes.
  void generateChanges();

  /// \brief Stores \p Text as the replacement for the whitespace in \p Range.
  void storeReplacement(SourceRange Range, StringRef Text);
  void appendNewlineText(std::string &Text, unsigned Newlines);
  void appendNewlineText(std::string &Text, unsigned Newlines,
                         unsigned PreviousEndOfTokenColumn,
                         unsigned EscapedNewlineColumn);
  void appendIndentText(std::string &Text, unsigned IndentLevel,
                        unsigned Spaces, unsigned WhitespaceStartColumn);

  SmallVector<Change, 16> Changes;
  const SourceManager &SourceMgr;
  tooling::Replacements Replaces;
  const FormatStyle &Style;
  bool UseCRLF;
};

} // namespace format
} // namespace clang

#endif
