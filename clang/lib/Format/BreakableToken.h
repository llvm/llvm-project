//===--- BreakableToken.h - Format C++ code -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares BreakableToken, BreakableStringLiteral, BreakableComment,
/// BreakableBlockComment and BreakableLineCommentSection classes, that contain
/// token type-specific logic to break long lines in tokens and reflow content
/// between tokens.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_BREAKABLETOKEN_H
#define LLVM_CLANG_LIB_FORMAT_BREAKABLETOKEN_H

#include "Encoding.h"
#include "TokenAnnotator.h"
#include "WhitespaceManager.h"
#include "llvm/Support/Regex.h"
#include <utility>

namespace clang {
namespace format {

/// \brief Checks if \p Token switches formatting, like /* clang-format off */.
/// \p Token must be a comment.
bool switchesFormatting(const FormatToken &Token);

struct FormatStyle;

/// \brief Base class for strategies on how to break tokens.
///
/// This is organised around the concept of a \c Split, which is a whitespace
/// range that signifies a position of the content of a token where a
/// reformatting might be done. Operating with splits is divided into 3
/// operations:
/// - getSplit, for finding a split starting at a position,
/// - getLineLengthAfterSplit, for calculating the size in columns of the rest
///   of the content after a split has been used for breaking, and
/// - insertBreak, for executing the split using a whitespace manager.
///
/// There is a pair of operations that are used to compress a long whitespace
/// range with a single space if that will bring the line lenght under the
/// column limit:
/// - getLineLengthAfterCompression, for calculating the size in columns of the
///   line after a whitespace range has been compressed, and
/// - compressWhitespace, for executing the whitespace compression using a
///   whitespace manager; note that the compressed whitespace may be in the
///   middle of the original line and of the reformatted line.
///
/// For tokens where the whitespace before each line needs to be also
/// reformatted, for example for tokens supporting reflow, there are analogous
/// operations that might be executed before the main line breaking occurs:
/// - getSplitBefore, for finding a split such that the content preceding it
///   needs to be specially reflown,
/// - getLineLengthAfterSplitBefore, for calculating the line length in columns
///   of the remainder of the content after the beginning of the content has
///   been reformatted, and
/// - replaceWhitespaceBefore, for executing the reflow using a whitespace
///   manager.
///
/// FIXME: The interface seems set in stone, so we might want to just pull the
/// strategy into the class, instead of controlling it from the outside.
class BreakableToken {
public:
  /// \brief Contains starting character index and length of split.
  typedef std::pair<StringRef::size_type, unsigned> Split;

  virtual ~BreakableToken() {}

  /// \brief Returns the number of lines in this token in the original code.
  virtual unsigned getLineCount() const = 0;

  /// \brief Returns the number of columns required to format the piece of line
  /// at \p LineIndex, from byte offset \p TailOffset with length \p Length.
  ///
  /// Note that previous breaks are not taken into account. \p TailOffset is
  /// always specified from the start of the (original) line.
  /// \p Length can be set to StringRef::npos, which means "to the end of line".
  virtual unsigned
  getLineLengthAfterSplit(unsigned LineIndex, unsigned TailOffset,
                          StringRef::size_type Length) const = 0;

  /// \brief Returns a range (offset, length) at which to break the line at
  /// \p LineIndex, if previously broken at \p TailOffset. If possible, do not
  /// violate \p ColumnLimit.
  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const = 0;

  /// \brief Emits the previously retrieved \p Split via \p Whitespaces.
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           WhitespaceManager &Whitespaces) = 0;

  /// \brief Returns the number of columns required to format the piece of line
  /// at \p LineIndex, from byte offset \p TailOffset after the whitespace range
  /// \p Split has been compressed into a single space.
  unsigned getLineLengthAfterCompression(unsigned RemainingTokenColumns,
                                         Split Split) const;

  /// \brief Replaces the whitespace range described by \p Split with a single
  /// space.
  virtual void compressWhitespace(unsigned LineIndex, unsigned TailOffset,
                                  Split Split,
                                  WhitespaceManager &Whitespaces) = 0;

  /// \brief Returns a whitespace range (offset, length) of the content at
  /// \p LineIndex such that the content preceding this range needs to be
  /// reformatted before any breaks are made to this line.
  ///
  /// \p PreviousEndColumn is the end column of the previous line after
  /// formatting.
  ///
  /// A result having offset == StringRef::npos means that no piece of the line
  /// needs to be reformatted before any breaks are made.
  virtual Split getSplitBefore(unsigned LineIndex,
                               unsigned PreviousEndColumn,
                               unsigned ColumnLimit,
                               llvm::Regex& CommentPragmasRegex) const {
    return Split(StringRef::npos, 0);
  }

  /// \brief Returns the number of columns required to format the piece of line
  /// at \p LineIndex after the content preceding the whitespace range specified
  /// \p SplitBefore has been reformatted, but before any breaks are made to
  /// this line.
  virtual unsigned getLineLengthAfterSplitBefore(unsigned LineIndex,
                                         unsigned TailOffset,
                                         unsigned PreviousEndColumn,
                                         unsigned ColumnLimit,
                                         Split SplitBefore) const {
    return getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
  }

  /// \brief Replaces the whitespace between \p LineIndex-1 and \p LineIndex.
  /// Performs a reformatting of the content at \p LineIndex preceding the
  /// whitespace range \p SplitBefore.
  virtual void replaceWhitespaceBefore(unsigned LineIndex,
                                       unsigned PreviousEndColumn,
                                       unsigned ColumnLimit,
                                       Split SplitBefore,
                                       WhitespaceManager &Whitespaces) {}

  /// \brief Updates the next token of \p State to the next token after this
  /// one. This can be used when this token manages a set of underlying tokens
  /// as a unit and is responsible for the formatting of the them.
  virtual void updateNextToken(LineState &State) const {}

protected:
  BreakableToken(const FormatToken &Tok, bool InPPDirective,
                 encoding::Encoding Encoding, const FormatStyle &Style)
      : Tok(Tok), InPPDirective(InPPDirective), Encoding(Encoding),
        Style(Style) {}

  const FormatToken &Tok;
  const bool InPPDirective;
  const encoding::Encoding Encoding;
  const FormatStyle &Style;
};

/// \brief Base class for single line tokens that can be broken.
///
/// \c getSplit() needs to be implemented by child classes.
class BreakableSingleLineToken : public BreakableToken {
public:
  unsigned getLineCount() const override;
  unsigned getLineLengthAfterSplit(unsigned LineIndex, unsigned TailOffset,
                                   StringRef::size_type Length) const override;

protected:
  BreakableSingleLineToken(const FormatToken &Tok, unsigned StartColumn,
                           StringRef Prefix, StringRef Postfix,
                           bool InPPDirective, encoding::Encoding Encoding,
                           const FormatStyle &Style);

  // The column in which the token starts.
  unsigned StartColumn;
  // The prefix a line needs after a break in the token.
  StringRef Prefix;
  // The postfix a line needs before introducing a break.
  StringRef Postfix;
  // The token text excluding the prefix and postfix.
  StringRef Line;
};

class BreakableStringLiteral : public BreakableSingleLineToken {
public:
  /// \brief Creates a breakable token for a single line string literal.
  ///
  /// \p StartColumn specifies the column in which the token will start
  /// after formatting.
  BreakableStringLiteral(const FormatToken &Tok, unsigned StartColumn,
                         StringRef Prefix, StringRef Postfix,
                         bool InPPDirective, encoding::Encoding Encoding,
                         const FormatStyle &Style);

  Split getSplit(unsigned LineIndex, unsigned TailOffset,
                 unsigned ColumnLimit) const override;
  void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                   WhitespaceManager &Whitespaces) override;
  void compressWhitespace(unsigned LineIndex, unsigned TailOffset, Split Split,
                          WhitespaceManager &Whitespaces) override {}
};

class BreakableComment : public BreakableToken {
protected:
  /// \brief Creates a breakable token for a comment.
  ///
  /// \p StartColumn specifies the column in which the comment will start
  /// after formatting, while \p OriginalStartColumn specifies in which
  /// column the comment started before formatting.
  /// If the comment starts a line after formatting, set \p FirstInLine to true.
  BreakableComment(const FormatToken &Token, unsigned StartColumn,
                   unsigned OriginalStartColumn, bool FirstInLine,
                   bool InPPDirective, encoding::Encoding Encoding,
                   const FormatStyle &Style);

public:
  unsigned getLineCount() const override;
  Split getSplit(unsigned LineIndex, unsigned TailOffset,
                 unsigned ColumnLimit) const override;
  void compressWhitespace(unsigned LineIndex, unsigned TailOffset, Split Split,
                          WhitespaceManager &Whitespaces) override;

protected:
  virtual unsigned getContentStartColumn(unsigned LineIndex,
                                         unsigned TailOffset) const = 0;

  // Returns a split that divides Text into a left and right parts, such that
  // the left part is suitable for reflowing after PreviousEndColumn.
  Split getReflowSplit(StringRef Text, StringRef ReflowPrefix,
                       unsigned PreviousEndColumn, unsigned ColumnLimit) const;

  // Returns the token containing the line at LineIndex.
  const FormatToken &tokenAt(unsigned LineIndex) const;

  // Checks if the content of line LineIndex may be reflown with the previous
  // line.
  virtual bool mayReflow(unsigned LineIndex,
                         llvm::Regex &CommentPragmasRegex) const = 0;

  // Contains the original text of the lines of the block comment.
  //
  // In case of a block comments, excludes the leading /* in the first line and
  // trailing */ in the last line. In case of line comments, excludes the
  // leading // and spaces.
  SmallVector<StringRef, 16> Lines;

  // Contains the text of the lines excluding all leading and trailing
  // whitespace between the lines. Note that the decoration (if present) is also
  // not considered part of the text.
  SmallVector<StringRef, 16> Content;

  // Tokens[i] contains a reference to the token containing Lines[i] if the
  // whitespace range before that token is managed by this block.
  // Otherwise, Tokens[i] is a null pointer.
  SmallVector<FormatToken *, 16> Tokens;

  // ContentColumn[i] is the target column at which Content[i] should be.
  // Note that this excludes a leading "* " or "*" in case of block comments
  // where all lines have a "*" prefix, or the leading "// " or "//" in case of
  // line comments.
  //
  // In block comments, the first line's target column is always positive. The
  // remaining lines' target columns are relative to the first line to allow
  // correct indentation of comments in \c WhitespaceManager. Thus they can be
  // negative as well (in case the first line needs to be unindented more than
  // there's actual whitespace in another line).
  SmallVector<int, 16> ContentColumn;

  // The intended start column of the first line of text from this section.
  unsigned StartColumn;

  // The original start column of the first line of text from this section.
  unsigned OriginalStartColumn;

  // Whether the first token of this section is the first token in its unwrapped
  // line.
  bool FirstInLine;

  // The prefix to use in front a line that has been reflown up.
  // For example, when reflowing the second line after the first here:
  // // comment 1
  // // comment 2
  // we expect:
  // // comment 1 comment 2
  // and not:
  // // comment 1comment 2
  StringRef ReflowPrefix = " ";
};

class BreakableBlockComment : public BreakableComment {
public:
  BreakableBlockComment(const FormatToken &Token, unsigned StartColumn,
                        unsigned OriginalStartColumn, bool FirstInLine,
                        bool InPPDirective, encoding::Encoding Encoding,
                        const FormatStyle &Style);

  unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                   unsigned TailOffset,
                                   StringRef::size_type Length) const override;
  void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                   WhitespaceManager &Whitespaces) override;
  Split getSplitBefore(unsigned LineIndex, unsigned PreviousEndColumn,
                       unsigned ColumnLimit,
                       llvm::Regex &CommentPragmasRegex) const override;
  unsigned getLineLengthAfterSplitBefore(unsigned LineIndex,
                                         unsigned TailOffset,
                                         unsigned PreviousEndColumn,
                                         unsigned ColumnLimit,
                                         Split SplitBefore) const override;
  void replaceWhitespaceBefore(unsigned LineIndex, unsigned PreviousEndColumn,
                               unsigned ColumnLimit,
                               Split SplitBefore,
                               WhitespaceManager &Whitespaces) override;
  bool mayReflow(unsigned LineIndex,
                 llvm::Regex &CommentPragmasRegex) const override;

private:
  // Rearranges the whitespace between Lines[LineIndex-1] and Lines[LineIndex].
  //
  // Updates Content[LineIndex-1] and Content[LineIndex] by stripping off
  // leading and trailing whitespace.
  //
  // Sets ContentColumn to the intended column in which the text at
  // Lines[LineIndex] starts (note that the decoration, if present, is not
  // considered part of the text).
  void adjustWhitespace(unsigned LineIndex, int IndentDelta);

  // Computes the end column if the full Content from LineIndex gets reflown
  // after PreviousEndColumn.
  unsigned getReflownColumn(StringRef Content,
                            unsigned LineIndex,
                            unsigned PreviousEndColumn) const;

  unsigned getContentStartColumn(unsigned LineIndex,
                                 unsigned TailOffset) const override;

  // The column at which the text of a broken line should start.
  // Note that an optional decoration would go before that column.
  // IndentAtLineBreak is a uniform position for all lines in a block comment,
  // regardless of their relative position.
  // FIXME: Revisit the decision to do this; the main reason was to support
  // patterns like
  // /**************//**
  //  * Comment
  // We could also support such patterns by special casing the first line
  // instead.
  unsigned IndentAtLineBreak;

  // This is to distinguish between the case when the last line was empty and
  // the case when it started with a decoration ("*" or "* ").
  bool LastLineNeedsDecoration;

  // Either "* " if all lines begin with a "*", or empty.
  StringRef Decoration;
};

class BreakableLineCommentSection : public BreakableComment {
public:
  BreakableLineCommentSection(const FormatToken &Token, unsigned StartColumn,
                              unsigned OriginalStartColumn, bool FirstInLine,
                              bool InPPDirective, encoding::Encoding Encoding,
                              const FormatStyle &Style);

  unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                   unsigned TailOffset,
                                   StringRef::size_type Length) const override;
  void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                   WhitespaceManager &Whitespaces) override;
  Split getSplitBefore(unsigned LineIndex, unsigned PreviousEndColumn,
                       unsigned ColumnLimit,
                       llvm::Regex &CommentPragmasRegex) const override;
  unsigned getLineLengthAfterSplitBefore(unsigned LineIndex, unsigned TailOffset,
                                         unsigned PreviousEndColumn,
                                         unsigned ColumnLimit,
                                         Split SplitBefore) const override;
  void replaceWhitespaceBefore(unsigned LineIndex, unsigned PreviousEndColumn,
                               unsigned ColumnLimit, Split SplitBefore,
                               WhitespaceManager &Whitespaces) override;
  void updateNextToken(LineState& State) const override;
  bool mayReflow(unsigned LineIndex,
                 llvm::Regex &CommentPragmasRegex) const override;

private:
  unsigned getContentStartColumn(unsigned LineIndex,
                                 unsigned TailOffset) const override;

  // OriginalPrefix[i] contains the original prefix of line i, including
  // trailing whitespace before the start of the content. The indentation
  // preceding the prefix is not included.
  // For example, if the line is:
  // // content
  // then the original prefix is "// ".
  SmallVector<StringRef, 16> OriginalPrefix;

  // Prefix[i] contains the intended leading "//" with trailing spaces to
  // account for the indentation of content within the comment at line i after
  // formatting. It can be different than the original prefix when the original
  // line starts like this:
  // //content
  // Then the original prefix is "//", but the prefix is "// ".
  SmallVector<StringRef, 16> Prefix;

  SmallVector<unsigned, 16> OriginalContentColumn;

  /// \brief The token to which the last line of this breakable token belongs
  /// to; nullptr if that token is the initial token.
  ///
  /// The distinction is because if the token of the last line of this breakable
  /// token is distinct from the initial token, this breakable token owns the
  /// whitespace before the token of the last line, and the whitespace manager
  /// must be able to modify it.
  FormatToken *LastLineTok = nullptr;
};
} // namespace format
} // namespace clang

#endif
