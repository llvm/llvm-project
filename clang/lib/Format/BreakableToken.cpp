//===--- BreakableToken.cpp - Format C++ code -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains implementation of BreakableToken class and classes derived
/// from it.
///
//===----------------------------------------------------------------------===//

#include "BreakableToken.h"
#include "Comments.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "format-token-breaker"

namespace clang {
namespace format {

static const char *const Blanks = " \t\v\f\r";
static bool IsBlank(char C) {
  switch (C) {
  case ' ':
  case '\t':
  case '\v':
  case '\f':
  case '\r':
    return true;
  default:
    return false;
  }
}

static BreakableToken::Split getCommentSplit(StringRef Text,
                                             unsigned ContentStartColumn,
                                             unsigned ColumnLimit,
                                             unsigned TabWidth,
                                             encoding::Encoding Encoding) {
  if (ColumnLimit <= ContentStartColumn + 1)
    return BreakableToken::Split(StringRef::npos, 0);

  unsigned MaxSplit = ColumnLimit - ContentStartColumn + 1;
  unsigned MaxSplitBytes = 0;

  for (unsigned NumChars = 0;
       NumChars < MaxSplit && MaxSplitBytes < Text.size();) {
    unsigned BytesInChar =
        encoding::getCodePointNumBytes(Text[MaxSplitBytes], Encoding);
    NumChars +=
        encoding::columnWidthWithTabs(Text.substr(MaxSplitBytes, BytesInChar),
                                      ContentStartColumn, TabWidth, Encoding);
    MaxSplitBytes += BytesInChar;
  }

  StringRef::size_type SpaceOffset = Text.find_last_of(Blanks, MaxSplitBytes);
  if (SpaceOffset == StringRef::npos ||
      // Don't break at leading whitespace.
      Text.find_last_not_of(Blanks, SpaceOffset) == StringRef::npos) {
    // Make sure that we don't break at leading whitespace that
    // reaches past MaxSplit.
    StringRef::size_type FirstNonWhitespace = Text.find_first_not_of(Blanks);
    if (FirstNonWhitespace == StringRef::npos)
      // If the comment is only whitespace, we cannot split.
      return BreakableToken::Split(StringRef::npos, 0);
    SpaceOffset = Text.find_first_of(
        Blanks, std::max<unsigned>(MaxSplitBytes, FirstNonWhitespace));
  }
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0) {
    StringRef BeforeCut = Text.substr(0, SpaceOffset).rtrim(Blanks);
    StringRef AfterCut = Text.substr(SpaceOffset).ltrim(Blanks);
    return BreakableToken::Split(BeforeCut.size(),
                                 AfterCut.begin() - BeforeCut.end());
  }
  return BreakableToken::Split(StringRef::npos, 0);
}

static BreakableToken::Split
getStringSplit(StringRef Text, unsigned UsedColumns, unsigned ColumnLimit,
               unsigned TabWidth, encoding::Encoding Encoding) {
  // FIXME: Reduce unit test case.
  if (Text.empty())
    return BreakableToken::Split(StringRef::npos, 0);
  if (ColumnLimit <= UsedColumns)
    return BreakableToken::Split(StringRef::npos, 0);
  unsigned MaxSplit = ColumnLimit - UsedColumns;
  StringRef::size_type SpaceOffset = 0;
  StringRef::size_type SlashOffset = 0;
  StringRef::size_type WordStartOffset = 0;
  StringRef::size_type SplitPoint = 0;
  for (unsigned Chars = 0;;) {
    unsigned Advance;
    if (Text[0] == '\\') {
      Advance = encoding::getEscapeSequenceLength(Text);
      Chars += Advance;
    } else {
      Advance = encoding::getCodePointNumBytes(Text[0], Encoding);
      Chars += encoding::columnWidthWithTabs(
          Text.substr(0, Advance), UsedColumns + Chars, TabWidth, Encoding);
    }

    if (Chars > MaxSplit || Text.size() <= Advance)
      break;

    if (IsBlank(Text[0]))
      SpaceOffset = SplitPoint;
    if (Text[0] == '/')
      SlashOffset = SplitPoint;
    if (Advance == 1 && !isAlphanumeric(Text[0]))
      WordStartOffset = SplitPoint;

    SplitPoint += Advance;
    Text = Text.substr(Advance);
  }

  if (SpaceOffset != 0)
    return BreakableToken::Split(SpaceOffset + 1, 0);
  if (SlashOffset != 0)
    return BreakableToken::Split(SlashOffset + 1, 0);
  if (WordStartOffset != 0)
    return BreakableToken::Split(WordStartOffset + 1, 0);
  if (SplitPoint != 0)
    return BreakableToken::Split(SplitPoint, 0);
  return BreakableToken::Split(StringRef::npos, 0);
}

unsigned BreakableSingleLineToken::getLineCount() const { return 1; }

unsigned BreakableSingleLineToken::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned Offset, StringRef::size_type Length) const {
  return StartColumn + Prefix.size() + Postfix.size() +
         encoding::columnWidthWithTabs(Line.substr(Offset, Length),
                                       StartColumn + Prefix.size(),
                                       Style.TabWidth, Encoding);
}

BreakableSingleLineToken::BreakableSingleLineToken(
    const FormatToken &Tok, unsigned IndentLevel, unsigned StartColumn,
    StringRef Prefix, StringRef Postfix, bool InPPDirective,
    encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableToken(Tok, IndentLevel, InPPDirective, Encoding, Style),
      StartColumn(StartColumn), Prefix(Prefix), Postfix(Postfix) {
  assert(Tok.TokenText.endswith(Postfix));
  Line = Tok.TokenText.substr(
      Prefix.size(), Tok.TokenText.size() - Prefix.size() - Postfix.size());
}

BreakableStringLiteral::BreakableStringLiteral(
    const FormatToken &Tok, unsigned IndentLevel, unsigned StartColumn,
    StringRef Prefix, StringRef Postfix, bool InPPDirective,
    encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableSingleLineToken(Tok, IndentLevel, StartColumn, Prefix, Postfix,
                               InPPDirective, Encoding, Style) {}

BreakableToken::Split
BreakableStringLiteral::getSplit(unsigned LineIndex, unsigned TailOffset,
                                 unsigned ColumnLimit) const {
  return getStringSplit(Line.substr(TailOffset),
                        StartColumn + Prefix.size() + Postfix.size(),
                        ColumnLimit, Style.TabWidth, Encoding);
}

void BreakableStringLiteral::insertBreak(unsigned LineIndex,
                                         unsigned TailOffset, Split Split,
                                         WhitespaceManager &Whitespaces) {
  unsigned LeadingSpaces = StartColumn;
  // The '@' of an ObjC string literal (@"Test") does not become part of the
  // string token.
  // FIXME: It might be a cleaner solution to merge the tokens as a
  // precomputation step.
  if (Prefix.startswith("@"))
    --LeadingSpaces;
  Whitespaces.replaceWhitespaceInToken(
      Tok, Prefix.size() + TailOffset + Split.first, Split.second, Postfix,
      Prefix, InPPDirective, 1, IndentLevel, LeadingSpaces);
}

BreakableLineComment::BreakableLineComment(
    const FormatToken &Token, unsigned IndentLevel, unsigned StartColumn,
    bool InPPDirective, encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableSingleLineToken(Token, IndentLevel, StartColumn,
                               getLineCommentIndentPrefix(Token.TokenText), "",
                               InPPDirective, Encoding, Style) {
  OriginalPrefix = Prefix;
  if (Token.TokenText.size() > Prefix.size() &&
      isAlphanumeric(Token.TokenText[Prefix.size()])) {
    if (Prefix == "//")
      Prefix = "// ";
    else if (Prefix == "///")
      Prefix = "/// ";
    else if (Prefix == "//!")
      Prefix = "//! ";
  }
}

BreakableToken::Split
BreakableLineComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                               unsigned ColumnLimit) const {
  return getCommentSplit(Line.substr(TailOffset), StartColumn + Prefix.size(),
                         ColumnLimit, Style.TabWidth, Encoding);
}

void BreakableLineComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                       Split Split,
                                       WhitespaceManager &Whitespaces) {
  Whitespaces.replaceWhitespaceInToken(
      Tok, OriginalPrefix.size() + TailOffset + Split.first, Split.second,
      Postfix, Prefix, InPPDirective, /*Newlines=*/1, IndentLevel, StartColumn);
}

void BreakableLineComment::replaceWhitespace(unsigned LineIndex,
                                             unsigned TailOffset, Split Split,
                                             WhitespaceManager &Whitespaces) {
  Whitespaces.replaceWhitespaceInToken(
      Tok, OriginalPrefix.size() + TailOffset + Split.first, Split.second, "",
      "", /*InPPDirective=*/false, /*Newlines=*/0, /*IndentLevel=*/0,
      /*Spaces=*/1);
}

void BreakableLineComment::replaceWhitespaceBefore(
    unsigned LineIndex, WhitespaceManager &Whitespaces) {
  if (OriginalPrefix != Prefix) {
    Whitespaces.replaceWhitespaceInToken(Tok, OriginalPrefix.size(), 0, "", "",
                                         /*InPPDirective=*/false,
                                         /*Newlines=*/0, /*IndentLevel=*/0,
                                         /*Spaces=*/1);
  }
}

BreakableBlockComment::BreakableBlockComment(
    const FormatToken &Token, unsigned IndentLevel, unsigned StartColumn,
    unsigned OriginalStartColumn, bool FirstInLine, bool InPPDirective,
    encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableToken(Token, IndentLevel, InPPDirective, Encoding, Style) {
  StringRef TokenText(Token.TokenText);
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));
  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  int IndentDelta = StartColumn - OriginalStartColumn;
  LeadingWhitespace.resize(Lines.size());
  StartOfLineColumn.resize(Lines.size());
  StartOfLineColumn[0] = StartColumn + 2;
  for (size_t i = 1; i < Lines.size(); ++i)
    adjustWhitespace(i, IndentDelta);

  Decoration = "* ";
  if (Lines.size() == 1 && !FirstInLine) {
    // Comments for which FirstInLine is false can start on arbitrary column,
    // and available horizontal space can be too small to align consecutive
    // lines with the first one.
    // FIXME: We could, probably, align them to current indentation level, but
    // now we just wrap them without stars.
    Decoration = "";
  }
  for (size_t i = 1, e = Lines.size(); i < e && !Decoration.empty(); ++i) {
    // If the last line is empty, the closing "*/" will have a star.
    if (i + 1 == e && Lines[i].empty())
      break;
    if (!Lines[i].empty() && i + 1 != e && Decoration.startswith(Lines[i]))
      continue;
    while (!Lines[i].startswith(Decoration))
      Decoration = Decoration.substr(0, Decoration.size() - 1);
  }

  LastLineNeedsDecoration = true;
  IndentAtLineBreak = StartOfLineColumn[0] + 1;
  for (size_t i = 1; i < Lines.size(); ++i) {
    if (Lines[i].empty()) {
      if (i + 1 == Lines.size()) {
        // Empty last line means that we already have a star as a part of the
        // trailing */. We also need to preserve whitespace, so that */ is
        // correctly indented.
        LastLineNeedsDecoration = false;
      } else if (Decoration.empty()) {
        // For all other lines, set the start column to 0 if they're empty, so
        // we do not insert trailing whitespace anywhere.
        StartOfLineColumn[i] = 0;
      }
      continue;
    }

    // The first line already excludes the star.
    // For all other lines, adjust the line to exclude the star and
    // (optionally) the first whitespace.
    unsigned DecorationSize =
        Decoration.startswith(Lines[i]) ? Lines[i].size() : Decoration.size();
    StartOfLineColumn[i] += DecorationSize;
    Lines[i] = Lines[i].substr(DecorationSize);
    LeadingWhitespace[i] += DecorationSize;
    if (!Decoration.startswith(Lines[i]))
      IndentAtLineBreak =
          std::min<int>(IndentAtLineBreak, std::max(0, StartOfLineColumn[i]));
  }
  IndentAtLineBreak = std::max<unsigned>(IndentAtLineBreak, Decoration.size());
  DEBUG({
    llvm::dbgs() << "IndentAtLineBreak " << IndentAtLineBreak << "\n";
    for (size_t i = 0; i < Lines.size(); ++i) {
      llvm::dbgs() << i << " |" << Lines[i] << "| " << LeadingWhitespace[i]
                   << "\n";
    }
  });
}

void BreakableBlockComment::adjustWhitespace(unsigned LineIndex,
                                             int IndentDelta) {
  // When in a preprocessor directive, the trailing backslash in a block comment
  // is not needed, but can serve a purpose of uniformity with necessary escaped
  // newlines outside the comment. In this case we remove it here before
  // trimming the trailing whitespace. The backslash will be re-added later when
  // inserting a line break.
  size_t EndOfPreviousLine = Lines[LineIndex - 1].size();
  if (InPPDirective && Lines[LineIndex - 1].endswith("\\"))
    --EndOfPreviousLine;

  // Calculate the end of the non-whitespace text in the previous line.
  EndOfPreviousLine =
      Lines[LineIndex - 1].find_last_not_of(Blanks, EndOfPreviousLine);
  if (EndOfPreviousLine == StringRef::npos)
    EndOfPreviousLine = 0;
  else
    ++EndOfPreviousLine;
  // Calculate the start of the non-whitespace text in the current line.
  size_t StartOfLine = Lines[LineIndex].find_first_not_of(Blanks);
  if (StartOfLine == StringRef::npos)
    StartOfLine = Lines[LineIndex].rtrim("\r\n").size();

  StringRef Whitespace = Lines[LineIndex].substr(0, StartOfLine);
  // Adjust Lines to only contain relevant text.
  Lines[LineIndex - 1] = Lines[LineIndex - 1].substr(0, EndOfPreviousLine);
  Lines[LineIndex] = Lines[LineIndex].substr(StartOfLine);
  // Adjust LeadingWhitespace to account all whitespace between the lines
  // to the current line.
  LeadingWhitespace[LineIndex] =
      Lines[LineIndex].begin() - Lines[LineIndex - 1].end();

  // Adjust the start column uniformly across all lines.
  StartOfLineColumn[LineIndex] =
      encoding::columnWidthWithTabs(Whitespace, 0, Style.TabWidth, Encoding) +
      IndentDelta;
}

unsigned BreakableBlockComment::getLineCount() const { return Lines.size(); }

unsigned BreakableBlockComment::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned Offset, StringRef::size_type Length) const {
  unsigned ContentStartColumn = getContentStartColumn(LineIndex, Offset);
  return ContentStartColumn +
         encoding::columnWidthWithTabs(Lines[LineIndex].substr(Offset, Length),
                                       ContentStartColumn, Style.TabWidth,
                                       Encoding) +
         // The last line gets a "*/" postfix.
         (LineIndex + 1 == Lines.size() ? 2 : 0);
}

BreakableToken::Split
BreakableBlockComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                                unsigned ColumnLimit) const {
  return getCommentSplit(Lines[LineIndex].substr(TailOffset),
                         getContentStartColumn(LineIndex, TailOffset),
                         ColumnLimit, Style.TabWidth, Encoding);
}

void BreakableBlockComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                        Split Split,
                                        WhitespaceManager &Whitespaces) {
  StringRef Text = Lines[LineIndex].substr(TailOffset);
  StringRef Prefix = Decoration;
  if (LineIndex + 1 == Lines.size() &&
      Text.size() == Split.first + Split.second) {
    // For the last line we need to break before "*/", but not to add "* ".
    Prefix = "";
  }

  unsigned BreakOffsetInToken =
      Text.data() - Tok.TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  assert(IndentAtLineBreak >= Decoration.size());
  Whitespaces.replaceWhitespaceInToken(
      Tok, BreakOffsetInToken, CharsToRemove, "", Prefix, InPPDirective, 1,
      IndentLevel, IndentAtLineBreak - Decoration.size());
}

void BreakableBlockComment::replaceWhitespace(unsigned LineIndex,
                                              unsigned TailOffset, Split Split,
                                              WhitespaceManager &Whitespaces) {
  StringRef Text = Lines[LineIndex].substr(TailOffset);
  unsigned BreakOffsetInToken =
      Text.data() - Tok.TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  Whitespaces.replaceWhitespaceInToken(
      Tok, BreakOffsetInToken, CharsToRemove, "", "", /*InPPDirective=*/false,
      /*Newlines=*/0, /*IndentLevel=*/0, /*Spaces=*/1);
}

void BreakableBlockComment::replaceWhitespaceBefore(
    unsigned LineIndex, WhitespaceManager &Whitespaces) {
  if (LineIndex == 0)
    return;
  StringRef Prefix = Decoration;
  if (Lines[LineIndex].empty()) {
    if (LineIndex + 1 == Lines.size()) {
      if (!LastLineNeedsDecoration) {
        // If the last line was empty, we don't need a prefix, as the */ will
        // line up with the decoration (if it exists).
        Prefix = "";
      }
    } else if (!Decoration.empty()) {
      // For other empty lines, if we do have a decoration, adapt it to not
      // contain a trailing whitespace.
      Prefix = Prefix.substr(0, 1);
    }
  } else {
    if (StartOfLineColumn[LineIndex] == 1) {
      // This line starts immediately after the decorating *.
      Prefix = Prefix.substr(0, 1);
    }
  }

  unsigned WhitespaceOffsetInToken = Lines[LineIndex].data() -
                                     Tok.TokenText.data() -
                                     LeadingWhitespace[LineIndex];
  Whitespaces.replaceWhitespaceInToken(
      Tok, WhitespaceOffsetInToken, LeadingWhitespace[LineIndex], "", Prefix,
      InPPDirective, 1, IndentLevel,
      StartOfLineColumn[LineIndex] - Prefix.size());
}

unsigned
BreakableBlockComment::getContentStartColumn(unsigned LineIndex,
                                             unsigned TailOffset) const {
  // If we break, we always break at the predefined indent.
  if (TailOffset != 0)
    return IndentAtLineBreak;
  return std::max(0, StartOfLineColumn[LineIndex]);
}

} // namespace format
} // namespace clang
