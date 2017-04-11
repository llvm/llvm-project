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
#include "ContinuationIndenter.h"
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

static StringRef getLineCommentIndentPrefix(StringRef Comment) {
  static const char *const KnownPrefixes[] = {"///", "//", "//!"};
  StringRef LongestPrefix;
  for (StringRef KnownPrefix : KnownPrefixes) {
    if (Comment.startswith(KnownPrefix)) {
      size_t PrefixLength = KnownPrefix.size();
      while (PrefixLength < Comment.size() && Comment[PrefixLength] == ' ')
        ++PrefixLength;
      if (PrefixLength > LongestPrefix.size())
        LongestPrefix = Comment.substr(0, PrefixLength);
    }
  }
  return LongestPrefix;
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

bool switchesFormatting(const FormatToken &Token) {
  assert((Token.is(TT_BlockComment) || Token.is(TT_LineComment)) &&
         "formatting regions are switched by comment tokens");
  StringRef Content = Token.TokenText.substr(2).ltrim();
  return Content.startswith("clang-format on") ||
         Content.startswith("clang-format off");
}

unsigned
BreakableToken::getLineLengthAfterCompression(unsigned RemainingTokenColumns,
                                              Split Split) const {
  // Example: consider the content
  // lala  lala
  // - RemainingTokenColumns is the original number of columns, 10;
  // - Split is (4, 2), denoting the two spaces between the two words;
  //
  // We compute the number of columns when the split is compressed into a single
  // space, like:
  // lala lala
  return RemainingTokenColumns + 1 - Split.second;
}

unsigned BreakableSingleLineToken::getLineCount() const { return 1; }

unsigned BreakableSingleLineToken::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned TailOffset,
    StringRef::size_type Length) const {
  return StartColumn + Prefix.size() + Postfix.size() +
         encoding::columnWidthWithTabs(Line.substr(TailOffset, Length),
                                       StartColumn + Prefix.size(),
                                       Style.TabWidth, Encoding);
}

BreakableSingleLineToken::BreakableSingleLineToken(
    const FormatToken &Tok, unsigned StartColumn, StringRef Prefix,
    StringRef Postfix, bool InPPDirective, encoding::Encoding Encoding,
    const FormatStyle &Style)
    : BreakableToken(Tok, InPPDirective, Encoding, Style),
      StartColumn(StartColumn), Prefix(Prefix), Postfix(Postfix) {
  assert(Tok.TokenText.startswith(Prefix) && Tok.TokenText.endswith(Postfix));
  Line = Tok.TokenText.substr(
      Prefix.size(), Tok.TokenText.size() - Prefix.size() - Postfix.size());
}

BreakableStringLiteral::BreakableStringLiteral(
    const FormatToken &Tok, unsigned StartColumn, StringRef Prefix,
    StringRef Postfix, bool InPPDirective, encoding::Encoding Encoding,
    const FormatStyle &Style)
    : BreakableSingleLineToken(Tok, StartColumn, Prefix, Postfix, InPPDirective,
                               Encoding, Style) {}

BreakableToken::Split
BreakableStringLiteral::getSplit(unsigned LineIndex, unsigned TailOffset,
                                 unsigned ColumnLimit,
                                 llvm::Regex &CommentPragmasRegex) const {
  return getStringSplit(Line.substr(TailOffset),
                        StartColumn + Prefix.size() + Postfix.size(),
                        ColumnLimit, Style.TabWidth, Encoding);
}

void BreakableStringLiteral::insertBreak(unsigned LineIndex,
                                         unsigned TailOffset, Split Split,
                                         WhitespaceManager &Whitespaces) {
  Whitespaces.replaceWhitespaceInToken(
      Tok, Prefix.size() + TailOffset + Split.first, Split.second, Postfix,
      Prefix, InPPDirective, 1, StartColumn);
}

BreakableComment::BreakableComment(const FormatToken &Token,
                                   unsigned StartColumn,
                                   bool InPPDirective,
                                   encoding::Encoding Encoding,
                                   const FormatStyle &Style)
    : BreakableToken(Token, InPPDirective, Encoding, Style),
      StartColumn(StartColumn) {}

unsigned BreakableComment::getLineCount() const { return Lines.size(); }

BreakableToken::Split
BreakableComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                           unsigned ColumnLimit,
                           llvm::Regex &CommentPragmasRegex) const {
  // Don't break lines matching the comment pragmas regex.
  if (CommentPragmasRegex.match(Content[LineIndex]))
    return Split(StringRef::npos, 0);
  return getCommentSplit(Content[LineIndex].substr(TailOffset),
                         getContentStartColumn(LineIndex, TailOffset),
                         ColumnLimit, Style.TabWidth, Encoding);
}

void BreakableComment::compressWhitespace(unsigned LineIndex,
                                          unsigned TailOffset, Split Split,
                                          WhitespaceManager &Whitespaces) {
  StringRef Text = Content[LineIndex].substr(TailOffset);
  // Text is relative to the content line, but Whitespaces operates relative to
  // the start of the corresponding token, so compute the start of the Split
  // that needs to be compressed into a single space relative to the start of
  // its token.
  unsigned BreakOffsetInToken =
      Text.data() - tokenAt(LineIndex).TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  Whitespaces.replaceWhitespaceInToken(
      tokenAt(LineIndex), BreakOffsetInToken, CharsToRemove, "", "",
      /*InPPDirective=*/false, /*Newlines=*/0, /*Spaces=*/1);
}

BreakableToken::Split
BreakableComment::getReflowSplit(StringRef Text, StringRef ReflowPrefix,
                                 unsigned PreviousEndColumn,
                                 unsigned ColumnLimit) const {
  unsigned ReflowStartColumn = PreviousEndColumn + ReflowPrefix.size();
  StringRef TrimmedText = Text.rtrim(Blanks);
  // This is the width of the resulting line in case the full line of Text gets
  // reflown up starting at ReflowStartColumn.
  unsigned FullWidth = ReflowStartColumn + encoding::columnWidthWithTabs(
                                               TrimmedText, ReflowStartColumn,
                                               Style.TabWidth, Encoding);
  // If the full line fits up, we return a reflow split after it,
  // otherwise we compute the largest piece of text that fits after
  // ReflowStartColumn.
  Split ReflowSplit =
      FullWidth <= ColumnLimit
          ? Split(TrimmedText.size(), Text.size() - TrimmedText.size())
          : getCommentSplit(Text, ReflowStartColumn, ColumnLimit,
                            Style.TabWidth, Encoding);

  // We need to be extra careful here, because while it's OK to keep a long line
  // if it can't be broken into smaller pieces (like when the first word of a
  // long line is longer than the column limit), it's not OK to reflow that long
  // word up. So we recompute the size of the previous line after reflowing and
  // only return the reflow split if that's under the line limit.
  if (ReflowSplit.first != StringRef::npos &&
      // Check if the width of the newly reflown line is under the limit.
      PreviousEndColumn + ReflowPrefix.size() +
              encoding::columnWidthWithTabs(Text.substr(0, ReflowSplit.first),
                                            PreviousEndColumn +
                                                ReflowPrefix.size(),
                                            Style.TabWidth, Encoding) <=
          ColumnLimit) {
    return ReflowSplit;
  }
  return Split(StringRef::npos, 0);
}

const FormatToken &BreakableComment::tokenAt(unsigned LineIndex) const {
  return Tokens[LineIndex] ? *Tokens[LineIndex] : Tok;
}

static bool mayReflowContent(StringRef Content) {
  Content = Content.trim(Blanks);
  // Lines starting with '@' commonly have special meaning.
  static const SmallVector<StringRef, 4> kSpecialMeaningPrefixes = {
      "@", "TODO", "FIXME", "XXX"};
  bool hasSpecialMeaningPrefix = false;
  for (StringRef Prefix : kSpecialMeaningPrefixes) {
    if (Content.startswith(Prefix)) {
      hasSpecialMeaningPrefix = true;
      break;
    }
  }
  // Simple heuristic for what to reflow: content should contain at least two
  // characters and either the first or second character must be
  // non-punctuation.
  return Content.size() >= 2 && !hasSpecialMeaningPrefix &&
         !Content.endswith("\\") &&
         // Note that this is UTF-8 safe, since if isPunctuation(Content[0]) is
         // true, then the first code point must be 1 byte long.
         (!isPunctuation(Content[0]) || !isPunctuation(Content[1]));
}

BreakableBlockComment::BreakableBlockComment(
    const FormatToken &Token, unsigned StartColumn,
    unsigned OriginalStartColumn, bool FirstInLine, bool InPPDirective,
    encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableComment(Token, StartColumn, InPPDirective, Encoding, Style) {
  assert(Tok.is(TT_BlockComment) &&
         "block comment section must start with a block comment");

  StringRef TokenText(Tok.TokenText);
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));
  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  int IndentDelta = StartColumn - OriginalStartColumn;
  Content.resize(Lines.size());
  Content[0] = Lines[0];
  ContentColumn.resize(Lines.size());
  // Account for the initial '/*'.
  ContentColumn[0] = StartColumn + 2;
  Tokens.resize(Lines.size());
  for (size_t i = 1; i < Lines.size(); ++i)
    adjustWhitespace(i, IndentDelta);

  // Align decorations with the column of the star on the first line,
  // that is one column after the start "/*".
  DecorationColumn = StartColumn + 1;

  // Account for comment decoration patterns like this:
  //
  // /*
  // ** blah blah blah
  // */
  if (Lines.size() >= 2 && Content[1].startswith("**") &&
      static_cast<unsigned>(ContentColumn[1]) == StartColumn) {
    DecorationColumn = StartColumn;
  }

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
    if (i + 1 == e && Content[i].empty())
      break;
    if (!Content[i].empty() && i + 1 != e &&
        Decoration.startswith(Content[i]))
      continue;
    while (!Content[i].startswith(Decoration))
      Decoration = Decoration.substr(0, Decoration.size() - 1);
  }

  LastLineNeedsDecoration = true;
  IndentAtLineBreak = ContentColumn[0] + 1;
  for (size_t i = 1, e = Lines.size(); i < e; ++i) {
    if (Content[i].empty()) {
      if (i + 1 == e) {
        // Empty last line means that we already have a star as a part of the
        // trailing */. We also need to preserve whitespace, so that */ is
        // correctly indented.
        LastLineNeedsDecoration = false;
        // Align the star in the last '*/' with the stars on the previous lines.
        if (e >= 2 && !Decoration.empty()) {
          ContentColumn[i] = DecorationColumn;
        }
      } else if (Decoration.empty()) {
        // For all other lines, set the start column to 0 if they're empty, so
        // we do not insert trailing whitespace anywhere.
        ContentColumn[i] = 0;
      }
      continue;
    }

    // The first line already excludes the star.
    // The last line excludes the star if LastLineNeedsDecoration is false.
    // For all other lines, adjust the line to exclude the star and
    // (optionally) the first whitespace.
    unsigned DecorationSize = Decoration.startswith(Content[i])
                                  ? Content[i].size()
                                  : Decoration.size();
    if (DecorationSize) {
      ContentColumn[i] = DecorationColumn + DecorationSize;
    }
    Content[i] = Content[i].substr(DecorationSize);
    if (!Decoration.startswith(Content[i]))
      IndentAtLineBreak =
          std::min<int>(IndentAtLineBreak, std::max(0, ContentColumn[i]));
  }
  IndentAtLineBreak =
      std::max<unsigned>(IndentAtLineBreak, Decoration.size());

  DEBUG({
    llvm::dbgs() << "IndentAtLineBreak " << IndentAtLineBreak << "\n";
    for (size_t i = 0; i < Lines.size(); ++i) {
      llvm::dbgs() << i << " |" << Content[i] << "| "
                   << "CC=" << ContentColumn[i] << "| "
                   << "IN=" << (Content[i].data() - Lines[i].data()) << "\n";
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
  size_t PreviousContentOffset =
      Content[LineIndex - 1].data() - Lines[LineIndex - 1].data();
  Content[LineIndex - 1] = Lines[LineIndex - 1].substr(
      PreviousContentOffset, EndOfPreviousLine - PreviousContentOffset);
  Content[LineIndex] = Lines[LineIndex].substr(StartOfLine);

  // Adjust the start column uniformly across all lines.
  ContentColumn[LineIndex] =
      encoding::columnWidthWithTabs(Whitespace, 0, Style.TabWidth, Encoding) +
      IndentDelta;
}

unsigned BreakableBlockComment::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned TailOffset,
    StringRef::size_type Length) const {
  unsigned ContentStartColumn = getContentStartColumn(LineIndex, TailOffset);
  unsigned LineLength =
      ContentStartColumn + encoding::columnWidthWithTabs(
                               Content[LineIndex].substr(TailOffset, Length),
                               ContentStartColumn, Style.TabWidth, Encoding);
  // The last line gets a "*/" postfix.
  if (LineIndex + 1 == Lines.size()) {
    LineLength += 2;
    // We never need a decoration when breaking just the trailing "*/" postfix.
    // Note that checking that Length == 0 is not enough, since Length could
    // also be StringRef::npos.
    if (Content[LineIndex].substr(TailOffset, Length).empty()) {
      LineLength -= Decoration.size();
    }
  }
  return LineLength;
}

void BreakableBlockComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                        Split Split,
                                        WhitespaceManager &Whitespaces) {
  StringRef Text = Content[LineIndex].substr(TailOffset);
  StringRef Prefix = Decoration;
  // We need this to account for the case when we have a decoration "* " for all
  // the lines except for the last one, where the star in "*/" acts as a
  // decoration.
  unsigned LocalIndentAtLineBreak = IndentAtLineBreak;
  if (LineIndex + 1 == Lines.size() &&
      Text.size() == Split.first + Split.second) {
    // For the last line we need to break before "*/", but not to add "* ".
    Prefix = "";
    if (LocalIndentAtLineBreak >= 2)
      LocalIndentAtLineBreak -= 2;
  }
  // The split offset is from the beginning of the line. Convert it to an offset
  // from the beginning of the token text.
  unsigned BreakOffsetInToken =
      Text.data() - tokenAt(LineIndex).TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  assert(LocalIndentAtLineBreak >= Prefix.size());
  Whitespaces.replaceWhitespaceInToken(
      tokenAt(LineIndex), BreakOffsetInToken, CharsToRemove, "", Prefix,
      InPPDirective, /*Newlines=*/1,
      /*Spaces=*/LocalIndentAtLineBreak - Prefix.size());
}

BreakableToken::Split BreakableBlockComment::getSplitBefore(
    unsigned LineIndex,
    unsigned PreviousEndColumn,
    unsigned ColumnLimit,
    llvm::Regex &CommentPragmasRegex) const {
  if (!mayReflow(LineIndex, CommentPragmasRegex))
    return Split(StringRef::npos, 0);
  StringRef TrimmedContent = Content[LineIndex].ltrim(Blanks);
  return getReflowSplit(TrimmedContent, ReflowPrefix, PreviousEndColumn,
                        ColumnLimit);
}

unsigned BreakableBlockComment::getReflownColumn(
    StringRef Content,
    unsigned LineIndex,
    unsigned PreviousEndColumn) const {
    unsigned StartColumn = PreviousEndColumn + ReflowPrefix.size();
    // If this is the last line, it will carry around its '*/' postfix.
    unsigned PostfixLength = (LineIndex + 1 == Lines.size() ? 2 : 0);
    // The line is composed of previous text, reflow prefix, reflown text and
    // postfix.
    unsigned ReflownColumn =
        StartColumn + encoding::columnWidthWithTabs(Content, StartColumn,
                                                    Style.TabWidth, Encoding) +
        PostfixLength;
    return ReflownColumn;
}

unsigned BreakableBlockComment::getLineLengthAfterSplitBefore(
    unsigned LineIndex, unsigned TailOffset,
    unsigned PreviousEndColumn,
    unsigned ColumnLimit,
    Split SplitBefore) const {
  if (SplitBefore.first == StringRef::npos ||
      // Block comment line contents contain the trailing whitespace after the
      // decoration, so the need of left trim. Note that this behavior is
      // consistent with the breaking of block comments where the indentation of
      // a broken line is uniform across all the lines of the block comment.
      SplitBefore.first + SplitBefore.second <
          Content[LineIndex].ltrim().size()) {
    // A piece of line, not the whole, gets reflown.
    return getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
  } else {
    // The whole line gets reflown, need to check if we need to insert a break
    // for the postfix or not.
    StringRef TrimmedContent = Content[LineIndex].ltrim(Blanks);
    unsigned ReflownColumn =
        getReflownColumn(TrimmedContent, LineIndex, PreviousEndColumn);
    if (ReflownColumn <= ColumnLimit) {
      return ReflownColumn;
    }
    return getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
  }
}
void BreakableBlockComment::replaceWhitespaceBefore(
    unsigned LineIndex, unsigned PreviousEndColumn, unsigned ColumnLimit,
    Split SplitBefore, WhitespaceManager &Whitespaces) {
  if (LineIndex == 0) return;
  StringRef TrimmedContent = Content[LineIndex].ltrim(Blanks);
  if (SplitBefore.first != StringRef::npos) {
    // Here we need to reflow.
    assert(Tokens[LineIndex - 1] == Tokens[LineIndex] &&
           "Reflowing whitespace within a token");
    // This is the offset of the end of the last line relative to the start of
    // the token text in the token.
    unsigned WhitespaceOffsetInToken = Content[LineIndex - 1].data() +
        Content[LineIndex - 1].size() -
        tokenAt(LineIndex).TokenText.data();
    unsigned WhitespaceLength = TrimmedContent.data() -
        tokenAt(LineIndex).TokenText.data() -
        WhitespaceOffsetInToken;
    Whitespaces.replaceWhitespaceInToken(
        tokenAt(LineIndex), WhitespaceOffsetInToken,
        /*ReplaceChars=*/WhitespaceLength, /*PreviousPostfix=*/"",
        /*CurrentPrefix=*/ReflowPrefix, InPPDirective, /*Newlines=*/0,
        /*Spaces=*/0);
    // Check if we need to also insert a break at the whitespace range.
    // For this we first adapt the reflow split relative to the beginning of the
    // content.
    // Note that we don't need a penalty for this break, since it doesn't change
    // the total number of lines.
    Split BreakSplit = SplitBefore;
    BreakSplit.first += TrimmedContent.data() - Content[LineIndex].data();
    unsigned ReflownColumn =
        getReflownColumn(TrimmedContent, LineIndex, PreviousEndColumn);
    if (ReflownColumn > ColumnLimit) {
      insertBreak(LineIndex, 0, BreakSplit, Whitespaces);
    }
    return;
  }

  // Here no reflow with the previous line will happen.
  // Fix the decoration of the line at LineIndex.
  StringRef Prefix = Decoration;
  if (Content[LineIndex].empty()) {
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
    if (ContentColumn[LineIndex] == 1) {
      // This line starts immediately after the decorating *.
      Prefix = Prefix.substr(0, 1);
    }
  }
  // This is the offset of the end of the last line relative to the start of the
  // token text in the token.
  unsigned WhitespaceOffsetInToken = Content[LineIndex - 1].data() +
                                     Content[LineIndex - 1].size() -
                                     tokenAt(LineIndex).TokenText.data();
  unsigned WhitespaceLength = Content[LineIndex].data() -
                              tokenAt(LineIndex).TokenText.data() -
                              WhitespaceOffsetInToken;
  Whitespaces.replaceWhitespaceInToken(
      tokenAt(LineIndex), WhitespaceOffsetInToken, WhitespaceLength, "", Prefix,
      InPPDirective, /*Newlines=*/1, ContentColumn[LineIndex] - Prefix.size());
}

bool BreakableBlockComment::mayReflow(unsigned LineIndex,
                                      llvm::Regex &CommentPragmasRegex) const {
  // Content[LineIndex] may exclude the indent after the '*' decoration. In that
  // case, we compute the start of the comment pragma manually.
  StringRef IndentContent = Content[LineIndex];
  if (Lines[LineIndex].ltrim(Blanks).startswith("*")) {
    IndentContent = Lines[LineIndex].ltrim(Blanks).substr(1);
  }
  return LineIndex > 0 && !CommentPragmasRegex.match(IndentContent) &&
         mayReflowContent(Content[LineIndex]) && !Tok.Finalized &&
         !switchesFormatting(tokenAt(LineIndex));
}

unsigned
BreakableBlockComment::getContentStartColumn(unsigned LineIndex,
                                             unsigned TailOffset) const {
  // If we break, we always break at the predefined indent.
  if (TailOffset != 0)
    return IndentAtLineBreak;
  return std::max(0, ContentColumn[LineIndex]);
}

BreakableLineCommentSection::BreakableLineCommentSection(
    const FormatToken &Token, unsigned StartColumn,
    unsigned OriginalStartColumn, bool FirstInLine, bool InPPDirective,
    encoding::Encoding Encoding, const FormatStyle &Style)
    : BreakableComment(Token, StartColumn, InPPDirective, Encoding, Style) {
  assert(Tok.is(TT_LineComment) &&
         "line comment section must start with a line comment");
  FormatToken *LineTok = nullptr;
  for (const FormatToken *CurrentTok = &Tok;
       CurrentTok && CurrentTok->is(TT_LineComment);
       CurrentTok = CurrentTok->Next) {
    LastLineTok = LineTok;
    StringRef TokenText(CurrentTok->TokenText);
    assert(TokenText.startswith("//"));
    size_t FirstLineIndex = Lines.size();
    TokenText.split(Lines, "\n");
    Content.resize(Lines.size());
    ContentColumn.resize(Lines.size());
    OriginalContentColumn.resize(Lines.size());
    Tokens.resize(Lines.size());
    Prefix.resize(Lines.size());
    OriginalPrefix.resize(Lines.size());
    for (size_t i = FirstLineIndex, e = Lines.size(); i < e; ++i) {
      // We need to trim the blanks in case this is not the first line in a
      // multiline comment. Then the indent is included in Lines[i].
      StringRef IndentPrefix =
          getLineCommentIndentPrefix(Lines[i].ltrim(Blanks));
      assert(IndentPrefix.startswith("//"));
      OriginalPrefix[i] = Prefix[i] = IndentPrefix;
      if (Lines[i].size() > Prefix[i].size() &&
          isAlphanumeric(Lines[i][Prefix[i].size()])) {
        if (Prefix[i] == "//")
          Prefix[i] = "// ";
        else if (Prefix[i] == "///")
          Prefix[i] = "/// ";
        else if (Prefix[i] == "//!")
          Prefix[i] = "//! ";
      }

      Tokens[i] = LineTok;
      Content[i] = Lines[i].substr(IndentPrefix.size());
      OriginalContentColumn[i] =
          StartColumn +
          encoding::columnWidthWithTabs(OriginalPrefix[i],
                                        StartColumn,
                                        Style.TabWidth,
                                        Encoding);
      ContentColumn[i] =
          StartColumn +
          encoding::columnWidthWithTabs(Prefix[i],
                                        StartColumn,
                                        Style.TabWidth,
                                        Encoding);

      // Calculate the end of the non-whitespace text in this line.
      size_t EndOfLine = Content[i].find_last_not_of(Blanks);
      if (EndOfLine == StringRef::npos)
        EndOfLine = Content[i].size();
      else
        ++EndOfLine;
      Content[i] = Content[i].substr(0, EndOfLine);
    }
    LineTok = CurrentTok->Next;
    if (CurrentTok->Next && !CurrentTok->Next->ContinuesLineCommentSection) {
      // A line comment section needs to broken by a line comment that is
      // preceded by at least two newlines. Note that we put this break here
      // instead of breaking at a previous stage during parsing, since that
      // would split the contents of the enum into two unwrapped lines in this
      // example, which is undesirable:
      // enum A {
      //   a, // comment about a
      //
      //   // comment about b
      //   b
      // };
      //
      // FIXME: Consider putting separate line comment sections as children to
      // the unwrapped line instead.
      break;
    }
  }
}

unsigned BreakableLineCommentSection::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned TailOffset,
    StringRef::size_type Length) const {
  unsigned ContentStartColumn =
      (TailOffset == 0 ? ContentColumn[LineIndex]
                       : OriginalContentColumn[LineIndex]);
  return ContentStartColumn + encoding::columnWidthWithTabs(
                                  Content[LineIndex].substr(TailOffset, Length),
                                  ContentStartColumn, Style.TabWidth, Encoding);
}

void BreakableLineCommentSection::insertBreak(unsigned LineIndex,
                                              unsigned TailOffset, Split Split,
                                              WhitespaceManager &Whitespaces) {
  StringRef Text = Content[LineIndex].substr(TailOffset);
  // Compute the offset of the split relative to the beginning of the token
  // text.
  unsigned BreakOffsetInToken =
      Text.data() - tokenAt(LineIndex).TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  // Compute the size of the new indent, including the size of the new prefix of
  // the newly broken line.
  unsigned IndentAtLineBreak = OriginalContentColumn[LineIndex] +
                               Prefix[LineIndex].size() -
                               OriginalPrefix[LineIndex].size();
  assert(IndentAtLineBreak >= Prefix[LineIndex].size());
  Whitespaces.replaceWhitespaceInToken(
      tokenAt(LineIndex), BreakOffsetInToken, CharsToRemove, "",
      Prefix[LineIndex], InPPDirective, /*Newlines=*/1,
      /*Spaces=*/IndentAtLineBreak - Prefix[LineIndex].size());
}

BreakableComment::Split BreakableLineCommentSection::getSplitBefore(
    unsigned LineIndex, unsigned PreviousEndColumn, unsigned ColumnLimit,
    llvm::Regex &CommentPragmasRegex) const {
  if (!mayReflow(LineIndex, CommentPragmasRegex))
    return Split(StringRef::npos, 0);
  return getReflowSplit(Content[LineIndex], ReflowPrefix, PreviousEndColumn,
                        ColumnLimit);
}

unsigned BreakableLineCommentSection::getLineLengthAfterSplitBefore(
    unsigned LineIndex, unsigned TailOffset,
    unsigned PreviousEndColumn,
    unsigned ColumnLimit,
    Split SplitBefore) const {
  if (SplitBefore.first == StringRef::npos ||
      SplitBefore.first + SplitBefore.second < Content[LineIndex].size()) {
    // A piece of line, not the whole line, gets reflown.
    return getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
  } else {
    // The whole line gets reflown.
    unsigned StartColumn = PreviousEndColumn + ReflowPrefix.size();
    return StartColumn + encoding::columnWidthWithTabs(Content[LineIndex],
                                                       StartColumn,
                                                       Style.TabWidth,
                                                       Encoding);
  }
}

void BreakableLineCommentSection::replaceWhitespaceBefore(
    unsigned LineIndex, unsigned PreviousEndColumn, unsigned ColumnLimit,
    Split SplitBefore, WhitespaceManager &Whitespaces) {
  // If this is the first line of a token, we need to inform Whitespace Manager
  // about it: either adapt the whitespace range preceding it, or mark it as an
  // untouchable token.
  // This happens for instance here:
  // // line 1 \
  // // line 2
  if (LineIndex > 0 && Tokens[LineIndex] != Tokens[LineIndex - 1]) {
    if (SplitBefore.first != StringRef::npos) {
      // Reflow happens between tokens. Replace the whitespace between the
      // tokens by the empty string.
      Whitespaces.replaceWhitespace(
          *Tokens[LineIndex], /*Newlines=*/0, /*Spaces=*/0,
          /*StartOfTokenColumn=*/StartColumn, /*InPPDirective=*/false);
      // Replace the indent and prefix of the token with the reflow prefix.
      unsigned WhitespaceLength =
          Content[LineIndex].data() - tokenAt(LineIndex).TokenText.data();
      Whitespaces.replaceWhitespaceInToken(*Tokens[LineIndex],
                                           /*Offset=*/0,
                                           /*ReplaceChars=*/WhitespaceLength,
                                           /*PreviousPostfix=*/"",
                                           /*CurrentPrefix=*/ReflowPrefix,
                                           /*InPPDirective=*/false,
                                           /*Newlines=*/0,
                                           /*Spaces=*/0);
    } else {
      // This is the first line for the current token, but no reflow with the
      // previous token is necessary. However, we still may need to adjust the
      // start column. Note that ContentColumn[LineIndex] is the expected
      // content column after a possible update to the prefix, hence the prefix
      // length change is included.
      unsigned LineColumn =
          ContentColumn[LineIndex] -
          (Content[LineIndex].data() - Lines[LineIndex].data()) +
          (OriginalPrefix[LineIndex].size() - Prefix[LineIndex].size());

      // We always want to create a replacement instead of adding an untouchable
      // token, even if LineColumn is the same as the original column of the
      // token. This is because WhitespaceManager doesn't align trailing
      // comments if they are untouchable.
      Whitespaces.replaceWhitespace(*Tokens[LineIndex],
                                    /*Newlines=*/1,
                                    /*Spaces=*/LineColumn,
                                    /*StartOfTokenColumn=*/LineColumn,
                                    /*InPPDirective=*/false);
    }
  }
  if (OriginalPrefix[LineIndex] != Prefix[LineIndex]) {
    // Adjust the prefix if necessary.

    // Take care of the space possibly introduced after a decoration.
    assert(Prefix[LineIndex] == (OriginalPrefix[LineIndex] + " ").str() &&
           "Expecting a line comment prefix to differ from original by at most "
           "a space");
    Whitespaces.replaceWhitespaceInToken(
        tokenAt(LineIndex), OriginalPrefix[LineIndex].size(), 0, "", "",
        /*InPPDirective=*/false, /*Newlines=*/0, /*Spaces=*/1);
  }
  // Add a break after a reflow split has been introduced, if necessary.
  // Note that this break doesn't need to be penalized, since it doesn't change
  // the number of lines.
  if (SplitBefore.first != StringRef::npos &&
      SplitBefore.first + SplitBefore.second < Content[LineIndex].size()) {
    insertBreak(LineIndex, 0, SplitBefore, Whitespaces);
  }
}

void BreakableLineCommentSection::updateNextToken(LineState& State) const {
  if (LastLineTok) {
    State.NextToken = LastLineTok->Next;
  }
}

bool BreakableLineCommentSection::mayReflow(
    unsigned LineIndex, llvm::Regex &CommentPragmasRegex) const {
  // Line comments have the indent as part of the prefix, so we need to
  // recompute the start of the line.
  StringRef IndentContent = Content[LineIndex];
  if (Lines[LineIndex].startswith("//")) {
    IndentContent = Lines[LineIndex].substr(2);
  }
  return LineIndex > 0 && !CommentPragmasRegex.match(IndentContent) &&
         mayReflowContent(Content[LineIndex]) && !Tok.Finalized &&
         !switchesFormatting(tokenAt(LineIndex)) &&
         OriginalPrefix[LineIndex] == OriginalPrefix[LineIndex - 1];
}

unsigned
BreakableLineCommentSection::getContentStartColumn(unsigned LineIndex,
                                                   unsigned TailOffset) const {
  if (TailOffset != 0) {
    return OriginalContentColumn[LineIndex];
  }
  return ContentColumn[LineIndex];
}

} // namespace format
} // namespace clang
