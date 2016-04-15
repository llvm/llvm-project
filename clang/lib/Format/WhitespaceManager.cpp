//===--- WhitespaceManager.cpp - Format C++ code --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WhitespaceManager class.
///
//===----------------------------------------------------------------------===//

#include "WhitespaceManager.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace format {

bool WhitespaceManager::Change::IsBeforeInFile::
operator()(const Change &C1, const Change &C2) const {
  return SourceMgr.isBeforeInTranslationUnit(
      C1.OriginalWhitespaceRange.getBegin(),
      C2.OriginalWhitespaceRange.getBegin());
}

WhitespaceManager::Change::Change(
    bool CreateReplacement, SourceRange OriginalWhitespaceRange,
    unsigned IndentLevel, int Spaces, unsigned StartOfTokenColumn,
    unsigned NewlinesBefore, StringRef PreviousLinePostfix,
    StringRef CurrentLinePrefix, tok::TokenKind Kind, bool ContinuesPPDirective,
    bool IsStartOfDeclName, bool IsInsideToken)
    : CreateReplacement(CreateReplacement),
      OriginalWhitespaceRange(OriginalWhitespaceRange),
      StartOfTokenColumn(StartOfTokenColumn), NewlinesBefore(NewlinesBefore),
      PreviousLinePostfix(PreviousLinePostfix),
      CurrentLinePrefix(CurrentLinePrefix), Kind(Kind),
      ContinuesPPDirective(ContinuesPPDirective),
      IsStartOfDeclName(IsStartOfDeclName), IndentLevel(IndentLevel),
      Spaces(Spaces), IsInsideToken(IsInsideToken), IsTrailingComment(false),
      TokenLength(0), PreviousEndOfTokenColumn(0), EscapedNewlineColumn(0),
      StartOfBlockComment(nullptr), IndentationOffset(0) {}

void WhitespaceManager::reset() {
  Changes.clear();
  Replaces.clear();
}

void WhitespaceManager::replaceWhitespace(FormatToken &Tok, unsigned Newlines,
                                          unsigned IndentLevel, unsigned Spaces,
                                          unsigned StartOfTokenColumn,
                                          bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Tok.Decision = (Newlines > 0) ? FD_Break : FD_Continue;
  Changes.push_back(
      Change(/*CreateReplacement=*/true, Tok.WhitespaceRange, IndentLevel,
             Spaces, StartOfTokenColumn, Newlines, "", "", Tok.Tok.getKind(),
             InPPDirective && !Tok.IsFirst,
             Tok.is(TT_StartOfName) || Tok.is(TT_FunctionDeclarationName),
             /*IsInsideToken=*/false));
}

void WhitespaceManager::addUntouchableToken(const FormatToken &Tok,
                                            bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Changes.push_back(Change(
      /*CreateReplacement=*/false, Tok.WhitespaceRange, /*IndentLevel=*/0,
      /*Spaces=*/0, Tok.OriginalColumn, Tok.NewlinesBefore, "", "",
      Tok.Tok.getKind(), InPPDirective && !Tok.IsFirst,
      Tok.is(TT_StartOfName) || Tok.is(TT_FunctionDeclarationName),
      /*IsInsideToken=*/false));
}

void WhitespaceManager::replaceWhitespaceInToken(
    const FormatToken &Tok, unsigned Offset, unsigned ReplaceChars,
    StringRef PreviousPostfix, StringRef CurrentPrefix, bool InPPDirective,
    unsigned Newlines, unsigned IndentLevel, int Spaces) {
  if (Tok.Finalized)
    return;
  SourceLocation Start = Tok.getStartOfNonWhitespace().getLocWithOffset(Offset);
  Changes.push_back(Change(
      true, SourceRange(Start, Start.getLocWithOffset(ReplaceChars)),
      IndentLevel, Spaces, std::max(0, Spaces), Newlines, PreviousPostfix,
      CurrentPrefix, Tok.is(TT_LineComment) ? tok::comment : tok::unknown,
      InPPDirective && !Tok.IsFirst,
      Tok.is(TT_StartOfName) || Tok.is(TT_FunctionDeclarationName),
      /*IsInsideToken=*/Newlines == 0));
}

const tooling::Replacements &WhitespaceManager::generateReplacements() {
  if (Changes.empty())
    return Replaces;

  std::sort(Changes.begin(), Changes.end(), Change::IsBeforeInFile(SourceMgr));
  calculateLineBreakInformation();
  alignConsecutiveDeclarations();
  alignConsecutiveAssignments();
  alignTrailingComments();
  alignEscapedNewlines();
  generateChanges();

  return Replaces;
}

void WhitespaceManager::calculateLineBreakInformation() {
  Changes[0].PreviousEndOfTokenColumn = 0;
  Change *LastOutsideTokenChange = &Changes[0];
  for (unsigned i = 1, e = Changes.size(); i != e; ++i) {
    unsigned OriginalWhitespaceStart =
        SourceMgr.getFileOffset(Changes[i].OriginalWhitespaceRange.getBegin());
    unsigned PreviousOriginalWhitespaceEnd = SourceMgr.getFileOffset(
        Changes[i - 1].OriginalWhitespaceRange.getEnd());
    Changes[i - 1].TokenLength = OriginalWhitespaceStart -
                                 PreviousOriginalWhitespaceEnd +
                                 Changes[i].PreviousLinePostfix.size() +
                                 Changes[i - 1].CurrentLinePrefix.size();

    // If there are multiple changes in this token, sum up all the changes until
    // the end of the line.
    if (Changes[i - 1].IsInsideToken)
      LastOutsideTokenChange->TokenLength +=
          Changes[i - 1].TokenLength + Changes[i - 1].Spaces;
    else
      LastOutsideTokenChange = &Changes[i - 1];

    Changes[i].PreviousEndOfTokenColumn =
        Changes[i - 1].StartOfTokenColumn + Changes[i - 1].TokenLength;

    Changes[i - 1].IsTrailingComment =
        (Changes[i].NewlinesBefore > 0 || Changes[i].Kind == tok::eof ||
         (Changes[i].IsInsideToken && Changes[i].Kind == tok::comment)) &&
        Changes[i - 1].Kind == tok::comment;
  }
  // FIXME: The last token is currently not always an eof token; in those
  // cases, setting TokenLength of the last token to 0 is wrong.
  Changes.back().TokenLength = 0;
  Changes.back().IsTrailingComment = Changes.back().Kind == tok::comment;

  const WhitespaceManager::Change *LastBlockComment = nullptr;
  for (auto &Change : Changes) {
    // Reset the IsTrailingComment flag for changes inside of trailing comments
    // so they don't get realigned later.
    if (Change.IsInsideToken)
      Change.IsTrailingComment = false;
    Change.StartOfBlockComment = nullptr;
    Change.IndentationOffset = 0;
    if (Change.Kind == tok::comment) {
      LastBlockComment = &Change;
    } else if (Change.Kind == tok::unknown) {
      if ((Change.StartOfBlockComment = LastBlockComment))
        Change.IndentationOffset =
            Change.StartOfTokenColumn -
            Change.StartOfBlockComment->StartOfTokenColumn;
    } else {
      LastBlockComment = nullptr;
    }
  }
}

// Align a single sequence of tokens, see AlignTokens below.
template <typename F>
static void
AlignTokenSequence(unsigned Start, unsigned End, unsigned Column, F &&Matches,
                   SmallVector<WhitespaceManager::Change, 16> &Changes) {
  bool FoundMatchOnLine = false;
  int Shift = 0;
  for (unsigned i = Start; i != End; ++i) {
    if (Changes[i].NewlinesBefore > 0) {
      FoundMatchOnLine = false;
      Shift = 0;
    }

    // If this is the first matching token to be aligned, remember by how many
    // spaces it has to be shifted, so the rest of the changes on the line are
    // shifted by the same amount
    if (!FoundMatchOnLine && Matches(Changes[i])) {
      FoundMatchOnLine = true;
      Shift = Column - Changes[i].StartOfTokenColumn;
      Changes[i].Spaces += Shift;
    }

    assert(Shift >= 0);
    Changes[i].StartOfTokenColumn += Shift;
    if (i + 1 != Changes.size())
      Changes[i + 1].PreviousEndOfTokenColumn += Shift;
  }
}

// Walk through all of the changes and find sequences of matching tokens to
// align. To do so, keep track of the lines and whether or not a matching token
// was found on a line. If a matching token is found, extend the current
// sequence. If the current line cannot be part of a sequence, e.g. because
// there is an empty line before it or it contains only non-matching tokens,
// finalize the previous sequence.
template <typename F>
static void AlignTokens(const FormatStyle &Style, F &&Matches,
                        SmallVector<WhitespaceManager::Change, 16> &Changes) {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;

  // Line number of the start and the end of the current token sequence.
  unsigned StartOfSequence = 0;
  unsigned EndOfSequence = 0;

  // Keep track of the nesting level of matching tokens, i.e. the number of
  // surrounding (), [], or {}. We will only align a sequence of matching
  // token that share the same scope depth.
  //
  // FIXME: This could use FormatToken::NestingLevel information, but there is
  // an outstanding issue wrt the brace scopes.
  unsigned NestingLevelOfLastMatch = 0;
  unsigned NestingLevel = 0;

  // Keep track of the number of commas before the matching tokens, we will only
  // align a sequence of matching tokens if they are preceded by the same number
  // of commas.
  unsigned CommasBeforeLastMatch = 0;
  unsigned CommasBeforeMatch = 0;

  // Whether a matching token has been found on the current line.
  bool FoundMatchOnLine = false;

  // Aligns a sequence of matching tokens, on the MinColumn column.
  //
  // Sequences start from the first matching token to align, and end at the
  // first token of the first line that doesn't need to be aligned.
  //
  // We need to adjust the StartOfTokenColumn of each Change that is on a line
  // containing any matching token to be aligned and located after such token.
  auto AlignCurrentSequence = [&] {
    if (StartOfSequence > 0 && StartOfSequence < EndOfSequence)
      AlignTokenSequence(StartOfSequence, EndOfSequence, MinColumn, Matches,
                         Changes);
    MinColumn = 0;
    MaxColumn = UINT_MAX;
    StartOfSequence = 0;
    EndOfSequence = 0;
  };

  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    if (Changes[i].NewlinesBefore != 0) {
      CommasBeforeMatch = 0;
      EndOfSequence = i;
      // If there is a blank line, or if the last line didn't contain any
      // matching token, the sequence ends here.
      if (Changes[i].NewlinesBefore > 1 || !FoundMatchOnLine)
        AlignCurrentSequence();

      FoundMatchOnLine = false;
    }

    if (Changes[i].Kind == tok::comma) {
      ++CommasBeforeMatch;
    } else if (Changes[i].Kind == tok::r_brace ||
               Changes[i].Kind == tok::r_paren ||
               Changes[i].Kind == tok::r_square) {
      --NestingLevel;
    } else if (Changes[i].Kind == tok::l_brace ||
               Changes[i].Kind == tok::l_paren ||
               Changes[i].Kind == tok::l_square) {
      // We want sequences to skip over child scopes if possible, but not the
      // other way around.
      NestingLevelOfLastMatch = std::min(NestingLevelOfLastMatch, NestingLevel);
      ++NestingLevel;
    }

    if (!Matches(Changes[i]))
      continue;

    // If there is more than one matching token per line, or if the number of
    // preceding commas, or the scope depth, do not match anymore, end the
    // sequence.
    if (FoundMatchOnLine || CommasBeforeMatch != CommasBeforeLastMatch ||
        NestingLevel != NestingLevelOfLastMatch)
      AlignCurrentSequence();

    CommasBeforeLastMatch = CommasBeforeMatch;
    NestingLevelOfLastMatch = NestingLevel;
    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = i;

    unsigned ChangeMinColumn = Changes[i].StartOfTokenColumn;
    int LineLengthAfter = -Changes[i].Spaces;
    for (unsigned j = i; j != e && Changes[j].NewlinesBefore == 0; ++j)
      LineLengthAfter += Changes[j].Spaces + Changes[j].TokenLength;
    unsigned ChangeMaxColumn = Style.ColumnLimit - LineLengthAfter;

    // If we are restricted by the maximum column width, end the sequence.
    if (ChangeMinColumn > MaxColumn || ChangeMaxColumn < MinColumn ||
        CommasBeforeLastMatch != CommasBeforeMatch) {
      AlignCurrentSequence();
      StartOfSequence = i;
    }

    MinColumn = std::max(MinColumn, ChangeMinColumn);
    MaxColumn = std::min(MaxColumn, ChangeMaxColumn);
  }

  EndOfSequence = Changes.size();
  AlignCurrentSequence();
}

void WhitespaceManager::alignConsecutiveAssignments() {
  if (!Style.AlignConsecutiveAssignments)
    return;

  AlignTokens(Style,
              [&](const Change &C) {
                // Do not align on equal signs that are first on a line.
                if (C.NewlinesBefore > 0)
                  return false;

                // Do not align on equal signs that are last on a line.
                if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
                  return false;

                return C.Kind == tok::equal;
              },
              Changes);
}

void WhitespaceManager::alignConsecutiveDeclarations() {
  if (!Style.AlignConsecutiveDeclarations)
    return;

  // FIXME: Currently we don't handle properly the PointerAlignment: Right
  // The * and & are not aligned and are left dangling. Something has to be done
  // about it, but it raises the question of alignment of code like:
  //   const char* const* v1;
  //   float const* v2;
  //   SomeVeryLongType const& v3;

  AlignTokens(Style, [](Change const &C) { return C.IsStartOfDeclName; },
              Changes);
}

void WhitespaceManager::alignTrailingComments() {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;
  unsigned StartOfSequence = 0;
  bool BreakBeforeNext = false;
  unsigned Newlines = 0;
  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    if (Changes[i].StartOfBlockComment)
      continue;
    Newlines += Changes[i].NewlinesBefore;
    if (!Changes[i].IsTrailingComment)
      continue;

    unsigned ChangeMinColumn = Changes[i].StartOfTokenColumn;
    unsigned ChangeMaxColumn = Style.ColumnLimit - Changes[i].TokenLength;

    // If we don't create a replacement for this change, we have to consider
    // it to be immovable.
    if (!Changes[i].CreateReplacement)
      ChangeMaxColumn = ChangeMinColumn;

    if (i + 1 != e && Changes[i + 1].ContinuesPPDirective)
      ChangeMaxColumn -= 2;
    // If this comment follows an } in column 0, it probably documents the
    // closing of a namespace and we don't want to align it.
    bool FollowsRBraceInColumn0 = i > 0 && Changes[i].NewlinesBefore == 0 &&
                                  Changes[i - 1].Kind == tok::r_brace &&
                                  Changes[i - 1].StartOfTokenColumn == 0;
    bool WasAlignedWithStartOfNextLine = false;
    if (Changes[i].NewlinesBefore == 1) { // A comment on its own line.
      unsigned CommentColumn = SourceMgr.getSpellingColumnNumber(
          Changes[i].OriginalWhitespaceRange.getEnd());
      for (unsigned j = i + 1; j != e; ++j) {
        if (Changes[j].Kind == tok::comment ||
            Changes[j].Kind == tok::unknown)
          // Skip over comments and unknown tokens. "unknown tokens are used for
          // the continuation of multiline comments.
          continue;

        unsigned NextColumn = SourceMgr.getSpellingColumnNumber(
            Changes[j].OriginalWhitespaceRange.getEnd());
        // The start of the next token was previously aligned with the
        // start of this comment.
        WasAlignedWithStartOfNextLine =
            CommentColumn == NextColumn ||
            CommentColumn == NextColumn + Style.IndentWidth;
        break;
      }
    }
    if (!Style.AlignTrailingComments || FollowsRBraceInColumn0) {
      alignTrailingComments(StartOfSequence, i, MinColumn);
      MinColumn = ChangeMinColumn;
      MaxColumn = ChangeMinColumn;
      StartOfSequence = i;
    } else if (BreakBeforeNext || Newlines > 1 ||
               (ChangeMinColumn > MaxColumn || ChangeMaxColumn < MinColumn) ||
               // Break the comment sequence if the previous line did not end
               // in a trailing comment.
               (Changes[i].NewlinesBefore == 1 && i > 0 &&
                !Changes[i - 1].IsTrailingComment) ||
               WasAlignedWithStartOfNextLine) {
      alignTrailingComments(StartOfSequence, i, MinColumn);
      MinColumn = ChangeMinColumn;
      MaxColumn = ChangeMaxColumn;
      StartOfSequence = i;
    } else {
      MinColumn = std::max(MinColumn, ChangeMinColumn);
      MaxColumn = std::min(MaxColumn, ChangeMaxColumn);
    }
    BreakBeforeNext =
        (i == 0) || (Changes[i].NewlinesBefore > 1) ||
        // Never start a sequence with a comment at the beginning of
        // the line.
        (Changes[i].NewlinesBefore == 1 && StartOfSequence == i);
    Newlines = 0;
  }
  alignTrailingComments(StartOfSequence, Changes.size(), MinColumn);
}

void WhitespaceManager::alignTrailingComments(unsigned Start, unsigned End,
                                              unsigned Column) {
  for (unsigned i = Start; i != End; ++i) {
    int Shift = 0;
    if (Changes[i].IsTrailingComment) {
      Shift = Column - Changes[i].StartOfTokenColumn;
    }
    if (Changes[i].StartOfBlockComment) {
      Shift = Changes[i].IndentationOffset +
              Changes[i].StartOfBlockComment->StartOfTokenColumn -
              Changes[i].StartOfTokenColumn;
    }
    assert(Shift >= 0);
    Changes[i].Spaces += Shift;
    if (i + 1 != End)
      Changes[i + 1].PreviousEndOfTokenColumn += Shift;
    Changes[i].StartOfTokenColumn += Shift;
  }
}

void WhitespaceManager::alignEscapedNewlines() {
  unsigned MaxEndOfLine =
      Style.AlignEscapedNewlinesLeft ? 0 : Style.ColumnLimit;
  unsigned StartOfMacro = 0;
  for (unsigned i = 1, e = Changes.size(); i < e; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      if (C.ContinuesPPDirective) {
        MaxEndOfLine = std::max(C.PreviousEndOfTokenColumn + 2, MaxEndOfLine);
      } else {
        alignEscapedNewlines(StartOfMacro + 1, i, MaxEndOfLine);
        MaxEndOfLine = Style.AlignEscapedNewlinesLeft ? 0 : Style.ColumnLimit;
        StartOfMacro = i;
      }
    }
  }
  alignEscapedNewlines(StartOfMacro + 1, Changes.size(), MaxEndOfLine);
}

void WhitespaceManager::alignEscapedNewlines(unsigned Start, unsigned End,
                                             unsigned Column) {
  for (unsigned i = Start; i < End; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      assert(C.ContinuesPPDirective);
      if (C.PreviousEndOfTokenColumn + 1 > Column)
        C.EscapedNewlineColumn = 0;
      else
        C.EscapedNewlineColumn = Column;
    }
  }
}

void WhitespaceManager::generateChanges() {
  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    const Change &C = Changes[i];
    if (i > 0) {
      assert(Changes[i - 1].OriginalWhitespaceRange.getBegin() !=
                 C.OriginalWhitespaceRange.getBegin() &&
             "Generating two replacements for the same location");
    }
    if (C.CreateReplacement) {
      std::string ReplacementText = C.PreviousLinePostfix;
      if (C.ContinuesPPDirective)
        appendNewlineText(ReplacementText, C.NewlinesBefore,
                          C.PreviousEndOfTokenColumn, C.EscapedNewlineColumn);
      else
        appendNewlineText(ReplacementText, C.NewlinesBefore);
      appendIndentText(ReplacementText, C.IndentLevel, std::max(0, C.Spaces),
                       C.StartOfTokenColumn - std::max(0, C.Spaces));
      ReplacementText.append(C.CurrentLinePrefix);
      storeReplacement(C.OriginalWhitespaceRange, ReplacementText);
    }
  }
}

void WhitespaceManager::storeReplacement(SourceRange Range,
                                         StringRef Text) {
  unsigned WhitespaceLength = SourceMgr.getFileOffset(Range.getEnd()) -
                              SourceMgr.getFileOffset(Range.getBegin());
  // Don't create a replacement, if it does not change anything.
  if (StringRef(SourceMgr.getCharacterData(Range.getBegin()),
                WhitespaceLength) == Text)
    return;
  Replaces.insert(tooling::Replacement(
      SourceMgr, CharSourceRange::getCharRange(Range), Text));
}

void WhitespaceManager::appendNewlineText(std::string &Text,
                                          unsigned Newlines) {
  for (unsigned i = 0; i < Newlines; ++i)
    Text.append(UseCRLF ? "\r\n" : "\n");
}

void WhitespaceManager::appendNewlineText(std::string &Text, unsigned Newlines,
                                          unsigned PreviousEndOfTokenColumn,
                                          unsigned EscapedNewlineColumn) {
  if (Newlines > 0) {
    unsigned Offset =
        std::min<int>(EscapedNewlineColumn - 1, PreviousEndOfTokenColumn);
    for (unsigned i = 0; i < Newlines; ++i) {
      Text.append(EscapedNewlineColumn - Offset - 1, ' ');
      Text.append(UseCRLF ? "\\\r\n" : "\\\n");
      Offset = 0;
    }
  }
}

void WhitespaceManager::appendIndentText(std::string &Text,
                                         unsigned IndentLevel, unsigned Spaces,
                                         unsigned WhitespaceStartColumn) {
  switch (Style.UseTab) {
  case FormatStyle::UT_Never:
    Text.append(Spaces, ' ');
    break;
  case FormatStyle::UT_Always: {
    unsigned FirstTabWidth =
        Style.TabWidth - WhitespaceStartColumn % Style.TabWidth;
    // Indent with tabs only when there's at least one full tab.
    if (FirstTabWidth + Style.TabWidth <= Spaces) {
      Spaces -= FirstTabWidth;
      Text.append("\t");
    }
    Text.append(Spaces / Style.TabWidth, '\t');
    Text.append(Spaces % Style.TabWidth, ' ');
    break;
  }
  case FormatStyle::UT_ForIndentation:
    if (WhitespaceStartColumn == 0) {
      unsigned Indentation = IndentLevel * Style.IndentWidth;
      // This happens, e.g. when a line in a block comment is indented less than
      // the first one.
      if (Indentation > Spaces)
        Indentation = Spaces;
      unsigned Tabs = Indentation / Style.TabWidth;
      Text.append(Tabs, '\t');
      Spaces -= Tabs * Style.TabWidth;
    }
    Text.append(Spaces, ' ');
    break;
  case FormatStyle::UT_ForContinuationAndIndentation:
    if (WhitespaceStartColumn == 0) {
      unsigned Tabs = Spaces / Style.TabWidth;
      Text.append(Tabs, '\t');
      Spaces -= Tabs * Style.TabWidth;
    }
    Text.append(Spaces, ' ');
    break;
  }
}

} // namespace format
} // namespace clang
