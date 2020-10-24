//===--- WhitespaceManager.cpp - Format C++ code --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements WhitespaceManager class.
///
//===----------------------------------------------------------------------===//

#include "WhitespaceManager.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace format {

bool WhitespaceManager::Change::IsBeforeInFile::operator()(
    const Change &C1, const Change &C2) const {
  return SourceMgr.isBeforeInTranslationUnit(
      C1.OriginalWhitespaceRange.getBegin(),
      C2.OriginalWhitespaceRange.getBegin());
}

WhitespaceManager::Change::Change(const FormatToken &Tok,
                                  bool CreateReplacement,
                                  SourceRange OriginalWhitespaceRange,
                                  int Spaces, unsigned StartOfTokenColumn,
                                  unsigned NewlinesBefore,
                                  StringRef PreviousLinePostfix,
                                  StringRef CurrentLinePrefix, bool IsAligned,
                                  bool ContinuesPPDirective, bool IsInsideToken)
    : Tok(&Tok), CreateReplacement(CreateReplacement),
      OriginalWhitespaceRange(OriginalWhitespaceRange),
      StartOfTokenColumn(StartOfTokenColumn), NewlinesBefore(NewlinesBefore),
      PreviousLinePostfix(PreviousLinePostfix),
      CurrentLinePrefix(CurrentLinePrefix), IsAligned(IsAligned),
      ContinuesPPDirective(ContinuesPPDirective), Spaces(Spaces),
      IsInsideToken(IsInsideToken), IsTrailingComment(false), TokenLength(0),
      PreviousEndOfTokenColumn(0), EscapedNewlineColumn(0),
      StartOfBlockComment(nullptr), IndentationOffset(0), ConditionalsLevel(0) {
}

void WhitespaceManager::replaceWhitespace(FormatToken &Tok, unsigned Newlines,
                                          unsigned Spaces,
                                          unsigned StartOfTokenColumn,
                                          bool IsAligned, bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Tok.setDecision((Newlines > 0) ? FD_Break : FD_Continue);
  Changes.push_back(Change(Tok, /*CreateReplacement=*/true, Tok.WhitespaceRange,
                           Spaces, StartOfTokenColumn, Newlines, "", "",
                           IsAligned, InPPDirective && !Tok.IsFirst,
                           /*IsInsideToken=*/false));
}

void WhitespaceManager::addUntouchableToken(const FormatToken &Tok,
                                            bool InPPDirective) {
  if (Tok.Finalized)
    return;
  Changes.push_back(Change(Tok, /*CreateReplacement=*/false,
                           Tok.WhitespaceRange, /*Spaces=*/0,
                           Tok.OriginalColumn, Tok.NewlinesBefore, "", "",
                           /*IsAligned=*/false, InPPDirective && !Tok.IsFirst,
                           /*IsInsideToken=*/false));
}

llvm::Error
WhitespaceManager::addReplacement(const tooling::Replacement &Replacement) {
  return Replaces.add(Replacement);
}

void WhitespaceManager::replaceWhitespaceInToken(
    const FormatToken &Tok, unsigned Offset, unsigned ReplaceChars,
    StringRef PreviousPostfix, StringRef CurrentPrefix, bool InPPDirective,
    unsigned Newlines, int Spaces) {
  if (Tok.Finalized)
    return;
  SourceLocation Start = Tok.getStartOfNonWhitespace().getLocWithOffset(Offset);
  Changes.push_back(
      Change(Tok, /*CreateReplacement=*/true,
             SourceRange(Start, Start.getLocWithOffset(ReplaceChars)), Spaces,
             std::max(0, Spaces), Newlines, PreviousPostfix, CurrentPrefix,
             /*IsAligned=*/true, InPPDirective && !Tok.IsFirst,
             /*IsInsideToken=*/true));
}

const tooling::Replacements &WhitespaceManager::generateReplacements() {
  if (Changes.empty())
    return Replaces;

  llvm::sort(Changes, Change::IsBeforeInFile(SourceMgr));
  calculateLineBreakInformation();
  alignConsecutiveMacros();
  alignConsecutiveDeclarations();
  alignConsecutiveBitFields();
  alignConsecutiveAssignments();
  alignChainedConditionals();
  alignTrailingComments();
  alignEscapedNewlines();
  generateChanges();

  return Replaces;
}

void WhitespaceManager::calculateLineBreakInformation() {
  Changes[0].PreviousEndOfTokenColumn = 0;
  Change *LastOutsideTokenChange = &Changes[0];
  for (unsigned i = 1, e = Changes.size(); i != e; ++i) {
    SourceLocation OriginalWhitespaceStart =
        Changes[i].OriginalWhitespaceRange.getBegin();
    SourceLocation PreviousOriginalWhitespaceEnd =
        Changes[i - 1].OriginalWhitespaceRange.getEnd();
    unsigned OriginalWhitespaceStartOffset =
        SourceMgr.getFileOffset(OriginalWhitespaceStart);
    unsigned PreviousOriginalWhitespaceEndOffset =
        SourceMgr.getFileOffset(PreviousOriginalWhitespaceEnd);
    assert(PreviousOriginalWhitespaceEndOffset <=
           OriginalWhitespaceStartOffset);
    const char *const PreviousOriginalWhitespaceEndData =
        SourceMgr.getCharacterData(PreviousOriginalWhitespaceEnd);
    StringRef Text(PreviousOriginalWhitespaceEndData,
                   SourceMgr.getCharacterData(OriginalWhitespaceStart) -
                       PreviousOriginalWhitespaceEndData);
    // Usually consecutive changes would occur in consecutive tokens. This is
    // not the case however when analyzing some preprocessor runs of the
    // annotated lines. For example, in this code:
    //
    // #if A // line 1
    // int i = 1;
    // #else B // line 2
    // int i = 2;
    // #endif // line 3
    //
    // one of the runs will produce the sequence of lines marked with line 1, 2
    // and 3. So the two consecutive whitespace changes just before '// line 2'
    // and before '#endif // line 3' span multiple lines and tokens:
    //
    // #else B{change X}[// line 2
    // int i = 2;
    // ]{change Y}#endif // line 3
    //
    // For this reason, if the text between consecutive changes spans multiple
    // newlines, the token length must be adjusted to the end of the original
    // line of the token.
    auto NewlinePos = Text.find_first_of('\n');
    if (NewlinePos == StringRef::npos) {
      Changes[i - 1].TokenLength = OriginalWhitespaceStartOffset -
                                   PreviousOriginalWhitespaceEndOffset +
                                   Changes[i].PreviousLinePostfix.size() +
                                   Changes[i - 1].CurrentLinePrefix.size();
    } else {
      Changes[i - 1].TokenLength =
          NewlinePos + Changes[i - 1].CurrentLinePrefix.size();
    }

    // If there are multiple changes in this token, sum up all the changes until
    // the end of the line.
    if (Changes[i - 1].IsInsideToken && Changes[i - 1].NewlinesBefore == 0)
      LastOutsideTokenChange->TokenLength +=
          Changes[i - 1].TokenLength + Changes[i - 1].Spaces;
    else
      LastOutsideTokenChange = &Changes[i - 1];

    Changes[i].PreviousEndOfTokenColumn =
        Changes[i - 1].StartOfTokenColumn + Changes[i - 1].TokenLength;

    Changes[i - 1].IsTrailingComment =
        (Changes[i].NewlinesBefore > 0 || Changes[i].Tok->is(tok::eof) ||
         (Changes[i].IsInsideToken && Changes[i].Tok->is(tok::comment))) &&
        Changes[i - 1].Tok->is(tok::comment) &&
        // FIXME: This is a dirty hack. The problem is that
        // BreakableLineCommentSection does comment reflow changes and here is
        // the aligning of trailing comments. Consider the case where we reflow
        // the second line up in this example:
        //
        // // line 1
        // // line 2
        //
        // That amounts to 2 changes by BreakableLineCommentSection:
        //  - the first, delimited by (), for the whitespace between the tokens,
        //  - and second, delimited by [], for the whitespace at the beginning
        //  of the second token:
        //
        // // line 1(
        // )[// ]line 2
        //
        // So in the end we have two changes like this:
        //
        // // line1()[ ]line 2
        //
        // Note that the OriginalWhitespaceStart of the second change is the
        // same as the PreviousOriginalWhitespaceEnd of the first change.
        // In this case, the below check ensures that the second change doesn't
        // get treated as a trailing comment change here, since this might
        // trigger additional whitespace to be wrongly inserted before "line 2"
        // by the comment aligner here.
        //
        // For a proper solution we need a mechanism to say to WhitespaceManager
        // that a particular change breaks the current sequence of trailing
        // comments.
        OriginalWhitespaceStart != PreviousOriginalWhitespaceEnd;
  }
  // FIXME: The last token is currently not always an eof token; in those
  // cases, setting TokenLength of the last token to 0 is wrong.
  Changes.back().TokenLength = 0;
  Changes.back().IsTrailingComment = Changes.back().Tok->is(tok::comment);

  const WhitespaceManager::Change *LastBlockComment = nullptr;
  for (auto &Change : Changes) {
    // Reset the IsTrailingComment flag for changes inside of trailing comments
    // so they don't get realigned later. Comment line breaks however still need
    // to be aligned.
    if (Change.IsInsideToken && Change.NewlinesBefore == 0)
      Change.IsTrailingComment = false;
    Change.StartOfBlockComment = nullptr;
    Change.IndentationOffset = 0;
    if (Change.Tok->is(tok::comment)) {
      if (Change.Tok->is(TT_LineComment) || !Change.IsInsideToken)
        LastBlockComment = &Change;
      else {
        if ((Change.StartOfBlockComment = LastBlockComment))
          Change.IndentationOffset =
              Change.StartOfTokenColumn -
              Change.StartOfBlockComment->StartOfTokenColumn;
      }
    } else {
      LastBlockComment = nullptr;
    }
  }

  // Compute conditional nesting level
  // Level is increased for each conditional, unless this conditional continues
  // a chain of conditional, i.e. starts immediately after the colon of another
  // conditional.
  SmallVector<bool, 16> ScopeStack;
  int ConditionalsLevel = 0;
  for (auto &Change : Changes) {
    for (unsigned i = 0, e = Change.Tok->FakeLParens.size(); i != e; ++i) {
      bool isNestedConditional =
          Change.Tok->FakeLParens[e - 1 - i] == prec::Conditional &&
          !(i == 0 && Change.Tok->Previous &&
            Change.Tok->Previous->is(TT_ConditionalExpr) &&
            Change.Tok->Previous->is(tok::colon));
      if (isNestedConditional)
        ++ConditionalsLevel;
      ScopeStack.push_back(isNestedConditional);
    }

    Change.ConditionalsLevel = ConditionalsLevel;

    for (unsigned i = Change.Tok->FakeRParens; i > 0 && ScopeStack.size();
         --i) {
      if (ScopeStack.pop_back_val())
        --ConditionalsLevel;
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

  // ScopeStack keeps track of the current scope depth. It contains indices of
  // the first token on each scope.
  // We only run the "Matches" function on tokens from the outer-most scope.
  // However, we do need to pay special attention to one class of tokens
  // that are not in the outer-most scope, and that is function parameters
  // which are split across multiple lines, as illustrated by this example:
  //   double a(int x);
  //   int    b(int  y,
  //          double z);
  // In the above example, we need to take special care to ensure that
  // 'double z' is indented along with it's owning function 'b'.
  // Special handling is required for 'nested' ternary operators.
  SmallVector<unsigned, 16> ScopeStack;

  for (unsigned i = Start; i != End; ++i) {
    if (ScopeStack.size() != 0 &&
        Changes[i].indentAndNestingLevel() <
            Changes[ScopeStack.back()].indentAndNestingLevel())
      ScopeStack.pop_back();

    // Compare current token to previous non-comment token to ensure whether
    // it is in a deeper scope or not.
    unsigned PreviousNonComment = i - 1;
    while (PreviousNonComment > Start &&
           Changes[PreviousNonComment].Tok->is(tok::comment))
      PreviousNonComment--;
    if (i != Start && Changes[i].indentAndNestingLevel() >
                          Changes[PreviousNonComment].indentAndNestingLevel())
      ScopeStack.push_back(i);

    bool InsideNestedScope = ScopeStack.size() != 0;

    if (Changes[i].NewlinesBefore > 0 && !InsideNestedScope) {
      Shift = 0;
      FoundMatchOnLine = false;
    }

    // If this is the first matching token to be aligned, remember by how many
    // spaces it has to be shifted, so the rest of the changes on the line are
    // shifted by the same amount
    if (!FoundMatchOnLine && !InsideNestedScope && Matches(Changes[i])) {
      FoundMatchOnLine = true;
      Shift = Column - Changes[i].StartOfTokenColumn;
      Changes[i].Spaces += Shift;
    }

    // This is for function parameters that are split across multiple lines,
    // as mentioned in the ScopeStack comment.
    if (InsideNestedScope && Changes[i].NewlinesBefore > 0) {
      unsigned ScopeStart = ScopeStack.back();
      if (Changes[ScopeStart - 1].Tok->is(TT_FunctionDeclarationName) ||
          (ScopeStart > Start + 1 &&
           Changes[ScopeStart - 2].Tok->is(TT_FunctionDeclarationName)) ||
          Changes[i].Tok->is(TT_ConditionalExpr) ||
          (Changes[i].Tok->Previous &&
           Changes[i].Tok->Previous->is(TT_ConditionalExpr)))
        Changes[i].Spaces += Shift;
    }

    assert(Shift >= 0);
    Changes[i].StartOfTokenColumn += Shift;
    if (i + 1 != Changes.size())
      Changes[i + 1].PreviousEndOfTokenColumn += Shift;
  }
}

// Walk through a subset of the changes, starting at StartAt, and find
// sequences of matching tokens to align. To do so, keep track of the lines and
// whether or not a matching token was found on a line. If a matching token is
// found, extend the current sequence. If the current line cannot be part of a
// sequence, e.g. because there is an empty line before it or it contains only
// non-matching tokens, finalize the previous sequence.
// The value returned is the token on which we stopped, either because we
// exhausted all items inside Changes, or because we hit a scope level higher
// than our initial scope.
// This function is recursive. Each invocation processes only the scope level
// equal to the initial level, which is the level of Changes[StartAt].
// If we encounter a scope level greater than the initial level, then we call
// ourselves recursively, thereby avoiding the pollution of the current state
// with the alignment requirements of the nested sub-level. This recursive
// behavior is necessary for aligning function prototypes that have one or more
// arguments.
// If this function encounters a scope level less than the initial level,
// it returns the current position.
// There is a non-obvious subtlety in the recursive behavior: Even though we
// defer processing of nested levels to recursive invocations of this
// function, when it comes time to align a sequence of tokens, we run the
// alignment on the entire sequence, including the nested levels.
// When doing so, most of the nested tokens are skipped, because their
// alignment was already handled by the recursive invocations of this function.
// However, the special exception is that we do NOT skip function parameters
// that are split across multiple lines. See the test case in FormatTest.cpp
// that mentions "split function parameter alignment" for an example of this.
template <typename F>
static unsigned AlignTokens(const FormatStyle &Style, F &&Matches,
                            SmallVector<WhitespaceManager::Change, 16> &Changes,
                            unsigned StartAt) {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;

  // Line number of the start and the end of the current token sequence.
  unsigned StartOfSequence = 0;
  unsigned EndOfSequence = 0;

  // Measure the scope level (i.e. depth of (), [], {}) of the first token, and
  // abort when we hit any token in a higher scope than the starting one.
  auto IndentAndNestingLevel = StartAt < Changes.size()
                                   ? Changes[StartAt].indentAndNestingLevel()
                                   : std::tuple<unsigned, unsigned, unsigned>();

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

  unsigned i = StartAt;
  for (unsigned e = Changes.size(); i != e; ++i) {
    if (Changes[i].indentAndNestingLevel() < IndentAndNestingLevel)
      break;

    if (Changes[i].NewlinesBefore != 0) {
      CommasBeforeMatch = 0;
      EndOfSequence = i;
      // If there is a blank line, or if the last line didn't contain any
      // matching token, the sequence ends here.
      if (Changes[i].NewlinesBefore > 1 || !FoundMatchOnLine)
        AlignCurrentSequence();

      FoundMatchOnLine = false;
    }

    if (Changes[i].Tok->is(tok::comma)) {
      ++CommasBeforeMatch;
    } else if (Changes[i].indentAndNestingLevel() > IndentAndNestingLevel) {
      // Call AlignTokens recursively, skipping over this scope block.
      unsigned StoppedAt = AlignTokens(Style, Matches, Changes, i);
      i = StoppedAt - 1;
      continue;
    }

    if (!Matches(Changes[i]))
      continue;

    // If there is more than one matching token per line, or if the number of
    // preceding commas, do not match anymore, end the sequence.
    if (FoundMatchOnLine || CommasBeforeMatch != CommasBeforeLastMatch)
      AlignCurrentSequence();

    CommasBeforeLastMatch = CommasBeforeMatch;
    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = i;

    unsigned ChangeMinColumn = Changes[i].StartOfTokenColumn;
    int LineLengthAfter = Changes[i].TokenLength;
    for (unsigned j = i + 1; j != e && Changes[j].NewlinesBefore == 0; ++j) {
      LineLengthAfter += Changes[j].Spaces;
      // Changes are generally 1:1 with the tokens, but a change could also be
      // inside of a token, in which case it's counted more than once: once for
      // the whitespace surrounding the token (!IsInsideToken) and once for
      // each whitespace change within it (IsInsideToken).
      // Therefore, changes inside of a token should only count the space.
      if (!Changes[j].IsInsideToken)
        LineLengthAfter += Changes[j].TokenLength;
    }
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

  EndOfSequence = i;
  AlignCurrentSequence();
  return i;
}

// Aligns a sequence of matching tokens, on the MinColumn column.
//
// Sequences start from the first matching token to align, and end at the
// first token of the first line that doesn't need to be aligned.
//
// We need to adjust the StartOfTokenColumn of each Change that is on a line
// containing any matching token to be aligned and located after such token.
static void AlignMacroSequence(
    unsigned &StartOfSequence, unsigned &EndOfSequence, unsigned &MinColumn,
    unsigned &MaxColumn, bool &FoundMatchOnLine,
    std::function<bool(const WhitespaceManager::Change &C)> AlignMacrosMatches,
    SmallVector<WhitespaceManager::Change, 16> &Changes) {
  if (StartOfSequence > 0 && StartOfSequence < EndOfSequence) {

    FoundMatchOnLine = false;
    int Shift = 0;

    for (unsigned I = StartOfSequence; I != EndOfSequence; ++I) {
      if (Changes[I].NewlinesBefore > 0) {
        Shift = 0;
        FoundMatchOnLine = false;
      }

      // If this is the first matching token to be aligned, remember by how many
      // spaces it has to be shifted, so the rest of the changes on the line are
      // shifted by the same amount
      if (!FoundMatchOnLine && AlignMacrosMatches(Changes[I])) {
        FoundMatchOnLine = true;
        Shift = MinColumn - Changes[I].StartOfTokenColumn;
        Changes[I].Spaces += Shift;
      }

      assert(Shift >= 0);
      Changes[I].StartOfTokenColumn += Shift;
      if (I + 1 != Changes.size())
        Changes[I + 1].PreviousEndOfTokenColumn += Shift;
    }
  }

  MinColumn = 0;
  MaxColumn = UINT_MAX;
  StartOfSequence = 0;
  EndOfSequence = 0;
}

void WhitespaceManager::alignConsecutiveMacros() {
  if (!Style.AlignConsecutiveMacros)
    return;

  auto AlignMacrosMatches = [](const Change &C) {
    const FormatToken *Current = C.Tok;
    unsigned SpacesRequiredBefore = 1;

    if (Current->SpacesRequiredBefore == 0 || !Current->Previous)
      return false;

    Current = Current->Previous;

    // If token is a ")", skip over the parameter list, to the
    // token that precedes the "("
    if (Current->is(tok::r_paren) && Current->MatchingParen) {
      Current = Current->MatchingParen->Previous;
      SpacesRequiredBefore = 0;
    }

    if (!Current || !Current->is(tok::identifier))
      return false;

    if (!Current->Previous || !Current->Previous->is(tok::pp_define))
      return false;

    // For a macro function, 0 spaces are required between the
    // identifier and the lparen that opens the parameter list.
    // For a simple macro, 1 space is required between the
    // identifier and the first token of the defined value.
    return Current->Next->SpacesRequiredBefore == SpacesRequiredBefore;
  };

  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;

  // Start and end of the token sequence we're processing.
  unsigned StartOfSequence = 0;
  unsigned EndOfSequence = 0;

  // Whether a matching token has been found on the current line.
  bool FoundMatchOnLine = false;

  unsigned I = 0;
  for (unsigned E = Changes.size(); I != E; ++I) {
    if (Changes[I].NewlinesBefore != 0) {
      EndOfSequence = I;
      // If there is a blank line, or if the last line didn't contain any
      // matching token, the sequence ends here.
      if (Changes[I].NewlinesBefore > 1 || !FoundMatchOnLine)
        AlignMacroSequence(StartOfSequence, EndOfSequence, MinColumn, MaxColumn,
                           FoundMatchOnLine, AlignMacrosMatches, Changes);

      FoundMatchOnLine = false;
    }

    if (!AlignMacrosMatches(Changes[I]))
      continue;

    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = I;

    unsigned ChangeMinColumn = Changes[I].StartOfTokenColumn;
    int LineLengthAfter = -Changes[I].Spaces;
    for (unsigned j = I; j != E && Changes[j].NewlinesBefore == 0; ++j)
      LineLengthAfter += Changes[j].Spaces + Changes[j].TokenLength;
    unsigned ChangeMaxColumn = Style.ColumnLimit - LineLengthAfter;

    MinColumn = std::max(MinColumn, ChangeMinColumn);
    MaxColumn = std::min(MaxColumn, ChangeMaxColumn);
  }

  EndOfSequence = I;
  AlignMacroSequence(StartOfSequence, EndOfSequence, MinColumn, MaxColumn,
                     FoundMatchOnLine, AlignMacrosMatches, Changes);
}

void WhitespaceManager::alignConsecutiveAssignments() {
  if (!Style.AlignConsecutiveAssignments)
    return;

  AlignTokens(
      Style,
      [&](const Change &C) {
        // Do not align on equal signs that are first on a line.
        if (C.NewlinesBefore > 0)
          return false;

        // Do not align on equal signs that are last on a line.
        if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
          return false;

        return C.Tok->is(tok::equal);
      },
      Changes, /*StartAt=*/0);
}

void WhitespaceManager::alignConsecutiveBitFields() {
  if (!Style.AlignConsecutiveBitFields)
    return;

  AlignTokens(
      Style,
      [&](Change const &C) {
        // Do not align on ':' that is first on a line.
        if (C.NewlinesBefore > 0)
          return false;

        // Do not align on ':' that is last on a line.
        if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
          return false;

        return C.Tok->is(TT_BitFieldColon);
      },
      Changes, /*StartAt=*/0);
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
  AlignTokens(
      Style,
      [](Change const &C) {
        // tok::kw_operator is necessary for aligning operator overload
        // definitions.
        if (C.Tok->isOneOf(TT_FunctionDeclarationName, tok::kw_operator))
          return true;
        if (C.Tok->isNot(TT_StartOfName))
          return false;
        // Check if there is a subsequent name that starts the same declaration.
        for (FormatToken *Next = C.Tok->Next; Next; Next = Next->Next) {
          if (Next->is(tok::comment))
            continue;
          if (!Next->Tok.getIdentifierInfo())
            break;
          if (Next->isOneOf(TT_StartOfName, TT_FunctionDeclarationName,
                            tok::kw_operator))
            return false;
        }
        return true;
      },
      Changes, /*StartAt=*/0);
}

void WhitespaceManager::alignChainedConditionals() {
  if (Style.BreakBeforeTernaryOperators) {
    AlignTokens(
        Style,
        [](Change const &C) {
          // Align question operators and last colon
          return C.Tok->is(TT_ConditionalExpr) &&
                 ((C.Tok->is(tok::question) && !C.NewlinesBefore) ||
                  (C.Tok->is(tok::colon) && C.Tok->Next &&
                   (C.Tok->Next->FakeLParens.size() == 0 ||
                    C.Tok->Next->FakeLParens.back() != prec::Conditional)));
        },
        Changes, /*StartAt=*/0);
  } else {
    static auto AlignWrappedOperand = [](Change const &C) {
      auto Previous = C.Tok->getPreviousNonComment(); // Previous;
      return C.NewlinesBefore && Previous && Previous->is(TT_ConditionalExpr) &&
             (Previous->is(tok::question) ||
              (Previous->is(tok::colon) &&
               (C.Tok->FakeLParens.size() == 0 ||
                C.Tok->FakeLParens.back() != prec::Conditional)));
    };
    // Ensure we keep alignment of wrapped operands with non-wrapped operands
    // Since we actually align the operators, the wrapped operands need the
    // extra offset to be properly aligned.
    for (Change &C : Changes) {
      if (AlignWrappedOperand(C))
        C.StartOfTokenColumn -= 2;
    }
    AlignTokens(
        Style,
        [this](Change const &C) {
          // Align question operators if next operand is not wrapped, as
          // well as wrapped operands after question operator or last
          // colon in conditional sequence
          return (C.Tok->is(TT_ConditionalExpr) && C.Tok->is(tok::question) &&
                  &C != &Changes.back() && (&C + 1)->NewlinesBefore == 0 &&
                  !(&C + 1)->IsTrailingComment) ||
                 AlignWrappedOperand(C);
        },
        Changes, /*StartAt=*/0);
  }
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
    unsigned ChangeMaxColumn;

    if (Style.ColumnLimit == 0)
      ChangeMaxColumn = UINT_MAX;
    else if (Style.ColumnLimit >= Changes[i].TokenLength)
      ChangeMaxColumn = Style.ColumnLimit - Changes[i].TokenLength;
    else
      ChangeMaxColumn = ChangeMinColumn;

    // If we don't create a replacement for this change, we have to consider
    // it to be immovable.
    if (!Changes[i].CreateReplacement)
      ChangeMaxColumn = ChangeMinColumn;

    if (i + 1 != e && Changes[i + 1].ContinuesPPDirective)
      ChangeMaxColumn -= 2;
    // If this comment follows an } in column 0, it probably documents the
    // closing of a namespace and we don't want to align it.
    bool FollowsRBraceInColumn0 = i > 0 && Changes[i].NewlinesBefore == 0 &&
                                  Changes[i - 1].Tok->is(tok::r_brace) &&
                                  Changes[i - 1].StartOfTokenColumn == 0;
    bool WasAlignedWithStartOfNextLine = false;
    if (Changes[i].NewlinesBefore == 1) { // A comment on its own line.
      unsigned CommentColumn = SourceMgr.getSpellingColumnNumber(
          Changes[i].OriginalWhitespaceRange.getEnd());
      for (unsigned j = i + 1; j != e; ++j) {
        if (Changes[j].Tok->is(tok::comment))
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
    BreakBeforeNext = (i == 0) || (Changes[i].NewlinesBefore > 1) ||
                      // Never start a sequence with a comment at the beginning
                      // of the line.
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
    if (i + 1 != Changes.size())
      Changes[i + 1].PreviousEndOfTokenColumn += Shift;
    Changes[i].StartOfTokenColumn += Shift;
  }
}

void WhitespaceManager::alignEscapedNewlines() {
  if (Style.AlignEscapedNewlines == FormatStyle::ENAS_DontAlign)
    return;

  bool AlignLeft = Style.AlignEscapedNewlines == FormatStyle::ENAS_Left;
  unsigned MaxEndOfLine = AlignLeft ? 0 : Style.ColumnLimit;
  unsigned StartOfMacro = 0;
  for (unsigned i = 1, e = Changes.size(); i < e; ++i) {
    Change &C = Changes[i];
    if (C.NewlinesBefore > 0) {
      if (C.ContinuesPPDirective) {
        MaxEndOfLine = std::max(C.PreviousEndOfTokenColumn + 2, MaxEndOfLine);
      } else {
        alignEscapedNewlines(StartOfMacro + 1, i, MaxEndOfLine);
        MaxEndOfLine = AlignLeft ? 0 : Style.ColumnLimit;
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
        appendEscapedNewlineText(ReplacementText, C.NewlinesBefore,
                                 C.PreviousEndOfTokenColumn,
                                 C.EscapedNewlineColumn);
      else
        appendNewlineText(ReplacementText, C.NewlinesBefore);
      appendIndentText(
          ReplacementText, C.Tok->IndentLevel, std::max(0, C.Spaces),
          C.StartOfTokenColumn - std::max(0, C.Spaces), C.IsAligned);
      ReplacementText.append(C.CurrentLinePrefix);
      storeReplacement(C.OriginalWhitespaceRange, ReplacementText);
    }
  }
}

void WhitespaceManager::storeReplacement(SourceRange Range, StringRef Text) {
  unsigned WhitespaceLength = SourceMgr.getFileOffset(Range.getEnd()) -
                              SourceMgr.getFileOffset(Range.getBegin());
  // Don't create a replacement, if it does not change anything.
  if (StringRef(SourceMgr.getCharacterData(Range.getBegin()),
                WhitespaceLength) == Text)
    return;
  auto Err = Replaces.add(tooling::Replacement(
      SourceMgr, CharSourceRange::getCharRange(Range), Text));
  // FIXME: better error handling. For now, just print an error message in the
  // release version.
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
    assert(false);
  }
}

void WhitespaceManager::appendNewlineText(std::string &Text,
                                          unsigned Newlines) {
  for (unsigned i = 0; i < Newlines; ++i)
    Text.append(UseCRLF ? "\r\n" : "\n");
}

void WhitespaceManager::appendEscapedNewlineText(
    std::string &Text, unsigned Newlines, unsigned PreviousEndOfTokenColumn,
    unsigned EscapedNewlineColumn) {
  if (Newlines > 0) {
    unsigned Spaces =
        std::max<int>(1, EscapedNewlineColumn - PreviousEndOfTokenColumn - 1);
    for (unsigned i = 0; i < Newlines; ++i) {
      Text.append(Spaces, ' ');
      Text.append(UseCRLF ? "\\\r\n" : "\\\n");
      Spaces = std::max<int>(0, EscapedNewlineColumn - 1);
    }
  }
}

void WhitespaceManager::appendIndentText(std::string &Text,
                                         unsigned IndentLevel, unsigned Spaces,
                                         unsigned WhitespaceStartColumn,
                                         bool IsAligned) {
  switch (Style.UseTab) {
  case FormatStyle::UT_Never:
    Text.append(Spaces, ' ');
    break;
  case FormatStyle::UT_Always: {
    if (Style.TabWidth) {
      unsigned FirstTabWidth =
          Style.TabWidth - WhitespaceStartColumn % Style.TabWidth;

      // Insert only spaces when we want to end up before the next tab.
      if (Spaces < FirstTabWidth || Spaces == 1) {
        Text.append(Spaces, ' ');
        break;
      }
      // Align to the next tab.
      Spaces -= FirstTabWidth;
      Text.append("\t");

      Text.append(Spaces / Style.TabWidth, '\t');
      Text.append(Spaces % Style.TabWidth, ' ');
    } else if (Spaces == 1) {
      Text.append(Spaces, ' ');
    }
    break;
  }
  case FormatStyle::UT_ForIndentation:
    if (WhitespaceStartColumn == 0) {
      unsigned Indentation = IndentLevel * Style.IndentWidth;
      Spaces = appendTabIndent(Text, Spaces, Indentation);
    }
    Text.append(Spaces, ' ');
    break;
  case FormatStyle::UT_ForContinuationAndIndentation:
    if (WhitespaceStartColumn == 0)
      Spaces = appendTabIndent(Text, Spaces, Spaces);
    Text.append(Spaces, ' ');
    break;
  case FormatStyle::UT_AlignWithSpaces:
    if (WhitespaceStartColumn == 0) {
      unsigned Indentation =
          IsAligned ? IndentLevel * Style.IndentWidth : Spaces;
      Spaces = appendTabIndent(Text, Spaces, Indentation);
    }
    Text.append(Spaces, ' ');
    break;
  }
}

unsigned WhitespaceManager::appendTabIndent(std::string &Text, unsigned Spaces,
                                            unsigned Indentation) {
  // This happens, e.g. when a line in a block comment is indented less than the
  // first one.
  if (Indentation > Spaces)
    Indentation = Spaces;
  if (Style.TabWidth) {
    unsigned Tabs = Indentation / Style.TabWidth;
    Text.append(Tabs, '\t');
    Spaces -= Tabs * Style.TabWidth;
  }
  return Spaces;
}

} // namespace format
} // namespace clang
