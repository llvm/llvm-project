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
// clang-format off
#include "WhitespaceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

namespace clang {
namespace format {

bool WhitespaceManager::Change::IsBeforeInFile::operator()(
    const Change &C1, const Change &C2) const {
  return SourceMgr.isBeforeInTranslationUnit(
             C1.OriginalWhitespaceRange.getBegin(),
             C2.OriginalWhitespaceRange.getBegin()) ||
         (C1.OriginalWhitespaceRange.getBegin() ==
              C2.OriginalWhitespaceRange.getBegin() &&
          SourceMgr.isBeforeInTranslationUnit(
              C1.OriginalWhitespaceRange.getEnd(),
              C2.OriginalWhitespaceRange.getEnd()));
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
  if (Tok.Finalized || (Tok.MacroCtx && Tok.MacroCtx->Role == MR_ExpandedArg))
    return;
  Tok.setDecision((Newlines > 0) ? FD_Break : FD_Continue);
  Changes.push_back(Change(Tok, /*CreateReplacement=*/true, Tok.WhitespaceRange,
                           Spaces, StartOfTokenColumn, Newlines, "", "",
                           IsAligned, InPPDirective && !Tok.IsFirst,
                           /*IsInsideToken=*/false));
}

void WhitespaceManager::addUntouchableToken(const FormatToken &Tok,
                                            bool InPPDirective) {
  if (Tok.Finalized || (Tok.MacroCtx && Tok.MacroCtx->Role == MR_ExpandedArg))
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

bool WhitespaceManager::inputUsesCRLF(StringRef Text, bool DefaultToCRLF) {
  size_t LF = Text.count('\n');
  size_t CR = Text.count('\r') * 2;
  return LF == CR ? DefaultToCRLF : CR > LF;
}

void WhitespaceManager::replaceWhitespaceInToken(
    const FormatToken &Tok, unsigned Offset, unsigned ReplaceChars,
    StringRef PreviousPostfix, StringRef CurrentPrefix, bool InPPDirective,
    unsigned Newlines, int Spaces) {
  if (Tok.Finalized || (Tok.MacroCtx && Tok.MacroCtx->Role == MR_ExpandedArg))
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
  columnarizeKeywords();                                // TALLY
  columnarizeDeclarationSpecifierTokens();              // TALLY
  columnarizeDatatypeTokens();                          // TALLY
  columnarizeNoDiscardOrNoReturnOrTemplate ();          // TALLY
  columnarizeIdentifierTokens();                        // TALLY
  columnarizeLParenTokensAndSplitArgs();                // TALLY
  alignConsecutiveAssignmentsOnScopedVarName();         // TALLY
  alignConsecutiveAssignmentsOnVarNameAcrossSections(); // TALLY
  alignConsecutiveAssignmentsOnVarNameWithinSection();  // TALLY
  alignConsecutiveVarBitFields();                       // TALLY. We do NOT use alignConsecutiveBitFields().
  alignConsecutiveAssignmentsOnEqualsAcrossSections();  // TALLY
  alignConsecutiveAssignmentsOnEqualsWithinSection();   // TALLY
  alignConsecutiveLBraceOfVarDeclOrDef();               // TALLY
  alignConsecutiveAssignementsOnUsing ();               // TALLY
  alignConsecutiveMacros();
  alignConsecutiveShortCaseStatements();
  //alignConsecutiveDeclarations();
  alignConsecutiveBitFields();
  //alignConsecutiveAssignments();
  alignChainedConditionals();
  alignTrailingComments();
  alignEscapedNewlines();
  alignArrayInitializers();
  generateChanges();

  return Replaces;
}

// TALLY
bool WhitespaceManager::IsConditionBlock(const FormatToken *tkn)
{
    while (tkn) {
        if (tkn->Tok.isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::kw_for))
            return true;

        tkn = tkn->Previous;
    }

    return false;
}

void WhitespaceManager::calculateLineBreakInformation() {
  int prevTknStartOfColumnNWidth = 0;

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
    if (Changes[i - 1].IsInsideToken && Changes[i - 1].NewlinesBefore == 0) {
      LastOutsideTokenChange->TokenLength +=
          Changes[i - 1].TokenLength + Changes[i - 1].Spaces;
    } else {
      LastOutsideTokenChange = &Changes[i - 1];
    }

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
      if (Change.Tok->is(TT_LineComment) || !Change.IsInsideToken) {
        LastBlockComment = &Change;
      } else if ((Change.StartOfBlockComment = LastBlockComment)) {
        Change.IndentationOffset =
            Change.StartOfTokenColumn -
            Change.StartOfBlockComment->StartOfTokenColumn;
      }
    } else {
      LastBlockComment = nullptr;
    }

    if (Change.Tok->isOneOf(tok::r_paren, tok::kw_noexcept) &&
            Change.Tok->Next && Change.Tok->Next->is(tok::colon)) {
        Change.Tok->Next->SpacesRequiredBefore = 1;
        Change.Tok->Next->MustBreakBefore = false;
        Change.NewlinesBefore = 0;
        prevTknStartOfColumnNWidth = Change.StartOfTokenColumn + Change.Tok->ColumnWidth;
    }
    else if (Change.Tok->is(tok::colon) && Change.Tok->Previous &&
            Change.Tok->Previous->isOneOf(tok::r_paren, tok::kw_noexcept) &&
            Change.Tok->Next && Change.Tok->Next->is(tok::identifier)) {
        Change.Tok->Next->SpacesRequiredBefore = 1;
        Change.Tok->Next->MustBreakBefore = false;
        Change.Spaces = 1;
        Change.NewlinesBefore = 0;
        ++prevTknStartOfColumnNWidth;
        if (Change.StartOfTokenColumn > prevTknStartOfColumnNWidth)
            Change.StartOfTokenColumn = prevTknStartOfColumnNWidth;

        ++prevTknStartOfColumnNWidth;
    }
    else if (Change.Tok->is(tok::identifier) &&
            Change.Tok->Previous && Change.Tok->Previous->is(tok::colon) &&
            Change.Tok->Previous->Previous && Change.Tok->Previous->Previous->isOneOf(tok::r_paren, tok::kw_noexcept)) {
        Change.Spaces = 1;
        Change.NewlinesBefore = 0;
        ++prevTknStartOfColumnNWidth;
        if (Change.StartOfTokenColumn > prevTknStartOfColumnNWidth)
            Change.StartOfTokenColumn = prevTknStartOfColumnNWidth;
    }
/*
    if (Change.Spaces > 4 && Change.Tok->LbraceCount == 1 &&
            Change.Tok->IsInFunctionDefinitionScope && !Change.Tok->isTallyTrace() &&
            !Change.Tok->IsDatatype) {
        Change.Spaces = 4;
        Change.StartOfTokenColumn = 4;
        if (Change.NewlinesBefore > 2)
            Change.NewlinesBefore = 2;
    }
    */
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

    for (unsigned i = Change.Tok->FakeRParens; i > 0 && ScopeStack.size(); --i)
      if (ScopeStack.pop_back_val())
        --ConditionalsLevel;
  }
}

// Align a single sequence of tokens, see AlignTokens below.
// Column - The token for which Matches returns true is moved to this column.
// RightJustify - Whether it is the token's right end or left end that gets
// moved to that column.
template <typename F>
static void
AlignTokenSequence(const FormatStyle &Style, unsigned Start, unsigned End,
                   unsigned Column, bool RightJustify, F &&Matches,
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
  // The same holds for calling a function:
  //   double a = foo(x);
  //   int    b = bar(foo(y),
  //            foor(z));
  // Similar for broken string literals:
  //   double x = 3.14;
  //   auto s   = "Hello"
  //          "World";
  // Special handling is required for 'nested' ternary operators.
  SmallVector<unsigned, 16> ScopeStack;

  for (unsigned i = Start; i != End; ++i) {
    auto &CurrentChange = Changes[i];
    if (ScopeStack.size() != 0 &&
        CurrentChange.indentAndNestingLevel() <
            Changes[ScopeStack.back()].indentAndNestingLevel()) {
      ScopeStack.pop_back();
    }

    // Compare current token to previous non-comment token to ensure whether
    // it is in a deeper scope or not.
    unsigned PreviousNonComment = i - 1;
    while (PreviousNonComment > Start &&
           Changes[PreviousNonComment].Tok->is(tok::comment)) {
      --PreviousNonComment;
    }
    if (i != Start && CurrentChange.indentAndNestingLevel() >
                          Changes[PreviousNonComment].indentAndNestingLevel()) {
      ScopeStack.push_back(i);
    }

    bool InsideNestedScope = ScopeStack.size() != 0;
    bool ContinuedStringLiteral = i > Start &&
                                  CurrentChange.Tok->is(tok::string_literal) &&
                                  Changes[i - 1].Tok->is(tok::string_literal);
    bool SkipMatchCheck = InsideNestedScope || ContinuedStringLiteral;

    if (CurrentChange.NewlinesBefore > 0 && !SkipMatchCheck) {
      Shift = 0;
      FoundMatchOnLine = false;
    }

    // If this is the first matching token to be aligned, remember by how many
    // spaces it has to be shifted, so the rest of the changes on the line are
    // shifted by the same amount
    if (!FoundMatchOnLine && !SkipMatchCheck && Matches(CurrentChange)) {
      FoundMatchOnLine = true;
      Shift = Column - (RightJustify ? CurrentChange.TokenLength : 0) -
              CurrentChange.StartOfTokenColumn;
      CurrentChange.Spaces += Shift;
      // FIXME: This is a workaround that should be removed when we fix
      // http://llvm.org/PR53699. An assertion later below verifies this.
      if (CurrentChange.NewlinesBefore == 0) {
        CurrentChange.Spaces =
            std::max(CurrentChange.Spaces,
                     static_cast<int>(CurrentChange.Tok->SpacesRequiredBefore));
      }
    }

    if (Shift == 0)
      continue;

    // This is for function parameters that are split across multiple lines,
    // as mentioned in the ScopeStack comment.
    if (InsideNestedScope && CurrentChange.NewlinesBefore > 0) {
      unsigned ScopeStart = ScopeStack.back();
      auto ShouldShiftBeAdded = [&] {
        // Function declaration
        if (Changes[ScopeStart - 1].Tok->is(TT_FunctionDeclarationName))
          return true;

        // Lambda.
        if (Changes[ScopeStart - 1].Tok->is(TT_LambdaLBrace))
          return false;

        // Continued function declaration
        if (ScopeStart > Start + 1 &&
            Changes[ScopeStart - 2].Tok->is(TT_FunctionDeclarationName)) {
          return true;
        }

        // Continued (template) function call.
        if (ScopeStart > Start + 1 &&
            Changes[ScopeStart - 2].Tok->isOneOf(tok::identifier,
                                                 TT_TemplateCloser) &&
            Changes[ScopeStart - 1].Tok->is(tok::l_paren) &&
            Changes[ScopeStart].Tok->isNot(TT_LambdaLSquare)) {
          if (CurrentChange.Tok->MatchingParen &&
              CurrentChange.Tok->MatchingParen->is(TT_LambdaLBrace)) {
            return false;
          }
          if (Changes[ScopeStart].NewlinesBefore > 0)
            return false;
          if (CurrentChange.Tok->is(tok::l_brace) &&
              CurrentChange.Tok->is(BK_BracedInit)) {
            return true;
          }
          return Style.BinPackArguments;
        }

        // Ternary operator
        if (CurrentChange.Tok->is(TT_ConditionalExpr))
          return true;

        // Period Initializer .XXX = 1.
        if (CurrentChange.Tok->is(TT_DesignatedInitializerPeriod))
          return true;

        // Continued ternary operator
        if (CurrentChange.Tok->Previous &&
            CurrentChange.Tok->Previous->is(TT_ConditionalExpr)) {
          return true;
        }

        // Continued direct-list-initialization using braced list.
        if (ScopeStart > Start + 1 &&
            Changes[ScopeStart - 2].Tok->is(tok::identifier) &&
            Changes[ScopeStart - 1].Tok->is(tok::l_brace) &&
            CurrentChange.Tok->is(tok::l_brace) &&
            CurrentChange.Tok->is(BK_BracedInit)) {
          return true;
        }

        // Continued braced list.
        if (ScopeStart > Start + 1 &&
            Changes[ScopeStart - 2].Tok->isNot(tok::identifier) &&
            Changes[ScopeStart - 1].Tok->is(tok::l_brace) &&
            CurrentChange.Tok->isNot(tok::r_brace)) {
          for (unsigned OuterScopeStart : llvm::reverse(ScopeStack)) {
            // Lambda.
            if (OuterScopeStart > Start &&
                Changes[OuterScopeStart - 1].Tok->is(TT_LambdaLBrace)) {
              return false;
            }
          }
          if (Changes[ScopeStart].NewlinesBefore > 0)
            return false;
          return true;
        }

        // Continued template parameter.
        if (Changes[ScopeStart - 1].Tok->is(TT_TemplateOpener))
          return true;

        return false;
      };

      if (ShouldShiftBeAdded())
        CurrentChange.Spaces += Shift;
    }

    if (ContinuedStringLiteral)
      CurrentChange.Spaces += Shift;

    // We should not remove required spaces unless we break the line before.
    assert(Shift > 0 || Changes[i].NewlinesBefore > 0 ||
           CurrentChange.Spaces >=
               static_cast<int>(Changes[i].Tok->SpacesRequiredBefore) ||
           CurrentChange.Tok->is(tok::eof));

    CurrentChange.StartOfTokenColumn += Shift;
    if (i + 1 != Changes.size())
      Changes[i + 1].PreviousEndOfTokenColumn += Shift;

    // If PointerAlignment is PAS_Right, keep *s or &s next to the token
    if ((Style.PointerAlignment == FormatStyle::PAS_Right ||
         Style.ReferenceAlignment == FormatStyle::RAS_Right) &&
        CurrentChange.Spaces != 0) {
      const bool ReferenceNotRightAligned =
          Style.ReferenceAlignment != FormatStyle::RAS_Right &&
          Style.ReferenceAlignment != FormatStyle::RAS_Pointer;
      for (int Previous = i - 1;
           Previous >= 0 &&
           Changes[Previous].Tok->getType() == TT_PointerOrReference;
           --Previous) {
        assert(Changes[Previous].Tok->isPointerOrReference());
        if (Changes[Previous].Tok->isNot(tok::star)) {
          if (ReferenceNotRightAligned)
            continue;
        } else if (Style.PointerAlignment != FormatStyle::PAS_Right) {
          continue;
        }
        Changes[Previous + 1].Spaces -= Shift;
        Changes[Previous].Spaces += Shift;
        Changes[Previous].StartOfTokenColumn += Shift;
      }
    }
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
// When the parameter RightJustify is true, the operator will be
// right-justified. It is used to align compound assignments like `+=` and `=`.
// When RightJustify and ACS.PadOperators are true, operators in each block to
// be aligned will be padded on the left to the same length before aligning.
template <typename F>
static unsigned AlignTokens(const FormatStyle &Style, F &&Matches,
                            SmallVector<WhitespaceManager::Change, 16> &Changes,
                            unsigned StartAt,
                            const FormatStyle::AlignConsecutiveStyle &ACS = {},
                            bool RightJustify = false) {
  // We arrange each line in 3 parts. The operator to be aligned (the anchor),
  // and text to its left and right. In the aligned text the width of each part
  // will be the maximum of that over the block that has been aligned. Maximum
  // widths of each part so far. When RightJustify is true and ACS.PadOperators
  // is false, the part from start of line to the right end of the anchor.
  // Otherwise, only the part to the left of the anchor. Including the space
  // that exists on its left from the start. Not including the padding added on
  // the left to right-justify the anchor.
  unsigned WidthLeft = 0;
  // The operator to be aligned when RightJustify is true and ACS.PadOperators
  // is false. 0 otherwise.
  unsigned WidthAnchor = 0;
  // Width to the right of the anchor. Plus width of the anchor when
  // RightJustify is false.
  unsigned WidthRight = 0;

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

  // Whether the current line consists purely of comments.
  bool LineIsComment = true;

  // Aligns a sequence of matching tokens, on the MinColumn column.
  //
  // Sequences start from the first matching token to align, and end at the
  // first token of the first line that doesn't need to be aligned.
  //
  // We need to adjust the StartOfTokenColumn of each Change that is on a line
  // containing any matching token to be aligned and located after such token.
  auto AlignCurrentSequence = [&] {
    if (StartOfSequence > 0 && StartOfSequence < EndOfSequence) {
      AlignTokenSequence(Style, StartOfSequence, EndOfSequence,
                         WidthLeft + WidthAnchor, RightJustify, Matches,
                         Changes);
    }
    WidthLeft = 0;
    WidthAnchor = 0;
    WidthRight = 0;
    StartOfSequence = 0;
    EndOfSequence = 0;
  };

  unsigned i = StartAt;
  for (unsigned e = Changes.size(); i != e; ++i) {
    auto &CurrentChange = Changes[i];
    if (CurrentChange.indentAndNestingLevel() < IndentAndNestingLevel)
      break;

    if (CurrentChange.NewlinesBefore != 0) {
      CommasBeforeMatch = 0;
      EndOfSequence = i;

      // Whether to break the alignment sequence because of an empty line.
      bool EmptyLineBreak =
          (CurrentChange.NewlinesBefore > 1) && !ACS.AcrossEmptyLines;

      // Whether to break the alignment sequence because of a line without a
      // match.
      bool NoMatchBreak =
          !FoundMatchOnLine && !(LineIsComment && ACS.AcrossComments);

      if (EmptyLineBreak || NoMatchBreak)
        AlignCurrentSequence();

      // A new line starts, re-initialize line status tracking bools.
      // Keep the match state if a string literal is continued on this line.
      if (i == 0 || CurrentChange.Tok->isNot(tok::string_literal) ||
          Changes[i - 1].Tok->isNot(tok::string_literal)) {
        FoundMatchOnLine = false;
      }
      LineIsComment = true;
    }

    if (CurrentChange.Tok->isNot(tok::comment))
      LineIsComment = false;

    if (CurrentChange.Tok->is(tok::comma)) {
      ++CommasBeforeMatch;
    } else if (CurrentChange.indentAndNestingLevel() > IndentAndNestingLevel) {
      // Call AlignTokens recursively, skipping over this scope block.
      unsigned StoppedAt =
          AlignTokens(Style, Matches, Changes, i, ACS, RightJustify);
      i = StoppedAt - 1;
      continue;
    }

    if (!Matches(CurrentChange))
      continue;

    // If there is more than one matching token per line, or if the number of
    // preceding commas, do not match anymore, end the sequence.
    if (FoundMatchOnLine || CommasBeforeMatch != CommasBeforeLastMatch)
      AlignCurrentSequence();

    CommasBeforeLastMatch = CommasBeforeMatch;
    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = i;

    unsigned ChangeWidthLeft = CurrentChange.StartOfTokenColumn;
    unsigned ChangeWidthAnchor = 0;
    unsigned ChangeWidthRight = 0;
    if (RightJustify)
      if (ACS.PadOperators)
        ChangeWidthAnchor = CurrentChange.TokenLength;
      else
        ChangeWidthLeft += CurrentChange.TokenLength;
    else
      ChangeWidthRight = CurrentChange.TokenLength;
    for (unsigned j = i + 1; j != e && Changes[j].NewlinesBefore == 0; ++j) {
      ChangeWidthRight += Changes[j].Spaces;
      // Changes are generally 1:1 with the tokens, but a change could also be
      // inside of a token, in which case it's counted more than once: once for
      // the whitespace surrounding the token (!IsInsideToken) and once for
      // each whitespace change within it (IsInsideToken).
      // Therefore, changes inside of a token should only count the space.
      if (!Changes[j].IsInsideToken)
        ChangeWidthRight += Changes[j].TokenLength;
    }

    // If we are restricted by the maximum column width, end the sequence.
    unsigned NewLeft = std::max(ChangeWidthLeft, WidthLeft);
    unsigned NewAnchor = std::max(ChangeWidthAnchor, WidthAnchor);
    unsigned NewRight = std::max(ChangeWidthRight, WidthRight);
    // `ColumnLimit == 0` means there is no column limit.
    if (Style.ColumnLimit != 0 &&
        Style.ColumnLimit < NewLeft + NewAnchor + NewRight) {
      AlignCurrentSequence();
      StartOfSequence = i;
      WidthLeft = ChangeWidthLeft;
      WidthAnchor = ChangeWidthAnchor;
      WidthRight = ChangeWidthRight;
    } else {
      WidthLeft = NewLeft;
      WidthAnchor = NewAnchor;
      WidthRight = NewRight;
    }
  }

  EndOfSequence = i;
  AlignCurrentSequence();
  return i;
}

// TALLY: Align a single sequence of tokens, see AlignTokens below.
template <typename F>
static void
AlignTokenSequence(unsigned Start, unsigned End, unsigned Column, F&& Matches,
    SmallVector<WhitespaceManager::Change, 16>& Changes,
    /* TALLY */ bool IgnoreScope, /* TALLY */ bool IgnoreCommas) {

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
    SmallVector<unsigned, 16> ScopeStack;

    for (unsigned i = Start; i != End; ++i) {
        if (!IgnoreScope && ScopeStack.size() != 0 &&
            Changes[i].indentAndNestingLevel() <
            Changes[ScopeStack.back()].indentAndNestingLevel())
            ScopeStack.pop_back();

        // Compare current token to previous non-comment token to ensure whether
        // it is in a deeper scope or not.
        unsigned PreviousNonComment = i - 1;
        while (PreviousNonComment > Start &&
            Changes[PreviousNonComment].Tok->is(tok::comment))
            PreviousNonComment--;
        if (!IgnoreScope && i != Start && Changes[i].indentAndNestingLevel() >
            Changes[PreviousNonComment].indentAndNestingLevel())
            ScopeStack.push_back(i);

        bool InsideNestedScope = ScopeStack.size() != 0;

        if (Changes[i].NewlinesBefore > 0 &&
            (!InsideNestedScope || IgnoreScope)) {
            Shift = 0;
            FoundMatchOnLine = false;
        }

        if (!FoundMatchOnLine && (IgnoreScope || !InsideNestedScope) &&
            Matches(Changes[i])) {
            if (!(Changes[i].Tok->IsFunctionDefinitionLine && Changes[i].Tok->is(tok::identifier) && (i > 0)
                && Changes[i-1].Tok->isOneOf(tok::coloncolon, tok::l_square, tok::less,tok::r_brace,tok::kw_noexcept))
                && Changes[i].Tok->LparenCount == 0
                && !Changes[i-1].Tok->isOneOf(tok::r_brace, tok::kw_noexcept, tok::coloncolon)) {
                FoundMatchOnLine = true;
                Shift = Column - Changes[i].StartOfTokenColumn;
                Changes[i].Spaces += Shift;
            }
        }


        // This is for function parameters that are split across multiple lines,
        // as mentioned in the ScopeStack comment.
        if (InsideNestedScope && Changes[i].NewlinesBefore > 0) {
            unsigned ScopeStart = ScopeStack.back();
            if (Changes[ScopeStart - 1].Tok->is(TT_FunctionDeclarationName) ||
                (ScopeStart > Start + 1 &&
                    Changes[ScopeStart - 2].Tok->is(TT_FunctionDeclarationName)))
                Changes[i].Spaces += Shift;
        }

        assert(Shift >= 0);
        Changes[i].StartOfTokenColumn += Shift;
        if (i + 1 != Changes.size())
            Changes[i + 1].PreviousEndOfTokenColumn += Shift;
    }
}

// TALLY: Walk through a subset of the changes, starting at StartAt, and find
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
static unsigned AlignTokens(const FormatStyle& Style, F&& Matches,
    SmallVector<WhitespaceManager::Change, 16>& Changes,
    /* TALLY */ bool IgnoreScope,
    /* TALLY */ bool IgnoreCommas,
    unsigned StartAt,
    /* TALLY */ unsigned MaxNewlinesBeforeSectionBreak,
    /* TALLY */ bool NonMatchingLineBreaksSection,
    /* TALLY */ bool AllowBeyondColumnLimitForAlignment,
    /* TALLY */ unsigned MaxLeadingSpacesForAlignment,
    /* TALLY */ bool ForceAlignToFourSpaces 
) {
    // TALLY
    unsigned ColumnLimitInEffect = AllowBeyondColumnLimitForAlignment ? Style.ColumnLimitExtended : Style.ColumnLimit;

    unsigned MinColumn = 0;
    unsigned MaxColumn = UINT_MAX;

    // Line number of the start and the end of the current token sequence.
    unsigned StartOfSequence = 0;
    unsigned EndOfSequence = 0;

    // Measure the scope level (i.e. depth of (), [], {}) of the first token, and
    // abort when we hit any token in a higher scope than the starting one.
    auto IndentAndNestingLevel = StartAt < Changes.size()
        ? Changes[StartAt].indentAndNestingLevel()
        : std::tuple<unsigned, unsigned, unsigned>(0, 0, 0);

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
                Changes, IgnoreScope, IgnoreCommas);
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
            if (Changes[i].NewlinesBefore > MaxNewlinesBeforeSectionBreak || (NonMatchingLineBreaksSection && !FoundMatchOnLine))
                AlignCurrentSequence();

            FoundMatchOnLine = false;
        }

        if (Changes[i].Tok->is(tok::comma)) {
            ++CommasBeforeMatch;
        }
        else if (!IgnoreScope && (Changes[i].indentAndNestingLevel() > IndentAndNestingLevel)) {
            // Call AlignTokens recursively, skipping over this scope block.
            unsigned StoppedAt =
                AlignTokens(Style, Matches, Changes, IgnoreScope, IgnoreCommas, i,
                    MaxNewlinesBeforeSectionBreak, NonMatchingLineBreaksSection, 
                    AllowBeyondColumnLimitForAlignment, MaxLeadingSpacesForAlignment, ForceAlignToFourSpaces);
            i = StoppedAt - 1;
            continue;
        }

        if (!Matches(Changes[i]))
            continue;

        // If there is more than one matching token per line, or if the number of
        // preceding commas, do not match anymore, end the sequence.
        if (FoundMatchOnLine ||
            (!IgnoreCommas && (CommasBeforeMatch != CommasBeforeLastMatch)))
            AlignCurrentSequence();

        CommasBeforeLastMatch = CommasBeforeMatch;
        FoundMatchOnLine = true;

        if (StartOfSequence == 0)
            StartOfSequence = i;

        unsigned ChangeMinColumn = Changes[i].StartOfTokenColumn;
        int LineLengthAfter = -Changes[i].Spaces;
        for (unsigned j = i; j != e && Changes[j].NewlinesBefore == 0; ++j)
            LineLengthAfter += Changes[j].Spaces + Changes[j].TokenLength;
        unsigned ChangeMaxColumn = ColumnLimitInEffect - LineLengthAfter;

        int leadingSpacesReqd = ChangeMinColumn - MinColumn;
        if (leadingSpacesReqd < 0) {
            leadingSpacesReqd *= -1;
        }
        // If we are restricted by the maximum leading spaces limit or maximum column width, end the sequence.
        if (
            (unsigned)leadingSpacesReqd > MaxLeadingSpacesForAlignment ||
            ChangeMinColumn > MaxColumn ||
            ChangeMaxColumn < MinColumn ||
            (!IgnoreCommas && (CommasBeforeLastMatch != CommasBeforeMatch))
            ) {
            AlignCurrentSequence();
            StartOfSequence = i;
        }

        MinColumn = std::max(MinColumn, ChangeMinColumn);
        MaxColumn = std::min(MaxColumn, ChangeMaxColumn);

        // TALLY: Force align to four spaces
        if (ForceAlignToFourSpaces) {
            if (MinColumn % 4 != 0) {
                unsigned int pad = 4 - (MinColumn % 4);
                MinColumn += pad;
                MaxColumn += pad;
            }
        }
    }

    EndOfSequence = i;
    AlignCurrentSequence();
    return i;
}

// TALLY: Align assignments on scoped variable name (WITHIN SECTION)
void WhitespaceManager::alignConsecutiveAssignmentsOnScopedVarName() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {

            bool retval = 
                (C.Tok->isScopedVarNameInDecl() &&
                C.Tok->HasSemiColonInLine);

            return retval;
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/true,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX, 
            /*ForceAlignToFourSpaces*/false);
}

// TALLY : Align the consecutive constexprs
void WhitespaceManager::alignConsecutiveLBraceOfVarDeclOrDef() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {
            return C.Tok->isLBraceOfConstexprOrVarDelcOrDef() && 
                   C.Tok->HasSemiColonInLine; // Ensure is terminated by a semi colon.
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/true,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX, 
            /*ForceAlignToFourSpaces*/true);
}

void WhitespaceManager::alignConsecutiveAssignementsOnUsing () {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {
            bool retval =  (C.Tok->MyLine && C.Tok->MyLine->First &&
                            C.Tok->MyLine->First->is(tok::kw_using) &&
                            C.Tok->HasSemiColonInLine && 
                            C.Tok->Previous && C.Tok->Previous->Previous && C.Tok->Previous->Previous == C.Tok->MyLine->First &&
                            C.Tok->is(tok::equal));

            return retval;
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/true,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX, 
            /*ForceAlignToFourSpaces*/true);
}

// TALLY: Align assignments on variable name (ACROSS SECTIONS)
// Mutually exclusive with alignConsecutiveAssignmentsOnVarNameWithinSection()
void WhitespaceManager::alignConsecutiveAssignmentsOnVarNameAcrossSections() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {

            return
                C.Tok->isVarNameInDecl() &&
                C.Tok->HasSemiColonInLine &&
                C.Tok->LbraceCount > 0 &&
                C.Tok->IsClassScope;
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/false,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX,
            /*ForceAlignToFourSpaces*/false);
}

// TALLY: Align assignments on variable name (WITHIN SECTION)
// Mutually exclusive with alignConsecutiveAssignmentsOnVarNameAcrossSections()
void WhitespaceManager::alignConsecutiveAssignmentsOnVarNameWithinSection() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {

            bool retval = 
                C.Tok->isVarNameInDecl() &&
                C.Tok->HasSemiColonInLine &&
                !C.Tok->IsClassScope;

            return retval;
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/false,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX,
            /*ForceAlignToFourSpaces*/true);
}

// TALLY: Align on bit field colon in a variable declaration (ACROSS SECTIONS)
void WhitespaceManager::alignConsecutiveVarBitFields() {

    AlignTokens(Style,
        [&](const Change& C) {

            // Do not align on ':' that is first on a line.
            if (C.NewlinesBefore > 0)
                return false;

            // Do not align on ':' that is last on a line.
            if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
                return false;

            return C.Tok->is(TT_BitFieldColon);
            /*
            return
                C.Tok->is(tok::colon) &&
                C.Tok->HasSemiColonInLine &&
                C.Tok->getPreviousNonComment() != nullptr &&
                C.Tok->getPreviousNonComment()->isVarNameInDecl();
            */
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/false,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/UINT_MAX,
            /*ForceAlignToFourSpaces*/false);
}

/// TALLY: Align consecutive assignments over all \c Changes (ACROSS SECTIONS)
// Mutually exclusive with alignConsecutiveAssignmentsWithinSection()
void WhitespaceManager::alignConsecutiveAssignmentsOnEqualsAcrossSections() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {
            // Do not align on equal signs that are first on a line.
            if (C.NewlinesBefore > 0)
                return false;

            // Do not align on equal signs that are last on a line.
            if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
                return false;

            return
                C.Tok->is(tok::equal) &&
                C.Tok->HasSemiColonInLine &&
                C.Tok->getPreviousNonComment() != nullptr &&
                C.Tok->getPreviousNonComment()->isVarNameInDecl();
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/2, /*NonMatchingLineBreaksSection=*/false,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/16,
            /*ForceAlignToFourSpaces*/false);
}

/// TALLY: Align consecutive assignments over all \c Changes (WITHIN SECTION)
// Mutually exclusive with alignConsecutiveAssignmentsDeclAcrossSections()
void WhitespaceManager::alignConsecutiveAssignmentsOnEqualsWithinSection() {
    if (!Style.AlignConsecutiveAssignments.Enabled)
        return;

    AlignTokens(Style,
        [&](const Change& C) {
            // Do not align on equal signs that are first on a line.
            if (C.NewlinesBefore > 0)
                return false;

            // Do not align on equal signs that are last on a line.
            if (&C != &Changes.back() && (&C + 1)->NewlinesBefore > 0)
                return false;

            return
                C.Tok->is(tok::equal) &&
                C.Tok->HasSemiColonInLine &&
                C.Tok->isPrevBeforeInterimsVarWithoutDatatype();
        },
        Changes, /*IgnoreScope=*/false, /*IgnoreCommas=*/false, /*StartAt=*/0,
            /*MaxNewlinesBeforeSectionBreak=*/1, /*NonMatchingLineBreaksSection=*/true,
            /*AllowBeyondColumnLimitForAlignment=*/true, /*MaxLeadingSpacesForAlignment=*/12,
            /*ForceAlignToFourSpaces*/false);
}

/// TALLY: Columnarize specific tokens over all \c Changes.
void WhitespaceManager::columnarizePPKeywords() {
    // First loop
    for (int i = 0; i < Changes.size(); ++i) {
        const FormatToken* MyTok = Changes[i].Tok;

        if (MyTok->isPPKeywordAndPrevHash()) {
            size_t tokSize = ((StringRef)MyTok->TokenText).size() + 1;
            MaxPPKeywordLen = MaxPPKeywordLen < tokSize ? tokSize : MaxPPKeywordLen;
        }
    }

    unsigned toPad = MaxPPKeywordLen + 1;
    unsigned pad = 1;

    while (toPad % Style.TabWidth != 0) {
        toPad++;
        pad++;
    }

    // Second loop
    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* MyTok = Changes[i].Tok;
        const FormatToken* PrevTok = MyTok->getPreviousNonComment();

        if (PrevTok && PrevTok->isPPKeywordAndPrevHash()) {

            size_t tokSize = ((StringRef)PrevTok->TokenText).size() + 1;
            size_t lenDiff = MaxPPKeywordLen - tokSize;

            Changes[i].Spaces = pad + lenDiff;
        }
    }
}

/// TALLY: Columnarize specific tokens over all \c Changes.
void WhitespaceManager::columnarizePPDefineKeyword() {
    // First loop
    for (int i = 0; i < Changes.size(); ++i) {
        const FormatToken* MyTok = Changes[i].Tok;

        if (MyTok->isPPDefineKeywordAndPrevHash()) {
            const FormatToken* NextTok = MyTok->getNextNonComment();
            if (NextTok) {
                size_t tokSize = ((StringRef)NextTok->TokenText).size();
                MaxPPDefineLHSLen = MaxPPDefineLHSLen < tokSize ? tokSize : MaxPPDefineLHSLen;
            }
        }
    }

    unsigned toPad = MaxPPDefineLHSLen + 1;
    unsigned pad = 1;

    while (toPad % Style.TabWidth != 0) {
        toPad++;
        pad++;
    }

    // Second loop
    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* PrevTok = Changes[i].Tok->getPreviousNonComment();

        if (PrevTok) {
            const FormatToken* PrevPrevTok = PrevTok->getPreviousNonComment();

            if (PrevPrevTok && PrevPrevTok->isPPDefineKeywordAndPrevHash()) {
                // PrevTok is LHS
                size_t tokSize = ((StringRef)PrevTok->TokenText).size();
                size_t lenDiff = MaxPPDefineLHSLen - tokSize;
                // Spaces before RHS
                Changes[i].Spaces = pad + lenDiff;
            }
        }
    }
}

// TALLY: Columnarize keywords tokens over all \c Changes.
void WhitespaceManager::columnarizeKeywords()
{
    int spacebefswitch {};

    if (!Style.AlignConsecutiveDeclarations.Enabled)
        return;

    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* MyTok = Changes[i].Tok;

        if (MyTok->IsInFunctionDefinitionScope) {

            if (MyTok->is(tok::kw_switch)) {

                spacebefswitch = Changes[i].Spaces;
                continue;
            }

            if (MyTok->is(tok::kw_case)) {

                Changes[i].Spaces = spacebefswitch + 4;
                Changes[i].StartOfTokenColumn = spacebefswitch + 4;
                continue;
            }
        }
    }
}

/// TALLY: Columnarize specific tokens over all \c Changes.
// TODO: Check 'template' use-cases and adapt
// TODO: Works only if declaration specifiers and datatypes do not have inline comments between the tokens.
// TODO: Assumes tab size is 4. Need to fix to accept variable tab sizes.
void WhitespaceManager::columnarizeDeclarationSpecifierTokens() {
    unsigned dummy;

    if (!Style.AlignConsecutiveDeclarations.Enabled)
        return;

    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* MyTok = Changes[i].Tok;

        // 'const' is also applicable to params in addition to being a decl specifier, so filter out on LparenCount
        if ((!(MyTok->IsClassScope || MyTok->IsStructScope)) || MyTok->LbraceCount == 0 || MyTok->LparenCount > 0)
            continue;

        const FormatToken* PrevTok = MyTok->getPreviousNonComment();
        if (PrevTok && PrevTok->isAfterNoDiscardOrNoReturnOrTemplate(Changes[i].NewlinesBefore)) {
            PrevTok = nullptr;
        }

        // 'const' is also applicable after parens, so filter out such tokens
        if (MyTok->is(tok::kw_const) && PrevTok && PrevTok->is(tok::r_paren))
            continue;

        // Filter out the template token since they lie in a separate line itself.
        if (MyTok->isAfterNoDiscardOrNoReturnOrTemplate(dummy))
            continue;

        AnnotatedLine* MyLine = MyTok->MyLine;

        // As spaces before 'static' or 'virtual' has been set to zero, if static or virtual 
        // is not the first specifier in the list, then it will concatenate with the preceding
        // specifier.
        if ((MyTok->isDeclSpecStaticOrVirtual() && PrevTok == nullptr) || (MyTok->isDeclSpecInlineOrExtern() && MyTok->getNextNonComment()->isDeclSpecStaticOrVirtual())) {
            Changes[i].StartOfTokenColumn = 0;
            Changes[i].Spaces = 0;
            MyLine->LastSpecifierPadding = MyTok->is(tok::kw_static) ? 2 : 1; // len(static)=6, len(virtual)=7
            MyLine->LastSpecifierTabs += 2;
            MaxSpecifierTabs = MaxSpecifierTabs < MyLine->LastSpecifierTabs ? MyLine->LastSpecifierTabs : MaxSpecifierTabs;
        }
        else if (MyTok->isDeclarationSpecifier()) {

            if (PrevTok && PrevTok->isDeclSpecStaticOrVirtual()) {
                Changes[i].Spaces = MyLine->LastSpecifierPadding;
            }
            else {
                if (MyLine->LastSpecifierTabs == 0) {
                    MyLine->LastSpecifierTabs = 2;
                    if (MyTok->is(tok::kw_friend))
                        Changes[i].Spaces = MyTok->NewlinesBefore != 0 ? MyLine->LastSpecifierTabs * Style.TabWidth : 1;
                }
                else {
                    Changes[i].Spaces = MyLine->LastSpecifierPadding;
                }
            }

            Changes[i].StartOfTokenColumn = MyLine->LastSpecifierTabs * Style.TabWidth;

            // len=5
            if (MyTok->is(tok::kw_const)) {
                MyLine->LastSpecifierPadding = 3;
                if (PrevTok && PrevTok->isPointerOrRef()) {
                    MyLine->LastSpecifierPadding += 1;
                    Changes[i].Spaces = MyLine->LastSpecifierPadding;
                }

                if (!PrevTok && (MyLine->MightBeFunctionDecl || MyTok->IsStructScope)) {
                    Changes[i].Spaces += MyLine->LastSpecifierPadding + 1;
                    Changes[i].StartOfTokenColumn += 4;
                }
                MyLine->LastSpecifierTabs += 2;
            }
            // len=6
            else if (MyTok->isOneOf(tok::kw_inline, tok::kw_friend, tok::kw_extern)) {
                   MyLine->LastSpecifierPadding = 2;
                   MyLine->LastSpecifierTabs += 2;
            }
            // len=7
            else if (MyTok->is(tok::kw_mutable)) {
                MyLine->LastSpecifierPadding = 1;
                MyLine->LastSpecifierTabs += 2;
            }
            // len=8
            else if (MyTok->isOneOf(tok::kw_volatile, tok::kw_explicit, tok::kw_register)) {
                MyLine->LastSpecifierPadding = 4;
                MyLine->LastSpecifierTabs += 3;
            }
            // len=9
            else if (MyTok->is(tok::kw_constexpr)) {
                MyLine->LastSpecifierPadding = 3;
                if (!PrevTok && (MyLine->MightBeFunctionDecl || MyTok->IsStructScope)) {
                    Changes[i].Spaces += MyLine->LastSpecifierPadding + 1;
                    Changes[i].StartOfTokenColumn += 4;
                }
                MyLine->LastSpecifierTabs += 3;
            }
            // len=12
            else if (MyTok->is(tok::kw_thread_local)) {
                MyLine->LastSpecifierPadding = 4;
                MyLine->LastSpecifierTabs += 4;
            }
            // variable length
            else if (MyTok->is(tok::kw_alignas)) {
                const FormatToken* NextTok = MyTok->getNextNonComment();
                size_t interimSize = 0;
                while (NextTok && !NextTok->is(tok::r_paren))
                {
                    interimSize += NextTok->SpacesRequiredBefore;
                    interimSize += ((StringRef)NextTok->TokenText).size();
                    NextTok = NextTok->getNextNonComment();
                }
                if (NextTok && NextTok->is(tok::r_paren)) {
                    interimSize += NextTok->SpacesRequiredBefore;
                    interimSize++;
                    int toPad = 7 + interimSize;
                    while (toPad % Style.TabWidth != 0)
                        toPad++;
                    MyLine->LastSpecifierPadding = toPad - (7 + interimSize);
                    MyLine->LastSpecifierTabs += (toPad / 4);
                }
            }
            MaxSpecifierTabs = MaxSpecifierTabs < MyLine->LastSpecifierTabs ? MyLine->LastSpecifierTabs : MaxSpecifierTabs;
        }
    }
}

/// TALLY: Columnarize specific tokens over all \c Changes.
// TODO: Works only if declaration specifiers and datatypes do not have inline comments between the tokens.

void WhitespaceManager::columnarizeDatatypeTokens() {
  if (!Style.AlignConsecutiveDeclarations.Enabled)
    return;

  if (MaxSpecifierTabs < 4)
    MaxSpecifierTabs = 4;

  int bracecount = 0;

  for (int i = 0; i < Changes.size(); ++i) {

    const FormatToken *MyTok = Changes[i].Tok;

    if (!(MyTok->IsClassScope || MyTok->IsStructScope) || MyTok->LbraceCount == 0 || MyTok->LparenCount > 0) 
      continue;

    if (MyTok->is(tok::less) && MyTok->Previous->is(tok::kw_template)) {

      ++bracecount;
      continue;
    }

    if (bracecount) {

      if (MyTok->is(tok::greater))
        --bracecount;
      continue;
    }

    if (MyTok->IsDatatype) {

        AnnotatedLine *MyLine = MyTok->MyLine;

        bool functionNameAfterInterims = MyTok->isFunctionNameAfterInterims();
        bool memVarNameAfterInterims = MyTok->isMemberVariableNameAfterInterims();

        bool ismaybeunused = (MyTok->Previous && MyTok->Previous->Previous && MyTok->Previous->Previous->isMaybeUnused() && MyLine->First == MyTok->Previous->Previous);

      if (functionNameAfterInterims || memVarNameAfterInterims || ismaybeunused) {
          int j = ismaybeunused ? i - 2 : i;

          size_t tokSize = ((StringRef)MyTok->TokenText).size();

          FormatToken *NextTok = MyTok->getNextNonCommentNonConst();

          size_t interimSize = 0;

        while (NextTok && NextTok->IsInterimBeforeName) {

          interimSize += NextTok->SpacesRequiredBefore;
          interimSize += ((StringRef)NextTok->TokenText).size();
          NextTok = NextTok->getNextNonCommentNonConst();
        }

        if (ismaybeunused) {
          NextTok = NextTok->getNextNonCommentNonConst ();
          if (NextTok) {
            NextTok = NextTok->getNextNonCommentNonConst();
            NextTok->PrevTokenSizeForColumnarization = tokSize;
            NextTok->IsDatatype = true;
          }
        }

        tokSize = ismaybeunused ? 4 + tokSize : interimSize + tokSize;

        if (NextTok) {

          if ((functionNameAfterInterims && NextTok->isFunctionAndNextLeftParen()) ||
              (memVarNameAfterInterims && NextTok->isMemberVarNameInDecl())) {

            NextTok->PrevTokenSizeForColumnarization = tokSize;
          }
        }

        MaxDatatypeLen = MaxDatatypeLen < tokSize ? tokSize : MaxDatatypeLen;

        if (MyLine->LastSpecifierTabs == 0 || MyLine->First->isMaybeUnused()) {

          Changes[j].Spaces = MaxSpecifierTabs * Style.TabWidth;
          MyLine->LastSpecifierTabs = MaxSpecifierTabs;
        }
        else if (MyLine->LastSpecifierTabs < MaxSpecifierTabs) {

          Changes[j].Spaces = ((MaxSpecifierTabs - MyLine->LastSpecifierTabs) * Style.TabWidth) + MyLine->LastSpecifierPadding;
          MyLine->LastSpecifierTabs = MaxSpecifierTabs;
        }
        else if (MyLine->LastSpecifierTabs == MaxSpecifierTabs) {

          Changes[j].Spaces = MyLine->LastSpecifierPadding;
        }

        Changes[j].StartOfTokenColumn = MaxSpecifierTabs * Style.TabWidth;
      }
    }
  }
}

/// TALLY : For checking if the function decl in class body is a template/nodiscard/noreturn type.
void WhitespaceManager::columnarizeNoDiscardOrNoReturnOrTemplate() {
  int spacecount;
  int bracecount;

  if (!Style.AlignConsecutiveDeclarations.Enabled)
    return;

  if (MaxSpecifierTabs < 4)
    MaxSpecifierTabs = 4;

  for (int i = 0; i < Changes.size(); ++i) {

    const FormatToken *MyTok = Changes[i].Tok;

    // Space before [[maybe_unused]]
    if (MyTok->is(tok::l_square) && MyTok->Next && MyTok->Next->is(tok::l_square)
        && MyTok->LparenCount
        && (!MyTok->Previous || (MyTok->Previous && MyTok->Previous->is(tok::comment)))) {
      Changes[i].Spaces = 1;
    } // arrangement like LocatorType & pLocation
    else if (MyTok->is(tok::identifier) && MyTok->Previous
        && MyTok->Previous->isOneOf(tok::amp, tok::ampamp, tok::star)
        && MyTok->Previous->Previous && MyTok->Previous->Previous->is(tok::identifier)
        && MyTok->LparenCount && !IsConditionBlock(MyTok->Previous->Previous)) {
      Changes[i].Spaces = 1;
    }

    if ((!(MyTok->IsClassScope || MyTok->IsStructScope)) || MyTok->LbraceCount == 0 || MyTok->LparenCount > 0)
      continue;

    if (MyTok->isNoDiscardOrNoReturnOrTemplate()) {

      if (MyTok->is(tok::l_square)) {

        Changes[i].StartOfTokenColumn = MaxSpecifierTabs * Style.TabWidth;
        Changes[i].Spaces = MaxSpecifierTabs * Style.TabWidth;

          const FormatToken *next = MyTok->getNextNonComment();

        if (next && next->is(tok::l_square)) {

          ++i;
          Changes[i].StartOfTokenColumn = MaxSpecifierTabs * Style.TabWidth;
          Changes[i].Spaces = 0;

          next = next->getNextNonComment();
          if (next && (next->TokenText.startswith("nodiscard")
              || next->TokenText.startswith("noreturn")
              || next->TokenText.startswith("maybe_unused"))) {

            ++i;
            Changes[i].StartOfTokenColumn += MaxSpecifierTabs * Style.TabWidth;
            Changes[i].Spaces = 0;

            next = next->getNextNonComment();
            if (next && next->is(tok::r_square)) {

              ++i;
              Changes[i].StartOfTokenColumn += MaxSpecifierTabs * Style.TabWidth;
              Changes[i].Spaces = 0;

              next = next->getNextNonComment();
              if (next && next->is(tok::r_square)) {

                ++i;
                Changes[i].StartOfTokenColumn += MaxSpecifierTabs * Style.TabWidth;
                Changes[i].Spaces = 0;
              }
            }
          }
        }
      }

      FormatToken * aftertemplate = (FormatToken * )MyTok->walkTemplateBlockInClassDecl();
      bool isfrienddecl = aftertemplate ? aftertemplate->is(tok::kw_friend) : false;

      if (MyTok->is(tok::kw_template) &&
          MyTok->MyLine &&
          (MyTok->MyLine->MightBeFunctionDecl || isfrienddecl) &&
          MyTok->MyLine->endsWith(tok::semi) &&
          MyTok->LbraceCount > 0 &&
          MyTok->LparenCount == 0 &&
          MyTok->LArrowCount == 0) {

        spacecount = 0;
        bracecount = 0;
        
        Changes[i].StartOfTokenColumn = isfrienddecl ? 8 : MaxSpecifierTabs * Style.TabWidth;
        Changes[i].Spaces = isfrienddecl ? 8 : MaxSpecifierTabs * Style.TabWidth;;

        ++i;
        const FormatToken *curr = MyTok->Next;

        if (curr->is(tok::less)) {
            spacecount = 0;
            ++bracecount;
        }

        while (bracecount) {

            curr = curr->getNextNonComment();
            ++i;

            if (!curr)
                break;
            else if (curr->is(tok::less)) {
                spacecount = 0;
                ++bracecount;
            }
            else if (curr->is(tok::greater)) {
                spacecount = 0;
                --bracecount;
            }
            else if (curr->is(tok::comma) || curr->is(tok::ellipsis)) {
                spacecount = 0;
            }
            else if (curr->isOneOf (tok::kw_template, tok::kw_typename, tok::kw_class) ||
                curr->isDatatype()) {
              if (curr->Previous && curr->Previous->isOneOf(tok::ellipsis, tok::less, tok::coloncolon))
                spacecount = 0;
              else 
                spacecount = 1;
            }
            else if (curr->is(tok::identifier)) {
                if (curr->Previous && curr->Previous->isOneOf(tok::less, tok::coloncolon)) // tok::ellipsis,
                    spacecount = 0;
                else
                    spacecount = 1;
            }

            Changes[i].StartOfTokenColumn += MaxSpecifierTabs * Style.TabWidth;
            Changes[i].Spaces = spacecount;
        } while (bracecount != 0);
      }
    }
  }
}

/// TALLY: Columnarize specific tokens over all \c Changes.
void WhitespaceManager::columnarizeIdentifierTokens() {
    if (!Style.AlignConsecutiveDeclarations.Enabled)
        return;

    unsigned toPad = MaxDatatypeLen + 1;
    unsigned pad = 1;

    while (toPad % Style.TabWidth != 0) {
        toPad++;
        pad++;
    }

    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* MyTok = Changes[i].Tok;

        if ((!(MyTok->IsClassScope || MyTok->IsStructScope)) && (MyTok->LbraceCount == 0 || MyTok->LparenCount > 0))
            continue;

        // Dont align bitfield specifiers.
        if (MyTok->Previous && MyTok->Previous->is(TT_BitFieldColon))
            continue;

        FormatToken* NextTok = MyTok->getNextNonCommentNonConst();
        FormatToken* PrevTok = MyTok->getPreviousNonComment();

        if (MyTok->isMemberVarNameInDecl()) {

            size_t lenDiff = MaxDatatypeLen - MyTok->PrevTokenSizeForColumnarization;
            
            if (MyTok->Previous && MyTok->Previous->is(tok::l_brace) && MyTok->Next && !MyTok->Next->is(tok::comment)) {
                Changes[i].Spaces = 0;
            }
            else if (MyTok->Previous && MyTok->Previous->is(tok::coloncolon) && MyTok->is(tok::identifier)) {
                Changes[i].Spaces = 0;
            }
            else {
                Changes[i].Spaces = pad + lenDiff;
            }

            int j = i + 1;

            while (j < Changes.size() && Changes[j].NewlinesBefore == 0) {
                Changes [j].StartOfTokenColumn += lenDiff;
                ++j;
            }
        }
        else if (MyTok->isFunctionNameAndPrevIsPointerOrRefOrDatatype() && !MyTok->IsInFunctionDefinitionScope) {
            size_t tokSize = adjectIdentifierLocation (pad, i, MyTok);

            if (MyTok->is(tok::kw_operator)) {
                NextTok = NextTok->Next;
                ++i;
            }

            if (NextTok)
                NextTok->PrevTokenSizeForColumnarization = tokSize;
        }
        //else if (MyTok->is(tok::kw_operator)) {
        //    size_t tokSize = adjectIdentifierLocation (pad, i, MyTok);
        //    NextTok = NextTok->Next;
        //    if (NextTok)
        //        NextTok->PrevTokenSizeForColumnarization = tokSize;
        //}
        else if (MyTok->isConstructor()) {
            size_t tokSize = 0;

            if (PrevTok && PrevTok->is(tok::kw_constexpr)) { //  && (MyTok->IsClassScope || MyTok->IsStructScope)
                tokSize = adjectIdentifierLocation (pad, i, MyTok, true);
            }
            //else if (PrevTok && PrevTok->is(tok::kw_constexpr)) {
            //    Changes[i].Spaces = 0;
            //}
            else {
                Changes[i].Spaces = MaxSpecifierTabs * Style.TabWidth + MaxDatatypeLen + pad;
                tokSize = ((StringRef)MyTok->TokenText).size();
                MaxMemberNameLen = MaxMemberNameLen < tokSize ? tokSize : MaxMemberNameLen;
            }

            if (NextTok)
                NextTok->PrevTokenSizeForColumnarization = tokSize;
        }
        else if (MyTok->isDestructor()) {
            Changes[i].Spaces = MaxSpecifierTabs * Style.TabWidth + MaxDatatypeLen + pad;
            if (Changes[i].Spaces > 1)
                Changes[i].Spaces -= 1;

            if (NextTok) {
                // Size of 'next'
                size_t tokSize = ((StringRef)NextTok->TokenText).size();
                MaxMemberNameLen = MaxMemberNameLen < tokSize ? tokSize : MaxMemberNameLen;

                FormatToken* NextNextTok = NextTok->getNextNonCommentNonConst();

                if (NextNextTok)
                    NextNextTok->PrevTokenSizeForColumnarization = tokSize;
            }
        }
        else if (MyTok->is(tok::l_brace) && MyTok->Previous
                && MyTok->Previous->is(tok::identifier)
                && MyTok->Previous->Previous
                && MyTok->Previous->Previous->is(tok::star)) {
            // populate MaxGlobalVarNameLen
            if (MaxGlobalVarNameLen < MyTok->Previous->ColumnWidth)
                MaxGlobalVarNameLen = MyTok->Previous->ColumnWidth;
        }
    }
}

/// TALLY: Columnarize specific tokens over all \c Changes.
void WhitespaceManager::columnarizeLParenTokensAndSplitArgs() {
    bool insideargs = false;
    int newlineargssize = 0;

    if (!Style.AlignConsecutiveDeclarations.Enabled)
        return;

    unsigned toPad = MaxMemberNameLen + 1;
    unsigned pad = 1;

    while (toPad % Style.TabWidth != 0) {
        toPad++;
        pad++;
    }

    for (int i = 0; i < Changes.size(); ++i) {

        const FormatToken* MyTok = Changes[i].Tok;

        if ((!(MyTok->IsClassScope || MyTok->IsStructScope)) && MyTok->LbraceCount == 0)
            continue;

        FormatToken* PrevTok = MyTok->getPreviousNonComment();

        if (MyTok->is(tok::l_paren) && !MyTok->IsInFunctionDefinitionScope && PrevTok && PrevTok->isFunctionOrCtorOrPrevIsDtor()) {
            size_t lenDiff = MaxMemberNameLen - MyTok->PrevTokenSizeForColumnarization;
            Changes[i].Spaces = pad + lenDiff;
            newlineargssize = toPad + MaxSpecifierTabs * Style.TabWidth + MaxDatatypeLen + pad + 2;
            insideargs = true;
        }
        else if (MyTok->is(tok::l_brace) && PrevTok && PrevTok->is(tok::identifier)
                && MyTok->Previous->Previous && MyTok->Previous->Previous->is(tok::star)) {
            size_t lenDiff = MaxGlobalVarNameLen - PrevTok->ColumnWidth;
            Changes[i].Spaces = 1 + lenDiff;
            insideargs = false;
        }
        else if (MyTok->is(tok::l_brace) && PrevTok && PrevTok->is(tok::identifier)
                && MyTok->Previous->Previous && MyTok->Previous->Previous->is(tok::kw_operator)) {
            size_t lenDiff = MaxGlobalVarNameLen - PrevTok->ColumnWidth;
            Changes[i].Spaces = pad + lenDiff;
            insideargs = false;
        }

        if (insideargs && MyTok->is(tok::r_paren))
            insideargs = false;

        if (insideargs)
            if (MyTok->NewlinesBefore > 0)
                Changes[i].Spaces = newlineargssize;
    }
}

// Aligns a sequence of matching tokens, on the MinColumn column.
//
// Sequences start from the first matching token to align, and end at the
// first token of the first line that doesn't need to be aligned.
//
// We need to adjust the StartOfTokenColumn of each Change that is on a line
// containing any matching token to be aligned and located after such token.
static void AlignMatchingTokenSequence(
    unsigned &StartOfSequence, unsigned &EndOfSequence, unsigned &MinColumn,
    std::function<bool(const WhitespaceManager::Change &C)> Matches,
    SmallVector<WhitespaceManager::Change, 16> &Changes) {
  if (StartOfSequence > 0 && StartOfSequence < EndOfSequence) {
    bool FoundMatchOnLine = false;
    int Shift = 0;

    for (unsigned I = StartOfSequence; I != EndOfSequence; ++I) {
      if (Changes[I].NewlinesBefore > 0) {
        Shift = 0;
        FoundMatchOnLine = false;
      }

      // If this is the first matching token to be aligned, remember by how many
      // spaces it has to be shifted, so the rest of the changes on the line are
      // shifted by the same amount.
      if (!FoundMatchOnLine && Matches(Changes[I])) {
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
  StartOfSequence = 0;
  EndOfSequence = 0;
}

void WhitespaceManager::alignConsecutiveMacros() {
  if (!Style.AlignConsecutiveMacros.Enabled)
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

    if (!Current || Current->isNot(tok::identifier))
      return false;

    if (!Current->Previous || Current->Previous->isNot(tok::pp_define))
      return false;

    // For a macro function, 0 spaces are required between the
    // identifier and the lparen that opens the parameter list.
    // For a simple macro, 1 space is required between the
    // identifier and the first token of the defined value.
    return Current->Next->SpacesRequiredBefore == SpacesRequiredBefore;
  };

  unsigned MinColumn = 0;

  // Start and end of the token sequence we're processing.
  unsigned StartOfSequence = 0;
  unsigned EndOfSequence = 0;

  // Whether a matching token has been found on the current line.
  bool FoundMatchOnLine = false;

  // Whether the current line consists only of comments
  bool LineIsComment = true;

  unsigned I = 0;
  for (unsigned E = Changes.size(); I != E; ++I) {
    if (Changes[I].NewlinesBefore != 0) {
      EndOfSequence = I;

      // Whether to break the alignment sequence because of an empty line.
      bool EmptyLineBreak = (Changes[I].NewlinesBefore > 1) &&
                            !Style.AlignConsecutiveMacros.AcrossEmptyLines;

      // Whether to break the alignment sequence because of a line without a
      // match.
      bool NoMatchBreak =
          !FoundMatchOnLine &&
          !(LineIsComment && Style.AlignConsecutiveMacros.AcrossComments);

      if (EmptyLineBreak || NoMatchBreak) {
        AlignMatchingTokenSequence(StartOfSequence, EndOfSequence, MinColumn,
                                   AlignMacrosMatches, Changes);
      }

      // A new line starts, re-initialize line status tracking bools.
      FoundMatchOnLine = false;
      LineIsComment = true;
    }

    if (Changes[I].Tok->isNot(tok::comment))
      LineIsComment = false;

    if (!AlignMacrosMatches(Changes[I]))
      continue;

    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = I;

    unsigned ChangeMinColumn = Changes[I].StartOfTokenColumn;
    MinColumn = std::max(MinColumn, ChangeMinColumn);
  }

  EndOfSequence = I;
  AlignMatchingTokenSequence(StartOfSequence, EndOfSequence, MinColumn,
                             AlignMacrosMatches, Changes);
}

void WhitespaceManager::alignConsecutiveAssignments() {
  if (!Style.AlignConsecutiveAssignments.Enabled)
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

        // Do not align operator= overloads.
        FormatToken *Previous = C.Tok->getPreviousNonComment();
        if (Previous && Previous->is(tok::kw_operator))
          return false;

        return Style.AlignConsecutiveAssignments.AlignCompound
                   ? C.Tok->getPrecedence() == prec::Assignment
                   : (C.Tok->is(tok::equal) ||
                      // In Verilog the '<=' is not a compound assignment, thus
                      // it is aligned even when the AlignCompound option is not
                      // set.
                      (Style.isVerilog() && C.Tok->is(tok::lessequal) &&
                       C.Tok->getPrecedence() == prec::Assignment));
      },
      Changes, /*StartAt=*/0, Style.AlignConsecutiveAssignments,
      /*RightJustify=*/true);
}

void WhitespaceManager::alignConsecutiveBitFields() {
  if (!Style.AlignConsecutiveBitFields.Enabled)
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
      Changes, /*StartAt=*/0, Style.AlignConsecutiveBitFields);
}

void WhitespaceManager::alignConsecutiveShortCaseStatements() {
  if (!Style.AlignConsecutiveShortCaseStatements.Enabled ||
      !Style.AllowShortCaseLabelsOnASingleLine) {
    return;
  }

  auto Matches = [&](const Change &C) {
    if (Style.AlignConsecutiveShortCaseStatements.AlignCaseColons)
      return C.Tok->is(TT_CaseLabelColon);

    // Ignore 'IsInsideToken' to allow matching trailing comments which
    // need to be reflowed as that causes the token to appear in two
    // different changes, which will cause incorrect alignment as we'll
    // reflow early due to detecting multiple aligning tokens per line.
    return !C.IsInsideToken && C.Tok->Previous &&
           C.Tok->Previous->is(TT_CaseLabelColon);
  };

  unsigned MinColumn = 0;

  // Empty case statements don't break the alignment, but don't necessarily
  // match our predicate, so we need to track their column so they can push out
  // our alignment.
  unsigned MinEmptyCaseColumn = 0;

  // Start and end of the token sequence we're processing.
  unsigned StartOfSequence = 0;
  unsigned EndOfSequence = 0;

  // Whether a matching token has been found on the current line.
  bool FoundMatchOnLine = false;

  bool LineIsComment = true;
  bool LineIsEmptyCase = false;

  unsigned I = 0;
  for (unsigned E = Changes.size(); I != E; ++I) {
    if (Changes[I].NewlinesBefore != 0) {
      // Whether to break the alignment sequence because of an empty line.
      bool EmptyLineBreak =
          (Changes[I].NewlinesBefore > 1) &&
          !Style.AlignConsecutiveShortCaseStatements.AcrossEmptyLines;

      // Whether to break the alignment sequence because of a line without a
      // match.
      bool NoMatchBreak =
          !FoundMatchOnLine &&
          !(LineIsComment &&
            Style.AlignConsecutiveShortCaseStatements.AcrossComments) &&
          !LineIsEmptyCase;

      if (EmptyLineBreak || NoMatchBreak) {
        AlignMatchingTokenSequence(StartOfSequence, EndOfSequence, MinColumn,
                                   Matches, Changes);
        MinEmptyCaseColumn = 0;
      }

      // A new line starts, re-initialize line status tracking bools.
      FoundMatchOnLine = false;
      LineIsComment = true;
      LineIsEmptyCase = false;
    }

    if (Changes[I].Tok->isNot(tok::comment))
      LineIsComment = false;

    if (Changes[I].Tok->is(TT_CaseLabelColon)) {
      LineIsEmptyCase =
          !Changes[I].Tok->Next || Changes[I].Tok->Next->isTrailingComment();

      if (LineIsEmptyCase) {
        if (Style.AlignConsecutiveShortCaseStatements.AlignCaseColons) {
          MinEmptyCaseColumn =
              std::max(MinEmptyCaseColumn, Changes[I].StartOfTokenColumn);
        } else {
          MinEmptyCaseColumn =
              std::max(MinEmptyCaseColumn, Changes[I].StartOfTokenColumn + 2);
        }
      }
    }

    if (!Matches(Changes[I]))
      continue;

    if (LineIsEmptyCase)
      continue;

    FoundMatchOnLine = true;

    if (StartOfSequence == 0)
      StartOfSequence = I;

    EndOfSequence = I + 1;

    MinColumn = std::max(MinColumn, Changes[I].StartOfTokenColumn);

    // Allow empty case statements to push out our alignment.
    MinColumn = std::max(MinColumn, MinEmptyCaseColumn);
  }

  AlignMatchingTokenSequence(StartOfSequence, EndOfSequence, MinColumn, Matches,
                             Changes);
}

void WhitespaceManager::alignConsecutiveDeclarations() {
  if (!Style.AlignConsecutiveDeclarations.Enabled)
    return;

  AlignTokens(
      Style,
      [&](Change const &C) {
        if (Style.AlignConsecutiveDeclarations.AlignFunctionPointers) {
          for (const auto *Prev = C.Tok->Previous; Prev; Prev = Prev->Previous)
            if (Prev->is(tok::equal))
              return false;
          if (C.Tok->is(TT_FunctionTypeLParen))
            return true;
        }
        if (C.Tok->isOneOf(TT_FunctionDeclarationName, tok::kw_operator))
          return true;
        if (C.Tok->isNot(TT_StartOfName))
          return false;
        if (C.Tok->Previous &&
            C.Tok->Previous->is(TT_StatementAttributeLikeMacro))
          return false;
        // Check if there is a subsequent name that starts the same declaration.
        for (FormatToken *Next = C.Tok->Next; Next; Next = Next->Next) {
          if (Next->is(tok::comment))
            continue;
          if (Next->is(TT_PointerOrReference))
            return false;
          if (!Next->Tok.getIdentifierInfo())
            break;
          if (Next->isOneOf(TT_StartOfName, TT_FunctionDeclarationName,
                            tok::kw_operator)) {
            return false;
          }
        }
        return true;
      },
      Changes, /*StartAt=*/0, Style.AlignConsecutiveDeclarations);
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
      FormatToken *Previous = C.Tok->getPreviousNonComment();
      return C.NewlinesBefore && Previous && Previous->is(TT_ConditionalExpr) &&
             (Previous->is(tok::colon) &&
              (C.Tok->FakeLParens.size() == 0 ||
               C.Tok->FakeLParens.back() != prec::Conditional));
    };
    // Ensure we keep alignment of wrapped operands with non-wrapped operands
    // Since we actually align the operators, the wrapped operands need the
    // extra offset to be properly aligned.
    for (Change &C : Changes)
      if (AlignWrappedOperand(C))
        C.StartOfTokenColumn -= 2;
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
  if (Style.AlignTrailingComments.Kind == FormatStyle::TCAS_Never)
    return;

  const int Size = Changes.size();
  int MinColumn = 0;
  int StartOfSequence = 0;
  bool BreakBeforeNext = false;
  int NewLineThreshold = 1;
  if (Style.AlignTrailingComments.Kind == FormatStyle::TCAS_Always)
    NewLineThreshold = Style.AlignTrailingComments.OverEmptyLines + 1;

  for (int I = 0, MaxColumn = INT_MAX, Newlines = 0; I < Size; ++I) {
    auto &C = Changes[I];
    if (C.StartOfBlockComment)
      continue;
    Newlines += C.NewlinesBefore;
    if (!C.IsTrailingComment)
      continue;

    if (Style.AlignTrailingComments.Kind == FormatStyle::TCAS_Leave) {
      const int OriginalSpaces =
          C.OriginalWhitespaceRange.getEnd().getRawEncoding() -
          C.OriginalWhitespaceRange.getBegin().getRawEncoding() -
          C.Tok->LastNewlineOffset;
      assert(OriginalSpaces >= 0);
      const auto RestoredLineLength =
          C.StartOfTokenColumn + C.TokenLength + OriginalSpaces;
      // If leaving comments makes the line exceed the column limit, give up to
      // leave the comments.
      if (RestoredLineLength >= Style.ColumnLimit && Style.ColumnLimit > 0)
        break;
      C.Spaces = OriginalSpaces;
      continue;
    }

    const int ChangeMinColumn = C.StartOfTokenColumn;
    int ChangeMaxColumn;

    // If we don't create a replacement for this change, we have to consider
    // it to be immovable.
    if (!C.CreateReplacement)
      ChangeMaxColumn = ChangeMinColumn;
    else if (Style.ColumnLimit == 0)
      ChangeMaxColumn = INT_MAX;
    else if (Style.ColumnLimit >= C.TokenLength)
      ChangeMaxColumn = Style.ColumnLimit - C.TokenLength;
    else
      ChangeMaxColumn = ChangeMinColumn;

    if (I + 1 < Size && Changes[I + 1].ContinuesPPDirective &&
        ChangeMaxColumn >= 2) {
      ChangeMaxColumn -= 2;
    }

    bool WasAlignedWithStartOfNextLine = false;
    if (C.NewlinesBefore >= 1) { // A comment on its own line.
      const auto CommentColumn =
          SourceMgr.getSpellingColumnNumber(C.OriginalWhitespaceRange.getEnd());
      for (int J = I + 1; J < Size; ++J) {
        if (Changes[J].Tok->is(tok::comment))
          continue;

        const auto NextColumn = SourceMgr.getSpellingColumnNumber(
            Changes[J].OriginalWhitespaceRange.getEnd());
        // The start of the next token was previously aligned with the
        // start of this comment.
        WasAlignedWithStartOfNextLine =
            CommentColumn == NextColumn ||
            CommentColumn == NextColumn + Style.IndentWidth;
        break;
      }
    }

    // We don't want to align comments which end a scope, which are here
    // identified by most closing braces.
    auto DontAlignThisComment = [](const auto *Tok) {
      if (Tok->is(tok::semi)) {
        Tok = Tok->getPreviousNonComment();
        if (!Tok)
          return false;
      }
      if (Tok->is(tok::r_paren)) {
        // Back up past the parentheses and a `TT_DoWhile` that may precede.
        Tok = Tok->MatchingParen;
        if (!Tok)
          return false;
        Tok = Tok->getPreviousNonComment();
        if (!Tok)
          return false;
        if (Tok->is(TT_DoWhile)) {
          const auto *Prev = Tok->getPreviousNonComment();
          if (!Prev) {
            // A do-while-loop without braces.
            return true;
          }
          Tok = Prev;
        }
      }

      if (Tok->isNot(tok::r_brace))
        return false;

      while (Tok->Previous && Tok->Previous->is(tok::r_brace))
        Tok = Tok->Previous;
      return Tok->NewlinesBefore > 0;
    };

    if (I > 0 && C.NewlinesBefore == 0 &&
        DontAlignThisComment(Changes[I - 1].Tok)) {
      alignTrailingComments(StartOfSequence, I, MinColumn);
      // Reset to initial values, but skip this change for the next alignment
      // pass.
      MinColumn = 0;
      MaxColumn = INT_MAX;
      StartOfSequence = I + 1;
    } else if (BreakBeforeNext || Newlines > NewLineThreshold ||
               (ChangeMinColumn > MaxColumn || ChangeMaxColumn < MinColumn) ||
               // Break the comment sequence if the previous line did not end
               // in a trailing comment.
               (C.NewlinesBefore == 1 && I > 0 &&
                !Changes[I - 1].IsTrailingComment) ||
               WasAlignedWithStartOfNextLine) {
      alignTrailingComments(StartOfSequence, I, MinColumn);
      MinColumn = ChangeMinColumn;
      MaxColumn = ChangeMaxColumn;
      StartOfSequence = I;
    } else {
      MinColumn = std::max(MinColumn, ChangeMinColumn);
      MaxColumn = std::min(MaxColumn, ChangeMaxColumn);
    }
    BreakBeforeNext = (I == 0) || (C.NewlinesBefore > 1) ||
                      // Never start a sequence with a comment at the beginning
                      // of the line.
                      (C.NewlinesBefore == 1 && StartOfSequence == I);
    Newlines = 0;
  }
  alignTrailingComments(StartOfSequence, Size, MinColumn);
}

void WhitespaceManager::alignTrailingComments(unsigned Start, unsigned End,
                                              unsigned Column) {
  for (unsigned i = Start; i != End; ++i) {
    int Shift = 0;
    if (Changes[i].IsTrailingComment)
      Shift = Column - Changes[i].StartOfTokenColumn;
    if (Changes[i].StartOfBlockComment) {
      Shift = Changes[i].IndentationOffset +
              Changes[i].StartOfBlockComment->StartOfTokenColumn -
              Changes[i].StartOfTokenColumn;
    }
    if (Shift <= 0)
      continue;
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

void WhitespaceManager::alignArrayInitializers() {
  if (Style.AlignArrayOfStructures == FormatStyle::AIAS_None)
    return;

  for (unsigned ChangeIndex = 1U, ChangeEnd = Changes.size();
       ChangeIndex < ChangeEnd; ++ChangeIndex) {
    auto &C = Changes[ChangeIndex];
    if (C.Tok->IsArrayInitializer) {
      bool FoundComplete = false;
      for (unsigned InsideIndex = ChangeIndex + 1; InsideIndex < ChangeEnd;
           ++InsideIndex) {
        if (Changes[InsideIndex].Tok == C.Tok->MatchingParen) {
          alignArrayInitializers(ChangeIndex, InsideIndex + 1);
          ChangeIndex = InsideIndex + 1;
          FoundComplete = true;
          break;
        }
      }
      if (!FoundComplete)
        ChangeIndex = ChangeEnd;
    }
  }
}

void WhitespaceManager::alignArrayInitializers(unsigned Start, unsigned End) {

  if (Style.AlignArrayOfStructures == FormatStyle::AIAS_Right)
    alignArrayInitializersRightJustified(getCells(Start, End));
  else if (Style.AlignArrayOfStructures == FormatStyle::AIAS_Left)
    alignArrayInitializersLeftJustified(getCells(Start, End));
}

void WhitespaceManager::alignArrayInitializersRightJustified(
    CellDescriptions &&CellDescs) {
  if (!CellDescs.isRectangular())
    return;

  const int BracePadding = Style.Cpp11BracedListStyle ? 0 : 1;
  auto &Cells = CellDescs.Cells;
  // Now go through and fixup the spaces.
  auto *CellIter = Cells.begin();
  for (auto i = 0U; i < CellDescs.CellCounts[0]; ++i, ++CellIter) {
    unsigned NetWidth = 0U;
    if (isSplitCell(*CellIter))
      NetWidth = getNetWidth(Cells.begin(), CellIter, CellDescs.InitialSpaces);
    auto CellWidth = getMaximumCellWidth(CellIter, NetWidth);

    if (Changes[CellIter->Index].Tok->is(tok::r_brace)) {
      // So in here we want to see if there is a brace that falls
      // on a line that was split. If so on that line we make sure that
      // the spaces in front of the brace are enough.
      const auto *Next = CellIter;
      do {
        const FormatToken *Previous = Changes[Next->Index].Tok->Previous;
        if (Previous && Previous->isNot(TT_LineComment)) {
          Changes[Next->Index].Spaces = BracePadding;
          Changes[Next->Index].NewlinesBefore = 0;
        }
        Next = Next->NextColumnElement;
      } while (Next);
      // Unless the array is empty, we need the position of all the
      // immediately adjacent cells
      if (CellIter != Cells.begin()) {
        auto ThisNetWidth =
            getNetWidth(Cells.begin(), CellIter, CellDescs.InitialSpaces);
        auto MaxNetWidth = getMaximumNetWidth(
            Cells.begin(), CellIter, CellDescs.InitialSpaces,
            CellDescs.CellCounts[0], CellDescs.CellCounts.size());
        if (ThisNetWidth < MaxNetWidth)
          Changes[CellIter->Index].Spaces = (MaxNetWidth - ThisNetWidth);
        auto RowCount = 1U;
        auto Offset = std::distance(Cells.begin(), CellIter);
        for (const auto *Next = CellIter->NextColumnElement; Next;
             Next = Next->NextColumnElement) {
          if (RowCount >= CellDescs.CellCounts.size())
            break;
          auto *Start = (Cells.begin() + RowCount * CellDescs.CellCounts[0]);
          auto *End = Start + Offset;
          ThisNetWidth = getNetWidth(Start, End, CellDescs.InitialSpaces);
          if (ThisNetWidth < MaxNetWidth)
            Changes[Next->Index].Spaces = (MaxNetWidth - ThisNetWidth);
          ++RowCount;
        }
      }
    } else {
      auto ThisWidth =
          calculateCellWidth(CellIter->Index, CellIter->EndIndex, true) +
          NetWidth;
      if (Changes[CellIter->Index].NewlinesBefore == 0) {
        Changes[CellIter->Index].Spaces = (CellWidth - (ThisWidth + NetWidth));
        Changes[CellIter->Index].Spaces += (i > 0) ? 1 : BracePadding;
      }
      alignToStartOfCell(CellIter->Index, CellIter->EndIndex);
      for (const auto *Next = CellIter->NextColumnElement; Next;
           Next = Next->NextColumnElement) {
        ThisWidth =
            calculateCellWidth(Next->Index, Next->EndIndex, true) + NetWidth;
        if (Changes[Next->Index].NewlinesBefore == 0) {
          Changes[Next->Index].Spaces = (CellWidth - ThisWidth);
          Changes[Next->Index].Spaces += (i > 0) ? 1 : BracePadding;
        }
        alignToStartOfCell(Next->Index, Next->EndIndex);
      }
    }
  }
}

void WhitespaceManager::alignArrayInitializersLeftJustified(
    CellDescriptions &&CellDescs) {

  if (!CellDescs.isRectangular())
    return;

  const int BracePadding = Style.Cpp11BracedListStyle ? 0 : 1;
  auto &Cells = CellDescs.Cells;
  // Now go through and fixup the spaces.
  auto *CellIter = Cells.begin();
  // The first cell of every row needs to be against the left brace.
  for (const auto *Next = CellIter; Next; Next = Next->NextColumnElement) {
    auto &Change = Changes[Next->Index];
    Change.Spaces =
        Change.NewlinesBefore == 0 ? BracePadding : CellDescs.InitialSpaces;
  }
  ++CellIter;
  for (auto i = 1U; i < CellDescs.CellCounts[0]; i++, ++CellIter) {
    auto MaxNetWidth = getMaximumNetWidth(
        Cells.begin(), CellIter, CellDescs.InitialSpaces,
        CellDescs.CellCounts[0], CellDescs.CellCounts.size());
    auto ThisNetWidth =
        getNetWidth(Cells.begin(), CellIter, CellDescs.InitialSpaces);
    if (Changes[CellIter->Index].NewlinesBefore == 0) {
      Changes[CellIter->Index].Spaces =
          MaxNetWidth - ThisNetWidth +
          (Changes[CellIter->Index].Tok->isNot(tok::r_brace) ? 1
                                                             : BracePadding);
    }
    auto RowCount = 1U;
    auto Offset = std::distance(Cells.begin(), CellIter);
    for (const auto *Next = CellIter->NextColumnElement; Next;
         Next = Next->NextColumnElement) {
      if (RowCount >= CellDescs.CellCounts.size())
        break;
      auto *Start = (Cells.begin() + RowCount * CellDescs.CellCounts[0]);
      auto *End = Start + Offset;
      auto ThisNetWidth = getNetWidth(Start, End, CellDescs.InitialSpaces);
      if (Changes[Next->Index].NewlinesBefore == 0) {
        Changes[Next->Index].Spaces =
            MaxNetWidth - ThisNetWidth +
            (Changes[Next->Index].Tok->isNot(tok::r_brace) ? 1 : BracePadding);
      }
      ++RowCount;
    }
  }
}

bool WhitespaceManager::isSplitCell(const CellDescription &Cell) {
  if (Cell.HasSplit)
    return true;
  for (const auto *Next = Cell.NextColumnElement; Next;
       Next = Next->NextColumnElement) {
    if (Next->HasSplit)
      return true;
  }
  return false;
}

WhitespaceManager::CellDescriptions WhitespaceManager::getCells(unsigned Start,
                                                                unsigned End) {

  unsigned Depth = 0;
  unsigned Cell = 0;
  SmallVector<unsigned> CellCounts;
  unsigned InitialSpaces = 0;
  unsigned InitialTokenLength = 0;
  unsigned EndSpaces = 0;
  SmallVector<CellDescription> Cells;
  const FormatToken *MatchingParen = nullptr;
  for (unsigned i = Start; i < End; ++i) {
    auto &C = Changes[i];
    if (C.Tok->is(tok::l_brace))
      ++Depth;
    else if (C.Tok->is(tok::r_brace))
      --Depth;
    if (Depth == 2) {
      if (C.Tok->is(tok::l_brace)) {
        Cell = 0;
        MatchingParen = C.Tok->MatchingParen;
        if (InitialSpaces == 0) {
          InitialSpaces = C.Spaces + C.TokenLength;
          InitialTokenLength = C.TokenLength;
          auto j = i - 1;
          for (; Changes[j].NewlinesBefore == 0 && j > Start; --j) {
            InitialSpaces += Changes[j].Spaces + Changes[j].TokenLength;
            InitialTokenLength += Changes[j].TokenLength;
          }
          if (C.NewlinesBefore == 0) {
            InitialSpaces += Changes[j].Spaces + Changes[j].TokenLength;
            InitialTokenLength += Changes[j].TokenLength;
          }
        }
      } else if (C.Tok->is(tok::comma)) {
        if (!Cells.empty())
          Cells.back().EndIndex = i;
        if (const auto *Next = C.Tok->getNextNonComment();
            Next && Next->isNot(tok::r_brace)) { // dangling comma
          ++Cell;
        }
      }
    } else if (Depth == 1) {
      if (C.Tok == MatchingParen) {
        if (!Cells.empty())
          Cells.back().EndIndex = i;
        Cells.push_back(CellDescription{i, ++Cell, i + 1, false, nullptr});
        CellCounts.push_back(C.Tok->Previous->isNot(tok::comma) ? Cell + 1
                                                                : Cell);
        // Go to the next non-comment and ensure there is a break in front
        const auto *NextNonComment = C.Tok->getNextNonComment();
        while (NextNonComment->is(tok::comma))
          NextNonComment = NextNonComment->getNextNonComment();
        auto j = i;
        while (j < End && Changes[j].Tok != NextNonComment)
          ++j;
        if (j < End && Changes[j].NewlinesBefore == 0 &&
            Changes[j].Tok->isNot(tok::r_brace)) {
          Changes[j].NewlinesBefore = 1;
          // Account for the added token lengths
          Changes[j].Spaces = InitialSpaces - InitialTokenLength;
        }
      } else if (C.Tok->is(tok::comment) && C.Tok->NewlinesBefore == 0) {
        // Trailing comments stay at a space past the last token
        C.Spaces = Changes[i - 1].Tok->is(tok::comma) ? 1 : 2;
      } else if (C.Tok->is(tok::l_brace)) {
        // We need to make sure that the ending braces is aligned to the
        // start of our initializer
        auto j = i - 1;
        for (; j > 0 && !Changes[j].Tok->ArrayInitializerLineStart; --j)
          ; // Nothing the loop does the work
        EndSpaces = Changes[j].Spaces;
      }
    } else if (Depth == 0 && C.Tok->is(tok::r_brace)) {
      C.NewlinesBefore = 1;
      C.Spaces = EndSpaces;
    }
    if (C.Tok->StartsColumn) {
      // This gets us past tokens that have been split over multiple
      // lines
      bool HasSplit = false;
      if (Changes[i].NewlinesBefore > 0) {
        // So if we split a line previously and the tail line + this token is
        // less then the column limit we remove the split here and just put
        // the column start at a space past the comma
        //
        // FIXME This if branch covers the cases where the column is not
        // the first column. This leads to weird pathologies like the formatting
        // auto foo = Items{
        //     Section{
        //             0, bar(),
        //     }
        // };
        // Well if it doesn't lead to that it's indicative that the line
        // breaking should be revisited. Unfortunately alot of other options
        // interact with this
        auto j = i - 1;
        if ((j - 1) > Start && Changes[j].Tok->is(tok::comma) &&
            Changes[j - 1].NewlinesBefore > 0) {
          --j;
          auto LineLimit = Changes[j].Spaces + Changes[j].TokenLength;
          if (LineLimit < Style.ColumnLimit) {
            Changes[i].NewlinesBefore = 0;
            Changes[i].Spaces = 1;
          }
        }
      }
      while (Changes[i].NewlinesBefore > 0 && Changes[i].Tok == C.Tok) {
        Changes[i].Spaces = InitialSpaces;
        ++i;
        HasSplit = true;
      }
      if (Changes[i].Tok != C.Tok)
        --i;
      Cells.push_back(CellDescription{i, Cell, i, HasSplit, nullptr});
    }
  }

  return linkCells({Cells, CellCounts, InitialSpaces});
}

unsigned WhitespaceManager::calculateCellWidth(unsigned Start, unsigned End,
                                               bool WithSpaces) const {
  unsigned CellWidth = 0;
  for (auto i = Start; i < End; i++) {
    if (Changes[i].NewlinesBefore > 0)
      CellWidth = 0;
    CellWidth += Changes[i].TokenLength;
    CellWidth += (WithSpaces ? Changes[i].Spaces : 0);
  }
  return CellWidth;
}

void WhitespaceManager::alignToStartOfCell(unsigned Start, unsigned End) {
  if ((End - Start) <= 1)
    return;
  // If the line is broken anywhere in there make sure everything
  // is aligned to the parent
  for (auto i = Start + 1; i < End; i++)
    if (Changes[i].NewlinesBefore > 0)
      Changes[i].Spaces = Changes[Start].Spaces;
}

WhitespaceManager::CellDescriptions
WhitespaceManager::linkCells(CellDescriptions &&CellDesc) {
  auto &Cells = CellDesc.Cells;
  for (auto *CellIter = Cells.begin(); CellIter != Cells.end(); ++CellIter) {
    if (!CellIter->NextColumnElement && (CellIter + 1) != Cells.end()) {
      for (auto *NextIter = CellIter + 1; NextIter != Cells.end(); ++NextIter) {
        if (NextIter->Cell == CellIter->Cell) {
          CellIter->NextColumnElement = &(*NextIter);
          break;
        }
      }
    }
  }
  return std::move(CellDesc);
}

void WhitespaceManager::generateChanges() {
  for (unsigned i = 0, e = Changes.size(); i != e; ++i) {
    const Change &C = Changes[i];
    if (i > 0) {
      auto Last = Changes[i - 1].OriginalWhitespaceRange;
      auto New = Changes[i].OriginalWhitespaceRange;
      // Do not generate two replacements for the same location.  As a special
      // case, it is allowed if there is a replacement for the empty range
      // between 2 tokens and another non-empty range at the start of the second
      // token.  We didn't implement logic to combine replacements for 2
      // consecutive source ranges into a single replacement, because the
      // program works fine without it.
      //
      // We can't eliminate empty original whitespace ranges.  They appear when
      // 2 tokens have no whitespace in between in the input.  It does not
      // matter whether whitespace is to be added.  If no whitespace is to be
      // added, the replacement will be empty, and it gets eliminated after this
      // step in storeReplacement.  For example, if the input is `foo();`,
      // there will be a replacement for the range between every consecutive
      // pair of tokens.
      //
      // A replacement at the start of a token can be added by
      // BreakableStringLiteralUsingOperators::insertBreak when it adds braces
      // around the string literal.  Say Verilog code is being formatted and the
      // first line is to become the next 2 lines.
      //     x("long string");
      //     x({"long ",
      //        "string"});
      // There will be a replacement for the empty range between the parenthesis
      // and the string and another replacement for the quote character.  The
      // replacement for the empty range between the parenthesis and the quote
      // comes from ContinuationIndenter::addTokenOnCurrentLine when it changes
      // the original empty range between the parenthesis and the string to
      // another empty one.  The replacement for the quote character comes from
      // BreakableStringLiteralUsingOperators::insertBreak when it adds the
      // brace.  In the example, the replacement for the empty range is the same
      // as the original text.  However, eliminating replacements that are same
      // as the original does not help in general.  For example, a newline can
      // be inserted, causing the first line to become the next 3 lines.
      //     xxxxxxxxxxx("long string");
      //     xxxxxxxxxxx(
      //         {"long ",
      //          "string"});
      // In that case, the empty range between the parenthesis and the string
      // will be replaced by a newline and 4 spaces.  So we will still have to
      // deal with a replacement for an empty source range followed by a
      // replacement for a non-empty source range.
      if (Last.getBegin() == New.getBegin() &&
          (Last.getEnd() != Last.getBegin() ||
           New.getEnd() == New.getBegin())) {
        continue;
      }
    }
    if (C.CreateReplacement) {
      std::string ReplacementText = C.PreviousLinePostfix;
      if (C.ContinuesPPDirective) {
        appendEscapedNewlineText(ReplacementText, C.NewlinesBefore,
                                 C.PreviousEndOfTokenColumn,
                                 C.EscapedNewlineColumn);
      } else {
        appendNewlineText(ReplacementText, C.NewlinesBefore);
      }
      // FIXME: This assert should hold if we computed the column correctly.
      // assert((int)C.StartOfTokenColumn >= C.Spaces);
      appendIndentText(
          ReplacementText, C.Tok->IndentLevel, std::max(0, C.Spaces),
          std::max((int)C.StartOfTokenColumn, C.Spaces) - std::max(0, C.Spaces),
          C.IsAligned);
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
                WhitespaceLength) == Text) {
    return;
  }
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
  if (UseCRLF) {
    Text.reserve(Text.size() + 2 * Newlines);
    for (unsigned i = 0; i < Newlines; ++i)
      Text.append("\r\n");
  } else {
    Text.append(Newlines, '\n');
  }
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

//TALLY
size_t WhitespaceManager::adjectIdentifierLocation (unsigned pad, unsigned idx, const FormatToken * tkn, bool isConstructor)
{
    static const StringRef lparanthese ("(");
    size_t lenDiff = MaxDatatypeLen - tkn->PrevTokenSizeForColumnarization;
    int j = idx + 1;

    if (isConstructor)
        Changes[idx].Spaces = MaxSpecifierTabs * Style.TabWidth + MaxDatatypeLen + pad - (Changes[idx - 1].Spaces + Changes[idx - 1].TokenLength);
    else if (tkn->Previous && tkn->Previous->is(tok::kw_const) && tkn->Next && tkn->Next->TokenText.equals(lparanthese))
        Changes[idx].Spaces += pad;// > lenDiff ? pad - lenDiff : lenDiff - pad;
    else
        Changes[idx].Spaces = pad + lenDiff;

    size_t tokSize = ((StringRef)tkn->TokenText).size();

    if (tkn->is(tok::kw_operator)) {
        tokSize += tkn->Next->ColumnWidth;
        Changes[idx + 1].Spaces = 0; // TODO : Vivek changed from -1 to 0
        ++j;
    }

    MaxMemberNameLen = MaxMemberNameLen < tokSize ? tokSize : MaxMemberNameLen;

    while (j < Changes.size() && Changes[j].NewlinesBefore == 0) {
        Changes[j].StartOfTokenColumn += lenDiff;
        ++j;
    }

    return tokSize;
}

} // namespace format
} // namespace clang
