//===--- ContinuationIndenter.cpp - Format C++ code -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the continuation indenter.
///
//===----------------------------------------------------------------------===//

#include "BreakableToken.h"
#include "ContinuationIndenter.h"
#include "WhitespaceManager.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "format-formatter"

namespace clang {
namespace format {

// Returns the length of everything up to the first possible line break after
// the ), ], } or > matching \c Tok.
static unsigned getLengthToMatchingParen(const FormatToken &Tok) {
  if (!Tok.MatchingParen)
    return 0;
  FormatToken *End = Tok.MatchingParen;
  while (End->Next && !End->Next->CanBreakBefore) {
    End = End->Next;
  }
  return End->TotalLength - Tok.TotalLength + 1;
}

static unsigned getLengthToNextOperator(const FormatToken &Tok) {
  if (!Tok.NextOperator)
    return 0;
  return Tok.NextOperator->TotalLength - Tok.TotalLength;
}

// Returns \c true if \c Tok is the "." or "->" of a call and starts the next
// segment of a builder type call.
static bool startsSegmentOfBuilderTypeCall(const FormatToken &Tok) {
  return Tok.isMemberAccess() && Tok.Previous && Tok.Previous->closesScope();
}

// Returns \c true if \c Current starts a new parameter.
static bool startsNextParameter(const FormatToken &Current,
                                const FormatStyle &Style) {
  const FormatToken &Previous = *Current.Previous;
  if (Current.is(TT_CtorInitializerComma) &&
      Style.BreakConstructorInitializersBeforeComma)
    return true;
  return Previous.is(tok::comma) && !Current.isTrailingComment() &&
         (Previous.isNot(TT_CtorInitializerComma) ||
          !Style.BreakConstructorInitializersBeforeComma);
}

ContinuationIndenter::ContinuationIndenter(const FormatStyle &Style,
                                           const AdditionalKeywords &Keywords,
                                           SourceManager &SourceMgr,
                                           WhitespaceManager &Whitespaces,
                                           encoding::Encoding Encoding,
                                           bool BinPackInconclusiveFunctions)
    : Style(Style), Keywords(Keywords), SourceMgr(SourceMgr),
      Whitespaces(Whitespaces), Encoding(Encoding),
      BinPackInconclusiveFunctions(BinPackInconclusiveFunctions),
      CommentPragmasRegex(Style.CommentPragmas) {}

LineState ContinuationIndenter::getInitialState(unsigned FirstIndent,
                                                const AnnotatedLine *Line,
                                                bool DryRun) {
  LineState State;
  State.FirstIndent = FirstIndent;
  State.Column = FirstIndent;
  State.Line = Line;
  State.NextToken = Line->First;
  State.Stack.push_back(ParenState(FirstIndent, Line->Level, FirstIndent,
                                   /*AvoidBinPacking=*/false,
                                   /*NoLineBreak=*/false));
  State.LineContainsContinuedForLoopSection = false;
  State.StartOfStringLiteral = 0;
  State.StartOfLineLevel = 0;
  State.LowestLevelOnLine = 0;
  State.IgnoreStackForComparison = false;

  // The first token has already been indented and thus consumed.
  moveStateToNextToken(State, DryRun, /*Newline=*/false);
  return State;
}

bool ContinuationIndenter::canBreak(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *Current.Previous;
  assert(&Previous == Current.Previous);
  if (!Current.CanBreakBefore &&
      !(State.Stack.back().BreakBeforeClosingBrace &&
        Current.closesBlockOrBlockTypeList(Style)))
    return false;
  // The opening "{" of a braced list has to be on the same line as the first
  // element if it is nested in another braced init list or function call.
  if (!Current.MustBreakBefore && Previous.is(tok::l_brace) &&
      Previous.isNot(TT_DictLiteral) && Previous.BlockKind == BK_BracedInit &&
      Previous.Previous &&
      Previous.Previous->isOneOf(tok::l_brace, tok::l_paren, tok::comma))
    return false;
  // This prevents breaks like:
  //   ...
  //   SomeParameter, OtherParameter).DoSomething(
  //   ...
  // As they hide "DoSomething" and are generally bad for readability.
  if (Previous.opensScope() && Previous.isNot(tok::l_brace) &&
      State.LowestLevelOnLine < State.StartOfLineLevel &&
      State.LowestLevelOnLine < Current.NestingLevel)
    return false;
  if (Current.isMemberAccess() && State.Stack.back().ContainsUnwrappedBuilder)
    return false;

  // Don't create a 'hanging' indent if there are multiple blocks in a single
  // statement.
  if (Previous.is(tok::l_brace) && State.Stack.size() > 1 &&
      State.Stack[State.Stack.size() - 2].NestedBlockInlined &&
      State.Stack[State.Stack.size() - 2].HasMultipleNestedBlocks)
    return false;

  // Don't break after very short return types (e.g. "void") as that is often
  // unexpected.
  if (Current.is(TT_FunctionDeclarationName) && State.Column < 6) {
    if (Style.AlwaysBreakAfterReturnType == FormatStyle::RTBS_None)
      return false;
  }

  return !State.Stack.back().NoLineBreak;
}

bool ContinuationIndenter::mustBreak(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *Current.Previous;
  if (Current.MustBreakBefore || Current.is(TT_InlineASMColon))
    return true;
  if (State.Stack.back().BreakBeforeClosingBrace &&
      Current.closesBlockOrBlockTypeList(Style))
    return true;
  if (Previous.is(tok::semi) && State.LineContainsContinuedForLoopSection)
    return true;
  if ((startsNextParameter(Current, Style) || Previous.is(tok::semi) ||
       (Previous.is(TT_TemplateCloser) && Current.is(TT_StartOfName) &&
        Style.Language == FormatStyle::LK_Cpp &&
        // FIXME: This is a temporary workaround for the case where clang-format
        // sets BreakBeforeParameter to avoid bin packing and this creates a
        // completely unnecessary line break after a template type that isn't
        // line-wrapped.
        (Previous.NestingLevel == 1 || Style.BinPackParameters)) ||
       (Style.BreakBeforeTernaryOperators && Current.is(TT_ConditionalExpr) &&
        Previous.isNot(tok::question)) ||
       (!Style.BreakBeforeTernaryOperators &&
        Previous.is(TT_ConditionalExpr))) &&
      State.Stack.back().BreakBeforeParameter && !Current.isTrailingComment() &&
      !Current.isOneOf(tok::r_paren, tok::r_brace))
    return true;
  if (((Previous.is(TT_DictLiteral) && Previous.is(tok::l_brace)) ||
       (Previous.is(TT_ArrayInitializerLSquare) &&
        Previous.ParameterCount > 1)) &&
      Style.ColumnLimit > 0 &&
      getLengthToMatchingParen(Previous) + State.Column - 1 >
          getColumnLimit(State))
    return true;
  if (Current.is(TT_CtorInitializerColon) &&
      (State.Column + State.Line->Last->TotalLength - Current.TotalLength + 2 >
           getColumnLimit(State) ||
       State.Stack.back().BreakBeforeParameter) &&
      ((Style.AllowShortFunctionsOnASingleLine != FormatStyle::SFS_All) ||
       Style.BreakConstructorInitializersBeforeComma || Style.ColumnLimit != 0))
    return true;
  if (Current.is(TT_SelectorName) && State.Stack.back().ObjCSelectorNameFound &&
      State.Stack.back().BreakBeforeParameter)
    return true;

  unsigned NewLineColumn = getNewLineColumn(State);
  if (Current.isMemberAccess() && Style.ColumnLimit != 0 &&
      State.Column + getLengthToNextOperator(Current) > Style.ColumnLimit &&
      (State.Column > NewLineColumn ||
       Current.NestingLevel < State.StartOfLineLevel))
    return true;

  if (State.Column <= NewLineColumn)
    return false;

  if (Style.AlwaysBreakBeforeMultilineStrings &&
      (NewLineColumn == State.FirstIndent + Style.ContinuationIndentWidth ||
       Previous.is(tok::comma) || Current.NestingLevel < 2) &&
      !Previous.isOneOf(tok::kw_return, tok::lessless, tok::at) &&
      !Previous.isOneOf(TT_InlineASMColon, TT_ConditionalExpr) &&
      nextIsMultilineString(State))
    return true;

  // Using CanBreakBefore here and below takes care of the decision whether the
  // current style uses wrapping before or after operators for the given
  // operator.
  if (Previous.is(TT_BinaryOperator) && Current.CanBreakBefore) {
    // If we need to break somewhere inside the LHS of a binary expression, we
    // should also break after the operator. Otherwise, the formatting would
    // hide the operator precedence, e.g. in:
    //   if (aaaaaaaaaaaaaa ==
    //           bbbbbbbbbbbbbb && c) {..
    // For comparisons, we only apply this rule, if the LHS is a binary
    // expression itself as otherwise, the line breaks seem superfluous.
    // We need special cases for ">>" which we have split into two ">" while
    // lexing in order to make template parsing easier.
    bool IsComparison = (Previous.getPrecedence() == prec::Relational ||
                         Previous.getPrecedence() == prec::Equality) &&
                        Previous.Previous &&
                        Previous.Previous->isNot(TT_BinaryOperator); // For >>.
    bool LHSIsBinaryExpr =
        Previous.Previous && Previous.Previous->EndsBinaryExpression;
    if ((!IsComparison || LHSIsBinaryExpr) && !Current.isTrailingComment() &&
        Previous.getPrecedence() != prec::Assignment &&
        State.Stack.back().BreakBeforeParameter)
      return true;
  } else if (Current.is(TT_BinaryOperator) && Current.CanBreakBefore &&
             State.Stack.back().BreakBeforeParameter) {
    return true;
  }

  // Same as above, but for the first "<<" operator.
  if (Current.is(tok::lessless) && Current.isNot(TT_OverloadedOperator) &&
      State.Stack.back().BreakBeforeParameter &&
      State.Stack.back().FirstLessLess == 0)
    return true;

  if (Current.NestingLevel == 0 && !Current.isTrailingComment()) {
    // Always break after "template <...>" and leading annotations. This is only
    // for cases where the entire line does not fit on a single line as a
    // different LineFormatter would be used otherwise.
    if (Previous.ClosesTemplateDeclaration)
      return true;
    if (Previous.is(TT_FunctionAnnotationRParen))
      return true;
    if (Previous.is(TT_LeadingJavaAnnotation) && Current.isNot(tok::l_paren) &&
        Current.isNot(TT_LeadingJavaAnnotation))
      return true;
  }

  // If the return type spans multiple lines, wrap before the function name.
  if ((Current.is(TT_FunctionDeclarationName) ||
       (Current.is(tok::kw_operator) && !Previous.is(tok::coloncolon))) &&
      State.Stack.back().BreakBeforeParameter)
    return true;

  if (startsSegmentOfBuilderTypeCall(Current) &&
      (State.Stack.back().CallContinuation != 0 ||
       State.Stack.back().BreakBeforeParameter))
    return true;

  // The following could be precomputed as they do not depend on the state.
  // However, as they should take effect only if the UnwrappedLine does not fit
  // into the ColumnLimit, they are checked here in the ContinuationIndenter.
  if (Style.ColumnLimit != 0 && Previous.BlockKind == BK_Block &&
      Previous.is(tok::l_brace) && !Current.isOneOf(tok::r_brace, tok::comment))
    return true;

  if (Current.is(tok::lessless) &&
      ((Previous.is(tok::identifier) && Previous.TokenText == "endl") ||
       (Previous.Tok.isLiteral() && (Previous.TokenText.endswith("\\n\"") ||
                                     Previous.TokenText == "\'\\n\'"))))
    return true;

  return false;
}

unsigned ContinuationIndenter::addTokenToState(LineState &State, bool Newline,
                                               bool DryRun,
                                               unsigned ExtraSpaces) {
  const FormatToken &Current = *State.NextToken;

  assert(!State.Stack.empty());
  if ((Current.is(TT_ImplicitStringLiteral) &&
       (Current.Previous->Tok.getIdentifierInfo() == nullptr ||
        Current.Previous->Tok.getIdentifierInfo()->getPPKeywordID() ==
            tok::pp_not_keyword))) {
    unsigned EndColumn =
        SourceMgr.getSpellingColumnNumber(Current.WhitespaceRange.getEnd());
    if (Current.LastNewlineOffset != 0) {
      // If there is a newline within this token, the final column will solely
      // determined by the current end column.
      State.Column = EndColumn;
    } else {
      unsigned StartColumn =
          SourceMgr.getSpellingColumnNumber(Current.WhitespaceRange.getBegin());
      assert(EndColumn >= StartColumn);
      State.Column += EndColumn - StartColumn;
    }
    moveStateToNextToken(State, DryRun, /*Newline=*/false);
    return 0;
  }

  unsigned Penalty = 0;
  if (Newline)
    Penalty = addTokenOnNewLine(State, DryRun);
  else
    addTokenOnCurrentLine(State, DryRun, ExtraSpaces);

  return moveStateToNextToken(State, DryRun, Newline) + Penalty;
}

void ContinuationIndenter::addTokenOnCurrentLine(LineState &State, bool DryRun,
                                                 unsigned ExtraSpaces) {
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *State.NextToken->Previous;
  if (Current.is(tok::equal) &&
      (State.Line->First->is(tok::kw_for) || Current.NestingLevel == 0) &&
      State.Stack.back().VariablePos == 0) {
    State.Stack.back().VariablePos = State.Column;
    // Move over * and & if they are bound to the variable name.
    const FormatToken *Tok = &Previous;
    while (Tok && State.Stack.back().VariablePos >= Tok->ColumnWidth) {
      State.Stack.back().VariablePos -= Tok->ColumnWidth;
      if (Tok->SpacesRequiredBefore != 0)
        break;
      Tok = Tok->Previous;
    }
    if (Previous.PartOfMultiVariableDeclStmt)
      State.Stack.back().LastSpace = State.Stack.back().VariablePos;
  }

  unsigned Spaces = Current.SpacesRequiredBefore + ExtraSpaces;

  if (!DryRun)
    Whitespaces.replaceWhitespace(Current, /*Newlines=*/0, /*IndentLevel=*/0,
                                  Spaces, State.Column + Spaces);

  if (Current.is(TT_SelectorName) &&
      !State.Stack.back().ObjCSelectorNameFound) {
    unsigned MinIndent =
        std::max(State.FirstIndent + Style.ContinuationIndentWidth,
                 State.Stack.back().Indent);
    unsigned FirstColonPos = State.Column + Spaces + Current.ColumnWidth;
    if (Current.LongestObjCSelectorName == 0)
      State.Stack.back().AlignColons = false;
    else if (MinIndent + Current.LongestObjCSelectorName > FirstColonPos)
      State.Stack.back().ColonPos = MinIndent + Current.LongestObjCSelectorName;
    else
      State.Stack.back().ColonPos = FirstColonPos;
  }

  // In "AlwaysBreak" mode, enforce wrapping directly after the parenthesis by
  // disallowing any further line breaks if there is no line break after the
  // opening parenthesis. Don't break if it doesn't conserve columns.
  if (Style.AlignAfterOpenBracket == FormatStyle::BAS_AlwaysBreak &&
      Previous.is(tok::l_paren) && State.Column > getNewLineColumn(State) &&
      (!Previous.Previous ||
       !Previous.Previous->isOneOf(tok::kw_for, tok::kw_while, tok::kw_switch)))
    State.Stack.back().NoLineBreak = true;

  if (Style.AlignAfterOpenBracket != FormatStyle::BAS_DontAlign &&
      Previous.opensScope() && Previous.isNot(TT_ObjCMethodExpr) &&
      (Current.isNot(TT_LineComment) || Previous.BlockKind == BK_BracedInit))
    State.Stack.back().Indent = State.Column + Spaces;
  if (State.Stack.back().AvoidBinPacking && startsNextParameter(Current, Style))
    State.Stack.back().NoLineBreak = true;
  if (startsSegmentOfBuilderTypeCall(Current) &&
      State.Column > getNewLineColumn(State))
    State.Stack.back().ContainsUnwrappedBuilder = true;

  if (Current.is(TT_LambdaArrow) && Style.Language == FormatStyle::LK_Java)
    State.Stack.back().NoLineBreak = true;
  if (Current.isMemberAccess() && Previous.is(tok::r_paren) &&
      (Previous.MatchingParen &&
       (Previous.TotalLength - Previous.MatchingParen->TotalLength > 10))) {
    // If there is a function call with long parameters, break before trailing
    // calls. This prevents things like:
    //   EXPECT_CALL(SomeLongParameter).Times(
    //       2);
    // We don't want to do this for short parameters as they can just be
    // indexes.
    State.Stack.back().NoLineBreak = true;
  }

  State.Column += Spaces;
  if (Current.isNot(tok::comment) && Previous.is(tok::l_paren) &&
      Previous.Previous &&
      Previous.Previous->isOneOf(tok::kw_if, tok::kw_for)) {
    // Treat the condition inside an if as if it was a second function
    // parameter, i.e. let nested calls have a continuation indent.
    State.Stack.back().LastSpace = State.Column;
    State.Stack.back().NestedBlockIndent = State.Column;
  } else if (!Current.isOneOf(tok::comment, tok::caret) &&
             ((Previous.is(tok::comma) &&
               !Previous.is(TT_OverloadedOperator)) ||
              (Previous.is(tok::colon) && Previous.is(TT_ObjCMethodExpr)))) {
    State.Stack.back().LastSpace = State.Column;
  } else if ((Previous.isOneOf(TT_BinaryOperator, TT_ConditionalExpr,
                               TT_CtorInitializerColon)) &&
             ((Previous.getPrecedence() != prec::Assignment &&
               (Previous.isNot(tok::lessless) || Previous.OperatorIndex != 0 ||
                Previous.NextOperator)) ||
              Current.StartsBinaryExpression)) {
    // Always indent relative to the RHS of the expression unless this is a
    // simple assignment without binary expression on the RHS. Also indent
    // relative to unary operators and the colons of constructor initializers.
    State.Stack.back().LastSpace = State.Column;
  } else if (Previous.is(TT_InheritanceColon)) {
    State.Stack.back().Indent = State.Column;
    State.Stack.back().LastSpace = State.Column;
  } else if (Previous.opensScope()) {
    // If a function has a trailing call, indent all parameters from the
    // opening parenthesis. This avoids confusing indents like:
    //   OuterFunction(InnerFunctionCall( // break
    //       ParameterToInnerFunction))   // break
    //       .SecondInnerFunctionCall();
    bool HasTrailingCall = false;
    if (Previous.MatchingParen) {
      const FormatToken *Next = Previous.MatchingParen->getNextNonComment();
      HasTrailingCall = Next && Next->isMemberAccess();
    }
    if (HasTrailingCall && State.Stack.size() > 1 &&
        State.Stack[State.Stack.size() - 2].CallContinuation == 0)
      State.Stack.back().LastSpace = State.Column;
  }
}

unsigned ContinuationIndenter::addTokenOnNewLine(LineState &State,
                                                 bool DryRun) {
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *State.NextToken->Previous;

  // Extra penalty that needs to be added because of the way certain line
  // breaks are chosen.
  unsigned Penalty = 0;

  const FormatToken *PreviousNonComment = Current.getPreviousNonComment();
  const FormatToken *NextNonComment = Previous.getNextNonComment();
  if (!NextNonComment)
    NextNonComment = &Current;
  // The first line break on any NestingLevel causes an extra penalty in order
  // prefer similar line breaks.
  if (!State.Stack.back().ContainsLineBreak)
    Penalty += 15;
  State.Stack.back().ContainsLineBreak = true;

  Penalty += State.NextToken->SplitPenalty;

  // Breaking before the first "<<" is generally not desirable if the LHS is
  // short. Also always add the penalty if the LHS is split over mutliple lines
  // to avoid unnecessary line breaks that just work around this penalty.
  if (NextNonComment->is(tok::lessless) &&
      State.Stack.back().FirstLessLess == 0 &&
      (State.Column <= Style.ColumnLimit / 3 ||
       State.Stack.back().BreakBeforeParameter))
    Penalty += Style.PenaltyBreakFirstLessLess;

  State.Column = getNewLineColumn(State);

  // Indent nested blocks relative to this column, unless in a very specific
  // JavaScript special case where:
  //
  //   var loooooong_name =
  //       function() {
  //     // code
  //   }
  //
  // is common and should be formatted like a free-standing function.
  if (Style.Language != FormatStyle::LK_JavaScript ||
      Current.NestingLevel != 0 || !PreviousNonComment->is(tok::equal) ||
      !Current.is(Keywords.kw_function))
    State.Stack.back().NestedBlockIndent = State.Column;

  if (NextNonComment->isMemberAccess()) {
    if (State.Stack.back().CallContinuation == 0)
      State.Stack.back().CallContinuation = State.Column;
  } else if (NextNonComment->is(TT_SelectorName)) {
    if (!State.Stack.back().ObjCSelectorNameFound) {
      if (NextNonComment->LongestObjCSelectorName == 0) {
        State.Stack.back().AlignColons = false;
      } else {
        State.Stack.back().ColonPos =
            (Style.IndentWrappedFunctionNames
                 ? std::max(State.Stack.back().Indent,
                            State.FirstIndent + Style.ContinuationIndentWidth)
                 : State.Stack.back().Indent) +
            NextNonComment->LongestObjCSelectorName;
      }
    } else if (State.Stack.back().AlignColons &&
               State.Stack.back().ColonPos <= NextNonComment->ColumnWidth) {
      State.Stack.back().ColonPos = State.Column + NextNonComment->ColumnWidth;
    }
  } else if (PreviousNonComment && PreviousNonComment->is(tok::colon) &&
             PreviousNonComment->isOneOf(TT_ObjCMethodExpr, TT_DictLiteral)) {
    // FIXME: This is hacky, find a better way. The problem is that in an ObjC
    // method expression, the block should be aligned to the line starting it,
    // e.g.:
    //   [aaaaaaaaaaaaaaa aaaaaaaaa: \\ break for some reason
    //                        ^(int *i) {
    //                            // ...
    //                        }];
    // Thus, we set LastSpace of the next higher NestingLevel, to which we move
    // when we consume all of the "}"'s FakeRParens at the "{".
    if (State.Stack.size() > 1)
      State.Stack[State.Stack.size() - 2].LastSpace =
          std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) +
          Style.ContinuationIndentWidth;
  }

  if ((Previous.isOneOf(tok::comma, tok::semi) &&
       !State.Stack.back().AvoidBinPacking) ||
      Previous.is(TT_BinaryOperator))
    State.Stack.back().BreakBeforeParameter = false;
  if (Previous.isOneOf(TT_TemplateCloser, TT_JavaAnnotation) &&
      Current.NestingLevel == 0)
    State.Stack.back().BreakBeforeParameter = false;
  if (NextNonComment->is(tok::question) ||
      (PreviousNonComment && PreviousNonComment->is(tok::question)))
    State.Stack.back().BreakBeforeParameter = true;
  if (Current.is(TT_BinaryOperator) && Current.CanBreakBefore)
    State.Stack.back().BreakBeforeParameter = false;

  if (!DryRun) {
    unsigned Newlines = std::max(
        1u, std::min(Current.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1));
    Whitespaces.replaceWhitespace(Current, Newlines,
                                  State.Stack.back().IndentLevel, State.Column,
                                  State.Column, State.Line->InPPDirective);
  }

  if (!Current.isTrailingComment())
    State.Stack.back().LastSpace = State.Column;
  State.StartOfLineLevel = Current.NestingLevel;
  State.LowestLevelOnLine = Current.NestingLevel;

  // Any break on this level means that the parent level has been broken
  // and we need to avoid bin packing there.
  bool NestedBlockSpecialCase =
      Style.Language != FormatStyle::LK_Cpp &&
      Current.is(tok::r_brace) && State.Stack.size() > 1 &&
      State.Stack[State.Stack.size() - 2].NestedBlockInlined;
  if (!NestedBlockSpecialCase)
    for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i)
      State.Stack[i].BreakBeforeParameter = true;

  if (PreviousNonComment &&
      !PreviousNonComment->isOneOf(tok::comma, tok::semi) &&
      (PreviousNonComment->isNot(TT_TemplateCloser) ||
       Current.NestingLevel != 0) &&
      !PreviousNonComment->isOneOf(
          TT_BinaryOperator, TT_FunctionAnnotationRParen, TT_JavaAnnotation,
          TT_LeadingJavaAnnotation) &&
      Current.isNot(TT_BinaryOperator) && !PreviousNonComment->opensScope())
    State.Stack.back().BreakBeforeParameter = true;

  // If we break after { or the [ of an array initializer, we should also break
  // before the corresponding } or ].
  if (PreviousNonComment &&
      (PreviousNonComment->isOneOf(tok::l_brace, TT_ArrayInitializerLSquare)))
    State.Stack.back().BreakBeforeClosingBrace = true;

  if (State.Stack.back().AvoidBinPacking) {
    // If we are breaking after '(', '{', '<', this is not bin packing
    // unless AllowAllParametersOfDeclarationOnNextLine is false or this is a
    // dict/object literal.
    if (!Previous.isOneOf(tok::l_paren, tok::l_brace, TT_BinaryOperator) ||
        (!Style.AllowAllParametersOfDeclarationOnNextLine &&
         State.Line->MustBeDeclaration) ||
        Previous.is(TT_DictLiteral))
      State.Stack.back().BreakBeforeParameter = true;
  }

  return Penalty;
}

unsigned ContinuationIndenter::getNewLineColumn(const LineState &State) {
  if (!State.NextToken || !State.NextToken->Previous)
    return 0;
  FormatToken &Current = *State.NextToken;
  const FormatToken &Previous = *Current.Previous;
  // If we are continuing an expression, we want to use the continuation indent.
  unsigned ContinuationIndent =
      std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) +
      Style.ContinuationIndentWidth;
  const FormatToken *PreviousNonComment = Current.getPreviousNonComment();
  const FormatToken *NextNonComment = Previous.getNextNonComment();
  if (!NextNonComment)
    NextNonComment = &Current;

  // Java specific bits.
  if (Style.Language == FormatStyle::LK_Java &&
      Current.isOneOf(Keywords.kw_implements, Keywords.kw_extends))
    return std::max(State.Stack.back().LastSpace,
                    State.Stack.back().Indent + Style.ContinuationIndentWidth);

  if (NextNonComment->is(tok::l_brace) && NextNonComment->BlockKind == BK_Block)
    return Current.NestingLevel == 0 ? State.FirstIndent
                                     : State.Stack.back().Indent;
  if (Current.isOneOf(tok::r_brace, tok::r_square) && State.Stack.size() > 1) {
    if (Current.closesBlockOrBlockTypeList(Style))
      return State.Stack[State.Stack.size() - 2].NestedBlockIndent;
    if (Current.MatchingParen &&
        Current.MatchingParen->BlockKind == BK_BracedInit)
      return State.Stack[State.Stack.size() - 2].LastSpace;
    return State.FirstIndent;
  }
  if (Current.is(tok::identifier) && Current.Next &&
      Current.Next->is(TT_DictLiteral))
    return State.Stack.back().Indent;
  if (NextNonComment->isStringLiteral() && State.StartOfStringLiteral != 0)
    return State.StartOfStringLiteral;
  if (NextNonComment->is(TT_ObjCStringLiteral) &&
      State.StartOfStringLiteral != 0)
    return State.StartOfStringLiteral - 1;
  if (NextNonComment->is(tok::lessless) &&
      State.Stack.back().FirstLessLess != 0)
    return State.Stack.back().FirstLessLess;
  if (NextNonComment->isMemberAccess()) {
    if (State.Stack.back().CallContinuation == 0)
      return ContinuationIndent;
    return State.Stack.back().CallContinuation;
  }
  if (State.Stack.back().QuestionColumn != 0 &&
      ((NextNonComment->is(tok::colon) &&
        NextNonComment->is(TT_ConditionalExpr)) ||
       Previous.is(TT_ConditionalExpr)))
    return State.Stack.back().QuestionColumn;
  if (Previous.is(tok::comma) && State.Stack.back().VariablePos != 0)
    return State.Stack.back().VariablePos;
  if ((PreviousNonComment &&
       (PreviousNonComment->ClosesTemplateDeclaration ||
        PreviousNonComment->isOneOf(
            TT_AttributeParen, TT_FunctionAnnotationRParen, TT_JavaAnnotation,
            TT_LeadingJavaAnnotation))) ||
      (!Style.IndentWrappedFunctionNames &&
       NextNonComment->isOneOf(tok::kw_operator, TT_FunctionDeclarationName)))
    return std::max(State.Stack.back().LastSpace, State.Stack.back().Indent);
  if (NextNonComment->is(TT_SelectorName)) {
    if (!State.Stack.back().ObjCSelectorNameFound) {
      if (NextNonComment->LongestObjCSelectorName == 0)
        return State.Stack.back().Indent;
      return (Style.IndentWrappedFunctionNames
                  ? std::max(State.Stack.back().Indent,
                             State.FirstIndent + Style.ContinuationIndentWidth)
                  : State.Stack.back().Indent) +
             NextNonComment->LongestObjCSelectorName -
             NextNonComment->ColumnWidth;
    }
    if (!State.Stack.back().AlignColons)
      return State.Stack.back().Indent;
    if (State.Stack.back().ColonPos > NextNonComment->ColumnWidth)
      return State.Stack.back().ColonPos - NextNonComment->ColumnWidth;
    return State.Stack.back().Indent;
  }
  if (NextNonComment->is(TT_ArraySubscriptLSquare)) {
    if (State.Stack.back().StartOfArraySubscripts != 0)
      return State.Stack.back().StartOfArraySubscripts;
    return ContinuationIndent;
  }

  // This ensure that we correctly format ObjC methods calls without inputs,
  // i.e. where the last element isn't selector like: [callee method];
  if (NextNonComment->is(tok::identifier) && NextNonComment->FakeRParens == 0 &&
      NextNonComment->Next && NextNonComment->Next->is(TT_ObjCMethodExpr))
    return State.Stack.back().Indent;

  if (NextNonComment->isOneOf(TT_StartOfName, TT_PointerOrReference) ||
      Previous.isOneOf(tok::coloncolon, tok::equal, TT_JsTypeColon))
    return ContinuationIndent;
  if (PreviousNonComment && PreviousNonComment->is(tok::colon) &&
      PreviousNonComment->isOneOf(TT_ObjCMethodExpr, TT_DictLiteral))
    return ContinuationIndent;
  if (NextNonComment->is(TT_CtorInitializerColon))
    return State.FirstIndent + Style.ConstructorInitializerIndentWidth;
  if (NextNonComment->is(TT_CtorInitializerComma))
    return State.Stack.back().Indent;
  if (Previous.is(tok::r_paren) && !Current.isBinaryOperator() &&
      !Current.isOneOf(tok::colon, tok::comment))
    return ContinuationIndent;
  if (State.Stack.back().Indent == State.FirstIndent && PreviousNonComment &&
      PreviousNonComment->isNot(tok::r_brace))
    // Ensure that we fall back to the continuation indent width instead of
    // just flushing continuations left.
    return State.Stack.back().Indent + Style.ContinuationIndentWidth;
  return State.Stack.back().Indent;
}

unsigned ContinuationIndenter::moveStateToNextToken(LineState &State,
                                                    bool DryRun, bool Newline) {
  assert(State.Stack.size());
  const FormatToken &Current = *State.NextToken;

  if (Current.is(TT_InheritanceColon))
    State.Stack.back().AvoidBinPacking = true;
  if (Current.is(tok::lessless) && Current.isNot(TT_OverloadedOperator)) {
    if (State.Stack.back().FirstLessLess == 0)
      State.Stack.back().FirstLessLess = State.Column;
    else
      State.Stack.back().LastOperatorWrapped = Newline;
  }
  if ((Current.is(TT_BinaryOperator) && Current.isNot(tok::lessless)) ||
      Current.is(TT_ConditionalExpr))
    State.Stack.back().LastOperatorWrapped = Newline;
  if (Current.is(TT_ArraySubscriptLSquare) &&
      State.Stack.back().StartOfArraySubscripts == 0)
    State.Stack.back().StartOfArraySubscripts = State.Column;
  if ((Current.is(tok::question) && Style.BreakBeforeTernaryOperators) ||
      (Current.getPreviousNonComment() && Current.isNot(tok::colon) &&
       Current.getPreviousNonComment()->is(tok::question) &&
       !Style.BreakBeforeTernaryOperators))
    State.Stack.back().QuestionColumn = State.Column;
  if (!Current.opensScope() && !Current.closesScope())
    State.LowestLevelOnLine =
        std::min(State.LowestLevelOnLine, Current.NestingLevel);
  if (Current.isMemberAccess())
    State.Stack.back().StartOfFunctionCall =
        !Current.NextOperator ? 0 : State.Column;
  if (Current.is(TT_SelectorName)) {
    State.Stack.back().ObjCSelectorNameFound = true;
    if (Style.IndentWrappedFunctionNames) {
      State.Stack.back().Indent =
          State.FirstIndent + Style.ContinuationIndentWidth;
    }
  }
  if (Current.is(TT_CtorInitializerColon)) {
    // Indent 2 from the column, so:
    // SomeClass::SomeClass()
    //     : First(...), ...
    //       Next(...)
    //       ^ line up here.
    State.Stack.back().Indent =
        State.Column + (Style.BreakConstructorInitializersBeforeComma ? 0 : 2);
    State.Stack.back().NestedBlockIndent = State.Stack.back().Indent;
    if (Style.ConstructorInitializerAllOnOneLineOrOnePerLine)
      State.Stack.back().AvoidBinPacking = true;
    State.Stack.back().BreakBeforeParameter = false;
  }
  if (Current.isOneOf(TT_BinaryOperator, TT_ConditionalExpr) && Newline)
    State.Stack.back().NestedBlockIndent =
        State.Column + Current.ColumnWidth + 1;

  // Insert scopes created by fake parenthesis.
  const FormatToken *Previous = Current.getPreviousNonComment();

  // Add special behavior to support a format commonly used for JavaScript
  // closures:
  //   SomeFunction(function() {
  //     foo();
  //     bar();
  //   }, a, b, c);
  if (Current.isNot(tok::comment) && Previous &&
      Previous->isOneOf(tok::l_brace, TT_ArrayInitializerLSquare) &&
      !Previous->is(TT_DictLiteral) && State.Stack.size() > 1) {
    if (State.Stack[State.Stack.size() - 2].NestedBlockInlined && Newline)
      for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i)
        State.Stack[i].NoLineBreak = true;
    State.Stack[State.Stack.size() - 2].NestedBlockInlined = false;
  }
  if (Previous && (Previous->isOneOf(tok::l_paren, tok::comma, tok::colon) ||
                   Previous->isOneOf(TT_BinaryOperator, TT_ConditionalExpr)) &&
      !Previous->isOneOf(TT_DictLiteral, TT_ObjCMethodExpr)) {
    State.Stack.back().NestedBlockInlined =
        !Newline &&
        (Previous->isNot(tok::l_paren) || Previous->ParameterCount > 1);
  }

  moveStatePastFakeLParens(State, Newline);
  moveStatePastScopeOpener(State, Newline);
  moveStatePastScopeCloser(State);
  moveStatePastFakeRParens(State);

  if (Current.isStringLiteral() && State.StartOfStringLiteral == 0)
    State.StartOfStringLiteral = State.Column;
  if (Current.is(TT_ObjCStringLiteral) && State.StartOfStringLiteral == 0)
    State.StartOfStringLiteral = State.Column + 1;
  else if (!Current.isOneOf(tok::comment, tok::identifier, tok::hash) &&
           !Current.isStringLiteral())
    State.StartOfStringLiteral = 0;

  State.Column += Current.ColumnWidth;
  State.NextToken = State.NextToken->Next;
  unsigned Penalty = breakProtrudingToken(Current, State, DryRun);
  if (State.Column > getColumnLimit(State)) {
    unsigned ExcessCharacters = State.Column - getColumnLimit(State);
    Penalty += Style.PenaltyExcessCharacter * ExcessCharacters;
  }

  if (Current.Role)
    Current.Role->formatFromToken(State, this, DryRun);
  // If the previous has a special role, let it consume tokens as appropriate.
  // It is necessary to start at the previous token for the only implemented
  // role (comma separated list). That way, the decision whether or not to break
  // after the "{" is already done and both options are tried and evaluated.
  // FIXME: This is ugly, find a better way.
  if (Previous && Previous->Role)
    Penalty += Previous->Role->formatAfterToken(State, this, DryRun);

  return Penalty;
}

void ContinuationIndenter::moveStatePastFakeLParens(LineState &State,
                                                    bool Newline) {
  const FormatToken &Current = *State.NextToken;
  const FormatToken *Previous = Current.getPreviousNonComment();

  // Don't add extra indentation for the first fake parenthesis after
  // 'return', assignments or opening <({[. The indentation for these cases
  // is special cased.
  bool SkipFirstExtraIndent =
      (Previous && (Previous->opensScope() ||
                    Previous->isOneOf(tok::semi, tok::kw_return) ||
                    (Previous->getPrecedence() == prec::Assignment &&
                     Style.AlignOperands) ||
                    Previous->is(TT_ObjCMethodExpr)));
  for (SmallVectorImpl<prec::Level>::const_reverse_iterator
           I = Current.FakeLParens.rbegin(),
           E = Current.FakeLParens.rend();
       I != E; ++I) {
    ParenState NewParenState = State.Stack.back();
    NewParenState.ContainsLineBreak = false;

    // Indent from 'LastSpace' unless these are fake parentheses encapsulating
    // a builder type call after 'return' or, if the alignment after opening
    // brackets is disabled.
    if (!Current.isTrailingComment() &&
        (Style.AlignOperands || *I < prec::Assignment) &&
        (!Previous || Previous->isNot(tok::kw_return) ||
         (Style.Language != FormatStyle::LK_Java && *I > 0)) &&
        (Style.AlignAfterOpenBracket != FormatStyle::BAS_DontAlign ||
         *I != prec::Comma || Current.NestingLevel == 0))
      NewParenState.Indent =
          std::max(std::max(State.Column, NewParenState.Indent),
                   State.Stack.back().LastSpace);

    // Don't allow the RHS of an operator to be split over multiple lines unless
    // there is a line-break right after the operator.
    // Exclude relational operators, as there, it is always more desirable to
    // have the LHS 'left' of the RHS.
    if (Previous && Previous->getPrecedence() > prec::Assignment &&
        Previous->isOneOf(TT_BinaryOperator, TT_ConditionalExpr) &&
        Previous->getPrecedence() != prec::Relational) {
      bool BreakBeforeOperator =
          Previous->is(tok::lessless) ||
          (Previous->is(TT_BinaryOperator) &&
           Style.BreakBeforeBinaryOperators != FormatStyle::BOS_None) ||
          (Previous->is(TT_ConditionalExpr) &&
           Style.BreakBeforeTernaryOperators);
      if ((!Newline && !BreakBeforeOperator) ||
          (!State.Stack.back().LastOperatorWrapped && BreakBeforeOperator))
        NewParenState.NoLineBreak = true;
    }

    // Do not indent relative to the fake parentheses inserted for "." or "->".
    // This is a special case to make the following to statements consistent:
    //   OuterFunction(InnerFunctionCall( // break
    //       ParameterToInnerFunction));
    //   OuterFunction(SomeObject.InnerFunctionCall( // break
    //       ParameterToInnerFunction));
    if (*I > prec::Unknown)
      NewParenState.LastSpace = std::max(NewParenState.LastSpace, State.Column);
    if (*I != prec::Conditional && !Current.is(TT_UnaryOperator))
      NewParenState.StartOfFunctionCall = State.Column;

    // Always indent conditional expressions. Never indent expression where
    // the 'operator' is ',', ';' or an assignment (i.e. *I <=
    // prec::Assignment) as those have different indentation rules. Indent
    // other expression, unless the indentation needs to be skipped.
    if (*I == prec::Conditional ||
        (!SkipFirstExtraIndent && *I > prec::Assignment &&
         !Current.isTrailingComment()))
      NewParenState.Indent += Style.ContinuationIndentWidth;
    if ((Previous && !Previous->opensScope()) || *I != prec::Comma)
      NewParenState.BreakBeforeParameter = false;
    State.Stack.push_back(NewParenState);
    SkipFirstExtraIndent = false;
  }
}

void ContinuationIndenter::moveStatePastFakeRParens(LineState &State) {
  for (unsigned i = 0, e = State.NextToken->FakeRParens; i != e; ++i) {
    unsigned VariablePos = State.Stack.back().VariablePos;
    if (State.Stack.size() == 1) {
      // Do not pop the last element.
      break;
    }
    State.Stack.pop_back();
    State.Stack.back().VariablePos = VariablePos;
  }
}

void ContinuationIndenter::moveStatePastScopeOpener(LineState &State,
                                                    bool Newline) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.opensScope())
    return;

  if (Current.MatchingParen && Current.BlockKind == BK_Block) {
    moveStateToNewBlock(State);
    return;
  }

  unsigned NewIndent;
  unsigned NewIndentLevel = State.Stack.back().IndentLevel;
  unsigned LastSpace = State.Stack.back().LastSpace;
  bool AvoidBinPacking;
  bool BreakBeforeParameter = false;
  unsigned NestedBlockIndent = std::max(State.Stack.back().StartOfFunctionCall,
                                        State.Stack.back().NestedBlockIndent);
  if (Current.isOneOf(tok::l_brace, TT_ArrayInitializerLSquare)) {
    if (Current.opensBlockOrBlockTypeList(Style)) {
      NewIndent = State.Stack.back().NestedBlockIndent + Style.IndentWidth;
      NewIndent = std::min(State.Column + 2, NewIndent);
      ++NewIndentLevel;
    } else {
      NewIndent = State.Stack.back().LastSpace + Style.ContinuationIndentWidth;
    }
    const FormatToken *NextNoComment = Current.getNextNonComment();
    bool EndsInComma = Current.MatchingParen &&
                       Current.MatchingParen->Previous &&
                       Current.MatchingParen->Previous->is(tok::comma);
    AvoidBinPacking =
        (Current.is(TT_ArrayInitializerLSquare) && EndsInComma) ||
        Current.is(TT_DictLiteral) ||
        Style.Language == FormatStyle::LK_Proto || !Style.BinPackArguments ||
        (NextNoComment && NextNoComment->is(TT_DesignatedInitializerPeriod));
    if (Current.ParameterCount > 1)
      NestedBlockIndent = std::max(NestedBlockIndent, State.Column + 1);
  } else {
    NewIndent = Style.ContinuationIndentWidth +
                std::max(State.Stack.back().LastSpace,
                         State.Stack.back().StartOfFunctionCall);

    // Ensure that different different brackets force relative alignment, e.g.:
    // void SomeFunction(vector<  // break
    //                       int> v);
    // FIXME: We likely want to do this for more combinations of brackets.
    // Verify that it is wanted for ObjC, too.
    if (Current.Tok.getKind() == tok::less &&
        Current.ParentBracket == tok::l_paren) {
      NewIndent = std::max(NewIndent, State.Stack.back().Indent);
      LastSpace = std::max(LastSpace, State.Stack.back().Indent);
    }

    AvoidBinPacking =
        (State.Line->MustBeDeclaration && !Style.BinPackParameters) ||
        (!State.Line->MustBeDeclaration && !Style.BinPackArguments) ||
        (Style.ExperimentalAutoDetectBinPacking &&
         (Current.PackingKind == PPK_OnePerLine ||
          (!BinPackInconclusiveFunctions &&
           Current.PackingKind == PPK_Inconclusive)));
    if (Current.is(TT_ObjCMethodExpr) && Current.MatchingParen) {
      if (Style.ColumnLimit) {
        // If this '[' opens an ObjC call, determine whether all parameters fit
        // into one line and put one per line if they don't.
        if (getLengthToMatchingParen(Current) + State.Column >
            getColumnLimit(State))
          BreakBeforeParameter = true;
      } else {
        // For ColumnLimit = 0, we have to figure out whether there is or has to
        // be a line break within this call.
        for (const FormatToken *Tok = &Current;
             Tok && Tok != Current.MatchingParen; Tok = Tok->Next) {
          if (Tok->MustBreakBefore ||
              (Tok->CanBreakBefore && Tok->NewlinesBefore > 0)) {
            BreakBeforeParameter = true;
            break;
          }
        }
      }
    }
  }
  // Generally inherit NoLineBreak from the current scope to nested scope.
  // However, don't do this for non-empty nested blocks, dict literals and
  // array literals as these follow different indentation rules.
  bool NoLineBreak =
      Current.Children.empty() &&
      !Current.isOneOf(TT_DictLiteral, TT_ArrayInitializerLSquare) &&
      (State.Stack.back().NoLineBreak ||
       (Current.is(TT_TemplateOpener) &&
        State.Stack.back().ContainsUnwrappedBuilder));
  State.Stack.push_back(ParenState(NewIndent, NewIndentLevel, LastSpace,
                                   AvoidBinPacking, NoLineBreak));
  State.Stack.back().NestedBlockIndent = NestedBlockIndent;
  State.Stack.back().BreakBeforeParameter = BreakBeforeParameter;
  State.Stack.back().HasMultipleNestedBlocks = Current.BlockParameterCount > 1;
}

void ContinuationIndenter::moveStatePastScopeCloser(LineState &State) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.closesScope())
    return;

  // If we encounter a closing ), ], } or >, we can remove a level from our
  // stacks.
  if (State.Stack.size() > 1 &&
      (Current.isOneOf(tok::r_paren, tok::r_square) ||
       (Current.is(tok::r_brace) && State.NextToken != State.Line->First) ||
       State.NextToken->is(TT_TemplateCloser)))
    State.Stack.pop_back();

  if (Current.is(tok::r_square)) {
    // If this ends the array subscript expr, reset the corresponding value.
    const FormatToken *NextNonComment = Current.getNextNonComment();
    if (NextNonComment && NextNonComment->isNot(tok::l_square))
      State.Stack.back().StartOfArraySubscripts = 0;
  }
}

void ContinuationIndenter::moveStateToNewBlock(LineState &State) {
  unsigned NestedBlockIndent = State.Stack.back().NestedBlockIndent;
  // ObjC block sometimes follow special indentation rules.
  unsigned NewIndent =
      NestedBlockIndent + (State.NextToken->is(TT_ObjCBlockLBrace)
                               ? Style.ObjCBlockIndentWidth
                               : Style.IndentWidth);
  State.Stack.push_back(ParenState(
      NewIndent, /*NewIndentLevel=*/State.Stack.back().IndentLevel + 1,
      State.Stack.back().LastSpace, /*AvoidBinPacking=*/true,
      /*NoLineBreak=*/false));
  State.Stack.back().NestedBlockIndent = NestedBlockIndent;
  State.Stack.back().BreakBeforeParameter = true;
}

unsigned ContinuationIndenter::addMultilineToken(const FormatToken &Current,
                                                 LineState &State) {
  // Break before further function parameters on all levels.
  for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
    State.Stack[i].BreakBeforeParameter = true;

  unsigned ColumnsUsed = State.Column;
  // We can only affect layout of the first and the last line, so the penalty
  // for all other lines is constant, and we ignore it.
  State.Column = Current.LastLineColumnWidth;

  if (ColumnsUsed > getColumnLimit(State))
    return Style.PenaltyExcessCharacter * (ColumnsUsed - getColumnLimit(State));
  return 0;
}

unsigned ContinuationIndenter::breakProtrudingToken(const FormatToken &Current,
                                                    LineState &State,
                                                    bool DryRun) {
  // Don't break multi-line tokens other than block comments. Instead, just
  // update the state.
  if (Current.isNot(TT_BlockComment) && Current.IsMultiline)
    return addMultilineToken(Current, State);

  // Don't break implicit string literals or import statements.
  if (Current.is(TT_ImplicitStringLiteral) ||
      State.Line->Type == LT_ImportStatement)
    return 0;

  if (!Current.isStringLiteral() && !Current.is(tok::comment))
    return 0;

  std::unique_ptr<BreakableToken> Token;
  unsigned StartColumn = State.Column - Current.ColumnWidth;
  unsigned ColumnLimit = getColumnLimit(State);

  if (Current.isStringLiteral()) {
    // FIXME: String literal breaking is currently disabled for Java and JS, as
    // it requires strings to be merged using "+" which we don't support.
    if (Style.Language == FormatStyle::LK_Java ||
        Style.Language == FormatStyle::LK_JavaScript ||
        !Style.BreakStringLiterals)
      return 0;

    // Don't break string literals inside preprocessor directives (except for
    // #define directives, as their contents are stored in separate lines and
    // are not affected by this check).
    // This way we avoid breaking code with line directives and unknown
    // preprocessor directives that contain long string literals.
    if (State.Line->Type == LT_PreprocessorDirective)
      return 0;
    // Exempts unterminated string literals from line breaking. The user will
    // likely want to terminate the string before any line breaking is done.
    if (Current.IsUnterminatedLiteral)
      return 0;

    StringRef Text = Current.TokenText;
    StringRef Prefix;
    StringRef Postfix;
    bool IsNSStringLiteral = false;
    // FIXME: Handle whitespace between '_T', '(', '"..."', and ')'.
    // FIXME: Store Prefix and Suffix (or PrefixLength and SuffixLength to
    // reduce the overhead) for each FormatToken, which is a string, so that we
    // don't run multiple checks here on the hot path.
    if (Text.startswith("\"") && Current.Previous &&
        Current.Previous->is(tok::at)) {
      IsNSStringLiteral = true;
      Prefix = "@\"";
    }
    if ((Text.endswith(Postfix = "\"") &&
         (IsNSStringLiteral || Text.startswith(Prefix = "\"") ||
          Text.startswith(Prefix = "u\"") || Text.startswith(Prefix = "U\"") ||
          Text.startswith(Prefix = "u8\"") ||
          Text.startswith(Prefix = "L\""))) ||
        (Text.startswith(Prefix = "_T(\"") && Text.endswith(Postfix = "\")"))) {
      Token.reset(new BreakableStringLiteral(
          Current, State.Line->Level, StartColumn, Prefix, Postfix,
          State.Line->InPPDirective, Encoding, Style));
    } else {
      return 0;
    }
  } else if (Current.is(TT_BlockComment) && Current.isTrailingComment()) {
    if (!Style.ReflowComments ||
        CommentPragmasRegex.match(Current.TokenText.substr(2)))
      return 0;
    Token.reset(new BreakableBlockComment(
        Current, State.Line->Level, StartColumn, Current.OriginalColumn,
        !Current.Previous, State.Line->InPPDirective, Encoding, Style));
  } else if (Current.is(TT_LineComment) &&
             (Current.Previous == nullptr ||
              Current.Previous->isNot(TT_ImplicitStringLiteral))) {
    if (!Style.ReflowComments ||
        CommentPragmasRegex.match(Current.TokenText.substr(2)))
      return 0;
    Token.reset(new BreakableLineComment(Current, State.Line->Level,
                                         StartColumn, /*InPPDirective=*/false,
                                         Encoding, Style));
    // We don't insert backslashes when breaking line comments.
    ColumnLimit = Style.ColumnLimit;
  } else {
    return 0;
  }
  if (Current.UnbreakableTailLength >= ColumnLimit)
    return 0;

  unsigned RemainingSpace = ColumnLimit - Current.UnbreakableTailLength;
  bool BreakInserted = false;
  unsigned Penalty = 0;
  unsigned RemainingTokenColumns = 0;
  for (unsigned LineIndex = 0, EndIndex = Token->getLineCount();
       LineIndex != EndIndex; ++LineIndex) {
    if (!DryRun)
      Token->replaceWhitespaceBefore(LineIndex, Whitespaces);
    unsigned TailOffset = 0;
    RemainingTokenColumns =
        Token->getLineLengthAfterSplit(LineIndex, TailOffset, StringRef::npos);
    while (RemainingTokenColumns > RemainingSpace) {
      BreakableToken::Split Split =
          Token->getSplit(LineIndex, TailOffset, ColumnLimit);
      if (Split.first == StringRef::npos) {
        // The last line's penalty is handled in addNextStateToQueue().
        if (LineIndex < EndIndex - 1)
          Penalty += Style.PenaltyExcessCharacter *
                     (RemainingTokenColumns - RemainingSpace);
        break;
      }
      assert(Split.first != 0);
      unsigned NewRemainingTokenColumns = Token->getLineLengthAfterSplit(
          LineIndex, TailOffset + Split.first + Split.second, StringRef::npos);

      // We can remove extra whitespace instead of breaking the line.
      if (RemainingTokenColumns + 1 - Split.second <= RemainingSpace) {
        RemainingTokenColumns = 0;
        if (!DryRun)
          Token->replaceWhitespace(LineIndex, TailOffset, Split, Whitespaces);
        break;
      }

      // When breaking before a tab character, it may be moved by a few columns,
      // but will still be expanded to the next tab stop, so we don't save any
      // columns.
      if (NewRemainingTokenColumns == RemainingTokenColumns)
        break;

      assert(NewRemainingTokenColumns < RemainingTokenColumns);
      if (!DryRun)
        Token->insertBreak(LineIndex, TailOffset, Split, Whitespaces);
      Penalty += Current.SplitPenalty;
      unsigned ColumnsUsed =
          Token->getLineLengthAfterSplit(LineIndex, TailOffset, Split.first);
      if (ColumnsUsed > ColumnLimit) {
        Penalty += Style.PenaltyExcessCharacter * (ColumnsUsed - ColumnLimit);
      }
      TailOffset += Split.first + Split.second;
      RemainingTokenColumns = NewRemainingTokenColumns;
      BreakInserted = true;
    }
  }

  State.Column = RemainingTokenColumns;

  if (BreakInserted) {
    // If we break the token inside a parameter list, we need to break before
    // the next parameter on all levels, so that the next parameter is clearly
    // visible. Line comments already introduce a break.
    if (Current.isNot(TT_LineComment)) {
      for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
        State.Stack[i].BreakBeforeParameter = true;
    }

    Penalty += Current.isStringLiteral() ? Style.PenaltyBreakString
                                         : Style.PenaltyBreakComment;

    State.Stack.back().LastSpace = StartColumn;
  }
  return Penalty;
}

unsigned ContinuationIndenter::getColumnLimit(const LineState &State) const {
  // In preprocessor directives reserve two chars for trailing " \"
  return Style.ColumnLimit - (State.Line->InPPDirective ? 2 : 0);
}

bool ContinuationIndenter::nextIsMultilineString(const LineState &State) {
  const FormatToken &Current = *State.NextToken;
  if (!Current.isStringLiteral() || Current.is(TT_ImplicitStringLiteral))
    return false;
  // We never consider raw string literals "multiline" for the purpose of
  // AlwaysBreakBeforeMultilineStrings implementation as they are special-cased
  // (see TokenAnnotator::mustBreakBefore().
  if (Current.TokenText.startswith("R\""))
    return false;
  if (Current.IsMultiline)
    return true;
  if (Current.getNextNonComment() &&
      Current.getNextNonComment()->isStringLiteral())
    return true; // Implicit concatenation.
  if (Style.ColumnLimit != 0 &&
      State.Column + Current.ColumnWidth + Current.UnbreakableTailLength >
          Style.ColumnLimit)
    return true; // String will be split.
  return false;
}

} // namespace format
} // namespace clang
