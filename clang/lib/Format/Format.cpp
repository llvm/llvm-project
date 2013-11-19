//===--- Format.cpp - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements functions declared in Format.h. This will be
/// split into separate files as we go.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-formatter"

#include "ContinuationIndenter.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "WhitespaceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/Path.h"
#include <queue>
#include <string>

namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<clang::format::FormatStyle::LanguageStandard> {
  static void enumeration(IO &IO,
                          clang::format::FormatStyle::LanguageStandard &Value) {
    IO.enumCase(Value, "Cpp03", clang::format::FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "C++03", clang::format::FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "Cpp11", clang::format::FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "C++11", clang::format::FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "Auto", clang::format::FormatStyle::LS_Auto);
  }
};

template <>
struct ScalarEnumerationTraits<clang::format::FormatStyle::UseTabStyle> {
  static void enumeration(IO &IO,
                          clang::format::FormatStyle::UseTabStyle &Value) {
    IO.enumCase(Value, "Never", clang::format::FormatStyle::UT_Never);
    IO.enumCase(Value, "false", clang::format::FormatStyle::UT_Never);
    IO.enumCase(Value, "Always", clang::format::FormatStyle::UT_Always);
    IO.enumCase(Value, "true", clang::format::FormatStyle::UT_Always);
    IO.enumCase(Value, "ForIndentation",
                clang::format::FormatStyle::UT_ForIndentation);
  }
};

template <>
struct ScalarEnumerationTraits<clang::format::FormatStyle::BraceBreakingStyle> {
  static void
  enumeration(IO &IO, clang::format::FormatStyle::BraceBreakingStyle &Value) {
    IO.enumCase(Value, "Attach", clang::format::FormatStyle::BS_Attach);
    IO.enumCase(Value, "Linux", clang::format::FormatStyle::BS_Linux);
    IO.enumCase(Value, "Stroustrup", clang::format::FormatStyle::BS_Stroustrup);
    IO.enumCase(Value, "Allman", clang::format::FormatStyle::BS_Allman);
  }
};

template <>
struct ScalarEnumerationTraits<
    clang::format::FormatStyle::NamespaceIndentationKind> {
  static void
  enumeration(IO &IO,
              clang::format::FormatStyle::NamespaceIndentationKind &Value) {
    IO.enumCase(Value, "None", clang::format::FormatStyle::NI_None);
    IO.enumCase(Value, "Inner", clang::format::FormatStyle::NI_Inner);
    IO.enumCase(Value, "All", clang::format::FormatStyle::NI_All);
  }
};

template <> struct MappingTraits<clang::format::FormatStyle> {
  static void mapping(llvm::yaml::IO &IO, clang::format::FormatStyle &Style) {
    if (IO.outputting()) {
      StringRef StylesArray[] = { "LLVM",    "Google", "Chromium",
                                  "Mozilla", "WebKit" };
      ArrayRef<StringRef> Styles(StylesArray);
      for (size_t i = 0, e = Styles.size(); i < e; ++i) {
        StringRef StyleName(Styles[i]);
        clang::format::FormatStyle PredefinedStyle;
        if (clang::format::getPredefinedStyle(StyleName, &PredefinedStyle) &&
            Style == PredefinedStyle) {
          IO.mapOptional("# BasedOnStyle", StyleName);
          break;
        }
      }
    } else {
      StringRef BasedOnStyle;
      IO.mapOptional("BasedOnStyle", BasedOnStyle);
      if (!BasedOnStyle.empty())
        if (!clang::format::getPredefinedStyle(BasedOnStyle, &Style)) {
          IO.setError(Twine("Unknown value for BasedOnStyle: ", BasedOnStyle));
          return;
        }
    }

    IO.mapOptional("AccessModifierOffset", Style.AccessModifierOffset);
    IO.mapOptional("ConstructorInitializerIndentWidth",
                   Style.ConstructorInitializerIndentWidth);
    IO.mapOptional("AlignEscapedNewlinesLeft", Style.AlignEscapedNewlinesLeft);
    IO.mapOptional("AlignTrailingComments", Style.AlignTrailingComments);
    IO.mapOptional("AllowAllParametersOfDeclarationOnNextLine",
                   Style.AllowAllParametersOfDeclarationOnNextLine);
    IO.mapOptional("AllowShortIfStatementsOnASingleLine",
                   Style.AllowShortIfStatementsOnASingleLine);
    IO.mapOptional("AllowShortLoopsOnASingleLine",
                   Style.AllowShortLoopsOnASingleLine);
    IO.mapOptional("AlwaysBreakTemplateDeclarations",
                   Style.AlwaysBreakTemplateDeclarations);
    IO.mapOptional("AlwaysBreakBeforeMultilineStrings",
                   Style.AlwaysBreakBeforeMultilineStrings);
    IO.mapOptional("BreakBeforeBinaryOperators",
                   Style.BreakBeforeBinaryOperators);
    IO.mapOptional("BreakBeforeTernaryOperators",
                   Style.BreakBeforeTernaryOperators);
    IO.mapOptional("BreakConstructorInitializersBeforeComma",
                   Style.BreakConstructorInitializersBeforeComma);
    IO.mapOptional("BinPackParameters", Style.BinPackParameters);
    IO.mapOptional("ColumnLimit", Style.ColumnLimit);
    IO.mapOptional("ConstructorInitializerAllOnOneLineOrOnePerLine",
                   Style.ConstructorInitializerAllOnOneLineOrOnePerLine);
    IO.mapOptional("DerivePointerBinding", Style.DerivePointerBinding);
    IO.mapOptional("ExperimentalAutoDetectBinPacking",
                   Style.ExperimentalAutoDetectBinPacking);
    IO.mapOptional("IndentCaseLabels", Style.IndentCaseLabels);
    IO.mapOptional("MaxEmptyLinesToKeep", Style.MaxEmptyLinesToKeep);
    IO.mapOptional("NamespaceIndentation", Style.NamespaceIndentation);
    IO.mapOptional("ObjCSpaceBeforeProtocolList",
                   Style.ObjCSpaceBeforeProtocolList);
    IO.mapOptional("PenaltyBreakBeforeFirstCallParameter",
                   Style.PenaltyBreakBeforeFirstCallParameter);
    IO.mapOptional("PenaltyBreakComment", Style.PenaltyBreakComment);
    IO.mapOptional("PenaltyBreakString", Style.PenaltyBreakString);
    IO.mapOptional("PenaltyBreakFirstLessLess",
                   Style.PenaltyBreakFirstLessLess);
    IO.mapOptional("PenaltyExcessCharacter", Style.PenaltyExcessCharacter);
    IO.mapOptional("PenaltyReturnTypeOnItsOwnLine",
                   Style.PenaltyReturnTypeOnItsOwnLine);
    IO.mapOptional("PointerBindsToType", Style.PointerBindsToType);
    IO.mapOptional("SpacesBeforeTrailingComments",
                   Style.SpacesBeforeTrailingComments);
    IO.mapOptional("Cpp11BracedListStyle", Style.Cpp11BracedListStyle);
    IO.mapOptional("Standard", Style.Standard);
    IO.mapOptional("IndentWidth", Style.IndentWidth);
    IO.mapOptional("TabWidth", Style.TabWidth);
    IO.mapOptional("UseTab", Style.UseTab);
    IO.mapOptional("BreakBeforeBraces", Style.BreakBeforeBraces);
    IO.mapOptional("IndentFunctionDeclarationAfterType",
                   Style.IndentFunctionDeclarationAfterType);
    IO.mapOptional("SpacesInParentheses", Style.SpacesInParentheses);
    IO.mapOptional("SpacesInAngles", Style.SpacesInAngles);
    IO.mapOptional("SpaceInEmptyParentheses", Style.SpaceInEmptyParentheses);
    IO.mapOptional("SpacesInCStyleCastParentheses",
                   Style.SpacesInCStyleCastParentheses);
    IO.mapOptional("SpaceAfterControlStatementKeyword",
                   Style.SpaceAfterControlStatementKeyword);
    IO.mapOptional("SpaceBeforeAssignmentOperators",
                   Style.SpaceBeforeAssignmentOperators);
    IO.mapOptional("ContinuationIndentWidth", Style.ContinuationIndentWidth);
  }
};
}
}

namespace clang {
namespace format {

void setDefaultPenalties(FormatStyle &Style) {
  Style.PenaltyBreakComment = 60;
  Style.PenaltyBreakFirstLessLess = 120;
  Style.PenaltyBreakString = 1000;
  Style.PenaltyExcessCharacter = 1000000;
}

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.AlignEscapedNewlinesLeft = false;
  LLVMStyle.AlignTrailingComments = true;
  LLVMStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.AllowShortLoopsOnASingleLine = false;
  LLVMStyle.AlwaysBreakBeforeMultilineStrings = false;
  LLVMStyle.AlwaysBreakTemplateDeclarations = false;
  LLVMStyle.BinPackParameters = true;
  LLVMStyle.BreakBeforeBinaryOperators = false;
  LLVMStyle.BreakBeforeTernaryOperators = true;
  LLVMStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  LLVMStyle.BreakConstructorInitializersBeforeComma = false;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.ConstructorInitializerIndentWidth = 4;
  LLVMStyle.Cpp11BracedListStyle = false;
  LLVMStyle.DerivePointerBinding = false;
  LLVMStyle.ExperimentalAutoDetectBinPacking = false;
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.IndentFunctionDeclarationAfterType = false;
  LLVMStyle.IndentWidth = 2;
  LLVMStyle.TabWidth = 8;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.NamespaceIndentation = FormatStyle::NI_None;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PointerBindsToType = false;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  LLVMStyle.Standard = FormatStyle::LS_Cpp03;
  LLVMStyle.UseTab = FormatStyle::UT_Never;
  LLVMStyle.SpacesInParentheses = false;
  LLVMStyle.SpaceInEmptyParentheses = false;
  LLVMStyle.SpacesInCStyleCastParentheses = false;
  LLVMStyle.SpaceAfterControlStatementKeyword = true;
  LLVMStyle.SpaceBeforeAssignmentOperators = true;
  LLVMStyle.ContinuationIndentWidth = 4;
  LLVMStyle.SpacesInAngles = false;

  setDefaultPenalties(LLVMStyle);
  LLVMStyle.PenaltyReturnTypeOnItsOwnLine = 60;
  LLVMStyle.PenaltyBreakBeforeFirstCallParameter = 19;

  return LLVMStyle;
}

FormatStyle getGoogleStyle() {
  FormatStyle GoogleStyle;
  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.AlignEscapedNewlinesLeft = true;
  GoogleStyle.AlignTrailingComments = true;
  GoogleStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = true;
  GoogleStyle.AllowShortLoopsOnASingleLine = true;
  GoogleStyle.AlwaysBreakBeforeMultilineStrings = true;
  GoogleStyle.AlwaysBreakTemplateDeclarations = true;
  GoogleStyle.BinPackParameters = true;
  GoogleStyle.BreakBeforeBinaryOperators = false;
  GoogleStyle.BreakBeforeTernaryOperators = true;
  GoogleStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  GoogleStyle.BreakConstructorInitializersBeforeComma = false;
  GoogleStyle.ColumnLimit = 80;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.ConstructorInitializerIndentWidth = 4;
  GoogleStyle.Cpp11BracedListStyle = true;
  GoogleStyle.DerivePointerBinding = true;
  GoogleStyle.ExperimentalAutoDetectBinPacking = false;
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.IndentFunctionDeclarationAfterType = true;
  GoogleStyle.IndentWidth = 2;
  GoogleStyle.TabWidth = 8;
  GoogleStyle.MaxEmptyLinesToKeep = 1;
  GoogleStyle.NamespaceIndentation = FormatStyle::NI_None;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PointerBindsToType = true;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  GoogleStyle.Standard = FormatStyle::LS_Auto;
  GoogleStyle.UseTab = FormatStyle::UT_Never;
  GoogleStyle.SpacesInParentheses = false;
  GoogleStyle.SpaceInEmptyParentheses = false;
  GoogleStyle.SpacesInCStyleCastParentheses = false;
  GoogleStyle.SpaceAfterControlStatementKeyword = true;
  GoogleStyle.SpaceBeforeAssignmentOperators = true;
  GoogleStyle.ContinuationIndentWidth = 4;
  GoogleStyle.SpacesInAngles = false;

  setDefaultPenalties(GoogleStyle);
  GoogleStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  GoogleStyle.PenaltyBreakBeforeFirstCallParameter = 1;

  return GoogleStyle;
}

FormatStyle getChromiumStyle() {
  FormatStyle ChromiumStyle = getGoogleStyle();
  ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  ChromiumStyle.AllowShortIfStatementsOnASingleLine = false;
  ChromiumStyle.AllowShortLoopsOnASingleLine = false;
  ChromiumStyle.BinPackParameters = false;
  ChromiumStyle.DerivePointerBinding = false;
  ChromiumStyle.Standard = FormatStyle::LS_Cpp03;
  return ChromiumStyle;
}

FormatStyle getMozillaStyle() {
  FormatStyle MozillaStyle = getLLVMStyle();
  MozillaStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  MozillaStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  MozillaStyle.DerivePointerBinding = true;
  MozillaStyle.IndentCaseLabels = true;
  MozillaStyle.ObjCSpaceBeforeProtocolList = false;
  MozillaStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  MozillaStyle.PointerBindsToType = true;
  return MozillaStyle;
}

FormatStyle getWebKitStyle() {
  FormatStyle Style = getLLVMStyle();
  Style.AccessModifierOffset = -4;
  Style.AlignTrailingComments = false;
  Style.BreakBeforeBinaryOperators = true;
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  Style.BreakConstructorInitializersBeforeComma = true;
  Style.ColumnLimit = 0;
  Style.IndentWidth = 4;
  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  Style.PointerBindsToType = true;
  return Style;
}

bool getPredefinedStyle(StringRef Name, FormatStyle *Style) {
  if (Name.equals_lower("llvm"))
    *Style = getLLVMStyle();
  else if (Name.equals_lower("chromium"))
    *Style = getChromiumStyle();
  else if (Name.equals_lower("mozilla"))
    *Style = getMozillaStyle();
  else if (Name.equals_lower("google"))
    *Style = getGoogleStyle();
  else if (Name.equals_lower("webkit"))
    *Style = getWebKitStyle();
  else
    return false;

  return true;
}

llvm::error_code parseConfiguration(StringRef Text, FormatStyle *Style) {
  if (Text.trim().empty())
    return llvm::make_error_code(llvm::errc::invalid_argument);
  llvm::yaml::Input Input(Text);
  Input >> *Style;
  return Input.error();
}

std::string configurationAsText(const FormatStyle &Style) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  FormatStyle NonConstStyle = Style;
  Output << NonConstStyle;
  return Stream.str();
}

namespace {

class NoColumnLimitFormatter {
public:
  NoColumnLimitFormatter(ContinuationIndenter *Indenter) : Indenter(Indenter) {}

  /// \brief Formats the line starting at \p State, simply keeping all of the
  /// input's line breaking decisions.
  void format(unsigned FirstIndent, const AnnotatedLine *Line) {
    LineState State =
        Indenter->getInitialState(FirstIndent, Line, /*DryRun=*/false);
    while (State.NextToken != NULL) {
      bool Newline =
          Indenter->mustBreak(State) ||
          (Indenter->canBreak(State) && State.NextToken->NewlinesBefore > 0);
      Indenter->addTokenToState(State, Newline, /*DryRun=*/false);
    }
  }

private:
  ContinuationIndenter *Indenter;
};

class LineJoiner {
public:
  LineJoiner(const FormatStyle &Style) : Style(Style) {}

  /// \brief Calculates how many lines can be merged into 1 starting at \p I.
  unsigned
  tryFitMultipleLinesInOne(unsigned Indent,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator &I,
                           SmallVectorImpl<AnnotatedLine *>::const_iterator E) {
    // We can never merge stuff if there are trailing line comments.
    AnnotatedLine *TheLine = *I;
    if (TheLine->Last->Type == TT_LineComment)
      return 0;

    if (Indent > Style.ColumnLimit)
      return 0;

    unsigned Limit =
        Style.ColumnLimit == 0 ? UINT_MAX : Style.ColumnLimit - Indent;
    // If we already exceed the column limit, we set 'Limit' to 0. The different
    // tryMerge..() functions can then decide whether to still do merging.
    Limit = TheLine->Last->TotalLength > Limit
                ? 0
                : Limit - TheLine->Last->TotalLength;

    if (I + 1 == E || I[1]->Type == LT_Invalid)
      return 0;

    if (TheLine->Last->is(tok::l_brace)) {
      return tryMergeSimpleBlock(I, E, Limit);
    } else if (Style.AllowShortIfStatementsOnASingleLine &&
               TheLine->First->is(tok::kw_if)) {
      return tryMergeSimpleControlStatement(I, E, Limit);
    } else if (Style.AllowShortLoopsOnASingleLine &&
               TheLine->First->isOneOf(tok::kw_for, tok::kw_while)) {
      return tryMergeSimpleControlStatement(I, E, Limit);
    } else if (TheLine->InPPDirective && (TheLine->First->HasUnescapedNewline ||
                                          TheLine->First->IsFirst)) {
      return tryMergeSimplePPDirective(I, E, Limit);
    }
    return 0;
  }

private:
  unsigned
  tryMergeSimplePPDirective(SmallVectorImpl<AnnotatedLine *>::const_iterator &I,
                            SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                            unsigned Limit) {
    if (Limit == 0)
      return 0;
    if (!I[1]->InPPDirective || I[1]->First->HasUnescapedNewline)
      return 0;
    if (I + 2 != E && I[2]->InPPDirective && !I[2]->First->HasUnescapedNewline)
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    return 1;
  }

  unsigned tryMergeSimpleControlStatement(
      SmallVectorImpl<AnnotatedLine *>::const_iterator &I,
      SmallVectorImpl<AnnotatedLine *>::const_iterator E, unsigned Limit) {
    if (Limit == 0)
      return 0;
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman &&
        I[1]->First->is(tok::l_brace))
      return 0;
    if (I[1]->InPPDirective != (*I)->InPPDirective ||
        (I[1]->InPPDirective && I[1]->First->HasUnescapedNewline))
      return 0;
    AnnotatedLine &Line = **I;
    if (Line.Last->isNot(tok::r_paren))
      return 0;
    if (1 + I[1]->Last->TotalLength > Limit)
      return 0;
    if (I[1]->First->isOneOf(tok::semi, tok::kw_if, tok::kw_for,
                                   tok::kw_while) ||
        I[1]->First->Type == TT_LineComment)
      return 0;
    // Only inline simple if's (no nested if or else).
    if (I + 2 != E && Line.First->is(tok::kw_if) &&
        I[2]->First->is(tok::kw_else))
      return 0;
    return 1;
  }

  unsigned
  tryMergeSimpleBlock(SmallVectorImpl<AnnotatedLine *>::const_iterator &I,
                      SmallVectorImpl<AnnotatedLine *>::const_iterator E,
                      unsigned Limit) {
    // No merging if the brace already is on the next line.
    if (Style.BreakBeforeBraces != FormatStyle::BS_Attach)
      return 0;

    // First, check that the current line allows merging. This is the case if
    // we're not in a control flow statement and the last token is an opening
    // brace.
    AnnotatedLine &Line = **I;
    if (Line.First->isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::r_brace,
                            tok::kw_else, tok::kw_try, tok::kw_catch,
                            tok::kw_for,
                            // This gets rid of all ObjC @ keywords and methods.
                            tok::at, tok::minus, tok::plus))
      return 0;

    FormatToken *Tok = I[1]->First;
    if (Tok->is(tok::r_brace) && !Tok->MustBreakBefore &&
        (Tok->getNextNonComment() == NULL ||
         Tok->getNextNonComment()->is(tok::semi))) {
      // We merge empty blocks even if the line exceeds the column limit.
      Tok->SpacesRequiredBefore = 0;
      Tok->CanBreakBefore = true;
      return 1;
    } else if (Limit != 0 && Line.First->isNot(tok::kw_namespace)) {
      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || I[2]->Type == LT_Invalid)
        return 0;

      if (!nextTwoLinesFitInto(I, Limit))
        return 0;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if (I[1]->Last->Type == TT_LineComment || Tok->MustBreakBefore)
        return 0;
      do {
        if (Tok->isOneOf(tok::l_brace, tok::r_brace))
          return 0;
        Tok = Tok->Next;
      } while (Tok != NULL);

      // Last, check that the third line contains a single closing brace.
      Tok = I[2]->First;
      if (Tok->getNextNonComment() != NULL || Tok->isNot(tok::r_brace) ||
          Tok->MustBreakBefore)
        return 0;

      return 2;
    }
    return 0;
  }

  bool nextTwoLinesFitInto(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                           unsigned Limit) {
    return 1 + I[1]->Last->TotalLength + 1 + I[2]->Last->TotalLength <= Limit;
  }

  const FormatStyle &Style;
};

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(SourceManager &SourceMgr,
                         SmallVectorImpl<CharSourceRange> &Ranges,
                         ContinuationIndenter *Indenter,
                         WhitespaceManager *Whitespaces,
                         const FormatStyle &Style)
      : SourceMgr(SourceMgr), Ranges(Ranges), Indenter(Indenter),
        Whitespaces(Whitespaces), Style(Style), Joiner(Style) {}

  unsigned format(const SmallVectorImpl<AnnotatedLine *> &Lines, bool DryRun,
                  int AdditionalIndent = 0) {
    assert(!Lines.empty());
    unsigned Penalty = 0;
    std::vector<int> IndentForLevel;
    for (unsigned i = 0, e = Lines[0]->Level; i != e; ++i)
      IndentForLevel.push_back(Style.IndentWidth * i + AdditionalIndent);
    bool PreviousLineWasTouched = false;
    const AnnotatedLine *PreviousLine = NULL;
    bool FormatPPDirective = false;
    for (SmallVectorImpl<AnnotatedLine *>::const_iterator I = Lines.begin(),
                                                          E = Lines.end();
         I != E; ++I) {
      const AnnotatedLine &TheLine = **I;
      const FormatToken *FirstTok = TheLine.First;
      int Offset = getIndentOffset(*FirstTok);

      // Check whether this line is part of a formatted preprocessor directive.
      if (FirstTok->HasUnescapedNewline)
        FormatPPDirective = false;
      if (!FormatPPDirective && TheLine.InPPDirective &&
          (touchesLine(TheLine) || touchesPPDirective(I + 1, E)))
        FormatPPDirective = true;

      // Determine indent and try to merge multiple unwrapped lines.
      while (IndentForLevel.size() <= TheLine.Level)
        IndentForLevel.push_back(-1);
      IndentForLevel.resize(TheLine.Level + 1);
      unsigned Indent = getIndent(IndentForLevel, TheLine.Level);
      if (static_cast<int>(Indent) + Offset >= 0)
        Indent += Offset;
      unsigned MergedLines = Joiner.tryFitMultipleLinesInOne(Indent, I, E);
      if (!DryRun) {
        for (unsigned i = 0; i < MergedLines; ++i) {
          join(*I[i], *I[i + 1]);
        }
      }
      I += MergedLines;

      bool WasMoved = PreviousLineWasTouched && FirstTok->NewlinesBefore == 0;
      if (TheLine.First->is(tok::eof)) {
        if (PreviousLineWasTouched && !DryRun) {
          unsigned Newlines = std::min(FirstTok->NewlinesBefore, 1u);
          Whitespaces->replaceWhitespace(*TheLine.First, Newlines,
                                         /*IndentLevel=*/0, /*Spaces=*/0,
                                         /*TargetColumn=*/0);
        }
      } else if (TheLine.Type != LT_Invalid &&
                 (WasMoved || FormatPPDirective || touchesLine(TheLine))) {
        unsigned LevelIndent =
            getIndent(IndentForLevel, TheLine.Level);
        if (FirstTok->WhitespaceRange.isValid()) {
          if (!DryRun)
            formatFirstToken(*TheLine.First, PreviousLine, TheLine.Level,
                             Indent, TheLine.InPPDirective);
        } else {
          Indent = LevelIndent = FirstTok->OriginalColumn;
        }

        // If everything fits on a single line, just put it there.
        unsigned ColumnLimit = Style.ColumnLimit;
        if (I + 1 != E) {
          AnnotatedLine *NextLine = I[1];
          if (NextLine->InPPDirective && !NextLine->First->HasUnescapedNewline)
            ColumnLimit = getColumnLimit(TheLine.InPPDirective);
        }

        if (TheLine.Last->TotalLength + Indent <= ColumnLimit) {
          LineState State = Indenter->getInitialState(Indent, &TheLine, DryRun);
          while (State.NextToken != NULL)
            Indenter->addTokenToState(State, /*Newline=*/false, DryRun);
        } else if (Style.ColumnLimit == 0) {
          NoColumnLimitFormatter Formatter(Indenter);
          if (!DryRun)
            Formatter.format(Indent, &TheLine);
        } else {
          Penalty += format(TheLine, Indent, DryRun);
        }

        IndentForLevel[TheLine.Level] = LevelIndent;
        PreviousLineWasTouched = true;
      } else {
        // Format the first token if necessary, and notify the WhitespaceManager
        // about the unchanged whitespace.
        for (FormatToken *Tok = TheLine.First; Tok != NULL; Tok = Tok->Next) {
          if (Tok == TheLine.First &&
              (Tok->NewlinesBefore > 0 || Tok->IsFirst)) {
            unsigned LevelIndent = Tok->OriginalColumn;
            if (!DryRun) {
              // Remove trailing whitespace of the previous line if it was
              // touched.
              if (PreviousLineWasTouched || touchesEmptyLineBefore(TheLine)) {
                formatFirstToken(*Tok, PreviousLine, TheLine.Level, LevelIndent,
                                 TheLine.InPPDirective);
              } else {
                Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
              }
            }

            if (static_cast<int>(LevelIndent) - Offset >= 0)
              LevelIndent -= Offset;
            if (Tok->isNot(tok::comment))
              IndentForLevel[TheLine.Level] = LevelIndent;
          } else if (!DryRun) {
            Whitespaces->addUntouchableToken(*Tok, TheLine.InPPDirective);
          }
        }
        // If we did not reformat this unwrapped line, the column at the end of
        // the last token is unchanged - thus, we can calculate the end of the
        // last token.
        PreviousLineWasTouched = false;
      }
      if (!DryRun) {
        for (FormatToken *Tok = TheLine.First; Tok != NULL; Tok = Tok->Next) {
          Tok->Finalized = true;
        }
      }
      PreviousLine = *I;
    }
    return Penalty;
  }

private:
  /// \brief Formats an \c AnnotatedLine and returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned format(const AnnotatedLine &Line, unsigned FirstIndent,
                  bool DryRun) {
    LineState State = Indenter->getInitialState(FirstIndent, &Line, DryRun);

    // If the ObjC method declaration does not fit on a line, we should format
    // it with one arg per line.
    if (State.Line->Type == LT_ObjCMethodDecl)
      State.Stack.back().BreakBeforeParameter = true;

    // Find best solution in solution space.
    return analyzeSolutionSpace(State, DryRun);
  }

  /// \brief An edge in the solution space from \c Previous->State to \c State,
  /// inserting a newline dependent on the \c NewLine.
  struct StateNode {
    StateNode(const LineState &State, bool NewLine, StateNode *Previous)
        : State(State), NewLine(NewLine), Previous(Previous) {}
    LineState State;
    bool NewLine;
    StateNode *Previous;
  };

  /// \brief A pair of <penalty, count> that is used to prioritize the BFS on.
  ///
  /// In case of equal penalties, we want to prefer states that were inserted
  /// first. During state generation we make sure that we insert states first
  /// that break the line as late as possible.
  typedef std::pair<unsigned, unsigned> OrderedPenalty;

  /// \brief An item in the prioritized BFS search queue. The \c StateNode's
  /// \c State has the given \c OrderedPenalty.
  typedef std::pair<OrderedPenalty, StateNode *> QueueItem;

  /// \brief The BFS queue type.
  typedef std::priority_queue<QueueItem, std::vector<QueueItem>,
                              std::greater<QueueItem> > QueueType;

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const FormatToken &RootToken) {
    if (RootToken.isAccessSpecifier(false) || RootToken.isObjCAccessSpecifier())
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  void formatFirstToken(FormatToken &RootToken,
                        const AnnotatedLine *PreviousLine, unsigned IndentLevel,
                        unsigned Indent, bool InPPDirective) {
    unsigned Newlines =
        std::min(RootToken.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    // Remove empty lines before "}" where applicable.
    if (RootToken.is(tok::r_brace) &&
        (!RootToken.Next ||
         (RootToken.Next->is(tok::semi) && !RootToken.Next->Next)))
      Newlines = std::min(Newlines, 1u);
    if (Newlines == 0 && !RootToken.IsFirst)
      Newlines = 1;

    // Insert extra new line before access specifiers.
    if (PreviousLine && PreviousLine->Last->isOneOf(tok::semi, tok::r_brace) &&
        RootToken.isAccessSpecifier() && RootToken.NewlinesBefore == 1)
      ++Newlines;

    // Remove empty lines after access specifiers.
    if (PreviousLine && PreviousLine->First->isAccessSpecifier())
      Newlines = std::min(1u, Newlines);

    Whitespaces->replaceWhitespace(
        RootToken, Newlines, IndentLevel, Indent, Indent,
        InPPDirective && !RootToken.HasUnescapedNewline);
  }

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(const std::vector<int> IndentForLevel, unsigned Level) {
    if (IndentForLevel[Level] != -1)
      return IndentForLevel[Level];
    if (Level == 0)
      return 0;
    return getIndent(IndentForLevel, Level - 1) + Style.IndentWidth;
  }

  void join(AnnotatedLine &A, const AnnotatedLine &B) {
    assert(!A.Last->Next);
    assert(!B.First->Previous);
    A.Last->Next = B.First;
    B.First->Previous = A.Last;
    B.First->CanBreakBefore = true;
    unsigned LengthA = A.Last->TotalLength + B.First->SpacesRequiredBefore;
    for (FormatToken *Tok = B.First; Tok; Tok = Tok->Next) {
      Tok->TotalLength += LengthA;
      A.Last = Tok;
    }
  }

  unsigned getColumnLimit(bool InPPDirective) const {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (InPPDirective ? 2 : 0);
  }

  bool touchesRanges(const CharSourceRange &Range) {
    for (SmallVectorImpl<CharSourceRange>::const_iterator I = Ranges.begin(),
                                                          E = Ranges.end();
         I != E; ++I) {
      if (!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(), I->getBegin()) &&
          !SourceMgr.isBeforeInTranslationUnit(I->getEnd(), Range.getBegin()))
        return true;
    }
    return false;
  }

  bool touchesLine(const AnnotatedLine &TheLine) {
    const FormatToken *First = TheLine.First;
    const FormatToken *Last = TheLine.Last;
    CharSourceRange LineRange = CharSourceRange::getCharRange(
        First->WhitespaceRange.getBegin().getLocWithOffset(
            First->LastNewlineOffset),
        Last->getStartOfNonWhitespace().getLocWithOffset(
            Last->TokenText.size() - 1));
    return touchesRanges(LineRange);
  }

  bool touchesPPDirective(SmallVectorImpl<AnnotatedLine *>::const_iterator I,
                          SmallVectorImpl<AnnotatedLine *>::const_iterator E) {
    for (; I != E; ++I) {
      if ((*I)->First->HasUnescapedNewline)
        return false;
      if (touchesLine(**I))
        return true;
    }
    return false;
  }

  bool touchesEmptyLineBefore(const AnnotatedLine &TheLine) {
    const FormatToken *First = TheLine.First;
    CharSourceRange LineRange = CharSourceRange::getCharRange(
        First->WhitespaceRange.getBegin(),
        First->WhitespaceRange.getBegin().getLocWithOffset(
            First->LastNewlineOffset));
    return touchesRanges(LineRange);
  }

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed. Returns the penalty.
  ///
  /// If \p DryRun is \c false, directly applies the changes.
  unsigned analyzeSolutionSpace(LineState &InitialState, bool DryRun = false) {
    std::set<LineState> Seen;

    // Increasing count of \c StateNode items we have created. This is used to
    // create a deterministic order independent of the container.
    unsigned Count = 0;
    QueueType Queue;

    // Insert start element into queue.
    StateNode *Node =
        new (Allocator.Allocate()) StateNode(InitialState, false, NULL);
    Queue.push(QueueItem(OrderedPenalty(0, Count), Node));
    ++Count;

    unsigned Penalty = 0;

    // While not empty, take first element and follow edges.
    while (!Queue.empty()) {
      Penalty = Queue.top().first.first;
      StateNode *Node = Queue.top().second;
      if (Node->State.NextToken == NULL) {
        DEBUG(llvm::dbgs() << "\n---\nPenalty for line: " << Penalty << "\n");
        break;
      }
      Queue.pop();

      // Cut off the analysis of certain solutions if the analysis gets too
      // complex. See description of IgnoreStackForComparison.
      if (Count > 10000)
        Node->State.IgnoreStackForComparison = true;

      if (!Seen.insert(Node->State).second)
        // State already examined with lower penalty.
        continue;

      FormatDecision LastFormat = Node->State.NextToken->Decision;
      if (LastFormat == FD_Unformatted || LastFormat == FD_Continue)
        addNextStateToQueue(Penalty, Node, /*NewLine=*/false, &Count, &Queue);
      if (LastFormat == FD_Unformatted || LastFormat == FD_Break)
        addNextStateToQueue(Penalty, Node, /*NewLine=*/true, &Count, &Queue);
    }

    if (Queue.empty()) {
      // We were unable to find a solution, do nothing.
      // FIXME: Add diagnostic?
      DEBUG(llvm::dbgs() << "Could not find a solution.\n");
      return 0;
    }

    // Reconstruct the solution.
    if (!DryRun)
      reconstructPath(InitialState, Queue.top().second);

    DEBUG(llvm::dbgs() << "Total number of analyzed states: " << Count << "\n");
    DEBUG(llvm::dbgs() << "---\n");

    return Penalty;
  }

  void reconstructPath(LineState &State, StateNode *Current) {
    std::deque<StateNode *> Path;
    // We do not need a break before the initial token.
    while (Current->Previous) {
      Path.push_front(Current);
      Current = Current->Previous;
    }
    for (std::deque<StateNode *>::iterator I = Path.begin(), E = Path.end();
         I != E; ++I) {
      unsigned Penalty = 0;
      formatChildren(State, (*I)->NewLine, /*DryRun=*/false, Penalty);
      Penalty += Indenter->addTokenToState(State, (*I)->NewLine, false);

      DEBUG({
        if ((*I)->NewLine) {
          llvm::dbgs() << "Penalty for placing "
                       << (*I)->Previous->State.NextToken->Tok.getName() << ": "
                       << Penalty << "\n";
        }
      });
    }
  }

  /// \brief Add the following state to the analysis queue \c Queue.
  ///
  /// Assume the current state is \p PreviousNode and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(unsigned Penalty, StateNode *PreviousNode,
                           bool NewLine, unsigned *Count, QueueType *Queue) {
    if (NewLine && !Indenter->canBreak(PreviousNode->State))
      return;
    if (!NewLine && Indenter->mustBreak(PreviousNode->State))
      return;

    StateNode *Node = new (Allocator.Allocate())
        StateNode(PreviousNode->State, NewLine, PreviousNode);
    if (!formatChildren(Node->State, NewLine, /*DryRun=*/true, Penalty))
      return;

    Penalty += Indenter->addTokenToState(Node->State, NewLine, true);

    Queue->push(QueueItem(OrderedPenalty(Penalty, *Count), Node));
    ++(*Count);
  }

  /// \brief If the \p State's next token is an r_brace closing a nested block,
  /// format the nested block before it.
  ///
  /// Returns \c true if all children could be placed successfully and adapts
  /// \p Penalty as well as \p State. If \p DryRun is false, also directly
  /// creates changes using \c Whitespaces.
  ///
  /// The crucial idea here is that children always get formatted upon
  /// encountering the closing brace right after the nested block. Now, if we
  /// are currently trying to keep the "}" on the same line (i.e. \p NewLine is
  /// \c false), the entire block has to be kept on the same line (which is only
  /// possible if it fits on the line, only contains a single statement, etc.
  ///
  /// If \p NewLine is true, we format the nested block on separate lines, i.e.
  /// break after the "{", format all lines with correct indentation and the put
  /// the closing "}" on yet another new line.
  ///
  /// This enables us to keep the simple structure of the
  /// \c UnwrappedLineFormatter, where we only have two options for each token:
  /// break or don't break.
  bool formatChildren(LineState &State, bool NewLine, bool DryRun,
                      unsigned &Penalty) {
    FormatToken &Previous = *State.NextToken->Previous;
    const FormatToken *LBrace = State.NextToken->getPreviousNonComment();
    if (!LBrace || LBrace->isNot(tok::l_brace) ||
        LBrace->BlockKind != BK_Block || Previous.Children.size() == 0)
      // The previous token does not open a block. Nothing to do. We don't
      // assert so that we can simply call this function for all tokens.
      return true;

    if (NewLine) {
      int AdditionalIndent = State.Stack.back().Indent -
                             Previous.Children[0]->Level * Style.IndentWidth;
      Penalty += format(Previous.Children, DryRun, AdditionalIndent);
      return true;
    }

    // Cannot merge multiple statements into a single line.
    if (Previous.Children.size() > 1)
      return false; 

    // We can't put the closing "}" on a line with a trailing comment.
    if (Previous.Children[0]->Last->isTrailingComment())
      return false;

    if (!DryRun) {
      Whitespaces->replaceWhitespace(
          *Previous.Children[0]->First,
          /*Newlines=*/0, /*IndentLevel=*/0, /*Spaces=*/1,
          /*StartOfTokenColumn=*/State.Column, State.Line->InPPDirective);
    }
    Penalty += format(*Previous.Children[0], State.Column + 1, DryRun);

    State.Column += 1 + Previous.Children[0]->Last->TotalLength;
    return true;
  }

  SourceManager &SourceMgr;
  SmallVectorImpl<CharSourceRange> &Ranges;
  ContinuationIndenter *Indenter;
  WhitespaceManager *Whitespaces;
  FormatStyle Style;
  LineJoiner Joiner;

  llvm::SpecificBumpPtrAllocator<StateNode> Allocator;
};

class FormatTokenLexer {
public:
  FormatTokenLexer(Lexer &Lex, SourceManager &SourceMgr, FormatStyle &Style,
                   encoding::Encoding Encoding)
      : FormatTok(NULL), IsFirstToken(true), GreaterStashed(false), Column(0),
        TrailingWhitespace(0), Lex(Lex), SourceMgr(SourceMgr), Style(Style),
        IdentTable(getFormattingLangOpts()), Encoding(Encoding) {
    Lex.SetKeepWhitespaceMode(true);
  }

  ArrayRef<FormatToken *> lex() {
    assert(Tokens.empty());
    do {
      Tokens.push_back(getNextToken());
      maybeJoinPreviousTokens();
    } while (Tokens.back()->Tok.isNot(tok::eof));
    return Tokens;
  }

  IdentifierTable &getIdentTable() { return IdentTable; }

private:
  void maybeJoinPreviousTokens() {
    if (Tokens.size() < 4)
      return;
    FormatToken *Last = Tokens.back();
    if (!Last->is(tok::r_paren))
      return;

    FormatToken *String = Tokens[Tokens.size() - 2];
    if (!String->is(tok::string_literal) || String->IsMultiline)
      return;

    if (!Tokens[Tokens.size() - 3]->is(tok::l_paren))
      return;

    FormatToken *Macro = Tokens[Tokens.size() - 4];
    if (Macro->TokenText != "_T")
      return;

    const char *Start = Macro->TokenText.data();
    const char *End = Last->TokenText.data() + Last->TokenText.size();
    String->TokenText = StringRef(Start, End - Start);
    String->IsFirst = Macro->IsFirst;
    String->LastNewlineOffset = Macro->LastNewlineOffset;
    String->WhitespaceRange = Macro->WhitespaceRange;
    String->OriginalColumn = Macro->OriginalColumn;
    String->ColumnWidth = encoding::columnWidthWithTabs(
        String->TokenText, String->OriginalColumn, Style.TabWidth, Encoding);

    Tokens.pop_back();
    Tokens.pop_back();
    Tokens.pop_back();
    Tokens.back() = String;
  }

  FormatToken *getNextToken() {
    if (GreaterStashed) {
      // Create a synthesized second '>' token.
      // FIXME: Increment Column and set OriginalColumn.
      Token Greater = FormatTok->Tok;
      FormatTok = new (Allocator.Allocate()) FormatToken;
      FormatTok->Tok = Greater;
      SourceLocation GreaterLocation =
          FormatTok->Tok.getLocation().getLocWithOffset(1);
      FormatTok->WhitespaceRange =
          SourceRange(GreaterLocation, GreaterLocation);
      FormatTok->TokenText = ">";
      FormatTok->ColumnWidth = 1;
      GreaterStashed = false;
      return FormatTok;
    }

    FormatTok = new (Allocator.Allocate()) FormatToken;
    readRawToken(*FormatTok);
    SourceLocation WhitespaceStart =
        FormatTok->Tok.getLocation().getLocWithOffset(-TrailingWhitespace);
    FormatTok->IsFirst = IsFirstToken;
    IsFirstToken = false;

    // Consume and record whitespace until we find a significant token.
    unsigned WhitespaceLength = TrailingWhitespace;
    while (FormatTok->Tok.is(tok::unknown)) {
      for (int i = 0, e = FormatTok->TokenText.size(); i != e; ++i) {
        switch (FormatTok->TokenText[i]) {
        case '\n':
          ++FormatTok->NewlinesBefore;
          // FIXME: This is technically incorrect, as it could also
          // be a literal backslash at the end of the line.
          if (i == 0 || (FormatTok->TokenText[i - 1] != '\\' &&
                         (FormatTok->TokenText[i - 1] != '\r' || i == 1 ||
                          FormatTok->TokenText[i - 2] != '\\')))
            FormatTok->HasUnescapedNewline = true;
          FormatTok->LastNewlineOffset = WhitespaceLength + i + 1;
          Column = 0;
          break;
        case '\r':
        case '\f':
        case '\v':
          Column = 0;
          break;
        case ' ':
          ++Column;
          break;
        case '\t':
          Column += Style.TabWidth - Column % Style.TabWidth;
          break;
        case '\\':
          ++Column;
          if (i + 1 == e || (FormatTok->TokenText[i + 1] != '\r' &&
                             FormatTok->TokenText[i + 1] != '\n'))
            FormatTok->Type = TT_ImplicitStringLiteral;
          break;
        default:
          FormatTok->Type = TT_ImplicitStringLiteral;
          ++Column;
          break;
        }
      }

      if (FormatTok->Type == TT_ImplicitStringLiteral)
        break;
      WhitespaceLength += FormatTok->Tok.getLength();

      readRawToken(*FormatTok);
    }

    // In case the token starts with escaped newlines, we want to
    // take them into account as whitespace - this pattern is quite frequent
    // in macro definitions.
    // FIXME: Add a more explicit test.
    while (FormatTok->TokenText.size() > 1 && FormatTok->TokenText[0] == '\\' &&
           FormatTok->TokenText[1] == '\n') {
      // FIXME: ++FormatTok->NewlinesBefore is missing...
      WhitespaceLength += 2;
      Column = 0;
      FormatTok->TokenText = FormatTok->TokenText.substr(2);
    }

    FormatTok->WhitespaceRange = SourceRange(
        WhitespaceStart, WhitespaceStart.getLocWithOffset(WhitespaceLength));

    FormatTok->OriginalColumn = Column;

    TrailingWhitespace = 0;
    if (FormatTok->Tok.is(tok::comment)) {
      // FIXME: Add the trimmed whitespace to Column.
      StringRef UntrimmedText = FormatTok->TokenText;
      FormatTok->TokenText = FormatTok->TokenText.rtrim(" \t\v\f");
      TrailingWhitespace = UntrimmedText.size() - FormatTok->TokenText.size();
    } else if (FormatTok->Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = IdentTable.get(FormatTok->TokenText);
      FormatTok->Tok.setIdentifierInfo(&Info);
      FormatTok->Tok.setKind(Info.getTokenID());
    } else if (FormatTok->Tok.is(tok::greatergreater)) {
      FormatTok->Tok.setKind(tok::greater);
      FormatTok->TokenText = FormatTok->TokenText.substr(0, 1);
      GreaterStashed = true;
    }

    // Now FormatTok is the next non-whitespace token.

    StringRef Text = FormatTok->TokenText;
    size_t FirstNewlinePos = Text.find('\n');
    if (FirstNewlinePos == StringRef::npos) {
      // FIXME: ColumnWidth actually depends on the start column, we need to
      // take this into account when the token is moved.
      FormatTok->ColumnWidth =
          encoding::columnWidthWithTabs(Text, Column, Style.TabWidth, Encoding);
      Column += FormatTok->ColumnWidth;
    } else {
      FormatTok->IsMultiline = true;
      // FIXME: ColumnWidth actually depends on the start column, we need to
      // take this into account when the token is moved.
      FormatTok->ColumnWidth = encoding::columnWidthWithTabs(
          Text.substr(0, FirstNewlinePos), Column, Style.TabWidth, Encoding);

      // The last line of the token always starts in column 0.
      // Thus, the length can be precomputed even in the presence of tabs.
      FormatTok->LastLineColumnWidth = encoding::columnWidthWithTabs(
          Text.substr(Text.find_last_of('\n') + 1), 0, Style.TabWidth,
          Encoding);
      Column = FormatTok->LastLineColumnWidth;
    }

    return FormatTok;
  }

  FormatToken *FormatTok;
  bool IsFirstToken;
  bool GreaterStashed;
  unsigned Column;
  unsigned TrailingWhitespace;
  Lexer &Lex;
  SourceManager &SourceMgr;
  FormatStyle &Style;
  IdentifierTable IdentTable;
  encoding::Encoding Encoding;
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  SmallVector<FormatToken *, 16> Tokens;

  void readRawToken(FormatToken &Tok) {
    Lex.LexFromRawLexer(Tok.Tok);
    Tok.TokenText = StringRef(SourceMgr.getCharacterData(Tok.Tok.getLocation()),
                              Tok.Tok.getLength());
    // For formatting, treat unterminated string literals like normal string
    // literals.
    if (Tok.is(tok::unknown) && !Tok.TokenText.empty() &&
        Tok.TokenText[0] == '"') {
      Tok.Tok.setKind(tok::string_literal);
      Tok.IsUnterminatedLiteral = true;
    }
  }
};

class Formatter : public UnwrappedLineConsumer {
public:
  Formatter(const FormatStyle &Style, Lexer &Lex, SourceManager &SourceMgr,
            const std::vector<CharSourceRange> &Ranges)
      : Style(Style), Lex(Lex), SourceMgr(SourceMgr),
        Whitespaces(SourceMgr, Style, inputUsesCRLF(Lex.getBuffer())),
        Ranges(Ranges.begin(), Ranges.end()), UnwrappedLines(1),
        Encoding(encoding::detectEncoding(Lex.getBuffer())) {
    DEBUG(llvm::dbgs() << "File encoding: "
                       << (Encoding == encoding::Encoding_UTF8 ? "UTF8"
                                                               : "unknown")
                       << "\n");
  }

  tooling::Replacements format() {
    tooling::Replacements Result;
    FormatTokenLexer Tokens(Lex, SourceMgr, Style, Encoding);

    UnwrappedLineParser Parser(Style, Tokens.lex(), *this);
    bool StructuralError = Parser.parse();
    assert(UnwrappedLines.rbegin()->empty());
    for (unsigned Run = 0, RunE = UnwrappedLines.size(); Run + 1 != RunE;
         ++Run) {
      DEBUG(llvm::dbgs() << "Run " << Run << "...\n");
      SmallVector<AnnotatedLine *, 16> AnnotatedLines;
      for (unsigned i = 0, e = UnwrappedLines[Run].size(); i != e; ++i) {
        AnnotatedLines.push_back(new AnnotatedLine(UnwrappedLines[Run][i]));
      }
      tooling::Replacements RunResult =
          format(AnnotatedLines, StructuralError, Tokens);
      DEBUG({
        llvm::dbgs() << "Replacements for run " << Run << ":\n";
        for (tooling::Replacements::iterator I = RunResult.begin(),
                                             E = RunResult.end();
             I != E; ++I) {
          llvm::dbgs() << I->toString() << "\n";
        }
      });
      for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
        delete AnnotatedLines[i];
      }
      Result.insert(RunResult.begin(), RunResult.end());
      Whitespaces.reset();
    }
    return Result;
  }

  tooling::Replacements format(SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
                               bool StructuralError, FormatTokenLexer &Tokens) {
    TokenAnnotator Annotator(Style, Tokens.getIdentTable().get("in"));
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.annotate(*AnnotatedLines[i]);
    }
    deriveLocalStyle(AnnotatedLines);
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.calculateFormattingInformation(*AnnotatedLines[i]);
    }

    Annotator.setCommentLineLevels(AnnotatedLines);
    ContinuationIndenter Indenter(Style, SourceMgr, Whitespaces, Encoding,
                                  BinPackInconclusiveFunctions);
    UnwrappedLineFormatter Formatter(SourceMgr, Ranges, &Indenter, &Whitespaces,
                                     Style);
    Formatter.format(AnnotatedLines, /*DryRun=*/false);
    return Whitespaces.generateReplacements();
  }

private:
  static bool inputUsesCRLF(StringRef Text) {
    return Text.count('\r') * 2 > Text.count('\n');
  }

  void
  deriveLocalStyle(const SmallVectorImpl<AnnotatedLine *> &AnnotatedLines) {
    unsigned CountBoundToVariable = 0;
    unsigned CountBoundToType = 0;
    bool HasCpp03IncompatibleFormat = false;
    bool HasBinPackedFunction = false;
    bool HasOnePerLineFunction = false;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      if (!AnnotatedLines[i]->First->Next)
        continue;
      FormatToken *Tok = AnnotatedLines[i]->First->Next;
      while (Tok->Next) {
        if (Tok->Type == TT_PointerOrReference) {
          bool SpacesBefore =
              Tok->WhitespaceRange.getBegin() != Tok->WhitespaceRange.getEnd();
          bool SpacesAfter = Tok->Next->WhitespaceRange.getBegin() !=
                             Tok->Next->WhitespaceRange.getEnd();
          if (SpacesBefore && !SpacesAfter)
            ++CountBoundToVariable;
          else if (!SpacesBefore && SpacesAfter)
            ++CountBoundToType;
        }

        if (Tok->WhitespaceRange.getBegin() == Tok->WhitespaceRange.getEnd()) {
          if (Tok->is(tok::coloncolon) &&
              Tok->Previous->Type == TT_TemplateOpener)
            HasCpp03IncompatibleFormat = true;
          if (Tok->Type == TT_TemplateCloser &&
              Tok->Previous->Type == TT_TemplateCloser)
            HasCpp03IncompatibleFormat = true;
        }

        if (Tok->PackingKind == PPK_BinPacked)
          HasBinPackedFunction = true;
        if (Tok->PackingKind == PPK_OnePerLine)
          HasOnePerLineFunction = true;

        Tok = Tok->Next;
      }
    }
    if (Style.DerivePointerBinding) {
      if (CountBoundToType > CountBoundToVariable)
        Style.PointerBindsToType = true;
      else if (CountBoundToType < CountBoundToVariable)
        Style.PointerBindsToType = false;
    }
    if (Style.Standard == FormatStyle::LS_Auto) {
      Style.Standard = HasCpp03IncompatibleFormat ? FormatStyle::LS_Cpp11
                                                  : FormatStyle::LS_Cpp03;
    }
    BinPackInconclusiveFunctions =
        HasBinPackedFunction || !HasOnePerLineFunction;
  }

  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    assert(!UnwrappedLines.empty());
    UnwrappedLines.back().push_back(TheLine);
  }

  virtual void finishRun() {
    UnwrappedLines.push_back(SmallVector<UnwrappedLine, 16>());
  }

  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  WhitespaceManager Whitespaces;
  SmallVector<CharSourceRange, 8> Ranges;
  SmallVector<SmallVector<UnwrappedLine, 16>, 2> UnwrappedLines;

  encoding::Encoding Encoding;
  bool BinPackInconclusiveFunctions;
};

} // end anonymous namespace

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges) {
  Formatter formatter(Style, Lex, SourceMgr, Ranges);
  return formatter.format();
}

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               std::vector<tooling::Range> Ranges,
                               StringRef FileName) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager SourceMgr(Diagnostics, Files);
  llvm::MemoryBuffer *Buf = llvm::MemoryBuffer::getMemBuffer(Code, FileName);
  const clang::FileEntry *Entry =
      Files.getVirtualFile(FileName, Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(Entry, Buf);
  FileID ID =
      SourceMgr.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);
  Lexer Lex(ID, SourceMgr.getBuffer(ID), SourceMgr,
            getFormattingLangOpts(Style.Standard));
  SourceLocation StartOfFile = SourceMgr.getLocForStartOfFile(ID);
  std::vector<CharSourceRange> CharRanges;
  for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
    SourceLocation Start = StartOfFile.getLocWithOffset(Ranges[i].getOffset());
    SourceLocation End = Start.getLocWithOffset(Ranges[i].getLength());
    CharRanges.push_back(CharSourceRange::getCharRange(Start, End));
  }
  return reformat(Style, Lex, SourceMgr, CharRanges);
}

LangOptions getFormattingLangOpts(FormatStyle::LanguageStandard Standard) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.LineComment = 1;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  return LangOpts;
}

const char *StyleOptionHelpDescription =
    "Coding style, currently supports:\n"
    "  LLVM, Google, Chromium, Mozilla, WebKit.\n"
    "Use -style=file to load style configuration from\n"
    ".clang-format file located in one of the parent\n"
    "directories of the source file (or current\n"
    "directory for stdin).\n"
    "Use -style=\"{key: value, ...}\" to set specific\n"
    "parameters, e.g.:\n"
    "  -style=\"{BasedOnStyle: llvm, IndentWidth: 8}\"";

FormatStyle getStyle(StringRef StyleName, StringRef FileName) {
  // Fallback style in case the rest of this function can't determine a style.
  StringRef FallbackStyle = "LLVM";
  FormatStyle Style;
  getPredefinedStyle(FallbackStyle, &Style);

  if (StyleName.startswith("{")) {
    // Parse YAML/JSON style from the command line.
    if (llvm::error_code ec = parseConfiguration(StyleName, &Style)) {
      llvm::errs() << "Error parsing -style: " << ec.message() << ", using "
                   << FallbackStyle << " style\n";
    }
    return Style;
  }

  if (!StyleName.equals_lower("file")) {
    if (!getPredefinedStyle(StyleName, &Style))
      llvm::errs() << "Invalid value for -style, using " << FallbackStyle
                   << " style\n";
    return Style;
  }

  SmallString<128> Path(FileName);
  llvm::sys::fs::make_absolute(Path);
  for (StringRef Directory = Path; !Directory.empty();
       Directory = llvm::sys::path::parent_path(Directory)) {
    if (!llvm::sys::fs::is_directory(Directory))
      continue;
    SmallString<128> ConfigFile(Directory);

    llvm::sys::path::append(ConfigFile, ".clang-format");
    DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
    bool IsFile = false;
    // Ignore errors from is_regular_file: we only need to know if we can read
    // the file or not.
    llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);

    if (!IsFile) {
      // Try _clang-format too, since dotfiles are not commonly used on Windows.
      ConfigFile = Directory;
      llvm::sys::path::append(ConfigFile, "_clang-format");
      DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
      llvm::sys::fs::is_regular_file(Twine(ConfigFile), IsFile);
    }

    if (IsFile) {
      OwningPtr<llvm::MemoryBuffer> Text;
      if (llvm::error_code ec =
              llvm::MemoryBuffer::getFile(ConfigFile.c_str(), Text)) {
        llvm::errs() << ec.message() << "\n";
        continue;
      }
      if (llvm::error_code ec = parseConfiguration(Text->getBuffer(), &Style)) {
        llvm::errs() << "Error reading " << ConfigFile << ": " << ec.message()
                     << "\n";
        continue;
      }
      DEBUG(llvm::dbgs() << "Using configuration file " << ConfigFile << "\n");
      return Style;
    }
  }
  llvm::errs() << "Can't find usable .clang-format, using " << FallbackStyle
               << " style\n";
  return Style;
}

} // namespace format
} // namespace clang
