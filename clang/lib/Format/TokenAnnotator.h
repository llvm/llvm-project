//===--- TokenAnnotator.h - Format C++ code ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a token annotator, i.e. creates
/// \c AnnotatedTokens out of \c FormatTokens with required extra information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_TOKENANNOTATOR_H
#define LLVM_CLANG_LIB_FORMAT_TOKENANNOTATOR_H

#include "UnwrappedLineParser.h"
#include "clang/Format/Format.h"

namespace clang {
namespace format {

enum LineType {
  LT_Invalid,
  LT_ImportStatement,
  LT_ObjCDecl, // An @interface, @implementation, or @protocol line.
  LT_ObjCMethodDecl,
  LT_ObjCProperty, // An @property line.
  LT_Other,
  LT_PreprocessorDirective,
  LT_VirtualFunctionDecl,
  LT_ArrayOfStructInitializer,
  LT_CommentAbovePPDirective,
};

class AnnotatedLine {
public:
  AnnotatedLine(const UnwrappedLine &Line)
      : First(Line.Tokens.front().Tok), Level(Line.Level),
        MatchingOpeningBlockLineIndex(Line.MatchingOpeningBlockLineIndex),
        MatchingClosingBlockLineIndex(Line.MatchingClosingBlockLineIndex),
        InPPDirective(Line.InPPDirective), InMacroBody(Line.InMacroBody),
        MustBeDeclaration(Line.MustBeDeclaration), MightBeFunctionDecl(false),
        IsMultiVariableDeclStmt(false), Affected(false),
        LeadingEmptyLinesAffected(false), ChildrenAffected(false),
        IsContinuation(Line.IsContinuation),
        FirstStartColumn(Line.FirstStartColumn) {
    assert(!Line.Tokens.empty());

    // Calculate Next and Previous for all tokens. Note that we must overwrite
    // Next and Previous for every token, as previous formatting runs might have
    // left them in a different state.
    First->Previous = nullptr;
    FormatToken *Current = First;
    for (const UnwrappedLineNode &Node : llvm::drop_begin(Line.Tokens)) {
      Current->Next = Node.Tok;
      Node.Tok->Previous = Current;
      Current = Current->Next;
      Current->Children.clear();
      for (const auto &Child : Node.Children) {
        Children.push_back(new AnnotatedLine(Child));
        Current->Children.push_back(Children.back());
      }
    }
    Last = Current;
    Last->Next = nullptr;
  }

  ~AnnotatedLine() {
    for (AnnotatedLine *Child : Children)
      delete Child;
    FormatToken *Current = First;
    while (Current) {
      Current->Children.clear();
      Current->Role.reset();
      Current = Current->Next;
    }
  }

  bool isComment() const {
    return First && First->is(tok::comment) && !First->getNextNonComment();
  }

  /// \c true if this line starts with the given tokens in order, ignoring
  /// comments.
  template <typename... Ts> bool startsWith(Ts... Tokens) const {
    return First && First->startsSequence(Tokens...);
  }

  /// \c true if this line ends with the given tokens in reversed order,
  /// ignoring comments.
  /// For example, given tokens [T1, T2, T3, ...], the function returns true if
  /// this line is like "... T3 T2 T1".
  template <typename... Ts> bool endsWith(Ts... Tokens) const {
    return Last && Last->endsSequence(Tokens...);
  }

  /// \c true if this line looks like a function definition instead of a
  /// function declaration. Asserts MightBeFunctionDecl.
  bool mightBeFunctionDefinition() const {
    assert(MightBeFunctionDecl);
    // Try to determine if the end of a stream of tokens is either the
    // Definition or the Declaration for a function. It does this by looking for
    // the ';' in foo(); and using that it ends with a ; to know this is the
    // Definition, however the line could end with
    //    foo(); /* comment */
    // or
    //    foo(); // comment
    // or
    //    foo() // comment
    // endsWith() ignores the comment.
    return !endsWith(tok::semi);
  }

  /// \c true if this line starts a namespace definition.
  bool startsWithNamespace() const {
    return startsWith(tok::kw_namespace) || startsWith(TT_NamespaceMacro) ||
           startsWith(tok::kw_inline, tok::kw_namespace) ||
           startsWith(tok::kw_export, tok::kw_namespace);
  }

  FormatToken *First;
  FormatToken *Last;

  SmallVector<AnnotatedLine *, 0> Children;

  LineType Type;
  unsigned Level;
  size_t MatchingOpeningBlockLineIndex;
  size_t MatchingClosingBlockLineIndex;
  bool InPPDirective;
  bool InMacroBody;
  bool MustBeDeclaration;
  bool MightBeFunctionDecl;
  bool IsMultiVariableDeclStmt;

  /// \c True if this line should be formatted, i.e. intersects directly or
  /// indirectly with one of the input ranges.
  bool Affected;

  /// \c True if the leading empty lines of this line intersect with one of the
  /// input ranges.
  bool LeadingEmptyLinesAffected;

  /// \c True if one of this line's children intersects with an input range.
  bool ChildrenAffected;

  /// \c True if this line should be indented by ContinuationIndent in addition
  /// to the normal indention level.
  bool IsContinuation;

  unsigned FirstStartColumn;

private:
  // Disallow copying.
  AnnotatedLine(const AnnotatedLine &) = delete;
  void operator=(const AnnotatedLine &) = delete;
};

/// Determines extra information about the tokens comprising an
/// \c UnwrappedLine.
class TokenAnnotator {
public:
  TokenAnnotator(const FormatStyle &Style, const AdditionalKeywords &Keywords)
      : Style(Style), Keywords(Keywords) {}

  /// Adapts the indent levels of comment lines to the indent of the
  /// subsequent line.
  // FIXME: Can/should this be done in the UnwrappedLineParser?
  void setCommentLineLevels(SmallVectorImpl<AnnotatedLine *> &Lines) const;

  void annotate(AnnotatedLine &Line) const;
  void calculateFormattingInformation(AnnotatedLine &Line) const;

private:
  /// Calculate the penalty for splitting before \c Tok.
  unsigned splitPenalty(const AnnotatedLine &Line, const FormatToken &Tok,
                        bool InFunctionDecl) const;

  bool spaceRequiredBeforeParens(const FormatToken &Right) const;

  bool spaceRequiredBetween(const AnnotatedLine &Line, const FormatToken &Left,
                            const FormatToken &Right) const;

  bool spaceRequiredBefore(const AnnotatedLine &Line,
                           const FormatToken &Right) const;

  bool mustBreakBefore(const AnnotatedLine &Line,
                       const FormatToken &Right) const;

  bool canBreakBefore(const AnnotatedLine &Line,
                      const FormatToken &Right) const;

  bool mustBreakForReturnType(const AnnotatedLine &Line) const;

  void printDebugInfo(const AnnotatedLine &Line) const;

  void calculateUnbreakableTailLengths(AnnotatedLine &Line) const;

  void calculateArrayInitializerColumnList(AnnotatedLine &Line) const;

  FormatToken *calculateInitializerColumnList(AnnotatedLine &Line,
                                              FormatToken *CurrentToken,
                                              unsigned Depth) const;
  FormatStyle::PointerAlignmentStyle
  getTokenReferenceAlignment(const FormatToken &PointerOrReference) const;

  FormatStyle::PointerAlignmentStyle getTokenPointerOrReferenceAlignment(
      const FormatToken &PointerOrReference) const;

  const FormatStyle &Style;

  const AdditionalKeywords &Keywords;
};

} // end namespace format
} // end namespace clang

#endif
