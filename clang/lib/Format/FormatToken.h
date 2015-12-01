//===--- FormatToken.h - Format C++ code ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the FormatToken, a wrapper
/// around Token with additional information related to formatting.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_FORMATTOKEN_H
#define LLVM_CLANG_LIB_FORMAT_FORMATTOKEN_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include <memory>

namespace clang {
namespace format {

#define LIST_TOKEN_TYPES \
  TYPE(ArrayInitializerLSquare) \
  TYPE(ArraySubscriptLSquare) \
  TYPE(AttributeParen) \
  TYPE(BinaryOperator) \
  TYPE(BitFieldColon) \
  TYPE(BlockComment) \
  TYPE(CastRParen) \
  TYPE(ConditionalExpr) \
  TYPE(ConflictAlternative) \
  TYPE(ConflictEnd) \
  TYPE(ConflictStart) \
  TYPE(CtorInitializerColon) \
  TYPE(CtorInitializerComma) \
  TYPE(DesignatedInitializerPeriod) \
  TYPE(DictLiteral) \
  TYPE(ForEachMacro) \
  TYPE(FunctionAnnotationRParen) \
  TYPE(FunctionDeclarationName) \
  TYPE(FunctionLBrace) \
  TYPE(FunctionTypeLParen) \
  TYPE(ImplicitStringLiteral) \
  TYPE(InheritanceColon) \
  TYPE(InlineASMBrace) \
  TYPE(InlineASMColon) \
  TYPE(JavaAnnotation) \
  TYPE(JsComputedPropertyName) \
  TYPE(JsFatArrow) \
  TYPE(JsTypeColon) \
  TYPE(JsTypeOptionalQuestion) \
  TYPE(LambdaArrow) \
  TYPE(LambdaLSquare) \
  TYPE(LeadingJavaAnnotation) \
  TYPE(LineComment) \
  TYPE(MacroBlockBegin) \
  TYPE(MacroBlockEnd) \
  TYPE(ObjCBlockLBrace) \
  TYPE(ObjCBlockLParen) \
  TYPE(ObjCDecl) \
  TYPE(ObjCForIn) \
  TYPE(ObjCMethodExpr) \
  TYPE(ObjCMethodSpecifier) \
  TYPE(ObjCProperty) \
  TYPE(ObjCStringLiteral) \
  TYPE(OverloadedOperator) \
  TYPE(OverloadedOperatorLParen) \
  TYPE(PointerOrReference) \
  TYPE(PureVirtualSpecifier) \
  TYPE(RangeBasedForLoopColon) \
  TYPE(RegexLiteral) \
  TYPE(SelectorName) \
  TYPE(StartOfName) \
  TYPE(TemplateCloser) \
  TYPE(TemplateOpener) \
  TYPE(TemplateString) \
  TYPE(TrailingAnnotation) \
  TYPE(TrailingReturnArrow) \
  TYPE(TrailingUnaryOperator) \
  TYPE(UnaryOperator) \
  TYPE(Unknown)

enum TokenType {
#define TYPE(X) TT_##X,
LIST_TOKEN_TYPES
#undef TYPE
  NUM_TOKEN_TYPES
};

/// \brief Determines the name of a token type.
const char *getTokenTypeName(TokenType Type);

// Represents what type of block a set of braces open.
enum BraceBlockKind { BK_Unknown, BK_Block, BK_BracedInit };

// The packing kind of a function's parameters.
enum ParameterPackingKind { PPK_BinPacked, PPK_OnePerLine, PPK_Inconclusive };

enum FormatDecision { FD_Unformatted, FD_Continue, FD_Break };

class TokenRole;
class AnnotatedLine;

/// \brief A wrapper around a \c Token storing information about the
/// whitespace characters preceding it.
struct FormatToken {
  FormatToken() {}

  /// \brief The \c Token.
  Token Tok;

  /// \brief The number of newlines immediately before the \c Token.
  ///
  /// This can be used to determine what the user wrote in the original code
  /// and thereby e.g. leave an empty line between two function definitions.
  unsigned NewlinesBefore = 0;

  /// \brief Whether there is at least one unescaped newline before the \c
  /// Token.
  bool HasUnescapedNewline = false;

  /// \brief The range of the whitespace immediately preceding the \c Token.
  SourceRange WhitespaceRange;

  /// \brief The offset just past the last '\n' in this token's leading
  /// whitespace (relative to \c WhiteSpaceStart). 0 if there is no '\n'.
  unsigned LastNewlineOffset = 0;

  /// \brief The width of the non-whitespace parts of the token (or its first
  /// line for multi-line tokens) in columns.
  /// We need this to correctly measure number of columns a token spans.
  unsigned ColumnWidth = 0;

  /// \brief Contains the width in columns of the last line of a multi-line
  /// token.
  unsigned LastLineColumnWidth = 0;

  /// \brief Whether the token text contains newlines (escaped or not).
  bool IsMultiline = false;

  /// \brief Indicates that this is the first token.
  bool IsFirst = false;

  /// \brief Whether there must be a line break before this token.
  ///
  /// This happens for example when a preprocessor directive ended directly
  /// before the token.
  bool MustBreakBefore = false;

  /// \brief The raw text of the token.
  ///
  /// Contains the raw token text without leading whitespace and without leading
  /// escaped newlines.
  StringRef TokenText;

  /// \brief Set to \c true if this token is an unterminated literal.
  bool IsUnterminatedLiteral = 0;

  /// \brief Contains the kind of block if this token is a brace.
  BraceBlockKind BlockKind = BK_Unknown;

  TokenType Type = TT_Unknown;

  /// \brief The number of spaces that should be inserted before this token.
  unsigned SpacesRequiredBefore = 0;

  /// \brief \c true if it is allowed to break before this token.
  bool CanBreakBefore = false;

  /// \brief \c true if this is the ">" of "template<..>".
  bool ClosesTemplateDeclaration = false;

  /// \brief Number of parameters, if this is "(", "[" or "<".
  ///
  /// This is initialized to 1 as we don't need to distinguish functions with
  /// 0 parameters from functions with 1 parameter. Thus, we can simply count
  /// the number of commas.
  unsigned ParameterCount = 0;

  /// \brief Number of parameters that are nested blocks,
  /// if this is "(", "[" or "<".
  unsigned BlockParameterCount = 0;

  /// \brief If this is a bracket ("<", "(", "[" or "{"), contains the kind of
  /// the surrounding bracket.
  tok::TokenKind ParentBracket = tok::unknown;

  /// \brief A token can have a special role that can carry extra information
  /// about the token's formatting.
  std::unique_ptr<TokenRole> Role;

  /// \brief If this is an opening parenthesis, how are the parameters packed?
  ParameterPackingKind PackingKind = PPK_Inconclusive;

  /// \brief The total length of the unwrapped line up to and including this
  /// token.
  unsigned TotalLength = 0;

  /// \brief The original 0-based column of this token, including expanded tabs.
  /// The configured TabWidth is used as tab width.
  unsigned OriginalColumn = 0;

  /// \brief The length of following tokens until the next natural split point,
  /// or the next token that can be broken.
  unsigned UnbreakableTailLength = 0;

  // FIXME: Come up with a 'cleaner' concept.
  /// \brief The binding strength of a token. This is a combined value of
  /// operator precedence, parenthesis nesting, etc.
  unsigned BindingStrength = 0;

  /// \brief The nesting level of this token, i.e. the number of surrounding (),
  /// [], {} or <>.
  unsigned NestingLevel = 0;

  /// \brief Penalty for inserting a line break before this token.
  unsigned SplitPenalty = 0;

  /// \brief If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the length of the longest name.
  ///
  /// This being set to 0 means that the selectors should not be colon-aligned,
  /// e.g. because several of them are block-type.
  unsigned LongestObjCSelectorName = 0;

  /// \brief Stores the number of required fake parentheses and the
  /// corresponding operator precedence.
  ///
  /// If multiple fake parentheses start at a token, this vector stores them in
  /// reverse order, i.e. inner fake parenthesis first.
  SmallVector<prec::Level, 4> FakeLParens;
  /// \brief Insert this many fake ) after this token for correct indentation.
  unsigned FakeRParens = 0;

  /// \brief \c true if this token starts a binary expression, i.e. has at least
  /// one fake l_paren with a precedence greater than prec::Unknown.
  bool StartsBinaryExpression = false;
  /// \brief \c true if this token ends a binary expression.
  bool EndsBinaryExpression = false;

  /// \brief Is this is an operator (or "."/"->") in a sequence of operators
  /// with the same precedence, contains the 0-based operator index.
  unsigned OperatorIndex = 0;

  /// \brief Is this the last operator (or "."/"->") in a sequence of operators
  /// with the same precedence?
  bool LastOperator = false;

  /// \brief Is this token part of a \c DeclStmt defining multiple variables?
  ///
  /// Only set if \c Type == \c TT_StartOfName.
  bool PartOfMultiVariableDeclStmt = false;

  /// \brief If this is a bracket, this points to the matching one.
  FormatToken *MatchingParen = nullptr;

  /// \brief The previous token in the unwrapped line.
  FormatToken *Previous = nullptr;

  /// \brief The next token in the unwrapped line.
  FormatToken *Next = nullptr;

  /// \brief If this token starts a block, this contains all the unwrapped lines
  /// in it.
  SmallVector<AnnotatedLine *, 1> Children;

  /// \brief Stores the formatting decision for the token once it was made.
  FormatDecision Decision = FD_Unformatted;

  /// \brief If \c true, this token has been fully formatted (indented and
  /// potentially re-formatted inside), and we do not allow further formatting
  /// changes.
  bool Finalized = false;

  bool is(tok::TokenKind Kind) const { return Tok.is(Kind); }
  bool is(TokenType TT) const { return Type == TT; }
  bool is(const IdentifierInfo *II) const {
    return II && II == Tok.getIdentifierInfo();
  }
  bool is(tok::PPKeywordKind Kind) const {
    return Tok.getIdentifierInfo() &&
           Tok.getIdentifierInfo()->getPPKeywordID() == Kind;
  }
  template <typename A, typename B> bool isOneOf(A K1, B K2) const {
    return is(K1) || is(K2);
  }
  template <typename A, typename B, typename... Ts>
  bool isOneOf(A K1, B K2, Ts... Ks) const {
    return is(K1) || isOneOf(K2, Ks...);
  }
  template <typename T> bool isNot(T Kind) const { return !is(Kind); }

  bool isStringLiteral() const { return tok::isStringLiteral(Tok.getKind()); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return Tok.isObjCAtKeyword(Kind);
  }

  bool isAccessSpecifier(bool ColonRequired = true) const {
    return isOneOf(tok::kw_public, tok::kw_protected, tok::kw_private) &&
           (!ColonRequired || (Next && Next->is(tok::colon)));
  }

  /// \brief Determine whether the token is a simple-type-specifier.
  bool isSimpleTypeSpecifier() const;

  bool isObjCAccessSpecifier() const {
    return is(tok::at) && Next && (Next->isObjCAtKeyword(tok::objc_public) ||
                                   Next->isObjCAtKeyword(tok::objc_protected) ||
                                   Next->isObjCAtKeyword(tok::objc_package) ||
                                   Next->isObjCAtKeyword(tok::objc_private));
  }

  /// \brief Returns whether \p Tok is ([{ or a template opening <.
  bool opensScope() const {
    return isOneOf(tok::l_paren, tok::l_brace, tok::l_square,
                   TT_TemplateOpener);
  }
  /// \brief Returns whether \p Tok is )]} or a template closing >.
  bool closesScope() const {
    return isOneOf(tok::r_paren, tok::r_brace, tok::r_square,
                   TT_TemplateCloser);
  }

  /// \brief Returns \c true if this is a "." or "->" accessing a member.
  bool isMemberAccess() const {
    return isOneOf(tok::arrow, tok::period, tok::arrowstar) &&
           !isOneOf(TT_DesignatedInitializerPeriod, TT_TrailingReturnArrow,
                    TT_LambdaArrow);
  }

  bool isUnaryOperator() const {
    switch (Tok.getKind()) {
    case tok::plus:
    case tok::plusplus:
    case tok::minus:
    case tok::minusminus:
    case tok::exclaim:
    case tok::tilde:
    case tok::kw_sizeof:
    case tok::kw_alignof:
      return true;
    default:
      return false;
    }
  }

  bool isBinaryOperator() const {
    // Comma is a binary operator, but does not behave as such wrt. formatting.
    return getPrecedence() > prec::Comma;
  }

  bool isTrailingComment() const {
    return is(tok::comment) &&
           (is(TT_LineComment) || !Next || Next->NewlinesBefore > 0);
  }

  /// \brief Returns \c true if this is a keyword that can be used
  /// like a function call (e.g. sizeof, typeid, ...).
  bool isFunctionLikeKeyword() const {
    switch (Tok.getKind()) {
    case tok::kw_throw:
    case tok::kw_typeid:
    case tok::kw_return:
    case tok::kw_sizeof:
    case tok::kw_alignof:
    case tok::kw_alignas:
    case tok::kw_decltype:
    case tok::kw_noexcept:
    case tok::kw_static_assert:
    case tok::kw___attribute:
      return true;
    default:
      return false;
    }
  }

  /// \brief Returns actual token start location without leading escaped
  /// newlines and whitespace.
  ///
  /// This can be different to Tok.getLocation(), which includes leading escaped
  /// newlines.
  SourceLocation getStartOfNonWhitespace() const {
    return WhitespaceRange.getEnd();
  }

  prec::Level getPrecedence() const {
    return getBinOpPrecedence(Tok.getKind(), true, true);
  }

  /// \brief Returns the previous token ignoring comments.
  FormatToken *getPreviousNonComment() const {
    FormatToken *Tok = Previous;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Previous;
    return Tok;
  }

  /// \brief Returns the next token ignoring comments.
  const FormatToken *getNextNonComment() const {
    const FormatToken *Tok = Next;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Next;
    return Tok;
  }

  /// \brief Returns \c true if this tokens starts a block-type list, i.e. a
  /// list that should be indented with a block indent.
  bool opensBlockOrBlockTypeList(const FormatStyle &Style) const {
    return is(TT_ArrayInitializerLSquare) ||
           (is(tok::l_brace) &&
            (BlockKind == BK_Block || is(TT_DictLiteral) ||
             (!Style.Cpp11BracedListStyle && NestingLevel == 0)));
  }

  /// \brief Same as opensBlockOrBlockTypeList, but for the closing token.
  bool closesBlockOrBlockTypeList(const FormatStyle &Style) const {
    return MatchingParen && MatchingParen->opensBlockOrBlockTypeList(Style);
  }

private:
  // Disallow copying.
  FormatToken(const FormatToken &) = delete;
  void operator=(const FormatToken &) = delete;
};

class ContinuationIndenter;
struct LineState;

class TokenRole {
public:
  TokenRole(const FormatStyle &Style) : Style(Style) {}
  virtual ~TokenRole();

  /// \brief After the \c TokenAnnotator has finished annotating all the tokens,
  /// this function precomputes required information for formatting.
  virtual void precomputeFormattingInfos(const FormatToken *Token);

  /// \brief Apply the special formatting that the given role demands.
  ///
  /// Assumes that the token having this role is already formatted.
  ///
  /// Continues formatting from \p State leaving indentation to \p Indenter and
  /// returns the total penalty that this formatting incurs.
  virtual unsigned formatFromToken(LineState &State,
                                   ContinuationIndenter *Indenter,
                                   bool DryRun) {
    return 0;
  }

  /// \brief Same as \c formatFromToken, but assumes that the first token has
  /// already been set thereby deciding on the first line break.
  virtual unsigned formatAfterToken(LineState &State,
                                    ContinuationIndenter *Indenter,
                                    bool DryRun) {
    return 0;
  }

  /// \brief Notifies the \c Role that a comma was found.
  virtual void CommaFound(const FormatToken *Token) {}

protected:
  const FormatStyle &Style;
};

class CommaSeparatedList : public TokenRole {
public:
  CommaSeparatedList(const FormatStyle &Style)
      : TokenRole(Style), HasNestedBracedList(false) {}

  void precomputeFormattingInfos(const FormatToken *Token) override;

  unsigned formatAfterToken(LineState &State, ContinuationIndenter *Indenter,
                            bool DryRun) override;

  unsigned formatFromToken(LineState &State, ContinuationIndenter *Indenter,
                           bool DryRun) override;

  /// \brief Adds \p Token as the next comma to the \c CommaSeparated list.
  void CommaFound(const FormatToken *Token) override {
    Commas.push_back(Token);
  }

private:
  /// \brief A struct that holds information on how to format a given list with
  /// a specific number of columns.
  struct ColumnFormat {
    /// \brief The number of columns to use.
    unsigned Columns;

    /// \brief The total width in characters.
    unsigned TotalWidth;

    /// \brief The number of lines required for this format.
    unsigned LineCount;

    /// \brief The size of each column in characters.
    SmallVector<unsigned, 8> ColumnSizes;
  };

  /// \brief Calculate which \c ColumnFormat fits best into
  /// \p RemainingCharacters.
  const ColumnFormat *getColumnFormat(unsigned RemainingCharacters) const;

  /// \brief The ordered \c FormatTokens making up the commas of this list.
  SmallVector<const FormatToken *, 8> Commas;

  /// \brief The length of each of the list's items in characters including the
  /// trailing comma.
  SmallVector<unsigned, 8> ItemLengths;

  /// \brief Precomputed formats that can be used for this list.
  SmallVector<ColumnFormat, 4> Formats;

  bool HasNestedBracedList;
};

/// \brief Encapsulates keywords that are context sensitive or for languages not
/// properly supported by Clang's lexer.
struct AdditionalKeywords {
  AdditionalKeywords(IdentifierTable &IdentTable) {
    kw_final = &IdentTable.get("final");
    kw_override = &IdentTable.get("override");
    kw_in = &IdentTable.get("in");
    kw_CF_ENUM = &IdentTable.get("CF_ENUM");
    kw_CF_OPTIONS = &IdentTable.get("CF_OPTIONS");
    kw_NS_ENUM = &IdentTable.get("NS_ENUM");
    kw_NS_OPTIONS = &IdentTable.get("NS_OPTIONS");

    kw_finally = &IdentTable.get("finally");
    kw_function = &IdentTable.get("function");
    kw_import = &IdentTable.get("import");
    kw_let = &IdentTable.get("let");
    kw_var = &IdentTable.get("var");

    kw_abstract = &IdentTable.get("abstract");
    kw_assert = &IdentTable.get("assert");
    kw_extends = &IdentTable.get("extends");
    kw_implements = &IdentTable.get("implements");
    kw_instanceof = &IdentTable.get("instanceof");
    kw_interface = &IdentTable.get("interface");
    kw_native = &IdentTable.get("native");
    kw_package = &IdentTable.get("package");
    kw_synchronized = &IdentTable.get("synchronized");
    kw_throws = &IdentTable.get("throws");
    kw___except = &IdentTable.get("__except");

    kw_mark = &IdentTable.get("mark");

    kw_extend = &IdentTable.get("extend");
    kw_option = &IdentTable.get("option");
    kw_optional = &IdentTable.get("optional");
    kw_repeated = &IdentTable.get("repeated");
    kw_required = &IdentTable.get("required");
    kw_returns = &IdentTable.get("returns");

    kw_signals = &IdentTable.get("signals");
    kw_qsignals = &IdentTable.get("Q_SIGNALS");
    kw_slots = &IdentTable.get("slots");
    kw_qslots = &IdentTable.get("Q_SLOTS");
  }

  // Context sensitive keywords.
  IdentifierInfo *kw_final;
  IdentifierInfo *kw_override;
  IdentifierInfo *kw_in;
  IdentifierInfo *kw_CF_ENUM;
  IdentifierInfo *kw_CF_OPTIONS;
  IdentifierInfo *kw_NS_ENUM;
  IdentifierInfo *kw_NS_OPTIONS;
  IdentifierInfo *kw___except;

  // JavaScript keywords.
  IdentifierInfo *kw_finally;
  IdentifierInfo *kw_function;
  IdentifierInfo *kw_import;
  IdentifierInfo *kw_let;
  IdentifierInfo *kw_var;

  // Java keywords.
  IdentifierInfo *kw_abstract;
  IdentifierInfo *kw_assert;
  IdentifierInfo *kw_extends;
  IdentifierInfo *kw_implements;
  IdentifierInfo *kw_instanceof;
  IdentifierInfo *kw_interface;
  IdentifierInfo *kw_native;
  IdentifierInfo *kw_package;
  IdentifierInfo *kw_synchronized;
  IdentifierInfo *kw_throws;

  // Pragma keywords.
  IdentifierInfo *kw_mark;

  // Proto keywords.
  IdentifierInfo *kw_extend;
  IdentifierInfo *kw_option;
  IdentifierInfo *kw_optional;
  IdentifierInfo *kw_repeated;
  IdentifierInfo *kw_required;
  IdentifierInfo *kw_returns;

  // QT keywords.
  IdentifierInfo *kw_signals;
  IdentifierInfo *kw_qsignals;
  IdentifierInfo *kw_slots;
  IdentifierInfo *kw_qslots;
};

} // namespace format
} // namespace clang

#endif
