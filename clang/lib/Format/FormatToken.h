//===--- FormatToken.h - Format C++ code ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the FormatToken, a wrapper
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
#include <unordered_set>

namespace clang {
namespace format {

#define LIST_TOKEN_TYPES                                                       \
  TYPE(ArrayInitializerLSquare)                                                \
  TYPE(ArraySubscriptLSquare)                                                  \
  TYPE(AttributeColon)                                                         \
  TYPE(AttributeMacro)                                                         \
  TYPE(AttributeParen)                                                         \
  TYPE(AttributeSquare)                                                        \
  TYPE(BinaryOperator)                                                         \
  TYPE(BitFieldColon)                                                          \
  TYPE(BlockComment)                                                           \
  TYPE(BracedListLBrace)                                                       \
  TYPE(CastRParen)                                                             \
  TYPE(ClassLBrace)                                                            \
  TYPE(CompoundRequirementLBrace)                                              \
  TYPE(ConditionalExpr)                                                        \
  TYPE(ConflictAlternative)                                                    \
  TYPE(ConflictEnd)                                                            \
  TYPE(ConflictStart)                                                          \
  TYPE(CtorInitializerColon)                                                   \
  TYPE(CtorInitializerComma)                                                   \
  TYPE(DesignatedInitializerLSquare)                                           \
  TYPE(DesignatedInitializerPeriod)                                            \
  TYPE(DictLiteral)                                                            \
  TYPE(EnumLBrace)                                                             \
  TYPE(FatArrow)                                                               \
  TYPE(ForEachMacro)                                                           \
  TYPE(FunctionAnnotationRParen)                                               \
  TYPE(FunctionDeclarationName)                                                \
  TYPE(FunctionLBrace)                                                         \
  TYPE(FunctionLikeOrFreestandingMacro)                                        \
  TYPE(FunctionTypeLParen)                                                     \
  TYPE(IfMacro)                                                                \
  TYPE(ImplicitStringLiteral)                                                  \
  TYPE(InheritanceColon)                                                       \
  TYPE(InheritanceComma)                                                       \
  TYPE(InlineASMBrace)                                                         \
  TYPE(InlineASMColon)                                                         \
  TYPE(InlineASMSymbolicNameLSquare)                                           \
  TYPE(JavaAnnotation)                                                         \
  TYPE(JsComputedPropertyName)                                                 \
  TYPE(JsExponentiation)                                                       \
  TYPE(JsExponentiationEqual)                                                  \
  TYPE(JsPipePipeEqual)                                                        \
  TYPE(JsPrivateIdentifier)                                                    \
  TYPE(JsTypeColon)                                                            \
  TYPE(JsTypeOperator)                                                         \
  TYPE(JsTypeOptionalQuestion)                                                 \
  TYPE(JsAndAndEqual)                                                          \
  TYPE(LambdaArrow)                                                            \
  TYPE(LambdaLBrace)                                                           \
  TYPE(LambdaLSquare)                                                          \
  TYPE(LeadingJavaAnnotation)                                                  \
  TYPE(LineComment)                                                            \
  TYPE(MacroBlockBegin)                                                        \
  TYPE(MacroBlockEnd)                                                          \
  TYPE(ModulePartitionColon)                                                   \
  TYPE(NamespaceMacro)                                                         \
  TYPE(NonNullAssertion)                                                       \
  TYPE(NullCoalescingEqual)                                                    \
  TYPE(NullCoalescingOperator)                                                 \
  TYPE(NullPropagatingOperator)                                                \
  TYPE(ObjCBlockLBrace)                                                        \
  TYPE(ObjCBlockLParen)                                                        \
  TYPE(ObjCDecl)                                                               \
  TYPE(ObjCForIn)                                                              \
  TYPE(ObjCMethodExpr)                                                         \
  TYPE(ObjCMethodSpecifier)                                                    \
  TYPE(ObjCProperty)                                                           \
  TYPE(ObjCStringLiteral)                                                      \
  TYPE(OverloadedOperator)                                                     \
  TYPE(OverloadedOperatorLParen)                                               \
  TYPE(PointerOrReference)                                                     \
  TYPE(PureVirtualSpecifier)                                                   \
  TYPE(RangeBasedForLoopColon)                                                 \
  TYPE(RecordLBrace)                                                           \
  TYPE(RegexLiteral)                                                           \
  TYPE(RequiresClause)                                                         \
  TYPE(RequiresClauseInARequiresExpression)                                    \
  TYPE(RequiresExpression)                                                     \
  TYPE(RequiresExpressionLBrace)                                               \
  TYPE(RequiresExpressionLParen)                                               \
  TYPE(SelectorName)                                                           \
  TYPE(StartOfName)                                                            \
  TYPE(StatementAttributeLikeMacro)                                            \
  TYPE(StatementMacro)                                                         \
  TYPE(StructLBrace)                                                           \
  TYPE(StructuredBindingLSquare)                                               \
  TYPE(TemplateCloser)                                                         \
  TYPE(TemplateOpener)                                                         \
  TYPE(TemplateString)                                                         \
  TYPE(ProtoExtensionLSquare)                                                  \
  TYPE(TrailingAnnotation)                                                     \
  TYPE(TrailingReturnArrow)                                                    \
  TYPE(TrailingUnaryOperator)                                                  \
  TYPE(TypeDeclarationParen)                                                   \
  TYPE(TypenameMacro)                                                          \
  TYPE(UnaryOperator)                                                          \
  TYPE(UnionLBrace)                                                            \
  TYPE(UntouchableMacroFunc)                                                   \
  TYPE(CSharpStringLiteral)                                                    \
  TYPE(CSharpNamedArgumentColon)                                               \
  TYPE(CSharpNullable)                                                         \
  TYPE(CSharpNullConditionalLSquare)                                           \
  TYPE(CSharpGenericTypeConstraint)                                            \
  TYPE(CSharpGenericTypeConstraintColon)                                       \
  TYPE(CSharpGenericTypeConstraintComma)                                       \
  TYPE(Unknown)

/// Sorted operators that can follow a C variable.
static const std::vector<clang::tok::TokenKind> COperatorsFollowingVar = [] {
  std::vector<clang::tok::TokenKind> ReturnVal = {
      tok::l_square,     tok::r_square,
      tok::l_paren,      tok::r_paren,
      tok::r_brace,      tok::period,
      tok::ellipsis,     tok::ampamp,
      tok::ampequal,     tok::star,
      tok::starequal,    tok::plus,
      tok::plusplus,     tok::plusequal,
      tok::minus,        tok::arrow,
      tok::minusminus,   tok::minusequal,
      tok::exclaim,      tok::exclaimequal,
      tok::slash,        tok::slashequal,
      tok::percent,      tok::percentequal,
      tok::less,         tok::lessless,
      tok::lessequal,    tok::lesslessequal,
      tok::greater,      tok::greatergreater,
      tok::greaterequal, tok::greatergreaterequal,
      tok::caret,        tok::caretequal,
      tok::pipe,         tok::pipepipe,
      tok::pipeequal,    tok::question,
      tok::semi,         tok::equal,
      tok::equalequal,   tok::comma};
  assert(std::is_sorted(ReturnVal.begin(), ReturnVal.end()));
  return ReturnVal;
}();

/// Determines the semantic type of a syntactic token, e.g. whether "<" is a
/// template opener or binary operator.
enum TokenType : uint8_t {
#define TYPE(X) TT_##X,
  LIST_TOKEN_TYPES
#undef TYPE
      NUM_TOKEN_TYPES
};

/// Determines the name of a token type.
const char *getTokenTypeName(TokenType Type);

// Represents what type of block a set of braces open.
enum BraceBlockKind { BK_Unknown, BK_Block, BK_BracedInit };

// The packing kind of a function's parameters.
enum ParameterPackingKind { PPK_BinPacked, PPK_OnePerLine, PPK_Inconclusive };

enum FormatDecision { FD_Unformatted, FD_Continue, FD_Break };

/// Roles a token can take in a configured macro expansion.
enum MacroRole {
  /// The token was expanded from a macro argument when formatting the expanded
  /// token sequence.
  MR_ExpandedArg,
  /// The token is part of a macro argument that was previously formatted as
  /// expansion when formatting the unexpanded macro call.
  MR_UnexpandedArg,
  /// The token was expanded from a macro definition, and is not visible as part
  /// of the macro call.
  MR_Hidden,
};

struct FormatToken;

/// Contains information on the token's role in a macro expansion.
///
/// Given the following definitions:
/// A(X) = [ X ]
/// B(X) = < X >
/// C(X) = X
///
/// Consider the macro call:
/// A({B(C(C(x)))}) -> [{<x>}]
///
/// In this case, the tokens of the unexpanded macro call will have the
/// following relevant entries in their macro context (note that formatting
/// the unexpanded macro call happens *after* formatting the expanded macro
/// call):
///                   A( { B( C( C(x) ) ) } )
/// Role:             NN U NN NN NNUN N N U N  (N=None, U=UnexpandedArg)
///
///                   [  { <       x    > } ]
/// Role:             H  E H       E    H E H  (H=Hidden, E=ExpandedArg)
/// ExpandedFrom[0]:  A  A A       A    A A A
/// ExpandedFrom[1]:       B       B    B
/// ExpandedFrom[2]:               C
/// ExpandedFrom[3]:               C
/// StartOfExpansion: 1  0 1       2    0 0 0
/// EndOfExpansion:   0  0 0       2    1 0 1
struct MacroExpansion {
  MacroExpansion(MacroRole Role) : Role(Role) {}

  /// The token's role in the macro expansion.
  /// When formatting an expanded macro, all tokens that are part of macro
  /// arguments will be MR_ExpandedArg, while all tokens that are not visible in
  /// the macro call will be MR_Hidden.
  /// When formatting an unexpanded macro call, all tokens that are part of
  /// macro arguments will be MR_UnexpandedArg.
  MacroRole Role;

  /// The stack of macro call identifier tokens this token was expanded from.
  llvm::SmallVector<FormatToken *, 1> ExpandedFrom;

  /// The number of expansions of which this macro is the first entry.
  unsigned StartOfExpansion = 0;

  /// The number of currently open expansions in \c ExpandedFrom this macro is
  /// the last token in.
  unsigned EndOfExpansion = 0;
};

class TokenRole;
class AnnotatedLine;

/// A wrapper around a \c Token storing information about the
/// whitespace characters preceding it.
struct FormatToken {
  FormatToken()
      : HasUnescapedNewline(false), IsMultiline(false), IsFirst(false),
        MustBreakBefore(false), IsUnterminatedLiteral(false),
        CanBreakBefore(false), ClosesTemplateDeclaration(false),
        StartsBinaryExpression(false), EndsBinaryExpression(false),
        PartOfMultiVariableDeclStmt(false), ContinuesLineCommentSection(false),
        Finalized(false), ClosesRequiresClause(false), BlockKind(BK_Unknown),
        Decision(FD_Unformatted), PackingKind(PPK_Inconclusive),
        Type(TT_Unknown) {}

  /// The \c Token.
  Token Tok;

  /// The raw text of the token.
  ///
  /// Contains the raw token text without leading whitespace and without leading
  /// escaped newlines.
  StringRef TokenText;

  /// A token can have a special role that can carry extra information
  /// about the token's formatting.
  /// FIXME: Make FormatToken for parsing and AnnotatedToken two different
  /// classes and make this a unique_ptr in the AnnotatedToken class.
  std::shared_ptr<TokenRole> Role;

  /// The range of the whitespace immediately preceding the \c Token.
  SourceRange WhitespaceRange;

  /// Whether there is at least one unescaped newline before the \c
  /// Token.
  unsigned HasUnescapedNewline : 1;

  /// Whether the token text contains newlines (escaped or not).
  unsigned IsMultiline : 1;

  /// Indicates that this is the first token of the file.
  unsigned IsFirst : 1;

  /// Whether there must be a line break before this token.
  ///
  /// This happens for example when a preprocessor directive ended directly
  /// before the token.
  unsigned MustBreakBefore : 1;

  /// Set to \c true if this token is an unterminated literal.
  unsigned IsUnterminatedLiteral : 1;

  /// \c true if it is allowed to break before this token.
  unsigned CanBreakBefore : 1;

  /// \c true if this is the ">" of "template<..>".
  unsigned ClosesTemplateDeclaration : 1;

  /// \c true if this token starts a binary expression, i.e. has at least
  /// one fake l_paren with a precedence greater than prec::Unknown.
  unsigned StartsBinaryExpression : 1;
  /// \c true if this token ends a binary expression.
  unsigned EndsBinaryExpression : 1;

  /// Is this token part of a \c DeclStmt defining multiple variables?
  ///
  /// Only set if \c Type == \c TT_StartOfName.
  unsigned PartOfMultiVariableDeclStmt : 1;

  /// Does this line comment continue a line comment section?
  ///
  /// Only set to true if \c Type == \c TT_LineComment.
  unsigned ContinuesLineCommentSection : 1;

  /// If \c true, this token has been fully formatted (indented and
  /// potentially re-formatted inside), and we do not allow further formatting
  /// changes.
  unsigned Finalized : 1;

  /// \c true if this is the last token within requires clause.
  unsigned ClosesRequiresClause : 1;

private:
  /// Contains the kind of block if this token is a brace.
  unsigned BlockKind : 2;

public:
  BraceBlockKind getBlockKind() const {
    return static_cast<BraceBlockKind>(BlockKind);
  }
  void setBlockKind(BraceBlockKind BBK) {
    BlockKind = BBK;
    assert(getBlockKind() == BBK && "BraceBlockKind overflow!");
  }

private:
  /// Stores the formatting decision for the token once it was made.
  unsigned Decision : 2;

public:
  FormatDecision getDecision() const {
    return static_cast<FormatDecision>(Decision);
  }
  void setDecision(FormatDecision D) {
    Decision = D;
    assert(getDecision() == D && "FormatDecision overflow!");
  }

private:
  /// If this is an opening parenthesis, how are the parameters packed?
  unsigned PackingKind : 2;

public:
  ParameterPackingKind getPackingKind() const {
    return static_cast<ParameterPackingKind>(PackingKind);
  }
  void setPackingKind(ParameterPackingKind K) {
    PackingKind = K;
    assert(getPackingKind() == K && "ParameterPackingKind overflow!");
  }

private:
  TokenType Type;

public:
  /// Returns the token's type, e.g. whether "<" is a template opener or
  /// binary operator.
  TokenType getType() const { return Type; }
  void setType(TokenType T) { Type = T; }

  /// The number of newlines immediately before the \c Token.
  ///
  /// This can be used to determine what the user wrote in the original code
  /// and thereby e.g. leave an empty line between two function definitions.
  unsigned NewlinesBefore = 0;

  /// The offset just past the last '\n' in this token's leading
  /// whitespace (relative to \c WhiteSpaceStart). 0 if there is no '\n'.
  unsigned LastNewlineOffset = 0;

  /// The width of the non-whitespace parts of the token (or its first
  /// line for multi-line tokens) in columns.
  /// We need this to correctly measure number of columns a token spans.
  unsigned ColumnWidth = 0;

  /// Contains the width in columns of the last line of a multi-line
  /// token.
  unsigned LastLineColumnWidth = 0;

  /// The number of spaces that should be inserted before this token.
  unsigned SpacesRequiredBefore = 0;

  /// Number of parameters, if this is "(", "[" or "<".
  unsigned ParameterCount = 0;

  /// Number of parameters that are nested blocks,
  /// if this is "(", "[" or "<".
  unsigned BlockParameterCount = 0;

  /// If this is a bracket ("<", "(", "[" or "{"), contains the kind of
  /// the surrounding bracket.
  tok::TokenKind ParentBracket = tok::unknown;

  /// The total length of the unwrapped line up to and including this
  /// token.
  unsigned TotalLength = 0;

  /// The original 0-based column of this token, including expanded tabs.
  /// The configured TabWidth is used as tab width.
  unsigned OriginalColumn = 0;

  /// The length of following tokens until the next natural split point,
  /// or the next token that can be broken.
  unsigned UnbreakableTailLength = 0;

  // FIXME: Come up with a 'cleaner' concept.
  /// The binding strength of a token. This is a combined value of
  /// operator precedence, parenthesis nesting, etc.
  unsigned BindingStrength = 0;

  /// The nesting level of this token, i.e. the number of surrounding (),
  /// [], {} or <>.
  unsigned NestingLevel = 0;

  /// The indent level of this token. Copied from the surrounding line.
  unsigned IndentLevel = 0;

  /// Penalty for inserting a line break before this token.
  unsigned SplitPenalty = 0;

  /// If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the length of the longest name.
  ///
  /// This being set to 0 means that the selectors should not be colon-aligned,
  /// e.g. because several of them are block-type.
  unsigned LongestObjCSelectorName = 0;

  /// If this is the first ObjC selector name in an ObjC method
  /// definition or call, this contains the number of parts that the whole
  /// selector consist of.
  unsigned ObjCSelectorNameParts = 0;

  /// The 0-based index of the parameter/argument. For ObjC it is set
  /// for the selector name token.
  /// For now calculated only for ObjC.
  unsigned ParameterIndex = 0;

  /// Stores the number of required fake parentheses and the
  /// corresponding operator precedence.
  ///
  /// If multiple fake parentheses start at a token, this vector stores them in
  /// reverse order, i.e. inner fake parenthesis first.
  SmallVector<prec::Level, 4> FakeLParens;
  /// Insert this many fake ) after this token for correct indentation.
  unsigned FakeRParens = 0;

  /// If this is an operator (or "."/"->") in a sequence of operators
  /// with the same precedence, contains the 0-based operator index.
  unsigned OperatorIndex = 0;

  /// If this is an operator (or "."/"->") in a sequence of operators
  /// with the same precedence, points to the next operator.
  FormatToken *NextOperator = nullptr;

  /// If this is a bracket, this points to the matching one.
  FormatToken *MatchingParen = nullptr;

  /// The previous token in the unwrapped line.
  FormatToken *Previous = nullptr;

  /// The next token in the unwrapped line.
  FormatToken *Next = nullptr;

  /// The first token in set of column elements.
  bool StartsColumn = false;

  /// This notes the start of the line of an array initializer.
  bool ArrayInitializerLineStart = false;

  /// This starts an array initializer.
  bool IsArrayInitializer = false;

  /// Is optional and can be removed.
  bool Optional = false;

  /// Number of optional braces to be inserted after this token:
  ///   -1: a single left brace
  ///    0: no braces
  ///   >0: number of right braces
  int8_t BraceCount = 0;

  /// If this token starts a block, this contains all the unwrapped lines
  /// in it.
  SmallVector<AnnotatedLine *, 1> Children;

  // Contains all attributes related to how this token takes part
  // in a configured macro expansion.
  llvm::Optional<MacroExpansion> MacroCtx;

  bool is(tok::TokenKind Kind) const { return Tok.is(Kind); }
  bool is(TokenType TT) const { return getType() == TT; }
  bool is(const IdentifierInfo *II) const {
    return II && II == Tok.getIdentifierInfo();
  }
  bool is(tok::PPKeywordKind Kind) const {
    return Tok.getIdentifierInfo() &&
           Tok.getIdentifierInfo()->getPPKeywordID() == Kind;
  }
  bool is(BraceBlockKind BBK) const { return getBlockKind() == BBK; }
  bool is(ParameterPackingKind PPK) const { return getPackingKind() == PPK; }

  template <typename A, typename B> bool isOneOf(A K1, B K2) const {
    return is(K1) || is(K2);
  }
  template <typename A, typename B, typename... Ts>
  bool isOneOf(A K1, B K2, Ts... Ks) const {
    return is(K1) || isOneOf(K2, Ks...);
  }
  template <typename T> bool isNot(T Kind) const { return !is(Kind); }

  bool isIf(bool AllowConstexprMacro = true) const {
    return is(tok::kw_if) || endsSequence(tok::kw_constexpr, tok::kw_if) ||
           (endsSequence(tok::identifier, tok::kw_if) && AllowConstexprMacro);
  }

  bool closesScopeAfterBlock() const {
    if (getBlockKind() == BK_Block)
      return true;
    if (closesScope())
      return Previous->closesScopeAfterBlock();
    return false;
  }

  /// \c true if this token starts a sequence with the given tokens in order,
  /// following the ``Next`` pointers, ignoring comments.
  template <typename A, typename... Ts>
  bool startsSequence(A K1, Ts... Tokens) const {
    return startsSequenceInternal(K1, Tokens...);
  }

  /// \c true if this token ends a sequence with the given tokens in order,
  /// following the ``Previous`` pointers, ignoring comments.
  /// For example, given tokens [T1, T2, T3], the function returns true if
  /// 3 tokens ending at this (ignoring comments) are [T3, T2, T1]. In other
  /// words, the tokens passed to this function need to the reverse of the
  /// order the tokens appear in code.
  template <typename A, typename... Ts>
  bool endsSequence(A K1, Ts... Tokens) const {
    return endsSequenceInternal(K1, Tokens...);
  }

  bool isStringLiteral() const { return tok::isStringLiteral(Tok.getKind()); }

  bool isObjCAtKeyword(tok::ObjCKeywordKind Kind) const {
    return Tok.isObjCAtKeyword(Kind);
  }

  bool isAccessSpecifier(bool ColonRequired = true) const {
    return isOneOf(tok::kw_public, tok::kw_protected, tok::kw_private) &&
           (!ColonRequired || (Next && Next->is(tok::colon)));
  }

  bool canBePointerOrReferenceQualifier() const {
    return isOneOf(tok::kw_const, tok::kw_restrict, tok::kw_volatile,
                   tok::kw___attribute, tok::kw__Nonnull, tok::kw__Nullable,
                   tok::kw__Null_unspecified, tok::kw___ptr32, tok::kw___ptr64,
                   TT_AttributeMacro);
  }

  /// Determine whether the token is a simple-type-specifier.
  LLVM_NODISCARD bool isSimpleTypeSpecifier() const;

  LLVM_NODISCARD bool isTypeOrIdentifier() const;

  bool isObjCAccessSpecifier() const {
    return is(tok::at) && Next &&
           (Next->isObjCAtKeyword(tok::objc_public) ||
            Next->isObjCAtKeyword(tok::objc_protected) ||
            Next->isObjCAtKeyword(tok::objc_package) ||
            Next->isObjCAtKeyword(tok::objc_private));
  }

  /// Returns whether \p Tok is ([{ or an opening < of a template or in
  /// protos.
  bool opensScope() const {
    if (is(TT_TemplateString) && TokenText.endswith("${"))
      return true;
    if (is(TT_DictLiteral) && is(tok::less))
      return true;
    return isOneOf(tok::l_paren, tok::l_brace, tok::l_square,
                   TT_TemplateOpener);
  }
  /// Returns whether \p Tok is )]} or a closing > of a template or in
  /// protos.
  bool closesScope() const {
    if (is(TT_TemplateString) && TokenText.startswith("}"))
      return true;
    if (is(TT_DictLiteral) && is(tok::greater))
      return true;
    return isOneOf(tok::r_paren, tok::r_brace, tok::r_square,
                   TT_TemplateCloser);
  }

  /// Returns \c true if this is a "." or "->" accessing a member.
  bool isMemberAccess() const {
    return isOneOf(tok::arrow, tok::period, tok::arrowstar) &&
           !isOneOf(TT_DesignatedInitializerPeriod, TT_TrailingReturnArrow,
                    TT_LambdaArrow, TT_LeadingJavaAnnotation);
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

  /// Returns \c true if this is a keyword that can be used
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
    case tok::kw__Atomic:
    case tok::kw___attribute:
    case tok::kw___underlying_type:
    case tok::kw_requires:
      return true;
    default:
      return false;
    }
  }

  /// Returns \c true if this is a string literal that's like a label,
  /// e.g. ends with "=" or ":".
  bool isLabelString() const {
    if (!is(tok::string_literal))
      return false;
    StringRef Content = TokenText;
    if (Content.startswith("\"") || Content.startswith("'"))
      Content = Content.drop_front(1);
    if (Content.endswith("\"") || Content.endswith("'"))
      Content = Content.drop_back(1);
    Content = Content.trim();
    return Content.size() > 1 &&
           (Content.back() == ':' || Content.back() == '=');
  }

  /// Returns actual token start location without leading escaped
  /// newlines and whitespace.
  ///
  /// This can be different to Tok.getLocation(), which includes leading escaped
  /// newlines.
  SourceLocation getStartOfNonWhitespace() const {
    return WhitespaceRange.getEnd();
  }

  /// Returns \c true if the range of whitespace immediately preceding the \c
  /// Token is not empty.
  bool hasWhitespaceBefore() const {
    return WhitespaceRange.getBegin() != WhitespaceRange.getEnd();
  }

  prec::Level getPrecedence() const {
    return getBinOpPrecedence(Tok.getKind(), /*GreaterThanIsOperator=*/true,
                              /*CPlusPlus11=*/true);
  }

  /// Returns the previous token ignoring comments.
  LLVM_NODISCARD FormatToken *getPreviousNonComment() const {
    FormatToken *Tok = Previous;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Previous;
    return Tok;
  }

  /// Returns the next token ignoring comments.
  LLVM_NODISCARD const FormatToken *getNextNonComment() const {
    const FormatToken *Tok = Next;
    while (Tok && Tok->is(tok::comment))
      Tok = Tok->Next;
    return Tok;
  }

  /// Returns \c true if this tokens starts a block-type list, i.e. a
  /// list that should be indented with a block indent.
  LLVM_NODISCARD bool opensBlockOrBlockTypeList(const FormatStyle &Style) const;

  /// Returns whether the token is the left square bracket of a C++
  /// structured binding declaration.
  bool isCppStructuredBinding(const FormatStyle &Style) const {
    if (!Style.isCpp() || isNot(tok::l_square))
      return false;
    const FormatToken *T = this;
    do {
      T = T->getPreviousNonComment();
    } while (T && T->isOneOf(tok::kw_const, tok::kw_volatile, tok::amp,
                             tok::ampamp));
    return T && T->is(tok::kw_auto);
  }

  /// Same as opensBlockOrBlockTypeList, but for the closing token.
  bool closesBlockOrBlockTypeList(const FormatStyle &Style) const {
    if (is(TT_TemplateString) && closesScope())
      return true;
    return MatchingParen && MatchingParen->opensBlockOrBlockTypeList(Style);
  }

  /// Return the actual namespace token, if this token starts a namespace
  /// block.
  const FormatToken *getNamespaceToken() const {
    const FormatToken *NamespaceTok = this;
    if (is(tok::comment))
      NamespaceTok = NamespaceTok->getNextNonComment();
    // Detect "(inline|export)? namespace" in the beginning of a line.
    if (NamespaceTok && NamespaceTok->isOneOf(tok::kw_inline, tok::kw_export))
      NamespaceTok = NamespaceTok->getNextNonComment();
    return NamespaceTok &&
                   NamespaceTok->isOneOf(tok::kw_namespace, TT_NamespaceMacro)
               ? NamespaceTok
               : nullptr;
  }

  void copyFrom(const FormatToken &Tok) { *this = Tok; }

private:
  // Only allow copying via the explicit copyFrom method.
  FormatToken(const FormatToken &) = delete;
  FormatToken &operator=(const FormatToken &) = default;

  template <typename A, typename... Ts>
  bool startsSequenceInternal(A K1, Ts... Tokens) const {
    if (is(tok::comment) && Next)
      return Next->startsSequenceInternal(K1, Tokens...);
    return is(K1) && Next && Next->startsSequenceInternal(Tokens...);
  }

  template <typename A> bool startsSequenceInternal(A K1) const {
    if (is(tok::comment) && Next)
      return Next->startsSequenceInternal(K1);
    return is(K1);
  }

  template <typename A, typename... Ts> bool endsSequenceInternal(A K1) const {
    if (is(tok::comment) && Previous)
      return Previous->endsSequenceInternal(K1);
    return is(K1);
  }

  template <typename A, typename... Ts>
  bool endsSequenceInternal(A K1, Ts... Tokens) const {
    if (is(tok::comment) && Previous)
      return Previous->endsSequenceInternal(K1, Tokens...);
    return is(K1) && Previous && Previous->endsSequenceInternal(Tokens...);
  }
};

class ContinuationIndenter;
struct LineState;

class TokenRole {
public:
  TokenRole(const FormatStyle &Style) : Style(Style) {}
  virtual ~TokenRole();

  /// After the \c TokenAnnotator has finished annotating all the tokens,
  /// this function precomputes required information for formatting.
  virtual void precomputeFormattingInfos(const FormatToken *Token);

  /// Apply the special formatting that the given role demands.
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

  /// Same as \c formatFromToken, but assumes that the first token has
  /// already been set thereby deciding on the first line break.
  virtual unsigned formatAfterToken(LineState &State,
                                    ContinuationIndenter *Indenter,
                                    bool DryRun) {
    return 0;
  }

  /// Notifies the \c Role that a comma was found.
  virtual void CommaFound(const FormatToken *Token) {}

  virtual const FormatToken *lastComma() { return nullptr; }

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

  /// Adds \p Token as the next comma to the \c CommaSeparated list.
  void CommaFound(const FormatToken *Token) override {
    Commas.push_back(Token);
  }

  const FormatToken *lastComma() override {
    if (Commas.empty())
      return nullptr;
    return Commas.back();
  }

private:
  /// A struct that holds information on how to format a given list with
  /// a specific number of columns.
  struct ColumnFormat {
    /// The number of columns to use.
    unsigned Columns;

    /// The total width in characters.
    unsigned TotalWidth;

    /// The number of lines required for this format.
    unsigned LineCount;

    /// The size of each column in characters.
    SmallVector<unsigned, 8> ColumnSizes;
  };

  /// Calculate which \c ColumnFormat fits best into
  /// \p RemainingCharacters.
  const ColumnFormat *getColumnFormat(unsigned RemainingCharacters) const;

  /// The ordered \c FormatTokens making up the commas of this list.
  SmallVector<const FormatToken *, 8> Commas;

  /// The length of each of the list's items in characters including the
  /// trailing comma.
  SmallVector<unsigned, 8> ItemLengths;

  /// Precomputed formats that can be used for this list.
  SmallVector<ColumnFormat, 4> Formats;

  bool HasNestedBracedList;
};

/// Encapsulates keywords that are context sensitive or for languages not
/// properly supported by Clang's lexer.
struct AdditionalKeywords {
  AdditionalKeywords(IdentifierTable &IdentTable) {
    kw_final = &IdentTable.get("final");
    kw_override = &IdentTable.get("override");
    kw_in = &IdentTable.get("in");
    kw_of = &IdentTable.get("of");
    kw_CF_CLOSED_ENUM = &IdentTable.get("CF_CLOSED_ENUM");
    kw_CF_ENUM = &IdentTable.get("CF_ENUM");
    kw_CF_OPTIONS = &IdentTable.get("CF_OPTIONS");
    kw_NS_CLOSED_ENUM = &IdentTable.get("NS_CLOSED_ENUM");
    kw_NS_ENUM = &IdentTable.get("NS_ENUM");
    kw_NS_OPTIONS = &IdentTable.get("NS_OPTIONS");

    kw_as = &IdentTable.get("as");
    kw_async = &IdentTable.get("async");
    kw_await = &IdentTable.get("await");
    kw_declare = &IdentTable.get("declare");
    kw_finally = &IdentTable.get("finally");
    kw_from = &IdentTable.get("from");
    kw_function = &IdentTable.get("function");
    kw_get = &IdentTable.get("get");
    kw_import = &IdentTable.get("import");
    kw_infer = &IdentTable.get("infer");
    kw_is = &IdentTable.get("is");
    kw_let = &IdentTable.get("let");
    kw_module = &IdentTable.get("module");
    kw_readonly = &IdentTable.get("readonly");
    kw_set = &IdentTable.get("set");
    kw_type = &IdentTable.get("type");
    kw_typeof = &IdentTable.get("typeof");
    kw_var = &IdentTable.get("var");
    kw_yield = &IdentTable.get("yield");

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
    kw___has_include = &IdentTable.get("__has_include");
    kw___has_include_next = &IdentTable.get("__has_include_next");

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

    // For internal clang-format use.
    kw_internal_ident_after_define =
        &IdentTable.get("__CLANG_FORMAT_INTERNAL_IDENT_AFTER_DEFINE__");

    // C# keywords
    kw_dollar = &IdentTable.get("dollar");
    kw_base = &IdentTable.get("base");
    kw_byte = &IdentTable.get("byte");
    kw_checked = &IdentTable.get("checked");
    kw_decimal = &IdentTable.get("decimal");
    kw_delegate = &IdentTable.get("delegate");
    kw_event = &IdentTable.get("event");
    kw_fixed = &IdentTable.get("fixed");
    kw_foreach = &IdentTable.get("foreach");
    kw_implicit = &IdentTable.get("implicit");
    kw_internal = &IdentTable.get("internal");
    kw_lock = &IdentTable.get("lock");
    kw_null = &IdentTable.get("null");
    kw_object = &IdentTable.get("object");
    kw_out = &IdentTable.get("out");
    kw_params = &IdentTable.get("params");
    kw_ref = &IdentTable.get("ref");
    kw_string = &IdentTable.get("string");
    kw_stackalloc = &IdentTable.get("stackalloc");
    kw_sbyte = &IdentTable.get("sbyte");
    kw_sealed = &IdentTable.get("sealed");
    kw_uint = &IdentTable.get("uint");
    kw_ulong = &IdentTable.get("ulong");
    kw_unchecked = &IdentTable.get("unchecked");
    kw_unsafe = &IdentTable.get("unsafe");
    kw_ushort = &IdentTable.get("ushort");
    kw_when = &IdentTable.get("when");
    kw_where = &IdentTable.get("where");

    // Keep this at the end of the constructor to make sure everything here
    // is
    // already initialized.
    JsExtraKeywords = std::unordered_set<IdentifierInfo *>(
        {kw_as, kw_async, kw_await, kw_declare, kw_finally, kw_from,
         kw_function, kw_get, kw_import, kw_is, kw_let, kw_module, kw_override,
         kw_readonly, kw_set, kw_type, kw_typeof, kw_var, kw_yield,
         // Keywords from the Java section.
         kw_abstract, kw_extends, kw_implements, kw_instanceof, kw_interface});

    CSharpExtraKeywords = std::unordered_set<IdentifierInfo *>(
        {kw_base, kw_byte, kw_checked, kw_decimal, kw_delegate, kw_event,
         kw_fixed, kw_foreach, kw_implicit, kw_in, kw_interface, kw_internal,
         kw_is, kw_lock, kw_null, kw_object, kw_out, kw_override, kw_params,
         kw_readonly, kw_ref, kw_string, kw_stackalloc, kw_sbyte, kw_sealed,
         kw_uint, kw_ulong, kw_unchecked, kw_unsafe, kw_ushort, kw_when,
         kw_where,
         // Keywords from the JavaScript section.
         kw_as, kw_async, kw_await, kw_declare, kw_finally, kw_from,
         kw_function, kw_get, kw_import, kw_is, kw_let, kw_module, kw_readonly,
         kw_set, kw_type, kw_typeof, kw_var, kw_yield,
         // Keywords from the Java section.
         kw_abstract, kw_extends, kw_implements, kw_instanceof, kw_interface});
  }

  // Context sensitive keywords.
  IdentifierInfo *kw_final;
  IdentifierInfo *kw_override;
  IdentifierInfo *kw_in;
  IdentifierInfo *kw_of;
  IdentifierInfo *kw_CF_CLOSED_ENUM;
  IdentifierInfo *kw_CF_ENUM;
  IdentifierInfo *kw_CF_OPTIONS;
  IdentifierInfo *kw_NS_CLOSED_ENUM;
  IdentifierInfo *kw_NS_ENUM;
  IdentifierInfo *kw_NS_OPTIONS;
  IdentifierInfo *kw___except;
  IdentifierInfo *kw___has_include;
  IdentifierInfo *kw___has_include_next;

  // JavaScript keywords.
  IdentifierInfo *kw_as;
  IdentifierInfo *kw_async;
  IdentifierInfo *kw_await;
  IdentifierInfo *kw_declare;
  IdentifierInfo *kw_finally;
  IdentifierInfo *kw_from;
  IdentifierInfo *kw_function;
  IdentifierInfo *kw_get;
  IdentifierInfo *kw_import;
  IdentifierInfo *kw_infer;
  IdentifierInfo *kw_is;
  IdentifierInfo *kw_let;
  IdentifierInfo *kw_module;
  IdentifierInfo *kw_readonly;
  IdentifierInfo *kw_set;
  IdentifierInfo *kw_type;
  IdentifierInfo *kw_typeof;
  IdentifierInfo *kw_var;
  IdentifierInfo *kw_yield;

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

  // For internal use by clang-format.
  IdentifierInfo *kw_internal_ident_after_define;

  // C# keywords
  IdentifierInfo *kw_dollar;
  IdentifierInfo *kw_base;
  IdentifierInfo *kw_byte;
  IdentifierInfo *kw_checked;
  IdentifierInfo *kw_decimal;
  IdentifierInfo *kw_delegate;
  IdentifierInfo *kw_event;
  IdentifierInfo *kw_fixed;
  IdentifierInfo *kw_foreach;
  IdentifierInfo *kw_implicit;
  IdentifierInfo *kw_internal;

  IdentifierInfo *kw_lock;
  IdentifierInfo *kw_null;
  IdentifierInfo *kw_object;
  IdentifierInfo *kw_out;

  IdentifierInfo *kw_params;

  IdentifierInfo *kw_ref;
  IdentifierInfo *kw_string;
  IdentifierInfo *kw_stackalloc;
  IdentifierInfo *kw_sbyte;
  IdentifierInfo *kw_sealed;
  IdentifierInfo *kw_uint;
  IdentifierInfo *kw_ulong;
  IdentifierInfo *kw_unchecked;
  IdentifierInfo *kw_unsafe;
  IdentifierInfo *kw_ushort;
  IdentifierInfo *kw_when;
  IdentifierInfo *kw_where;

  /// Returns \c true if \p Tok is a true JavaScript identifier, returns
  /// \c false if it is a keyword or a pseudo keyword.
  /// If \c AcceptIdentifierName is true, returns true not only for keywords,
  // but also for IdentifierName tokens (aka pseudo-keywords), such as
  // ``yield``.
  bool IsJavaScriptIdentifier(const FormatToken &Tok,
                              bool AcceptIdentifierName = true) const {
    // Based on the list of JavaScript & TypeScript keywords here:
    // https://github.com/microsoft/TypeScript/blob/main/src/compiler/scanner.ts#L74
    switch (Tok.Tok.getKind()) {
    case tok::kw_break:
    case tok::kw_case:
    case tok::kw_catch:
    case tok::kw_class:
    case tok::kw_continue:
    case tok::kw_const:
    case tok::kw_default:
    case tok::kw_delete:
    case tok::kw_do:
    case tok::kw_else:
    case tok::kw_enum:
    case tok::kw_export:
    case tok::kw_false:
    case tok::kw_for:
    case tok::kw_if:
    case tok::kw_import:
    case tok::kw_module:
    case tok::kw_new:
    case tok::kw_private:
    case tok::kw_protected:
    case tok::kw_public:
    case tok::kw_return:
    case tok::kw_static:
    case tok::kw_switch:
    case tok::kw_this:
    case tok::kw_throw:
    case tok::kw_true:
    case tok::kw_try:
    case tok::kw_typeof:
    case tok::kw_void:
    case tok::kw_while:
      // These are JS keywords that are lexed by LLVM/clang as keywords.
      return false;
    case tok::identifier: {
      // For identifiers, make sure they are true identifiers, excluding the
      // JavaScript pseudo-keywords (not lexed by LLVM/clang as keywords).
      bool IsPseudoKeyword =
          JsExtraKeywords.find(Tok.Tok.getIdentifierInfo()) !=
          JsExtraKeywords.end();
      return AcceptIdentifierName || !IsPseudoKeyword;
    }
    default:
      // Other keywords are handled in the switch below, to avoid problems due
      // to duplicate case labels when using the #include trick.
      break;
    }

    switch (Tok.Tok.getKind()) {
      // Handle C++ keywords not included above: these are all JS identifiers.
#define KEYWORD(X, Y) case tok::kw_##X:
#include "clang/Basic/TokenKinds.def"
      // #undef KEYWORD is not needed -- it's #undef-ed at the end of
      // TokenKinds.def
      return true;
    default:
      // All other tokens (punctuation etc) are not JS identifiers.
      return false;
    }
  }

  /// Returns \c true if \p Tok is a C# keyword, returns
  /// \c false if it is a anything else.
  bool isCSharpKeyword(const FormatToken &Tok) const {
    switch (Tok.Tok.getKind()) {
    case tok::kw_bool:
    case tok::kw_break:
    case tok::kw_case:
    case tok::kw_catch:
    case tok::kw_char:
    case tok::kw_class:
    case tok::kw_const:
    case tok::kw_continue:
    case tok::kw_default:
    case tok::kw_do:
    case tok::kw_double:
    case tok::kw_else:
    case tok::kw_enum:
    case tok::kw_explicit:
    case tok::kw_extern:
    case tok::kw_false:
    case tok::kw_float:
    case tok::kw_for:
    case tok::kw_goto:
    case tok::kw_if:
    case tok::kw_int:
    case tok::kw_long:
    case tok::kw_namespace:
    case tok::kw_new:
    case tok::kw_operator:
    case tok::kw_private:
    case tok::kw_protected:
    case tok::kw_public:
    case tok::kw_return:
    case tok::kw_short:
    case tok::kw_sizeof:
    case tok::kw_static:
    case tok::kw_struct:
    case tok::kw_switch:
    case tok::kw_this:
    case tok::kw_throw:
    case tok::kw_true:
    case tok::kw_try:
    case tok::kw_typeof:
    case tok::kw_using:
    case tok::kw_virtual:
    case tok::kw_void:
    case tok::kw_volatile:
    case tok::kw_while:
      return true;
    default:
      return Tok.is(tok::identifier) &&
             CSharpExtraKeywords.find(Tok.Tok.getIdentifierInfo()) ==
                 CSharpExtraKeywords.end();
    }
  }

private:
  /// The JavaScript keywords beyond the C++ keyword set.
  std::unordered_set<IdentifierInfo *> JsExtraKeywords;

  /// The C# keywords beyond the C++ keyword set
  std::unordered_set<IdentifierInfo *> CSharpExtraKeywords;
};

} // namespace format
} // namespace clang

#endif
