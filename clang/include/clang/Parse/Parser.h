//===--- Parser.h - C Language Parser ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Parser interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSER_H
#define LLVM_CLANG_PARSE_PARSER_H

#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaCodeCompletion.h"
#include "clang/Sema/SemaObjC.h"
#include "clang/Sema/SemaOpenMP.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPContext.h"
#include "llvm/Support/SaveAndRestore.h"
#include <optional>
#include <stack>

namespace clang {
class PragmaHandler;
class Scope;
class BalancedDelimiterTracker;
class CorrectionCandidateCallback;
class DeclGroupRef;
class DiagnosticBuilder;
struct LoopHint;
class Parser;
class ParsingDeclRAIIObject;
class ParsingDeclSpec;
class ParsingDeclarator;
class ParsingFieldDeclarator;
class ColonProtectionRAIIObject;
class InMessageExpressionRAIIObject;
class PoisonSEHIdentifiersRAIIObject;
class OMPClause;
class OpenACCClause;
class ObjCTypeParamList;
struct OMPTraitProperty;
struct OMPTraitSelector;
struct OMPTraitSet;
class OMPTraitInfo;

enum class AnnotatedNameKind {
  /// Annotation has failed and emitted an error.
  Error,
  /// The identifier is a tentatively-declared name.
  TentativeDecl,
  /// The identifier is a template name. FIXME: Add an annotation for that.
  TemplateName,
  /// The identifier can't be resolved.
  Unresolved,
  /// Annotation was successful.
  Success
};

/// The kind of extra semi diagnostic to emit.
enum class ExtraSemiKind {
  OutsideFunction = 0,
  InsideStruct = 1,
  InstanceVariableList = 2,
  AfterMemberFunctionDefinition = 3
};

/// The kind of template we are parsing.
enum class ParsedTemplateKind {
  /// We are not parsing a template at all.
  NonTemplate = 0,
  /// We are parsing a template declaration.
  Template,
  /// We are parsing an explicit specialization.
  ExplicitSpecialization,
  /// We are parsing an explicit instantiation.
  ExplicitInstantiation
};

enum class CachedInitKind { DefaultArgument, DefaultInitializer };

// Definitions for Objective-c context sensitive keywords recognition.
enum class ObjCTypeQual {
  in = 0,
  out,
  inout,
  oneway,
  bycopy,
  byref,
  nonnull,
  nullable,
  null_unspecified,
  NumQuals
};

/// If a typo should be encountered, should typo correction suggest type names,
/// non type names, or both?
enum class TypoCorrectionTypeBehavior {
  AllowNonTypes,
  AllowTypes,
  AllowBoth,
};

/// Control what ParseCastExpression will parse.
enum class CastParseKind { AnyCastExpr = 0, UnaryExprOnly, PrimaryExprOnly };

/// ParenParseOption - Control what ParseParenExpression will parse.
enum class ParenParseOption {
  SimpleExpr,      // Only parse '(' expression ')'
  FoldExpr,        // Also allow fold-expression <anything>
  CompoundStmt,    // Also allow '(' compound-statement ')'
  CompoundLiteral, // Also allow '(' type-name ')' '{' ... '}'
  CastExpr         // Also allow '(' type-name ')' <anything>
};

/// In a call to ParseParenExpression, are the initial parentheses part of an
/// operator that requires the parens be there (like typeof(int)) or could they
/// be something else, such as part of a compound literal or a sizeof
/// expression, etc.
enum class ParenExprKind {
  PartOfOperator, // typeof(int)
  Unknown,        // sizeof(int) or sizeof (int)1.0f, or compound literal, etc
};

/// Describes the behavior that should be taken for an __if_exists
/// block.
enum class IfExistsBehavior {
  /// Parse the block; this code is always used.
  Parse,
  /// Skip the block entirely; this code is never used.
  Skip,
  /// Parse the block as a dependent block, which may be used in
  /// some template instantiations but not others.
  Dependent
};

/// Specifies the context in which type-id/expression
/// disambiguation will occur.
enum class TentativeCXXTypeIdContext {
  InParens,
  Unambiguous,
  AsTemplateArgument,
  InTrailingReturnType,
  AsGenericSelectionArgument,
};

/// The kind of attribute specifier we have found.
enum class CXX11AttributeKind {
  /// This is not an attribute specifier.
  NotAttributeSpecifier,
  /// This should be treated as an attribute-specifier.
  AttributeSpecifier,
  /// The next tokens are '[[', but this is not an attribute-specifier. This
  /// is ill-formed by C++11 [dcl.attr.grammar]p6.
  InvalidAttributeSpecifier
};

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
/// \nosubgrouping
class Parser : public CodeCompletionHandler {
  // Table of Contents
  // -----------------
  // 1. Parsing (Parser.cpp)
  // 2. C++ Class Inline Methods (ParseCXXInlineMethods.cpp)
  // 3. Declarations (ParseDecl.cpp)
  // 4. C++ Declarations (ParseDeclCXX.cpp)
  // 5. Expressions (ParseExpr.cpp)
  // 6. C++ Expressions (ParseExprCXX.cpp)
  // 7. HLSL Constructs (ParseHLSL.cpp)
  // 8. Initializers (ParseInit.cpp)
  // 9. Objective-C Constructs (ParseObjc.cpp)
  // 10. OpenACC Constructs (ParseOpenACC.cpp)
  // 11. OpenMP Constructs (ParseOpenMP.cpp)
  // 12. Pragmas (ParsePragma.cpp)
  // 13. Statements (ParseStmt.cpp)
  // 14. `inline asm` Statement (ParseStmtAsm.cpp)
  // 15. C++ Templates (ParseTemplate.cpp)
  // 16. Tentative Parsing (ParseTentative.cpp)

  /// \name Parsing
  /// Implementations are in Parser.cpp
  ///@{

public:
  friend class ColonProtectionRAIIObject;
  friend class PoisonSEHIdentifiersRAIIObject;
  friend class ParenBraceBracketBalancer;
  friend class BalancedDelimiterTracker;

  Parser(Preprocessor &PP, Sema &Actions, bool SkipFunctionBodies);
  ~Parser() override;

  const LangOptions &getLangOpts() const { return PP.getLangOpts(); }
  const TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
  Preprocessor &getPreprocessor() const { return PP; }
  Sema &getActions() const { return Actions; }
  AttributeFactory &getAttrFactory() { return AttrFactory; }

  const Token &getCurToken() const { return Tok; }
  Scope *getCurScope() const { return Actions.getCurScope(); }

  void incrementMSManglingNumber() const {
    return Actions.incrementMSManglingNumber();
  }

  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
  typedef OpaquePtr<TemplateName> TemplateTy;

  /// Initialize - Warm up the parser.
  ///
  void Initialize();

  /// Parse the first top-level declaration in a translation unit.
  ///
  /// \verbatim
  ///   translation-unit:
  /// [C]     external-declaration
  /// [C]     translation-unit external-declaration
  /// [C++]   top-level-declaration-seq[opt]
  /// [C++20] global-module-fragment[opt] module-declaration
  ///                 top-level-declaration-seq[opt] private-module-fragment[opt]
  /// \endverbatim
  ///
  /// Note that in C, it is an error if there is no first declaration.
  bool ParseFirstTopLevelDecl(DeclGroupPtrTy &Result,
                              Sema::ModuleImportState &ImportState);

  /// ParseTopLevelDecl - Parse one top-level declaration, return whatever the
  /// action tells us to.  This returns true if the EOF was encountered.
  ///
  /// \verbatim
  ///   top-level-declaration:
  ///           declaration
  /// [C++20]   module-import-declaration
  /// \endverbatim
  bool ParseTopLevelDecl(DeclGroupPtrTy &Result,
                         Sema::ModuleImportState &ImportState);
  bool ParseTopLevelDecl() {
    DeclGroupPtrTy Result;
    Sema::ModuleImportState IS = Sema::ModuleImportState::NotACXX20Module;
    return ParseTopLevelDecl(Result, IS);
  }

  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work with special tokens: string literals, code completion,
  /// annotation tokens and balanced tokens must be handled using the specific
  /// consume methods.
  /// Returns the location of the consumed token.
  SourceLocation ConsumeToken() {
    assert(!isTokenSpecial() &&
           "Should consume special tokens with Consume*Token");
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  bool TryConsumeToken(tok::TokenKind Expected) {
    if (Tok.isNot(Expected))
      return false;
    assert(!isTokenSpecial() &&
           "Should consume special tokens with Consume*Token");
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return true;
  }

  bool TryConsumeToken(tok::TokenKind Expected, SourceLocation &Loc) {
    if (!TryConsumeToken(Expected))
      return false;
    Loc = PrevTokLocation;
    return true;
  }

  /// ConsumeAnyToken - Dispatch to the right Consume* method based on the
  /// current token type.  This should only be used in cases where the type of
  /// the token really isn't known, e.g. in error recovery.
  SourceLocation ConsumeAnyToken(bool ConsumeCodeCompletionTok = false) {
    if (isTokenParen())
      return ConsumeParen();
    if (isTokenBracket())
      return ConsumeBracket();
    if (isTokenBrace())
      return ConsumeBrace();
    if (isTokenStringLiteral())
      return ConsumeStringToken();
    if (Tok.is(tok::code_completion))
      return ConsumeCodeCompletionTok ? ConsumeCodeCompletionToken()
                                      : handleUnexpectedCodeCompletionToken();
    if (Tok.isAnnotation())
      return ConsumeAnnotationToken();
    return ConsumeToken();
  }

  SourceLocation getEndOfPreviousToken() const;

  /// GetLookAheadToken - This peeks ahead N tokens and returns that token
  /// without consuming any tokens.  LookAhead(0) returns 'Tok', LookAhead(1)
  /// returns the token after Tok, etc.
  ///
  /// Note that this differs from the Preprocessor's LookAhead method, because
  /// the Parser always has one token lexed that the preprocessor doesn't.
  ///
  const Token &GetLookAheadToken(unsigned N) {
    if (N == 0 || Tok.is(tok::eof))
      return Tok;
    return PP.LookAhead(N - 1);
  }

  /// NextToken - This peeks ahead one token and returns it without
  /// consuming it.
  const Token &NextToken() { return PP.LookAhead(0); }

  /// getTypeAnnotation - Read a parsed type out of an annotation token.
  static TypeResult getTypeAnnotation(const Token &Tok) {
    if (!Tok.getAnnotationValue())
      return TypeError();
    return ParsedType::getFromOpaquePtr(Tok.getAnnotationValue());
  }

  /// TryAnnotateTypeOrScopeToken - If the current token position is on a
  /// typename (possibly qualified in C++) or a C++ scope specifier not followed
  /// by a typename, TryAnnotateTypeOrScopeToken will replace one or more tokens
  /// with a single annotation token representing the typename or C++ scope
  /// respectively.
  /// This simplifies handling of C++ scope specifiers and allows efficient
  /// backtracking without the need to re-parse and resolve nested-names and
  /// typenames.
  /// It will mainly be called when we expect to treat identifiers as typenames
  /// (if they are typenames). For example, in C we do not expect identifiers
  /// inside expressions to be treated as typenames so it will not be called
  /// for expressions in C.
  /// The benefit for C/ObjC is that a typename will be annotated and
  /// Actions.getTypeName will not be needed to be called again (e.g.
  /// getTypeName will not be called twice, once to check whether we have a
  /// declaration specifier, and another one to get the actual type inside
  /// ParseDeclarationSpecifiers).
  ///
  /// This returns true if an error occurred.
  ///
  /// Note that this routine emits an error if you call it with ::new or
  /// ::delete as the current tokens, so only call it in contexts where these
  /// are invalid.
  bool
  TryAnnotateTypeOrScopeToken(ImplicitTypenameContext AllowImplicitTypename =
                                  ImplicitTypenameContext::No);

  /// Try to annotate a type or scope token, having already parsed an
  /// optional scope specifier. \p IsNewScope should be \c true unless the scope
  /// specifier was extracted from an existing tok::annot_cxxscope annotation.
  bool TryAnnotateTypeOrScopeTokenAfterScopeSpec(
      CXXScopeSpec &SS, bool IsNewScope,
      ImplicitTypenameContext AllowImplicitTypename);

  /// TryAnnotateScopeToken - Like TryAnnotateTypeOrScopeToken but only
  /// annotates C++ scope specifiers and template-ids.  This returns
  /// true if there was an error that could not be recovered from.
  ///
  /// Note that this routine emits an error if you call it with ::new or
  /// ::delete as the current tokens, so only call it in contexts where these
  /// are invalid.
  bool TryAnnotateCXXScopeToken(bool EnteringContext = false);

  bool MightBeCXXScopeToken() {
    return getLangOpts().CPlusPlus &&
           (Tok.is(tok::identifier) || Tok.is(tok::coloncolon) ||
            (Tok.is(tok::annot_template_id) &&
             NextToken().is(tok::coloncolon)) ||
            Tok.is(tok::kw_decltype) || Tok.is(tok::kw___super));
  }
  bool TryAnnotateOptionalCXXScopeToken(bool EnteringContext = false) {
    return MightBeCXXScopeToken() && TryAnnotateCXXScopeToken(EnteringContext);
  }

  //===--------------------------------------------------------------------===//
  // Scope manipulation

  /// ParseScope - Introduces a new scope for parsing. The kind of
  /// scope is determined by ScopeFlags. Objects of this type should
  /// be created on the stack to coincide with the position where the
  /// parser enters the new scope, and this object's constructor will
  /// create that new scope. Similarly, once the object is destroyed
  /// the parser will exit the scope.
  class ParseScope {
    Parser *Self;
    ParseScope(const ParseScope &) = delete;
    void operator=(const ParseScope &) = delete;

  public:
    // ParseScope - Construct a new object to manage a scope in the
    // parser Self where the new Scope is created with the flags
    // ScopeFlags, but only when we aren't about to enter a compound statement.
    ParseScope(Parser *Self, unsigned ScopeFlags, bool EnteredScope = true,
               bool BeforeCompoundStmt = false)
        : Self(Self) {
      if (EnteredScope && !BeforeCompoundStmt)
        Self->EnterScope(ScopeFlags);
      else {
        if (BeforeCompoundStmt)
          Self->incrementMSManglingNumber();

        this->Self = nullptr;
      }
    }

    // Exit - Exit the scope associated with this object now, rather
    // than waiting until the object is destroyed.
    void Exit() {
      if (Self) {
        Self->ExitScope();
        Self = nullptr;
      }
    }

    ~ParseScope() { Exit(); }
  };

  /// Introduces zero or more scopes for parsing. The scopes will all be exited
  /// when the object is destroyed.
  class MultiParseScope {
    Parser &Self;
    unsigned NumScopes = 0;

    MultiParseScope(const MultiParseScope &) = delete;

  public:
    MultiParseScope(Parser &Self) : Self(Self) {}
    void Enter(unsigned ScopeFlags) {
      Self.EnterScope(ScopeFlags);
      ++NumScopes;
    }
    void Exit() {
      while (NumScopes) {
        Self.ExitScope();
        --NumScopes;
      }
    }
    ~MultiParseScope() { Exit(); }
  };

  /// EnterScope - Start a new scope.
  void EnterScope(unsigned ScopeFlags);

  /// ExitScope - Pop a scope off the scope stack.
  void ExitScope();

  //===--------------------------------------------------------------------===//
  // Diagnostic Emission and Error recovery.

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);
  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID);
  DiagnosticBuilder Diag(unsigned DiagID) { return Diag(Tok, DiagID); }

  DiagnosticBuilder DiagCompat(SourceLocation Loc, unsigned CompatDiagId);
  DiagnosticBuilder DiagCompat(const Token &Tok, unsigned CompatDiagId);
  DiagnosticBuilder DiagCompat(unsigned CompatDiagId) {
    return DiagCompat(Tok, CompatDiagId);
  }

  /// Control flags for SkipUntil functions.
  enum SkipUntilFlags {
    StopAtSemi = 1 << 0, ///< Stop skipping at semicolon
    /// Stop skipping at specified token, but don't skip the token itself
    StopBeforeMatch = 1 << 1,
    StopAtCodeCompletion = 1 << 2 ///< Stop at code completion
  };

  friend constexpr SkipUntilFlags operator|(SkipUntilFlags L,
                                            SkipUntilFlags R) {
    return static_cast<SkipUntilFlags>(static_cast<unsigned>(L) |
                                       static_cast<unsigned>(R));
  }

  /// SkipUntil - Read tokens until we get to the specified token, then consume
  /// it (unless StopBeforeMatch is specified).  Because we cannot guarantee
  /// that the token will ever occur, this skips to the next token, or to some
  /// likely good stopping point.  If Flags has StopAtSemi flag, skipping will
  /// stop at a ';' character. Balances (), [], and {} delimiter tokens while
  /// skipping.
  ///
  /// If SkipUntil finds the specified token, it returns true, otherwise it
  /// returns false.
  bool SkipUntil(tok::TokenKind T,
                 SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
    return SkipUntil(llvm::ArrayRef(T), Flags);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2,
                 SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
    tok::TokenKind TokArray[] = {T1, T2};
    return SkipUntil(TokArray, Flags);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, tok::TokenKind T3,
                 SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0)) {
    tok::TokenKind TokArray[] = {T1, T2, T3};
    return SkipUntil(TokArray, Flags);
  }

  /// SkipUntil - Read tokens until we get to the specified token, then consume
  /// it (unless no flag StopBeforeMatch).  Because we cannot guarantee that the
  /// token will ever occur, this skips to the next token, or to some likely
  /// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
  /// character.
  ///
  /// If SkipUntil finds the specified token, it returns true, otherwise it
  /// returns false.
  bool SkipUntil(ArrayRef<tok::TokenKind> Toks,
                 SkipUntilFlags Flags = static_cast<SkipUntilFlags>(0));

private:
  Preprocessor &PP;

  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  Token Tok;

  // PrevTokLocation - The location of the token we previously
  // consumed. This token is used for diagnostics where we expected to
  // see a token following another token (e.g., the ';' at the end of
  // a statement).
  SourceLocation PrevTokLocation;

  /// Tracks an expected type for the current token when parsing an expression.
  /// Used by code completion for ranking.
  PreferredTypeBuilder PreferredType;

  unsigned short ParenCount = 0, BracketCount = 0, BraceCount = 0;
  unsigned short MisplacedModuleBeginCount = 0;

  /// Actions - These are the callbacks we invoke as we parse various constructs
  /// in the file.
  Sema &Actions;

  DiagnosticsEngine &Diags;

  StackExhaustionHandler StackHandler;

  /// ScopeCache - Cache scopes to reduce malloc traffic.
  static constexpr int ScopeCacheSize = 16;
  unsigned NumCachedScopes;
  Scope *ScopeCache[ScopeCacheSize];

  /// Identifiers used for SEH handling in Borland. These are only
  /// allowed in particular circumstances
  // __except block
  IdentifierInfo *Ident__exception_code, *Ident___exception_code,
      *Ident_GetExceptionCode;
  // __except filter expression
  IdentifierInfo *Ident__exception_info, *Ident___exception_info,
      *Ident_GetExceptionInfo;
  // __finally
  IdentifierInfo *Ident__abnormal_termination, *Ident___abnormal_termination,
      *Ident_AbnormalTermination;

  /// Contextual keywords for Microsoft extensions.
  IdentifierInfo *Ident__except;

  // C++2a contextual keywords.
  mutable IdentifierInfo *Ident_import;
  mutable IdentifierInfo *Ident_module;

  std::unique_ptr<CommentHandler> CommentSemaHandler;

  /// Gets set to true after calling ProduceSignatureHelp, it is for a
  /// workaround to make sure ProduceSignatureHelp is only called at the deepest
  /// function call.
  bool CalledSignatureHelp = false;

  IdentifierInfo *getSEHExceptKeyword();

  /// Whether to skip parsing of function bodies.
  ///
  /// This option can be used, for example, to speed up searches for
  /// declarations/definitions when indexing.
  bool SkipFunctionBodies;

  //===--------------------------------------------------------------------===//
  // Low-Level token peeking and consumption methods.
  //

  /// isTokenParen - Return true if the cur token is '(' or ')'.
  bool isTokenParen() const { return Tok.isOneOf(tok::l_paren, tok::r_paren); }
  /// isTokenBracket - Return true if the cur token is '[' or ']'.
  bool isTokenBracket() const {
    return Tok.isOneOf(tok::l_square, tok::r_square);
  }
  /// isTokenBrace - Return true if the cur token is '{' or '}'.
  bool isTokenBrace() const { return Tok.isOneOf(tok::l_brace, tok::r_brace); }
  /// isTokenStringLiteral - True if this token is a string-literal.
  bool isTokenStringLiteral() const {
    return tok::isStringLiteral(Tok.getKind());
  }
  /// isTokenSpecial - True if this token requires special consumption methods.
  bool isTokenSpecial() const {
    return isTokenStringLiteral() || isTokenParen() || isTokenBracket() ||
           isTokenBrace() || Tok.is(tok::code_completion) || Tok.isAnnotation();
  }

  /// Returns true if the current token is '=' or is a type of '='.
  /// For typos, give a fixit to '='
  bool isTokenEqualOrEqualTypo();

  /// Return the current token to the token stream and make the given
  /// token the current token.
  void UnconsumeToken(Token &Consumed) {
    Token Next = Tok;
    PP.EnterToken(Consumed, /*IsReinject*/ true);
    PP.Lex(Tok);
    PP.EnterToken(Next, /*IsReinject*/ true);
  }

  SourceLocation ConsumeAnnotationToken() {
    assert(Tok.isAnnotation() && "wrong consume method");
    SourceLocation Loc = Tok.getLocation();
    PrevTokLocation = Tok.getAnnotationEndLoc();
    PP.Lex(Tok);
    return Loc;
  }

  /// ConsumeParen - This consume method keeps the paren count up-to-date.
  ///
  SourceLocation ConsumeParen() {
    assert(isTokenParen() && "wrong consume method");
    if (Tok.getKind() == tok::l_paren)
      ++ParenCount;
    else if (ParenCount) {
      AngleBrackets.clear(*this);
      --ParenCount; // Don't let unbalanced )'s drive the count negative.
    }
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBracket - This consume method keeps the bracket count up-to-date.
  ///
  SourceLocation ConsumeBracket() {
    assert(isTokenBracket() && "wrong consume method");
    if (Tok.getKind() == tok::l_square)
      ++BracketCount;
    else if (BracketCount) {
      AngleBrackets.clear(*this);
      --BracketCount; // Don't let unbalanced ]'s drive the count negative.
    }

    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeBrace - This consume method keeps the brace count up-to-date.
  ///
  SourceLocation ConsumeBrace() {
    assert(isTokenBrace() && "wrong consume method");
    if (Tok.getKind() == tok::l_brace)
      ++BraceCount;
    else if (BraceCount) {
      AngleBrackets.clear(*this);
      --BraceCount; // Don't let unbalanced }'s drive the count negative.
    }

    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// ConsumeStringToken - Consume the current 'peek token', lexing a new one
  /// and returning the token kind.  This method is specific to strings, as it
  /// handles string literal concatenation, as per C99 5.1.1.2, translation
  /// phase #6.
  SourceLocation ConsumeStringToken() {
    assert(isTokenStringLiteral() &&
           "Should only consume string literals with this method");
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// Consume the current code-completion token.
  ///
  /// This routine can be called to consume the code-completion token and
  /// continue processing in special cases where \c cutOffParsing() isn't
  /// desired, such as token caching or completion with lookahead.
  SourceLocation ConsumeCodeCompletionToken() {
    assert(Tok.is(tok::code_completion));
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  /// When we are consuming a code-completion token without having matched
  /// specific position in the grammar, provide code-completion results based
  /// on context.
  ///
  /// \returns the source location of the code-completion token.
  SourceLocation handleUnexpectedCodeCompletionToken();

  /// Abruptly cut off parsing; mainly used when we have reached the
  /// code-completion point.
  void cutOffParsing() {
    if (PP.isCodeCompletionEnabled())
      PP.setCodeCompletionReached();
    // Cut off parsing by acting as if we reached the end-of-file.
    Tok.setKind(tok::eof);
  }

  /// Determine if we're at the end of the file or at a transition
  /// between modules.
  bool isEofOrEom() {
    tok::TokenKind Kind = Tok.getKind();
    return Kind == tok::eof || Kind == tok::annot_module_begin ||
           Kind == tok::annot_module_end || Kind == tok::annot_module_include ||
           Kind == tok::annot_repl_input_end;
  }

  static void setTypeAnnotation(Token &Tok, TypeResult T) {
    assert((T.isInvalid() || T.get()) &&
           "produced a valid-but-null type annotation?");
    Tok.setAnnotationValue(T.isInvalid() ? nullptr : T.get().getAsOpaquePtr());
  }

  static NamedDecl *getNonTypeAnnotation(const Token &Tok) {
    return static_cast<NamedDecl *>(Tok.getAnnotationValue());
  }

  static void setNonTypeAnnotation(Token &Tok, NamedDecl *ND) {
    Tok.setAnnotationValue(ND);
  }

  static IdentifierInfo *getIdentifierAnnotation(const Token &Tok) {
    return static_cast<IdentifierInfo *>(Tok.getAnnotationValue());
  }

  static void setIdentifierAnnotation(Token &Tok, IdentifierInfo *ND) {
    Tok.setAnnotationValue(ND);
  }

  /// Read an already-translated primary expression out of an annotation
  /// token.
  static ExprResult getExprAnnotation(const Token &Tok) {
    return ExprResult::getFromOpaquePointer(Tok.getAnnotationValue());
  }

  /// Set the primary expression corresponding to the given annotation
  /// token.
  static void setExprAnnotation(Token &Tok, ExprResult ER) {
    Tok.setAnnotationValue(ER.getAsOpaquePointer());
  }

  /// Attempt to classify the name at the current token position. This may
  /// form a type, scope or primary expression annotation, or replace the token
  /// with a typo-corrected keyword. This is only appropriate when the current
  /// name must refer to an entity which has already been declared.
  ///
  /// \param CCC Indicates how to perform typo-correction for this name. If
  ///        NULL, no typo correction will be performed.
  /// \param AllowImplicitTypename Whether we are in a context where a dependent
  ///        nested-name-specifier without typename is treated as a type (e.g.
  ///        T::type).
  AnnotatedNameKind
  TryAnnotateName(CorrectionCandidateCallback *CCC = nullptr,
                  ImplicitTypenameContext AllowImplicitTypename =
                      ImplicitTypenameContext::No);

  /// Push a tok::annot_cxxscope token onto the token stream.
  void AnnotateScopeToken(CXXScopeSpec &SS, bool IsNewAnnotation);

  /// TryKeywordIdentFallback - For compatibility with system headers using
  /// keywords as identifiers, attempt to convert the current token to an
  /// identifier and optionally disable the keyword for the remainder of the
  /// translation unit. This returns false if the token was not replaced,
  /// otherwise emits a diagnostic and returns true.
  bool TryKeywordIdentFallback(bool DisableKeyword);

  /// Get the TemplateIdAnnotation from the token and put it in the
  /// cleanup pool so that it gets destroyed when parsing the current top level
  /// declaration is finished.
  TemplateIdAnnotation *takeTemplateIdAnnotation(const Token &tok);

  /// ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
  /// input.  If so, it is consumed and false is returned.
  ///
  /// If a trivial punctuator misspelling is encountered, a FixIt error
  /// diagnostic is issued and false is returned after recovery.
  ///
  /// If the input is malformed, this emits the specified diagnostic and true is
  /// returned.
  bool ExpectAndConsume(tok::TokenKind ExpectedTok,
                        unsigned Diag = diag::err_expected,
                        StringRef DiagMsg = "");

  /// The parser expects a semicolon and, if present, will consume it.
  ///
  /// If the next token is not a semicolon, this emits the specified diagnostic,
  /// or, if there's just some closing-delimiter noise (e.g., ')' or ']') prior
  /// to the semicolon, consumes that extra token.
  bool ExpectAndConsumeSemi(unsigned DiagID, StringRef TokenUsed = "");

  /// Consume any extra semi-colons until the end of the line.
  void ConsumeExtraSemi(ExtraSemiKind Kind, DeclSpec::TST T = TST_unspecified);

  /// Return false if the next token is an identifier. An 'expected identifier'
  /// error is emitted otherwise.
  ///
  /// The parser tries to recover from the error by checking if the next token
  /// is a C++ keyword when parsing Objective-C++. Return false if the recovery
  /// was successful.
  bool expectIdentifier();

  /// Kinds of compound pseudo-tokens formed by a sequence of two real tokens.
  enum class CompoundToken {
    /// A '(' '{' beginning a statement-expression.
    StmtExprBegin,
    /// A '}' ')' ending a statement-expression.
    StmtExprEnd,
    /// A '[' '[' beginning a C++11 or C23 attribute.
    AttrBegin,
    /// A ']' ']' ending a C++11 or C23 attribute.
    AttrEnd,
    /// A '::' '*' forming a C++ pointer-to-member declaration.
    MemberPtr,
  };

  /// Check that a compound operator was written in a "sensible" way, and warn
  /// if not.
  void checkCompoundToken(SourceLocation FirstTokLoc,
                          tok::TokenKind FirstTokKind, CompoundToken Op);

  void diagnoseUseOfC11Keyword(const Token &Tok);

  /// RAII object used to modify the scope flags for the current scope.
  class ParseScopeFlags {
    Scope *CurScope;
    unsigned OldFlags = 0;
    ParseScopeFlags(const ParseScopeFlags &) = delete;
    void operator=(const ParseScopeFlags &) = delete;

  public:
    /// Set the flags for the current scope to ScopeFlags. If ManageFlags is
    /// false, this object does nothing.
    ParseScopeFlags(Parser *Self, unsigned ScopeFlags, bool ManageFlags = true);

    /// Restore the flags for the current scope to what they were before this
    /// object overrode them.
    ~ParseScopeFlags();
  };

  /// Emits a diagnostic suggesting parentheses surrounding a
  /// given range.
  ///
  /// \param Loc The location where we'll emit the diagnostic.
  /// \param DK The kind of diagnostic to emit.
  /// \param ParenRange Source range enclosing code that should be
  /// parenthesized.
  void SuggestParentheses(SourceLocation Loc, unsigned DK,
                          SourceRange ParenRange);

  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.

  /// ParseExternalDeclaration:
  ///
  /// The `Attrs` that are passed in are C++11 attributes and appertain to the
  /// declaration.
  ///
  /// \verbatim
  ///       external-declaration: [C99 6.9], declaration: [C++ dcl.dcl]
  ///         function-definition
  ///         declaration
  /// [GNU]   asm-definition
  /// [GNU]   __extension__ external-declaration
  /// [OBJC]  objc-class-definition
  /// [OBJC]  objc-class-declaration
  /// [OBJC]  objc-alias-declaration
  /// [OBJC]  objc-protocol-definition
  /// [OBJC]  objc-method-definition
  /// [OBJC]  @end
  /// [C++]   linkage-specification
  /// [GNU] asm-definition:
  ///         simple-asm-expr ';'
  /// [C++11] empty-declaration
  /// [C++11] attribute-declaration
  ///
  /// [C++11] empty-declaration:
  ///           ';'
  ///
  /// [C++0x/GNU] 'extern' 'template' declaration
  ///
  /// [C++20] module-import-declaration
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseExternalDeclaration(ParsedAttributes &DeclAttrs,
                                          ParsedAttributes &DeclSpecAttrs,
                                          ParsingDeclSpec *DS = nullptr);

  /// Determine whether the current token, if it occurs after a
  /// declarator, continues a declaration or declaration list.
  bool isDeclarationAfterDeclarator();

  /// Determine whether the current token, if it occurs after a
  /// declarator, indicates the start of a function definition.
  bool isStartOfFunctionDefinition(const ParsingDeclarator &Declarator);

  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(
      ParsedAttributes &DeclAttrs, ParsedAttributes &DeclSpecAttrs,
      ParsingDeclSpec *DS = nullptr, AccessSpecifier AS = AS_none);

  /// Parse either a function-definition or a declaration.  We can't tell which
  /// we have until we read up to the compound-statement in function-definition.
  /// TemplateParams, if non-NULL, provides the template parameters when we're
  /// parsing a C++ template-declaration.
  ///
  /// \verbatim
  ///       function-definition: [C99 6.9.1]
  ///         decl-specs      declarator declaration-list[opt] compound-statement
  /// [C90] function-definition: [C99 6.7.1] - implicit int result
  /// [C90]   decl-specs[opt] declarator declaration-list[opt] compound-statement
  ///
  ///       declaration: [C99 6.7]
  ///         declaration-specifiers init-declarator-list[opt] ';'
  /// [!C99]  init-declarator-list ';'                   [TODO: warn in c99 mode]
  /// [OMP]   threadprivate-directive
  /// [OMP]   allocate-directive                         [TODO]
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseDeclOrFunctionDefInternal(ParsedAttributes &Attrs,
                                                ParsedAttributes &DeclSpecAttrs,
                                                ParsingDeclSpec &DS,
                                                AccessSpecifier AS);

  void SkipFunctionBody();

  struct ParsedTemplateInfo;
  class LateParsedAttrList;

  /// ParseFunctionDefinition - We parsed and verified that the specified
  /// Declarator is well formed.  If this is a K&R-style function, read the
  /// parameters declaration-list, then start the compound-statement.
  ///
  /// \verbatim
  ///       function-definition: [C99 6.9.1]
  ///         decl-specs      declarator declaration-list[opt] compound-statement
  /// [C90] function-definition: [C99 6.7.1] - implicit int result
  /// [C90]   decl-specs[opt] declarator declaration-list[opt] compound-statement
  /// [C++] function-definition: [C++ 8.4]
  ///         decl-specifier-seq[opt] declarator ctor-initializer[opt]
  ///         function-body
  /// [C++] function-definition: [C++ 8.4]
  ///         decl-specifier-seq[opt] declarator function-try-block
  /// \endverbatim
  ///
  Decl *ParseFunctionDefinition(
      ParsingDeclarator &D,
      const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
      LateParsedAttrList *LateParsedAttrs = nullptr);

  /// ParseKNRParamDeclarations - Parse 'declaration-list[opt]' which provides
  /// types for a function with a K&R-style identifier list for arguments.
  void ParseKNRParamDeclarations(Declarator &D);

  /// ParseSimpleAsm
  ///
  /// \verbatim
  /// [GNU] simple-asm-expr:
  ///         'asm' '(' asm-string-literal ')'
  /// \endverbatim
  ///
  /// EndLoc is filled with the location of the last token of the simple-asm.
  ExprResult ParseSimpleAsm(bool ForAsmLabel, SourceLocation *EndLoc);

  /// ParseAsmStringLiteral - This is just a normal string-literal, but is not
  /// allowed to be a wide string, and is not subject to character translation.
  /// Unlike GCC, we also diagnose an empty string literal when parsing for an
  /// asm label as opposed to an asm statement, because such a construct does
  /// not behave well.
  ///
  /// \verbatim
  /// [GNU] asm-string-literal:
  ///         string-literal
  /// \endverbatim
  ///
  ExprResult ParseAsmStringLiteral(bool ForAsmLabel);

  /// Describes the condition of a Microsoft __if_exists or
  /// __if_not_exists block.
  struct IfExistsCondition {
    /// The location of the initial keyword.
    SourceLocation KeywordLoc;
    /// Whether this is an __if_exists block (rather than an
    /// __if_not_exists block).
    bool IsIfExists;

    /// Nested-name-specifier preceding the name.
    CXXScopeSpec SS;

    /// The name we're looking for.
    UnqualifiedId Name;

    /// The behavior of this __if_exists or __if_not_exists block
    /// should.
    IfExistsBehavior Behavior;
  };

  bool ParseMicrosoftIfExistsCondition(IfExistsCondition &Result);
  void ParseMicrosoftIfExistsExternalDeclaration();

  //===--------------------------------------------------------------------===//
  // Modules

  /// Parse a declaration beginning with the 'module' keyword or C++20
  /// context-sensitive keyword (optionally preceded by 'export').
  ///
  /// \verbatim
  ///   module-declaration:   [C++20]
  ///     'export'[opt] 'module' module-name attribute-specifier-seq[opt] ';'
  ///
  ///   global-module-fragment:  [C++2a]
  ///     'module' ';' top-level-declaration-seq[opt]
  ///   module-declaration:      [C++2a]
  ///     'export'[opt] 'module' module-name module-partition[opt]
  ///            attribute-specifier-seq[opt] ';'
  ///   private-module-fragment: [C++2a]
  ///     'module' ':' 'private' ';' top-level-declaration-seq[opt]
  /// \endverbatim
  DeclGroupPtrTy ParseModuleDecl(Sema::ModuleImportState &ImportState);

  /// Parse a module import declaration. This is essentially the same for
  /// Objective-C and C++20 except for the leading '@' (in ObjC) and the
  /// trailing optional attributes (in C++).
  ///
  /// \verbatim
  /// [ObjC]  @import declaration:
  ///           '@' 'import' module-name ';'
  /// [ModTS] module-import-declaration:
  ///           'import' module-name attribute-specifier-seq[opt] ';'
  /// [C++20] module-import-declaration:
  ///           'export'[opt] 'import' module-name
  ///                   attribute-specifier-seq[opt] ';'
  ///           'export'[opt] 'import' module-partition
  ///                   attribute-specifier-seq[opt] ';'
  ///           'export'[opt] 'import' header-name
  ///                   attribute-specifier-seq[opt] ';'
  /// \endverbatim
  Decl *ParseModuleImport(SourceLocation AtLoc,
                          Sema::ModuleImportState &ImportState);

  /// Try recover parser when module annotation appears where it must not
  /// be found.
  /// \returns false if the recover was successful and parsing may be continued,
  /// or true if parser must bail out to top level and handle the token there.
  bool parseMisplacedModuleImport();

  bool tryParseMisplacedModuleImport() {
    tok::TokenKind Kind = Tok.getKind();
    if (Kind == tok::annot_module_begin || Kind == tok::annot_module_end ||
        Kind == tok::annot_module_include)
      return parseMisplacedModuleImport();
    return false;
  }

  /// Parse a C++ / Objective-C module name (both forms use the same
  /// grammar).
  ///
  /// \verbatim
  ///         module-name:
  ///           module-name-qualifier[opt] identifier
  ///         module-name-qualifier:
  ///           module-name-qualifier[opt] identifier '.'
  /// \endverbatim
  bool ParseModuleName(SourceLocation UseLoc,
                       SmallVectorImpl<IdentifierLoc> &Path, bool IsImport);

  //===--------------------------------------------------------------------===//
  // Preprocessor code-completion pass-through
  void CodeCompleteDirective(bool InConditional) override;
  void CodeCompleteInConditionalExclusion() override;
  void CodeCompleteMacroName(bool IsDefinition) override;
  void CodeCompletePreprocessorExpression() override;
  void CodeCompleteMacroArgument(IdentifierInfo *Macro, MacroInfo *MacroInfo,
                                 unsigned ArgumentIndex) override;
  void CodeCompleteIncludedFile(llvm::StringRef Dir, bool IsAngled) override;
  void CodeCompleteNaturalLanguage() override;

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name C++ Class Inline Methods
  /// Implementations are in ParseCXXInlineMethods.cpp
  ///@{

private:
  struct ParsingClass;

  /// [class.mem]p1: "... the class is regarded as complete within
  /// - function bodies
  /// - default arguments
  /// - exception-specifications (TODO: C++0x)
  /// - and brace-or-equal-initializers for non-static data members
  /// (including such things in nested classes)."
  /// LateParsedDeclarations build the tree of those elements so they can
  /// be parsed after parsing the top-level class.
  class LateParsedDeclaration {
  public:
    virtual ~LateParsedDeclaration();

    virtual void ParseLexedMethodDeclarations();
    virtual void ParseLexedMemberInitializers();
    virtual void ParseLexedMethodDefs();
    virtual void ParseLexedAttributes();
    virtual void ParseLexedPragmas();
  };

  /// Inner node of the LateParsedDeclaration tree that parses
  /// all its members recursively.
  class LateParsedClass : public LateParsedDeclaration {
  public:
    LateParsedClass(Parser *P, ParsingClass *C);
    ~LateParsedClass() override;

    void ParseLexedMethodDeclarations() override;
    void ParseLexedMemberInitializers() override;
    void ParseLexedMethodDefs() override;
    void ParseLexedAttributes() override;
    void ParseLexedPragmas() override;

    // Delete copy constructor and copy assignment operator.
    LateParsedClass(const LateParsedClass &) = delete;
    LateParsedClass &operator=(const LateParsedClass &) = delete;

  private:
    Parser *Self;
    ParsingClass *Class;
  };

  /// Contains the lexed tokens of an attribute with arguments that
  /// may reference member variables and so need to be parsed at the
  /// end of the class declaration after parsing all other member
  /// member declarations.
  /// FIXME: Perhaps we should change the name of LateParsedDeclaration to
  /// LateParsedTokens.
  struct LateParsedAttribute : public LateParsedDeclaration {
    Parser *Self;
    CachedTokens Toks;
    IdentifierInfo &AttrName;
    IdentifierInfo *MacroII = nullptr;
    SourceLocation AttrNameLoc;
    SmallVector<Decl *, 2> Decls;

    explicit LateParsedAttribute(Parser *P, IdentifierInfo &Name,
                                 SourceLocation Loc)
        : Self(P), AttrName(Name), AttrNameLoc(Loc) {}

    void ParseLexedAttributes() override;

    void addDecl(Decl *D) { Decls.push_back(D); }
  };

  /// Contains the lexed tokens of a pragma with arguments that
  /// may reference member variables and so need to be parsed at the
  /// end of the class declaration after parsing all other member
  /// member declarations.
  class LateParsedPragma : public LateParsedDeclaration {
    Parser *Self = nullptr;
    AccessSpecifier AS = AS_none;
    CachedTokens Toks;

  public:
    explicit LateParsedPragma(Parser *P, AccessSpecifier AS)
        : Self(P), AS(AS) {}

    void takeToks(CachedTokens &Cached) { Toks.swap(Cached); }
    const CachedTokens &toks() const { return Toks; }
    AccessSpecifier getAccessSpecifier() const { return AS; }

    void ParseLexedPragmas() override;
  };

  // A list of late-parsed attributes.  Used by ParseGNUAttributes.
  class LateParsedAttrList : public SmallVector<LateParsedAttribute *, 2> {
  public:
    LateParsedAttrList(bool PSoon = false,
                       bool LateAttrParseExperimentalExtOnly = false)
        : ParseSoon(PSoon),
          LateAttrParseExperimentalExtOnly(LateAttrParseExperimentalExtOnly) {}

    bool parseSoon() { return ParseSoon; }
    /// returns true iff the attribute to be parsed should only be late parsed
    /// if it is annotated with `LateAttrParseExperimentalExt`
    bool lateAttrParseExperimentalExtOnly() {
      return LateAttrParseExperimentalExtOnly;
    }

  private:
    bool ParseSoon; // Are we planning to parse these shortly after creation?
    bool LateAttrParseExperimentalExtOnly;
  };

  /// Contains the lexed tokens of a member function definition
  /// which needs to be parsed at the end of the class declaration
  /// after parsing all other member declarations.
  struct LexedMethod : public LateParsedDeclaration {
    Parser *Self;
    Decl *D;
    CachedTokens Toks;

    explicit LexedMethod(Parser *P, Decl *MD) : Self(P), D(MD) {}

    void ParseLexedMethodDefs() override;
  };

  /// LateParsedDefaultArgument - Keeps track of a parameter that may
  /// have a default argument that cannot be parsed yet because it
  /// occurs within a member function declaration inside the class
  /// (C++ [class.mem]p2).
  struct LateParsedDefaultArgument {
    explicit LateParsedDefaultArgument(
        Decl *P, std::unique_ptr<CachedTokens> Toks = nullptr)
        : Param(P), Toks(std::move(Toks)) {}

    /// Param - The parameter declaration for this parameter.
    Decl *Param;

    /// Toks - The sequence of tokens that comprises the default
    /// argument expression, not including the '=' or the terminating
    /// ')' or ','. This will be NULL for parameters that have no
    /// default argument.
    std::unique_ptr<CachedTokens> Toks;
  };

  /// LateParsedMethodDeclaration - A method declaration inside a class that
  /// contains at least one entity whose parsing needs to be delayed
  /// until the class itself is completely-defined, such as a default
  /// argument (C++ [class.mem]p2).
  struct LateParsedMethodDeclaration : public LateParsedDeclaration {
    explicit LateParsedMethodDeclaration(Parser *P, Decl *M)
        : Self(P), Method(M), ExceptionSpecTokens(nullptr) {}

    void ParseLexedMethodDeclarations() override;

    Parser *Self;

    /// Method - The method declaration.
    Decl *Method;

    /// DefaultArgs - Contains the parameters of the function and
    /// their default arguments. At least one of the parameters will
    /// have a default argument, but all of the parameters of the
    /// method will be stored so that they can be reintroduced into
    /// scope at the appropriate times.
    SmallVector<LateParsedDefaultArgument, 8> DefaultArgs;

    /// The set of tokens that make up an exception-specification that
    /// has not yet been parsed.
    CachedTokens *ExceptionSpecTokens;
  };

  /// LateParsedMemberInitializer - An initializer for a non-static class data
  /// member whose parsing must to be delayed until the class is completely
  /// defined (C++11 [class.mem]p2).
  struct LateParsedMemberInitializer : public LateParsedDeclaration {
    LateParsedMemberInitializer(Parser *P, Decl *FD) : Self(P), Field(FD) {}

    void ParseLexedMemberInitializers() override;

    Parser *Self;

    /// Field - The field declaration.
    Decl *Field;

    /// CachedTokens - The sequence of tokens that comprises the initializer,
    /// including any leading '='.
    CachedTokens Toks;
  };

  /// LateParsedDeclarationsContainer - During parsing of a top (non-nested)
  /// C++ class, its method declarations that contain parts that won't be
  /// parsed until after the definition is completed (C++ [class.mem]p2),
  /// the method declarations and possibly attached inline definitions
  /// will be stored here with the tokens that will be parsed to create those
  /// entities.
  typedef SmallVector<LateParsedDeclaration *, 2>
      LateParsedDeclarationsContainer;

  /// Utility to re-enter a possibly-templated scope while parsing its
  /// late-parsed components.
  struct ReenterTemplateScopeRAII;

  /// Utility to re-enter a class scope while parsing its late-parsed
  /// components.
  struct ReenterClassScopeRAII;

  /// ParseCXXInlineMethodDef - We parsed and verified that the specified
  /// Declarator is a well formed C++ inline method definition. Now lex its body
  /// and store its tokens for parsing after the C++ class is complete.
  NamedDecl *ParseCXXInlineMethodDef(AccessSpecifier AS,
                                     const ParsedAttributesView &AccessAttrs,
                                     ParsingDeclarator &D,
                                     const ParsedTemplateInfo &TemplateInfo,
                                     const VirtSpecifiers &VS,
                                     SourceLocation PureSpecLoc);

  /// Parse the optional ("message") part of a deleted-function-body.
  StringLiteral *ParseCXXDeletedFunctionMessage();

  /// If we've encountered '= delete' in a context where it is ill-formed, such
  /// as in the declaration of a non-function, also skip the ("message") part if
  /// it is present to avoid issuing further diagnostics.
  void SkipDeletedFunctionBody();

  /// ParseCXXNonStaticMemberInitializer - We parsed and verified that the
  /// specified Declarator is a well formed C++ non-static data member
  /// declaration. Now lex its initializer and store its tokens for parsing
  /// after the class is complete.
  void ParseCXXNonStaticMemberInitializer(Decl *VarD);

  /// Wrapper class which calls ParseLexedAttribute, after setting up the
  /// scope appropriately.
  void ParseLexedAttributes(ParsingClass &Class);

  /// Parse all attributes in LAs, and attach them to Decl D.
  void ParseLexedAttributeList(LateParsedAttrList &LAs, Decl *D,
                               bool EnterScope, bool OnDefinition);

  /// Finish parsing an attribute for which parsing was delayed.
  /// This will be called at the end of parsing a class declaration
  /// for each LateParsedAttribute. We consume the saved tokens and
  /// create an attribute with the arguments filled in. We add this
  /// to the Attribute list for the decl.
  void ParseLexedAttribute(LateParsedAttribute &LA, bool EnterScope,
                           bool OnDefinition);

  /// ParseLexedMethodDeclarations - We finished parsing the member
  /// specification of a top (non-nested) C++ class. Now go over the
  /// stack of method declarations with some parts for which parsing was
  /// delayed (such as default arguments) and parse them.
  void ParseLexedMethodDeclarations(ParsingClass &Class);
  void ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM);

  /// ParseLexedMethodDefs - We finished parsing the member specification of a
  /// top (non-nested) C++ class. Now go over the stack of lexed methods that
  /// were collected during its parsing and parse them all.
  void ParseLexedMethodDefs(ParsingClass &Class);
  void ParseLexedMethodDef(LexedMethod &LM);

  /// ParseLexedMemberInitializers - We finished parsing the member
  /// specification of a top (non-nested) C++ class. Now go over the stack of
  /// lexed data member initializers that were collected during its parsing and
  /// parse them all.
  void ParseLexedMemberInitializers(ParsingClass &Class);
  void ParseLexedMemberInitializer(LateParsedMemberInitializer &MI);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Declarations
  /// Implementations are in ParseDecl.cpp
  ///@{

public:
  /// SkipMalformedDecl - Read tokens until we get to some likely good stopping
  /// point for skipping past a simple-declaration.
  ///
  /// Skip until we reach something which seems like a sensible place to pick
  /// up parsing after a malformed declaration. This will sometimes stop sooner
  /// than SkipUntil(tok::r_brace) would, but will never stop later.
  void SkipMalformedDecl();

  /// ParseTypeName
  /// \verbatim
  ///       type-name: [C99 6.7.6]
  ///         specifier-qualifier-list abstract-declarator[opt]
  /// \endverbatim
  ///
  /// Called type-id in C++.
  TypeResult
  ParseTypeName(SourceRange *Range = nullptr,
                DeclaratorContext Context = DeclaratorContext::TypeName,
                AccessSpecifier AS = AS_none, Decl **OwnedType = nullptr,
                ParsedAttributes *Attrs = nullptr);

private:
  /// Ident_vector, Ident_bool, Ident_Bool - cached IdentifierInfos for "vector"
  /// and "bool" fast comparison.  Only present if AltiVec or ZVector are
  /// enabled.
  IdentifierInfo *Ident_vector;
  IdentifierInfo *Ident_bool;
  IdentifierInfo *Ident_Bool;

  /// Ident_pixel - cached IdentifierInfos for "pixel" fast comparison.
  /// Only present if AltiVec enabled.
  IdentifierInfo *Ident_pixel;

  /// Identifier for "introduced".
  IdentifierInfo *Ident_introduced;

  /// Identifier for "deprecated".
  IdentifierInfo *Ident_deprecated;

  /// Identifier for "obsoleted".
  IdentifierInfo *Ident_obsoleted;

  /// Identifier for "unavailable".
  IdentifierInfo *Ident_unavailable;

  /// Identifier for "message".
  IdentifierInfo *Ident_message;

  /// Identifier for "strict".
  IdentifierInfo *Ident_strict;

  /// Identifier for "replacement".
  IdentifierInfo *Ident_replacement;

  /// Identifier for "environment".
  IdentifierInfo *Ident_environment;

  /// Identifiers used by the 'external_source_symbol' attribute.
  IdentifierInfo *Ident_language, *Ident_defined_in,
      *Ident_generated_declaration, *Ident_USR;

  /// Factory object for creating ParsedAttr objects.
  AttributeFactory AttrFactory;

  /// TryAltiVecToken - Check for context-sensitive AltiVec identifier tokens,
  /// replacing them with the non-context-sensitive keywords.  This returns
  /// true if the token was replaced.
  bool TryAltiVecToken(DeclSpec &DS, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID, bool &isInvalid) {
    if (!getLangOpts().AltiVec && !getLangOpts().ZVector)
      return false;

    if (Tok.getIdentifierInfo() != Ident_vector &&
        Tok.getIdentifierInfo() != Ident_bool &&
        Tok.getIdentifierInfo() != Ident_Bool &&
        (!getLangOpts().AltiVec || Tok.getIdentifierInfo() != Ident_pixel))
      return false;

    return TryAltiVecTokenOutOfLine(DS, Loc, PrevSpec, DiagID, isInvalid);
  }

  /// TryAltiVecVectorToken - Check for context-sensitive AltiVec vector
  /// identifier token, replacing it with the non-context-sensitive __vector.
  /// This returns true if the token was replaced.
  bool TryAltiVecVectorToken() {
    if ((!getLangOpts().AltiVec && !getLangOpts().ZVector) ||
        Tok.getIdentifierInfo() != Ident_vector)
      return false;
    return TryAltiVecVectorTokenOutOfLine();
  }

  /// TryAltiVecVectorTokenOutOfLine - Out of line body that should only be
  /// called from TryAltiVecVectorToken.
  bool TryAltiVecVectorTokenOutOfLine();
  bool TryAltiVecTokenOutOfLine(DeclSpec &DS, SourceLocation Loc,
                                const char *&PrevSpec, unsigned &DiagID,
                                bool &isInvalid);

  void ParseLexedCAttributeList(LateParsedAttrList &LA, bool EnterScope,
                                ParsedAttributes *OutAttrs = nullptr);

  /// Finish parsing an attribute for which parsing was delayed.
  /// This will be called at the end of parsing a class declaration
  /// for each LateParsedAttribute. We consume the saved tokens and
  /// create an attribute with the arguments filled in. We add this
  /// to the Attribute list for the decl.
  void ParseLexedCAttribute(LateParsedAttribute &LA, bool EnterScope,
                            ParsedAttributes *OutAttrs = nullptr);

  void ParseLexedPragmas(ParsingClass &Class);
  void ParseLexedPragma(LateParsedPragma &LP);

  /// Consume tokens and store them in the passed token container until
  /// we've passed the try keyword and constructor initializers and have
  /// consumed the opening brace of the function body. The opening brace will be
  /// consumed if and only if there was no error.
  ///
  /// \return True on error.
  bool ConsumeAndStoreFunctionPrologue(CachedTokens &Toks);

  /// ConsumeAndStoreInitializer - Consume and store the token at the passed
  /// token container until the end of the current initializer expression
  /// (either a default argument or an in-class initializer for a non-static
  /// data member).
  ///
  /// Returns \c true if we reached the end of something initializer-shaped,
  /// \c false if we bailed out.
  bool ConsumeAndStoreInitializer(CachedTokens &Toks, CachedInitKind CIK);

  /// Consume and store tokens from the '?' to the ':' in a conditional
  /// expression.
  bool ConsumeAndStoreConditional(CachedTokens &Toks);
  bool ConsumeAndStoreUntil(tok::TokenKind T1, CachedTokens &Toks,
                            bool StopAtSemi = true,
                            bool ConsumeFinalToken = true) {
    return ConsumeAndStoreUntil(T1, T1, Toks, StopAtSemi, ConsumeFinalToken);
  }

  /// ConsumeAndStoreUntil - Consume and store the token at the passed token
  /// container until the token 'T' is reached (which gets
  /// consumed/stored too, if ConsumeFinalToken).
  /// If StopAtSemi is true, then we will stop early at a ';' character.
  /// Returns true if token 'T1' or 'T2' was found.
  /// NOTE: This is a specialized version of Parser::SkipUntil.
  bool ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                            CachedTokens &Toks, bool StopAtSemi = true,
                            bool ConsumeFinalToken = true);

  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.

  /// A context for parsing declaration specifiers.  TODO: flesh this
  /// out, there are other significant restrictions on specifiers than
  /// would be best implemented in the parser.
  enum class DeclSpecContext {
    DSC_normal,         // normal context
    DSC_class,          // class context, enables 'friend'
    DSC_type_specifier, // C++ type-specifier-seq or C specifier-qualifier-list
    DSC_trailing, // C++11 trailing-type-specifier in a trailing return type
    DSC_alias_declaration,  // C++11 type-specifier-seq in an alias-declaration
    DSC_conv_operator,      // C++ type-specifier-seq in an conversion operator
    DSC_top_level,          // top-level/namespace declaration context
    DSC_template_param,     // template parameter context
    DSC_template_arg,       // template argument context
    DSC_template_type_arg,  // template type argument context
    DSC_objc_method_result, // ObjC method result context, enables
                            // 'instancetype'
    DSC_condition,          // condition declaration context
    DSC_association, // A _Generic selection expression's type association
    DSC_new,         // C++ new expression
  };

  /// Is this a context in which we are parsing just a type-specifier (or
  /// trailing-type-specifier)?
  static bool isTypeSpecifier(DeclSpecContext DSC) {
    switch (DSC) {
    case DeclSpecContext::DSC_normal:
    case DeclSpecContext::DSC_template_param:
    case DeclSpecContext::DSC_template_arg:
    case DeclSpecContext::DSC_class:
    case DeclSpecContext::DSC_top_level:
    case DeclSpecContext::DSC_objc_method_result:
    case DeclSpecContext::DSC_condition:
      return false;

    case DeclSpecContext::DSC_template_type_arg:
    case DeclSpecContext::DSC_type_specifier:
    case DeclSpecContext::DSC_conv_operator:
    case DeclSpecContext::DSC_trailing:
    case DeclSpecContext::DSC_alias_declaration:
    case DeclSpecContext::DSC_association:
    case DeclSpecContext::DSC_new:
      return true;
    }
    llvm_unreachable("Missing DeclSpecContext case");
  }

  /// Whether a defining-type-specifier is permitted in a given context.
  enum class AllowDefiningTypeSpec {
    /// The grammar doesn't allow a defining-type-specifier here, and we must
    /// not parse one (eg, because a '{' could mean something else).
    No,
    /// The grammar doesn't allow a defining-type-specifier here, but we permit
    /// one for error recovery purposes. Sema will reject.
    NoButErrorRecovery,
    /// The grammar allows a defining-type-specifier here, even though it's
    /// always invalid. Sema will reject.
    YesButInvalid,
    /// The grammar allows a defining-type-specifier here, and one can be valid.
    Yes
  };

  /// Is this a context in which we are parsing defining-type-specifiers (and
  /// so permit class and enum definitions in addition to non-defining class and
  /// enum elaborated-type-specifiers)?
  static AllowDefiningTypeSpec
  isDefiningTypeSpecifierContext(DeclSpecContext DSC, bool IsCPlusPlus) {
    switch (DSC) {
    case DeclSpecContext::DSC_normal:
    case DeclSpecContext::DSC_class:
    case DeclSpecContext::DSC_top_level:
    case DeclSpecContext::DSC_alias_declaration:
    case DeclSpecContext::DSC_objc_method_result:
      return AllowDefiningTypeSpec::Yes;

    case DeclSpecContext::DSC_condition:
    case DeclSpecContext::DSC_template_param:
      return AllowDefiningTypeSpec::YesButInvalid;

    case DeclSpecContext::DSC_template_type_arg:
    case DeclSpecContext::DSC_type_specifier:
      return AllowDefiningTypeSpec::NoButErrorRecovery;

    case DeclSpecContext::DSC_association:
      return IsCPlusPlus ? AllowDefiningTypeSpec::NoButErrorRecovery
                         : AllowDefiningTypeSpec::Yes;

    case DeclSpecContext::DSC_trailing:
    case DeclSpecContext::DSC_conv_operator:
    case DeclSpecContext::DSC_template_arg:
    case DeclSpecContext::DSC_new:
      return AllowDefiningTypeSpec::No;
    }
    llvm_unreachable("Missing DeclSpecContext case");
  }

  /// Is this a context in which an opaque-enum-declaration can appear?
  static bool isOpaqueEnumDeclarationContext(DeclSpecContext DSC) {
    switch (DSC) {
    case DeclSpecContext::DSC_normal:
    case DeclSpecContext::DSC_class:
    case DeclSpecContext::DSC_top_level:
      return true;

    case DeclSpecContext::DSC_alias_declaration:
    case DeclSpecContext::DSC_objc_method_result:
    case DeclSpecContext::DSC_condition:
    case DeclSpecContext::DSC_template_param:
    case DeclSpecContext::DSC_template_type_arg:
    case DeclSpecContext::DSC_type_specifier:
    case DeclSpecContext::DSC_trailing:
    case DeclSpecContext::DSC_association:
    case DeclSpecContext::DSC_conv_operator:
    case DeclSpecContext::DSC_template_arg:
    case DeclSpecContext::DSC_new:

      return false;
    }
    llvm_unreachable("Missing DeclSpecContext case");
  }

  /// Is this a context in which we can perform class template argument
  /// deduction?
  static bool isClassTemplateDeductionContext(DeclSpecContext DSC) {
    switch (DSC) {
    case DeclSpecContext::DSC_normal:
    case DeclSpecContext::DSC_template_param:
    case DeclSpecContext::DSC_template_arg:
    case DeclSpecContext::DSC_class:
    case DeclSpecContext::DSC_top_level:
    case DeclSpecContext::DSC_condition:
    case DeclSpecContext::DSC_type_specifier:
    case DeclSpecContext::DSC_association:
    case DeclSpecContext::DSC_conv_operator:
    case DeclSpecContext::DSC_new:
      return true;

    case DeclSpecContext::DSC_objc_method_result:
    case DeclSpecContext::DSC_template_type_arg:
    case DeclSpecContext::DSC_trailing:
    case DeclSpecContext::DSC_alias_declaration:
      return false;
    }
    llvm_unreachable("Missing DeclSpecContext case");
  }

  // Is this a context in which an implicit 'typename' is allowed?
  static ImplicitTypenameContext
  getImplicitTypenameContext(DeclSpecContext DSC) {
    switch (DSC) {
    case DeclSpecContext::DSC_class:
    case DeclSpecContext::DSC_top_level:
    case DeclSpecContext::DSC_type_specifier:
    case DeclSpecContext::DSC_template_type_arg:
    case DeclSpecContext::DSC_trailing:
    case DeclSpecContext::DSC_alias_declaration:
    case DeclSpecContext::DSC_template_param:
    case DeclSpecContext::DSC_new:
      return ImplicitTypenameContext::Yes;

    case DeclSpecContext::DSC_normal:
    case DeclSpecContext::DSC_objc_method_result:
    case DeclSpecContext::DSC_condition:
    case DeclSpecContext::DSC_template_arg:
    case DeclSpecContext::DSC_conv_operator:
    case DeclSpecContext::DSC_association:
      return ImplicitTypenameContext::No;
    }
    llvm_unreachable("Missing DeclSpecContext case");
  }

  /// Information on a C++0x for-range-initializer found while parsing a
  /// declaration which turns out to be a for-range-declaration.
  struct ForRangeInit {
    SourceLocation ColonLoc;
    ExprResult RangeExpr;
    SmallVector<MaterializeTemporaryExpr *, 8> LifetimeExtendTemps;
    bool ParsedForRangeDecl() { return !ColonLoc.isInvalid(); }
  };
  struct ForRangeInfo : ForRangeInit {
    StmtResult LoopVar;
  };

  /// ParseDeclaration - Parse a full 'declaration', which consists of
  /// declaration-specifiers, some number of declarators, and a semicolon.
  /// 'Context' should be a DeclaratorContext value.  This returns the
  /// location of the semicolon in DeclEnd.
  ///
  /// \verbatim
  ///       declaration: [C99 6.7]
  ///         block-declaration ->
  ///           simple-declaration
  ///           others                   [FIXME]
  /// [C++]   template-declaration
  /// [C++]   namespace-definition
  /// [C++]   using-directive
  /// [C++]   using-declaration
  /// [C++11/C11] static_assert-declaration
  ///         others... [FIXME]
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseDeclaration(DeclaratorContext Context,
                                  SourceLocation &DeclEnd,
                                  ParsedAttributes &DeclAttrs,
                                  ParsedAttributes &DeclSpecAttrs,
                                  SourceLocation *DeclSpecStart = nullptr);

  /// \verbatim
  ///       simple-declaration: [C99 6.7: declaration] [C++ 7p1: dcl.dcl]
  ///         declaration-specifiers init-declarator-list[opt] ';'
  /// [C++11] attribute-specifier-seq decl-specifier-seq[opt]
  ///             init-declarator-list ';'
  ///[C90/C++]init-declarator-list ';'                             [TODO]
  /// [OMP]   threadprivate-directive
  /// [OMP]   allocate-directive                                   [TODO]
  ///
  ///       for-range-declaration: [C++11 6.5p1: stmt.ranged]
  ///         attribute-specifier-seq[opt] type-specifier-seq declarator
  /// \endverbatim
  ///
  /// If RequireSemi is false, this does not check for a ';' at the end of the
  /// declaration.  If it is true, it checks for and eats it.
  ///
  /// If FRI is non-null, we might be parsing a for-range-declaration instead
  /// of a simple-declaration. If we find that we are, we also parse the
  /// for-range-initializer, and place it here.
  ///
  /// DeclSpecStart is used when decl-specifiers are parsed before parsing
  /// the Declaration. The SourceLocation for this Decl is set to
  /// DeclSpecStart if DeclSpecStart is non-null.
  DeclGroupPtrTy
  ParseSimpleDeclaration(DeclaratorContext Context, SourceLocation &DeclEnd,
                         ParsedAttributes &DeclAttrs,
                         ParsedAttributes &DeclSpecAttrs, bool RequireSemi,
                         ForRangeInit *FRI = nullptr,
                         SourceLocation *DeclSpecStart = nullptr);

  /// ParseDeclGroup - Having concluded that this is either a function
  /// definition or a group of object declarations, actually parse the
  /// result.
  ///
  /// Returns true if this might be the start of a declarator, or a common typo
  /// for a declarator.
  bool MightBeDeclarator(DeclaratorContext Context);
  DeclGroupPtrTy ParseDeclGroup(ParsingDeclSpec &DS, DeclaratorContext Context,
                                ParsedAttributes &Attrs,
                                ParsedTemplateInfo &TemplateInfo,
                                SourceLocation *DeclEnd = nullptr,
                                ForRangeInit *FRI = nullptr);

  /// Parse 'declaration' after parsing 'declaration-specifiers
  /// declarator'. This method parses the remainder of the declaration
  /// (including any attributes or initializer, among other things) and
  /// finalizes the declaration.
  ///
  /// \verbatim
  ///       init-declarator: [C99 6.7]
  ///         declarator
  ///         declarator '=' initializer
  /// [GNU]   declarator simple-asm-expr[opt] attributes[opt]
  /// [GNU]   declarator simple-asm-expr[opt] attributes[opt] '=' initializer
  /// [C++]   declarator initializer[opt]
  ///
  /// [C++] initializer:
  /// [C++]   '=' initializer-clause
  /// [C++]   '(' expression-list ')'
  /// [C++0x] '=' 'default'                                                [TODO]
  /// [C++0x] '=' 'delete'
  /// [C++0x] braced-init-list
  /// \endverbatim
  ///
  /// According to the standard grammar, =default and =delete are function
  /// definitions, but that definitely doesn't fit with the parser here.
  ///
  Decl *ParseDeclarationAfterDeclarator(
      Declarator &D,
      const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());

  /// Parse an optional simple-asm-expr and attributes, and attach them to a
  /// declarator. Returns true on an error.
  bool ParseAsmAttributesAfterDeclarator(Declarator &D);
  Decl *ParseDeclarationAfterDeclaratorAndAttributes(
      Declarator &D,
      const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
      ForRangeInit *FRI = nullptr);

  /// ParseImplicitInt - This method is called when we have an non-typename
  /// identifier in a declspec (which normally terminates the decl spec) when
  /// the declspec has no type specifier.  In this case, the declspec is either
  /// malformed or is "implicit int" (in K&R and C89).
  ///
  /// This method handles diagnosing this prettily and returns false if the
  /// declspec is done being processed.  If it recovers and thinks there may be
  /// other pieces of declspec after it, it returns true.
  ///
  bool ParseImplicitInt(DeclSpec &DS, CXXScopeSpec *SS,
                        ParsedTemplateInfo &TemplateInfo, AccessSpecifier AS,
                        DeclSpecContext DSC, ParsedAttributes &Attrs);

  /// Determine the declaration specifier context from the declarator
  /// context.
  ///
  /// \param Context the declarator context, which is one of the
  /// DeclaratorContext enumerator values.
  DeclSpecContext
  getDeclSpecContextFromDeclaratorContext(DeclaratorContext Context);
  void
  ParseDeclarationSpecifiers(DeclSpec &DS, ParsedTemplateInfo &TemplateInfo,
                             AccessSpecifier AS = AS_none,
                             DeclSpecContext DSC = DeclSpecContext::DSC_normal,
                             LateParsedAttrList *LateAttrs = nullptr) {
    return ParseDeclarationSpecifiers(DS, TemplateInfo, AS, DSC, LateAttrs,
                                      getImplicitTypenameContext(DSC));
  }

  /// ParseDeclarationSpecifiers
  /// \verbatim
  ///       declaration-specifiers: [C99 6.7]
  ///         storage-class-specifier declaration-specifiers[opt]
  ///         type-specifier declaration-specifiers[opt]
  /// [C99]   function-specifier declaration-specifiers[opt]
  /// [C11]   alignment-specifier declaration-specifiers[opt]
  /// [GNU]   attributes declaration-specifiers[opt]
  /// [Clang] '__module_private__' declaration-specifiers[opt]
  /// [ObjC1] '__kindof' declaration-specifiers[opt]
  ///
  ///       storage-class-specifier: [C99 6.7.1]
  ///         'typedef'
  ///         'extern'
  ///         'static'
  ///         'auto'
  ///         'register'
  /// [C++]   'mutable'
  /// [C++11] 'thread_local'
  /// [C11]   '_Thread_local'
  /// [GNU]   '__thread'
  ///       function-specifier: [C99 6.7.4]
  /// [C99]   'inline'
  /// [C++]   'virtual'
  /// [C++]   'explicit'
  /// [OpenCL] '__kernel'
  ///       'friend': [C++ dcl.friend]
  ///       'constexpr': [C++0x dcl.constexpr]
  /// \endverbatim
  void
  ParseDeclarationSpecifiers(DeclSpec &DS, ParsedTemplateInfo &TemplateInfo,
                             AccessSpecifier AS, DeclSpecContext DSC,
                             LateParsedAttrList *LateAttrs,
                             ImplicitTypenameContext AllowImplicitTypename);

  /// Determine whether we're looking at something that might be a declarator
  /// in a simple-declaration. If it can't possibly be a declarator, maybe
  /// diagnose a missing semicolon after a prior tag definition in the decl
  /// specifier.
  ///
  /// \return \c true if an error occurred and this can't be any kind of
  /// declaration.
  bool DiagnoseMissingSemiAfterTagDefinition(
      DeclSpec &DS, AccessSpecifier AS, DeclSpecContext DSContext,
      LateParsedAttrList *LateAttrs = nullptr);

  void ParseSpecifierQualifierList(
      DeclSpec &DS, AccessSpecifier AS = AS_none,
      DeclSpecContext DSC = DeclSpecContext::DSC_normal) {
    ParseSpecifierQualifierList(DS, getImplicitTypenameContext(DSC), AS, DSC);
  }

  /// ParseSpecifierQualifierList
  /// \verbatim
  ///        specifier-qualifier-list:
  ///          type-specifier specifier-qualifier-list[opt]
  ///          type-qualifier specifier-qualifier-list[opt]
  /// [GNU]    attributes     specifier-qualifier-list[opt]
  /// \endverbatim
  ///
  void ParseSpecifierQualifierList(
      DeclSpec &DS, ImplicitTypenameContext AllowImplicitTypename,
      AccessSpecifier AS = AS_none,
      DeclSpecContext DSC = DeclSpecContext::DSC_normal);

  /// ParseEnumSpecifier
  /// \verbatim
  ///       enum-specifier: [C99 6.7.2.2]
  ///         'enum' identifier[opt] '{' enumerator-list '}'
  ///[C99/C++]'enum' identifier[opt] '{' enumerator-list ',' '}'
  /// [GNU]   'enum' attributes[opt] identifier[opt] '{' enumerator-list ',' [opt]
  ///                                                 '}' attributes[opt]
  /// [MS]    'enum' __declspec[opt] identifier[opt] '{' enumerator-list ',' [opt]
  ///                                                 '}'
  ///         'enum' identifier
  /// [GNU]   'enum' attributes[opt] identifier
  ///
  /// [C++11] enum-head '{' enumerator-list[opt] '}'
  /// [C++11] enum-head '{' enumerator-list ','  '}'
  ///
  ///       enum-head: [C++11]
  ///         enum-key attribute-specifier-seq[opt] identifier[opt] enum-base[opt]
  ///         enum-key attribute-specifier-seq[opt] nested-name-specifier
  ///             identifier enum-base[opt]
  ///
  ///       enum-key: [C++11]
  ///         'enum'
  ///         'enum' 'class'
  ///         'enum' 'struct'
  ///
  ///       enum-base: [C++11]
  ///         ':' type-specifier-seq
  ///
  /// [C++] elaborated-type-specifier:
  /// [C++]   'enum' nested-name-specifier[opt] identifier
  /// \endverbatim
  ///
  void ParseEnumSpecifier(SourceLocation TagLoc, DeclSpec &DS,
                          const ParsedTemplateInfo &TemplateInfo,
                          AccessSpecifier AS, DeclSpecContext DSC);

  /// ParseEnumBody - Parse a {} enclosed enumerator-list.
  /// \verbatim
  ///       enumerator-list:
  ///         enumerator
  ///         enumerator-list ',' enumerator
  ///       enumerator:
  ///         enumeration-constant attributes[opt]
  ///         enumeration-constant attributes[opt] '=' constant-expression
  ///       enumeration-constant:
  ///         identifier
  /// \endverbatim
  ///
  void ParseEnumBody(SourceLocation StartLoc, Decl *TagDecl,
                     SkipBodyInfo *SkipBody = nullptr);

  /// ParseStructUnionBody
  /// \verbatim
  ///       struct-contents:
  ///         struct-declaration-list
  /// [EXT]   empty
  /// [GNU]   "struct-declaration-list" without terminating ';'
  ///       struct-declaration-list:
  ///         struct-declaration
  ///         struct-declaration-list struct-declaration
  /// [OBC]   '@' 'defs' '(' class-name ')'
  /// \endverbatim
  ///
  void ParseStructUnionBody(SourceLocation StartLoc, DeclSpec::TST TagType,
                            RecordDecl *TagDecl);

  /// ParseStructDeclaration - Parse a struct declaration without the
  /// terminating semicolon.
  ///
  /// Note that a struct declaration refers to a declaration in a struct,
  /// not to the declaration of a struct.
  ///
  /// \verbatim
  ///       struct-declaration:
  /// [C23]   attributes-specifier-seq[opt]
  ///           specifier-qualifier-list struct-declarator-list
  /// [GNU]   __extension__ struct-declaration
  /// [GNU]   specifier-qualifier-list
  ///       struct-declarator-list:
  ///         struct-declarator
  ///         struct-declarator-list ',' struct-declarator
  /// [GNU]   struct-declarator-list ',' attributes[opt] struct-declarator
  ///       struct-declarator:
  ///         declarator
  /// [GNU]   declarator attributes[opt]
  ///         declarator[opt] ':' constant-expression
  /// [GNU]   declarator[opt] ':' constant-expression attributes[opt]
  /// \endverbatim
  ///
  void ParseStructDeclaration(
      ParsingDeclSpec &DS,
      llvm::function_ref<Decl *(ParsingFieldDeclarator &)> FieldsCallback,
      LateParsedAttrList *LateFieldAttrs = nullptr);

  DeclGroupPtrTy ParseTopLevelStmtDecl();

  /// isDeclarationSpecifier() - Return true if the current token is part of a
  /// declaration specifier.
  ///
  /// \param AllowImplicitTypename whether this is a context where T::type [T
  /// dependent] can appear.
  /// \param DisambiguatingWithExpression True to indicate that the purpose of
  /// this check is to disambiguate between an expression and a declaration.
  bool isDeclarationSpecifier(ImplicitTypenameContext AllowImplicitTypename,
                              bool DisambiguatingWithExpression = false);

  /// isTypeSpecifierQualifier - Return true if the current token could be the
  /// start of a specifier-qualifier-list.
  bool isTypeSpecifierQualifier();

  /// isKnownToBeTypeSpecifier - Return true if we know that the specified token
  /// is definitely a type-specifier.  Return false if it isn't part of a type
  /// specifier or if we're not sure.
  bool isKnownToBeTypeSpecifier(const Token &Tok) const;

  /// Starting with a scope specifier, identifier, or
  /// template-id that refers to the current class, determine whether
  /// this is a constructor declarator.
  bool isConstructorDeclarator(
      bool Unqualified, bool DeductionGuide = false,
      DeclSpec::FriendSpecified IsFriend = DeclSpec::FriendSpecified::No,
      const ParsedTemplateInfo *TemplateInfo = nullptr);

  /// Diagnoses use of _ExtInt as being deprecated, and diagnoses use of
  /// _BitInt as an extension when appropriate.
  void DiagnoseBitIntUse(const Token &Tok);

  // Check for the start of an attribute-specifier-seq in a context where an
  // attribute is not allowed.
  bool CheckProhibitedCXX11Attribute() {
    assert(Tok.is(tok::l_square));
    if (NextToken().isNot(tok::l_square))
      return false;
    return DiagnoseProhibitedCXX11Attribute();
  }

  /// DiagnoseProhibitedCXX11Attribute - We have found the opening square
  /// brackets of a C++11 attribute-specifier in a location where an attribute
  /// is not permitted. By C++11 [dcl.attr.grammar]p6, this is ill-formed.
  /// Diagnose this situation.
  ///
  /// \return \c true if we skipped an attribute-like chunk of tokens, \c false
  /// if this doesn't appear to actually be an attribute-specifier, and the
  /// caller should try to parse it.
  bool DiagnoseProhibitedCXX11Attribute();

  void CheckMisplacedCXX11Attribute(ParsedAttributes &Attrs,
                                    SourceLocation CorrectLocation) {
    if (!Tok.isRegularKeywordAttribute() &&
        (Tok.isNot(tok::l_square) || NextToken().isNot(tok::l_square)) &&
        Tok.isNot(tok::kw_alignas))
      return;
    DiagnoseMisplacedCXX11Attribute(Attrs, CorrectLocation);
  }

  /// We have found the opening square brackets of a C++11
  /// attribute-specifier in a location where an attribute is not permitted, but
  /// we know where the attributes ought to be written. Parse them anyway, and
  /// provide a fixit moving them to the right place.
  void DiagnoseMisplacedCXX11Attribute(ParsedAttributes &Attrs,
                                       SourceLocation CorrectLocation);

  // Usually, `__attribute__((attrib)) class Foo {} var` means that attribute
  // applies to var, not the type Foo.
  // As an exception to the rule, __declspec(align(...)) before the
  // class-key affects the type instead of the variable.
  // Also, Microsoft-style [attributes] seem to affect the type instead of the
  // variable.
  // This function moves attributes that should apply to the type off DS to
  // Attrs.
  void stripTypeAttributesOffDeclSpec(ParsedAttributes &Attrs, DeclSpec &DS,
                                      TagUseKind TUK);

  // FixItLoc = possible correct location for the attributes
  void ProhibitAttributes(ParsedAttributes &Attrs,
                          SourceLocation FixItLoc = SourceLocation()) {
    if (Attrs.Range.isInvalid())
      return;
    DiagnoseProhibitedAttributes(Attrs, FixItLoc);
    Attrs.clear();
  }

  void ProhibitAttributes(ParsedAttributesView &Attrs,
                          SourceLocation FixItLoc = SourceLocation()) {
    if (Attrs.Range.isInvalid())
      return;
    DiagnoseProhibitedAttributes(Attrs, FixItLoc);
    Attrs.clearListOnly();
  }
  void DiagnoseProhibitedAttributes(const ParsedAttributesView &Attrs,
                                    SourceLocation FixItLoc);

  // Forbid C++11 and C23 attributes that appear on certain syntactic locations
  // which standard permits but we don't supported yet, for example, attributes
  // appertain to decl specifiers.
  // For the most cases we don't want to warn on unknown type attributes, but
  // left them to later diagnoses. However, for a few cases like module
  // declarations and module import declarations, we should do it.
  void ProhibitCXX11Attributes(ParsedAttributes &Attrs, unsigned AttrDiagID,
                               unsigned KeywordDiagId,
                               bool DiagnoseEmptyAttrs = false,
                               bool WarnOnUnknownAttrs = false);

  /// Emit warnings for C++11 and C23 attributes that are in a position that
  /// clang accepts as an extension.
  void DiagnoseCXX11AttributeExtension(ParsedAttributes &Attrs);

  ExprResult ParseUnevaluatedStringInAttribute(const IdentifierInfo &AttrName);

  bool
  ParseAttributeArgumentList(const clang::IdentifierInfo &AttrName,
                             SmallVectorImpl<Expr *> &Exprs,
                             ParsedAttributeArgumentsProperties ArgsProperties);

  /// Parses syntax-generic attribute arguments for attributes which are
  /// known to the implementation, and adds them to the given ParsedAttributes
  /// list with the given attribute syntax. Returns the number of arguments
  /// parsed for the attribute.
  unsigned
  ParseAttributeArgsCommon(IdentifierInfo *AttrName, SourceLocation AttrNameLoc,
                           ParsedAttributes &Attrs, SourceLocation *EndLoc,
                           IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                           ParsedAttr::Form Form);

  enum ParseAttrKindMask {
    PAKM_GNU = 1 << 0,
    PAKM_Declspec = 1 << 1,
    PAKM_CXX11 = 1 << 2,
  };

  /// \brief Parse attributes based on what syntaxes are desired, allowing for
  /// the order to vary. e.g. with PAKM_GNU | PAKM_Declspec:
  /// __attribute__((...)) __declspec(...) __attribute__((...)))
  /// Note that Microsoft attributes (spelled with single square brackets) are
  /// not supported by this because of parsing ambiguities with other
  /// constructs.
  ///
  /// There are some attribute parse orderings that should not be allowed in
  /// arbitrary order. e.g.,
  ///
  /// \verbatim
  ///   [[]] __attribute__(()) int i; // OK
  ///   __attribute__(()) [[]] int i; // Not OK
  /// \endverbatim
  ///
  /// Such situations should use the specific attribute parsing functionality.
  void ParseAttributes(unsigned WhichAttrKinds, ParsedAttributes &Attrs,
                       LateParsedAttrList *LateAttrs = nullptr);
  /// \brief Possibly parse attributes based on what syntaxes are desired,
  /// allowing for the order to vary.
  bool MaybeParseAttributes(unsigned WhichAttrKinds, ParsedAttributes &Attrs,
                            LateParsedAttrList *LateAttrs = nullptr) {
    if (Tok.isOneOf(tok::kw___attribute, tok::kw___declspec) ||
        isAllowedCXX11AttributeSpecifier()) {
      ParseAttributes(WhichAttrKinds, Attrs, LateAttrs);
      return true;
    }
    return false;
  }

  void MaybeParseGNUAttributes(Declarator &D,
                               LateParsedAttrList *LateAttrs = nullptr) {
    if (Tok.is(tok::kw___attribute)) {
      ParsedAttributes Attrs(AttrFactory);
      ParseGNUAttributes(Attrs, LateAttrs, &D);
      D.takeAttributes(Attrs);
    }
  }

  bool MaybeParseGNUAttributes(ParsedAttributes &Attrs,
                               LateParsedAttrList *LateAttrs = nullptr) {
    if (Tok.is(tok::kw___attribute)) {
      ParseGNUAttributes(Attrs, LateAttrs);
      return true;
    }
    return false;
  }

  /// ParseSingleGNUAttribute - Parse a single GNU attribute.
  ///
  /// \verbatim
  /// [GNU]  attrib:
  ///          empty
  ///          attrib-name
  ///          attrib-name '(' identifier ')'
  ///          attrib-name '(' identifier ',' nonempty-expr-list ')'
  ///          attrib-name '(' argument-expression-list [C99 6.5.2] ')'
  ///
  /// [GNU]  attrib-name:
  ///          identifier
  ///          typespec
  ///          typequal
  ///          storageclass
  /// \endverbatim
  bool ParseSingleGNUAttribute(ParsedAttributes &Attrs, SourceLocation &EndLoc,
                               LateParsedAttrList *LateAttrs = nullptr,
                               Declarator *D = nullptr);

  /// ParseGNUAttributes - Parse a non-empty attributes list.
  ///
  /// \verbatim
  /// [GNU] attributes:
  ///         attribute
  ///         attributes attribute
  ///
  /// [GNU]  attribute:
  ///          '__attribute__' '(' '(' attribute-list ')' ')'
  ///
  /// [GNU]  attribute-list:
  ///          attrib
  ///          attribute_list ',' attrib
  ///
  /// [GNU]  attrib:
  ///          empty
  ///          attrib-name
  ///          attrib-name '(' identifier ')'
  ///          attrib-name '(' identifier ',' nonempty-expr-list ')'
  ///          attrib-name '(' argument-expression-list [C99 6.5.2] ')'
  ///
  /// [GNU]  attrib-name:
  ///          identifier
  ///          typespec
  ///          typequal
  ///          storageclass
  /// \endverbatim
  ///
  /// Whether an attribute takes an 'identifier' is determined by the
  /// attrib-name. GCC's behavior here is not worth imitating:
  ///
  ///  * In C mode, if the attribute argument list starts with an identifier
  ///    followed by a ',' or an ')', and the identifier doesn't resolve to
  ///    a type, it is parsed as an identifier. If the attribute actually
  ///    wanted an expression, it's out of luck (but it turns out that no
  ///    attributes work that way, because C constant expressions are very
  ///    limited).
  ///  * In C++ mode, if the attribute argument list starts with an identifier,
  ///    and the attribute *wants* an identifier, it is parsed as an identifier.
  ///    At block scope, any additional tokens between the identifier and the
  ///    ',' or ')' are ignored, otherwise they produce a parse error.
  ///
  /// We follow the C++ model, but don't allow junk after the identifier.
  void ParseGNUAttributes(ParsedAttributes &Attrs,
                          LateParsedAttrList *LateAttrs = nullptr,
                          Declarator *D = nullptr);

  /// Parse the arguments to a parameterized GNU attribute or
  /// a C++11 attribute in "gnu" namespace.
  void ParseGNUAttributeArgs(IdentifierInfo *AttrName,
                             SourceLocation AttrNameLoc,
                             ParsedAttributes &Attrs, SourceLocation *EndLoc,
                             IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                             ParsedAttr::Form Form, Declarator *D);
  IdentifierLoc *ParseIdentifierLoc();

  unsigned
  ParseClangAttributeArgs(IdentifierInfo *AttrName, SourceLocation AttrNameLoc,
                          ParsedAttributes &Attrs, SourceLocation *EndLoc,
                          IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                          ParsedAttr::Form Form);

  void MaybeParseCXX11Attributes(Declarator &D) {
    if (isAllowedCXX11AttributeSpecifier()) {
      ParsedAttributes Attrs(AttrFactory);
      ParseCXX11Attributes(Attrs);
      D.takeAttributes(Attrs);
    }
  }

  bool MaybeParseCXX11Attributes(ParsedAttributes &Attrs,
                                 bool OuterMightBeMessageSend = false) {
    if (isAllowedCXX11AttributeSpecifier(false, OuterMightBeMessageSend)) {
      ParseCXX11Attributes(Attrs);
      return true;
    }
    return false;
  }

  bool MaybeParseMicrosoftAttributes(ParsedAttributes &Attrs) {
    bool AttrsParsed = false;
    if ((getLangOpts().MicrosoftExt || getLangOpts().HLSL) &&
        Tok.is(tok::l_square)) {
      ParsedAttributes AttrsWithRange(AttrFactory);
      ParseMicrosoftAttributes(AttrsWithRange);
      AttrsParsed = !AttrsWithRange.empty();
      Attrs.takeAllFrom(AttrsWithRange);
    }
    return AttrsParsed;
  }
  bool MaybeParseMicrosoftDeclSpecs(ParsedAttributes &Attrs) {
    if (getLangOpts().DeclSpecKeyword && Tok.is(tok::kw___declspec)) {
      ParseMicrosoftDeclSpecs(Attrs);
      return true;
    }
    return false;
  }

  /// \verbatim
  /// [MS] decl-specifier:
  ///             __declspec ( extended-decl-modifier-seq )
  ///
  /// [MS] extended-decl-modifier-seq:
  ///             extended-decl-modifier[opt]
  ///             extended-decl-modifier extended-decl-modifier-seq
  /// \endverbatim
  void ParseMicrosoftDeclSpecs(ParsedAttributes &Attrs);
  bool ParseMicrosoftDeclSpecArgs(IdentifierInfo *AttrName,
                                  SourceLocation AttrNameLoc,
                                  ParsedAttributes &Attrs);
  void ParseMicrosoftTypeAttributes(ParsedAttributes &attrs);
  void ParseWebAssemblyFuncrefTypeAttribute(ParsedAttributes &Attrs);
  void DiagnoseAndSkipExtendedMicrosoftTypeAttributes();
  SourceLocation SkipExtendedMicrosoftTypeAttributes();

  void ParseBorlandTypeAttributes(ParsedAttributes &attrs);
  void ParseOpenCLKernelAttributes(ParsedAttributes &attrs);
  void ParseOpenCLQualifiers(ParsedAttributes &Attrs);
  void ParseNullabilityTypeSpecifiers(ParsedAttributes &attrs);
  void ParseCUDAFunctionAttributes(ParsedAttributes &attrs);
  bool isHLSLQualifier(const Token &Tok) const;
  void ParseHLSLQualifiers(ParsedAttributes &Attrs);

  /// Parse a version number.
  ///
  /// \verbatim
  /// version:
  ///   simple-integer
  ///   simple-integer '.' simple-integer
  ///   simple-integer '_' simple-integer
  ///   simple-integer '.' simple-integer '.' simple-integer
  ///   simple-integer '_' simple-integer '_' simple-integer
  /// \endverbatim
  VersionTuple ParseVersionTuple(SourceRange &Range);

  /// Parse the contents of the "availability" attribute.
  ///
  /// \verbatim
  /// availability-attribute:
  ///   'availability' '(' platform ',' opt-strict version-arg-list,
  ///                      opt-replacement, opt-message')'
  ///
  /// platform:
  ///   identifier
  ///
  /// opt-strict:
  ///   'strict' ','
  ///
  /// version-arg-list:
  ///   version-arg
  ///   version-arg ',' version-arg-list
  ///
  /// version-arg:
  ///   'introduced' '=' version
  ///   'deprecated' '=' version
  ///   'obsoleted' = version
  ///   'unavailable'
  /// opt-replacement:
  ///   'replacement' '=' <string>
  /// opt-message:
  ///   'message' '=' <string>
  /// \endverbatim
  void ParseAvailabilityAttribute(IdentifierInfo &Availability,
                                  SourceLocation AvailabilityLoc,
                                  ParsedAttributes &attrs,
                                  SourceLocation *endLoc,
                                  IdentifierInfo *ScopeName,
                                  SourceLocation ScopeLoc,
                                  ParsedAttr::Form Form);

  /// Parse the contents of the "external_source_symbol" attribute.
  ///
  /// \verbatim
  /// external-source-symbol-attribute:
  ///   'external_source_symbol' '(' keyword-arg-list ')'
  ///
  /// keyword-arg-list:
  ///   keyword-arg
  ///   keyword-arg ',' keyword-arg-list
  ///
  /// keyword-arg:
  ///   'language' '=' <string>
  ///   'defined_in' '=' <string>
  ///   'USR' '=' <string>
  ///   'generated_declaration'
  /// \endverbatim
  void ParseExternalSourceSymbolAttribute(IdentifierInfo &ExternalSourceSymbol,
                                          SourceLocation Loc,
                                          ParsedAttributes &Attrs,
                                          SourceLocation *EndLoc,
                                          IdentifierInfo *ScopeName,
                                          SourceLocation ScopeLoc,
                                          ParsedAttr::Form Form);

  /// Parse the contents of the "objc_bridge_related" attribute.
  /// \verbatim
  /// objc_bridge_related '(' related_class ',' opt-class_method ',' opt-instance_method ')'
  /// related_class:
  ///     Identifier
  ///
  /// opt-class_method:
  ///     Identifier: | <empty>
  ///
  /// opt-instance_method:
  ///     Identifier | <empty>
  /// \endverbatim
  ///
  void ParseObjCBridgeRelatedAttribute(IdentifierInfo &ObjCBridgeRelated,
                                       SourceLocation ObjCBridgeRelatedLoc,
                                       ParsedAttributes &Attrs,
                                       SourceLocation *EndLoc,
                                       IdentifierInfo *ScopeName,
                                       SourceLocation ScopeLoc,
                                       ParsedAttr::Form Form);

  void ParseSwiftNewTypeAttribute(IdentifierInfo &AttrName,
                                  SourceLocation AttrNameLoc,
                                  ParsedAttributes &Attrs,
                                  SourceLocation *EndLoc,
                                  IdentifierInfo *ScopeName,
                                  SourceLocation ScopeLoc,
                                  ParsedAttr::Form Form);

  void ParseTypeTagForDatatypeAttribute(IdentifierInfo &AttrName,
                                        SourceLocation AttrNameLoc,
                                        ParsedAttributes &Attrs,
                                        SourceLocation *EndLoc,
                                        IdentifierInfo *ScopeName,
                                        SourceLocation ScopeLoc,
                                        ParsedAttr::Form Form);

  void ParseAttributeWithTypeArg(IdentifierInfo &AttrName,
                                 SourceLocation AttrNameLoc,
                                 ParsedAttributes &Attrs,
                                 IdentifierInfo *ScopeName,
                                 SourceLocation ScopeLoc,
                                 ParsedAttr::Form Form);

  void DistributeCLateParsedAttrs(Decl *Dcl, LateParsedAttrList *LateAttrs);

  /// Bounds attributes (e.g., counted_by):
  /// \verbatim
  ///   AttrName '(' expression ')'
  /// \endverbatim
  void ParseBoundsAttribute(IdentifierInfo &AttrName,
                            SourceLocation AttrNameLoc, ParsedAttributes &Attrs,
                            IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                            ParsedAttr::Form Form);

  /// \verbatim
  /// [GNU]   typeof-specifier:
  ///           typeof ( expressions )
  ///           typeof ( type-name )
  /// [GNU/C++] typeof unary-expression
  /// [C23]   typeof-specifier:
  ///           typeof '(' typeof-specifier-argument ')'
  ///           typeof_unqual '(' typeof-specifier-argument ')'
  ///
  ///         typeof-specifier-argument:
  ///           expression
  ///           type-name
  /// \endverbatim
  ///
  void ParseTypeofSpecifier(DeclSpec &DS);

  /// \verbatim
  /// [C11]   atomic-specifier:
  ///           _Atomic ( type-name )
  /// \endverbatim
  ///
  void ParseAtomicSpecifier(DeclSpec &DS);

  /// ParseAlignArgument - Parse the argument to an alignment-specifier.
  ///
  /// \verbatim
  /// [C11]   type-id
  /// [C11]   constant-expression
  /// [C++0x] type-id ...[opt]
  /// [C++0x] assignment-expression ...[opt]
  /// \endverbatim
  ExprResult ParseAlignArgument(StringRef KWName, SourceLocation Start,
                                SourceLocation &EllipsisLoc, bool &IsType,
                                ParsedType &Ty);

  /// ParseAlignmentSpecifier - Parse an alignment-specifier, and add the
  /// attribute to Attrs.
  ///
  /// \verbatim
  /// alignment-specifier:
  /// [C11]   '_Alignas' '(' type-id ')'
  /// [C11]   '_Alignas' '(' constant-expression ')'
  /// [C++11] 'alignas' '(' type-id ...[opt] ')'
  /// [C++11] 'alignas' '(' assignment-expression ...[opt] ')'
  /// \endverbatim
  void ParseAlignmentSpecifier(ParsedAttributes &Attrs,
                               SourceLocation *endLoc = nullptr);
  ExprResult ParseExtIntegerArgument();

  /// \verbatim
  /// type-qualifier:
  ///    ('__ptrauth') '(' constant-expression
  ///                   (',' constant-expression)[opt]
  ///                   (',' constant-expression)[opt] ')'
  /// \endverbatim
  void ParsePtrauthQualifier(ParsedAttributes &Attrs);

  /// DeclaratorScopeObj - RAII object used in Parser::ParseDirectDeclarator to
  /// enter a new C++ declarator scope and exit it when the function is
  /// finished.
  class DeclaratorScopeObj {
    Parser &P;
    CXXScopeSpec &SS;
    bool EnteredScope;
    bool CreatedScope;

  public:
    DeclaratorScopeObj(Parser &p, CXXScopeSpec &ss)
        : P(p), SS(ss), EnteredScope(false), CreatedScope(false) {}

    void EnterDeclaratorScope() {
      assert(!EnteredScope && "Already entered the scope!");
      assert(SS.isSet() && "C++ scope was not set!");

      CreatedScope = true;
      P.EnterScope(0); // Not a decl scope.

      if (!P.Actions.ActOnCXXEnterDeclaratorScope(P.getCurScope(), SS))
        EnteredScope = true;
    }

    ~DeclaratorScopeObj() {
      if (EnteredScope) {
        assert(SS.isSet() && "C++ scope was cleared ?");
        P.Actions.ActOnCXXExitDeclaratorScope(P.getCurScope(), SS);
      }
      if (CreatedScope)
        P.ExitScope();
    }
  };

  /// ParseDeclarator - Parse and verify a newly-initialized declarator.
  void ParseDeclarator(Declarator &D);
  /// A function that parses a variant of direct-declarator.
  typedef void (Parser::*DirectDeclParseFunction)(Declarator &);

  /// ParseDeclaratorInternal - Parse a C or C++ declarator. The
  /// direct-declarator is parsed by the function passed to it. Pass null, and
  /// the direct-declarator isn't parsed at all, making this function
  /// effectively parse the C++ ptr-operator production.
  ///
  /// If the grammar of this construct is extended, matching changes must also
  /// be made to TryParseDeclarator and MightBeDeclarator, and possibly to
  /// isConstructorDeclarator.
  ///
  /// \verbatim
  ///       declarator: [C99 6.7.5] [C++ 8p4, dcl.decl]
  /// [C]     pointer[opt] direct-declarator
  /// [C++]   direct-declarator
  /// [C++]   ptr-operator declarator
  ///
  ///       pointer: [C99 6.7.5]
  ///         '*' type-qualifier-list[opt]
  ///         '*' type-qualifier-list[opt] pointer
  ///
  ///       ptr-operator:
  ///         '*' cv-qualifier-seq[opt]
  ///         '&'
  /// [C++0x] '&&'
  /// [GNU]   '&' restrict[opt] attributes[opt]
  /// [GNU?]  '&&' restrict[opt] attributes[opt]
  ///         '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt]
  /// \endverbatim
  void ParseDeclaratorInternal(Declarator &D,
                               DirectDeclParseFunction DirectDeclParser);

  enum AttrRequirements {
    AR_NoAttributesParsed = 0, ///< No attributes are diagnosed.
    AR_GNUAttributesParsedAndRejected = 1 << 0, ///< Diagnose GNU attributes.
    AR_GNUAttributesParsed = 1 << 1,
    AR_CXX11AttributesParsed = 1 << 2,
    AR_DeclspecAttributesParsed = 1 << 3,
    AR_AllAttributesParsed = AR_GNUAttributesParsed | AR_CXX11AttributesParsed |
                             AR_DeclspecAttributesParsed,
    AR_VendorAttributesParsed =
        AR_GNUAttributesParsed | AR_DeclspecAttributesParsed
  };

  /// ParseTypeQualifierListOpt
  /// \verbatim
  ///          type-qualifier-list: [C99 6.7.5]
  ///            type-qualifier
  /// [vendor]   attributes
  ///              [ only if AttrReqs & AR_VendorAttributesParsed ]
  ///            type-qualifier-list type-qualifier
  /// [vendor]   type-qualifier-list attributes
  ///              [ only if AttrReqs & AR_VendorAttributesParsed ]
  /// [C++0x]    attribute-specifier[opt] is allowed before cv-qualifier-seq
  ///              [ only if AttReqs & AR_CXX11AttributesParsed ]
  /// \endverbatim
  /// Note: vendor can be GNU, MS, etc and can be explicitly controlled via
  /// AttrRequirements bitmask values.
  void ParseTypeQualifierListOpt(
      DeclSpec &DS, unsigned AttrReqs = AR_AllAttributesParsed,
      bool AtomicOrPtrauthAllowed = true, bool IdentifierRequired = false,
      llvm::function_ref<void()> CodeCompletionHandler = {});

  /// ParseDirectDeclarator
  /// \verbatim
  ///       direct-declarator: [C99 6.7.5]
  /// [C99]   identifier
  ///         '(' declarator ')'
  /// [GNU]   '(' attributes declarator ')'
  /// [C90]   direct-declarator '[' constant-expression[opt] ']'
  /// [C99]   direct-declarator '[' type-qual-list[opt] assignment-expr[opt] ']'
  /// [C99]   direct-declarator '[' 'static' type-qual-list[opt] assign-expr ']'
  /// [C99]   direct-declarator '[' type-qual-list 'static' assignment-expr ']'
  /// [C99]   direct-declarator '[' type-qual-list[opt] '*' ']'
  /// [C++11] direct-declarator '[' constant-expression[opt] ']'
  ///                    attribute-specifier-seq[opt]
  ///         direct-declarator '(' parameter-type-list ')'
  ///         direct-declarator '(' identifier-list[opt] ')'
  /// [GNU]   direct-declarator '(' parameter-forward-declarations
  ///                    parameter-type-list[opt] ')'
  /// [C++]   direct-declarator '(' parameter-declaration-clause ')'
  ///                    cv-qualifier-seq[opt] exception-specification[opt]
  /// [C++11] direct-declarator '(' parameter-declaration-clause ')'
  ///                    attribute-specifier-seq[opt] cv-qualifier-seq[opt]
  ///                    ref-qualifier[opt] exception-specification[opt]
  /// [C++]   declarator-id
  /// [C++11] declarator-id attribute-specifier-seq[opt]
  ///
  ///       declarator-id: [C++ 8]
  ///         '...'[opt] id-expression
  ///         '::'[opt] nested-name-specifier[opt] type-name
  ///
  ///       id-expression: [C++ 5.1]
  ///         unqualified-id
  ///         qualified-id
  ///
  ///       unqualified-id: [C++ 5.1]
  ///         identifier
  ///         operator-function-id
  ///         conversion-function-id
  ///          '~' class-name
  ///         template-id
  ///
  /// C++17 adds the following, which we also handle here:
  ///
  ///       simple-declaration:
  ///         <decl-spec> '[' identifier-list ']' brace-or-equal-initializer ';'
  /// \endverbatim
  ///
  /// Note, any additional constructs added here may need corresponding changes
  /// in isConstructorDeclarator.
  void ParseDirectDeclarator(Declarator &D);
  void ParseDecompositionDeclarator(Declarator &D);

  /// ParseParenDeclarator - We parsed the declarator D up to a paren.  This is
  /// only called before the identifier, so these are most likely just grouping
  /// parens for precedence.  If we find that these are actually function
  /// parameter parens in an abstract-declarator, we call
  /// ParseFunctionDeclarator.
  ///
  /// \verbatim
  ///       direct-declarator:
  ///         '(' declarator ')'
  /// [GNU]   '(' attributes declarator ')'
  ///         direct-declarator '(' parameter-type-list ')'
  ///         direct-declarator '(' identifier-list[opt] ')'
  /// [GNU]   direct-declarator '(' parameter-forward-declarations
  ///                    parameter-type-list[opt] ')'
  /// \endverbatim
  ///
  void ParseParenDeclarator(Declarator &D);

  /// ParseFunctionDeclarator - We are after the identifier and have parsed the
  /// declarator D up to a paren, which indicates that we are parsing function
  /// arguments.
  ///
  /// If FirstArgAttrs is non-null, then the caller parsed those attributes
  /// immediately after the open paren - they will be applied to the DeclSpec
  /// of the first parameter.
  ///
  /// If RequiresArg is true, then the first argument of the function is
  /// required to be present and required to not be an identifier list.
  ///
  /// For C++, after the parameter-list, it also parses the
  /// cv-qualifier-seq[opt], (C++11) ref-qualifier[opt],
  /// exception-specification[opt], (C++11) attribute-specifier-seq[opt],
  /// (C++11) trailing-return-type[opt] and (C++2a) the trailing
  /// requires-clause.
  ///
  /// \verbatim
  /// [C++11] exception-specification:
  ///           dynamic-exception-specification
  ///           noexcept-specification
  /// \endverbatim
  ///
  void ParseFunctionDeclarator(Declarator &D, ParsedAttributes &FirstArgAttrs,
                               BalancedDelimiterTracker &Tracker,
                               bool IsAmbiguous, bool RequiresArg = false);
  void InitCXXThisScopeForDeclaratorIfRelevant(
      const Declarator &D, const DeclSpec &DS,
      std::optional<Sema::CXXThisScopeRAII> &ThisScope);

  /// ParseRefQualifier - Parses a member function ref-qualifier. Returns
  /// true if a ref-qualifier is found.
  bool ParseRefQualifier(bool &RefQualifierIsLValueRef,
                         SourceLocation &RefQualifierLoc);

  /// isFunctionDeclaratorIdentifierList - This parameter list may have an
  /// identifier list form for a K&R-style function:  void foo(a,b,c)
  ///
  /// Note that identifier-lists are only allowed for normal declarators, not
  /// for abstract-declarators.
  bool isFunctionDeclaratorIdentifierList();

  /// ParseFunctionDeclaratorIdentifierList - While parsing a function
  /// declarator we found a K&R-style identifier list instead of a typed
  /// parameter list.
  ///
  /// After returning, ParamInfo will hold the parsed parameters.
  ///
  /// \verbatim
  ///       identifier-list: [C99 6.7.5]
  ///         identifier
  ///         identifier-list ',' identifier
  /// \endverbatim
  ///
  void ParseFunctionDeclaratorIdentifierList(
      Declarator &D, SmallVectorImpl<DeclaratorChunk::ParamInfo> &ParamInfo);
  void ParseParameterDeclarationClause(
      Declarator &D, ParsedAttributes &attrs,
      SmallVectorImpl<DeclaratorChunk::ParamInfo> &ParamInfo,
      SourceLocation &EllipsisLoc) {
    return ParseParameterDeclarationClause(
        D.getContext(), attrs, ParamInfo, EllipsisLoc,
        D.getCXXScopeSpec().isSet() &&
            D.isFunctionDeclaratorAFunctionDeclaration());
  }

  /// ParseParameterDeclarationClause - Parse a (possibly empty) parameter-list
  /// after the opening parenthesis. This function will not parse a K&R-style
  /// identifier list.
  ///
  /// DeclContext is the context of the declarator being parsed.  If
  /// FirstArgAttrs is non-null, then the caller parsed those attributes
  /// immediately after the open paren - they will be applied to the DeclSpec of
  /// the first parameter.
  ///
  /// After returning, ParamInfo will hold the parsed parameters. EllipsisLoc
  /// will be the location of the ellipsis, if any was parsed.
  ///
  /// \verbatim
  ///       parameter-type-list: [C99 6.7.5]
  ///         parameter-list
  ///         parameter-list ',' '...'
  /// [C++]   parameter-list '...'
  ///
  ///       parameter-list: [C99 6.7.5]
  ///         parameter-declaration
  ///         parameter-list ',' parameter-declaration
  ///
  ///       parameter-declaration: [C99 6.7.5]
  ///         declaration-specifiers declarator
  /// [C++]   declaration-specifiers declarator '=' assignment-expression
  /// [C++11]                                       initializer-clause
  /// [GNU]   declaration-specifiers declarator attributes
  ///         declaration-specifiers abstract-declarator[opt]
  /// [C++]   declaration-specifiers abstract-declarator[opt]
  ///           '=' assignment-expression
  /// [GNU]   declaration-specifiers abstract-declarator[opt] attributes
  /// [C++11] attribute-specifier-seq parameter-declaration
  /// [C++2b] attribute-specifier-seq 'this' parameter-declaration
  /// \endverbatim
  ///
  void ParseParameterDeclarationClause(
      DeclaratorContext DeclaratorContext, ParsedAttributes &attrs,
      SmallVectorImpl<DeclaratorChunk::ParamInfo> &ParamInfo,
      SourceLocation &EllipsisLoc, bool IsACXXFunctionDeclaration = false);

  /// \verbatim
  /// [C90]   direct-declarator '[' constant-expression[opt] ']'
  /// [C99]   direct-declarator '[' type-qual-list[opt] assignment-expr[opt] ']'
  /// [C99]   direct-declarator '[' 'static' type-qual-list[opt] assign-expr ']'
  /// [C99]   direct-declarator '[' type-qual-list 'static' assignment-expr ']'
  /// [C99]   direct-declarator '[' type-qual-list[opt] '*' ']'
  /// [C++11] direct-declarator '[' constant-expression[opt] ']'
  ///                           attribute-specifier-seq[opt]
  /// \endverbatim
  void ParseBracketDeclarator(Declarator &D);

  /// Diagnose brackets before an identifier.
  void ParseMisplacedBracketDeclarator(Declarator &D);

  /// Parse the given string as a type.
  ///
  /// This is a dangerous utility function currently employed only by API notes.
  /// It is not a general entry-point for safely parsing types from strings.
  ///
  /// \param TypeStr The string to be parsed as a type.
  /// \param Context The name of the context in which this string is being
  /// parsed, which will be used in diagnostics.
  /// \param IncludeLoc The location at which this parse was triggered.
  TypeResult ParseTypeFromString(StringRef TypeStr, StringRef Context,
                                 SourceLocation IncludeLoc);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name C++ Declarations
  /// Implementations are in ParseDeclCXX.cpp
  ///@{

private:
  /// Contextual keywords for Microsoft extensions.
  mutable IdentifierInfo *Ident_sealed;
  mutable IdentifierInfo *Ident_abstract;

  /// C++11 contextual keywords.
  mutable IdentifierInfo *Ident_final;
  mutable IdentifierInfo *Ident_GNU_final;
  mutable IdentifierInfo *Ident_override;
  mutable IdentifierInfo *Ident_trivially_relocatable_if_eligible;
  mutable IdentifierInfo *Ident_replaceable_if_eligible;

  /// Representation of a class that has been parsed, including
  /// any member function declarations or definitions that need to be
  /// parsed after the corresponding top-level class is complete.
  struct ParsingClass {
    ParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface)
        : TopLevelClass(TopLevelClass), IsInterface(IsInterface),
          TagOrTemplate(TagOrTemplate) {}

    /// Whether this is a "top-level" class, meaning that it is
    /// not nested within another class.
    bool TopLevelClass : 1;

    /// Whether this class is an __interface.
    bool IsInterface : 1;

    /// The class or class template whose definition we are parsing.
    Decl *TagOrTemplate;

    /// LateParsedDeclarations - Method declarations, inline definitions and
    /// nested classes that contain pieces whose parsing will be delayed until
    /// the top-level class is fully defined.
    LateParsedDeclarationsContainer LateParsedDeclarations;
  };

  /// The stack of classes that is currently being
  /// parsed. Nested and local classes will be pushed onto this stack
  /// when they are parsed, and removed afterward.
  std::stack<ParsingClass *> ClassStack;

  ParsingClass &getCurrentClass() {
    assert(!ClassStack.empty() && "No lexed method stacks!");
    return *ClassStack.top();
  }

  /// RAII object used to manage the parsing of a class definition.
  class ParsingClassDefinition {
    Parser &P;
    bool Popped;
    Sema::ParsingClassState State;

  public:
    ParsingClassDefinition(Parser &P, Decl *TagOrTemplate, bool TopLevelClass,
                           bool IsInterface)
        : P(P), Popped(false),
          State(P.PushParsingClass(TagOrTemplate, TopLevelClass, IsInterface)) {
    }

    /// Pop this class of the stack.
    void Pop() {
      assert(!Popped && "Nested class has already been popped");
      Popped = true;
      P.PopParsingClass(State);
    }

    ~ParsingClassDefinition() {
      if (!Popped)
        P.PopParsingClass(State);
    }
  };

  /// Parse a C++ exception-specification if present (C++0x [except.spec]).
  ///
  /// \verbatim
  ///       exception-specification:
  ///         dynamic-exception-specification
  ///         noexcept-specification
  ///
  ///       noexcept-specification:
  ///         'noexcept'
  ///         'noexcept' '(' constant-expression ')'
  /// \endverbatim
  ExceptionSpecificationType tryParseExceptionSpecification(
      bool Delayed, SourceRange &SpecificationRange,
      SmallVectorImpl<ParsedType> &DynamicExceptions,
      SmallVectorImpl<SourceRange> &DynamicExceptionRanges,
      ExprResult &NoexceptExpr, CachedTokens *&ExceptionSpecTokens);

  /// ParseDynamicExceptionSpecification - Parse a C++
  /// dynamic-exception-specification (C++ [except.spec]).
  /// EndLoc is filled with the location of the last token of the specification.
  ///
  /// \verbatim
  ///       dynamic-exception-specification:
  ///         'throw' '(' type-id-list [opt] ')'
  /// [MS]    'throw' '(' '...' ')'
  ///
  ///       type-id-list:
  ///         type-id ... [opt]
  ///         type-id-list ',' type-id ... [opt]
  /// \endverbatim
  ///
  ExceptionSpecificationType
  ParseDynamicExceptionSpecification(SourceRange &SpecificationRange,
                                     SmallVectorImpl<ParsedType> &Exceptions,
                                     SmallVectorImpl<SourceRange> &Ranges);

  //===--------------------------------------------------------------------===//
  // C++0x 8: Function declaration trailing-return-type

  /// ParseTrailingReturnType - Parse a trailing return type on a new-style
  /// function declaration.
  TypeResult ParseTrailingReturnType(SourceRange &Range,
                                     bool MayBeFollowedByDirectInit);

  /// Parse a requires-clause as part of a function declaration.
  void ParseTrailingRequiresClause(Declarator &D);

  void ParseMicrosoftIfExistsClassDeclaration(DeclSpec::TST TagType,
                                              ParsedAttributes &AccessAttrs,
                                              AccessSpecifier &CurAS);

  SourceLocation ParsePackIndexingType(DeclSpec &DS);
  void AnnotateExistingIndexedTypeNamePack(ParsedType T,
                                           SourceLocation StartLoc,
                                           SourceLocation EndLoc);

  /// Return true if the next token should be treated as a [[]] attribute,
  /// or as a keyword that behaves like one.  The former is only true if
  /// [[]] attributes are enabled, whereas the latter is true whenever
  /// such a keyword appears.  The arguments are as for
  /// isCXX11AttributeSpecifier.
  bool isAllowedCXX11AttributeSpecifier(bool Disambiguate = false,
                                        bool OuterMightBeMessageSend = false) {
    return (Tok.isRegularKeywordAttribute() ||
            isCXX11AttributeSpecifier(Disambiguate, OuterMightBeMessageSend) !=
                CXX11AttributeKind::NotAttributeSpecifier);
  }

  /// Skip C++11 and C23 attributes and return the end location of the
  /// last one.
  /// \returns SourceLocation() if there are no attributes.
  SourceLocation SkipCXX11Attributes();

  /// Diagnose and skip C++11 and C23 attributes that appear in syntactic
  /// locations where attributes are not allowed.
  void DiagnoseAndSkipCXX11Attributes();

  void ParseOpenMPAttributeArgs(const IdentifierInfo *AttrName,
                                CachedTokens &OpenMPTokens);

  /// Parse a C++11 or C23 attribute-specifier.
  ///
  /// \verbatim
  /// [C++11] attribute-specifier:
  ///         '[' '[' attribute-list ']' ']'
  ///         alignment-specifier
  ///
  /// [C++11] attribute-list:
  ///         attribute[opt]
  ///         attribute-list ',' attribute[opt]
  ///         attribute '...'
  ///         attribute-list ',' attribute '...'
  ///
  /// [C++11] attribute:
  ///         attribute-token attribute-argument-clause[opt]
  ///
  /// [C++11] attribute-token:
  ///         identifier
  ///         attribute-scoped-token
  ///
  /// [C++11] attribute-scoped-token:
  ///         attribute-namespace '::' identifier
  ///
  /// [C++11] attribute-namespace:
  ///         identifier
  /// \endverbatim
  void ParseCXX11AttributeSpecifierInternal(ParsedAttributes &Attrs,
                                            CachedTokens &OpenMPTokens,
                                            SourceLocation *EndLoc = nullptr);
  void ParseCXX11AttributeSpecifier(ParsedAttributes &Attrs,
                                    SourceLocation *EndLoc = nullptr) {
    CachedTokens OpenMPTokens;
    ParseCXX11AttributeSpecifierInternal(Attrs, OpenMPTokens, EndLoc);
    ReplayOpenMPAttributeTokens(OpenMPTokens);
  }

  /// ParseCXX11Attributes - Parse a C++11 or C23 attribute-specifier-seq.
  ///
  /// \verbatim
  /// attribute-specifier-seq:
  ///       attribute-specifier-seq[opt] attribute-specifier
  /// \endverbatim
  void ParseCXX11Attributes(ParsedAttributes &attrs);

  /// ParseCXX11AttributeArgs -- Parse a C++11 attribute-argument-clause.
  /// Parses a C++11 (or C23)-style attribute argument list. Returns true
  /// if this results in adding an attribute to the ParsedAttributes list.
  ///
  /// \verbatim
  /// [C++11] attribute-argument-clause:
  ///         '(' balanced-token-seq ')'
  ///
  /// [C++11] balanced-token-seq:
  ///         balanced-token
  ///         balanced-token-seq balanced-token
  ///
  /// [C++11] balanced-token:
  ///         '(' balanced-token-seq ')'
  ///         '[' balanced-token-seq ']'
  ///         '{' balanced-token-seq '}'
  ///         any token but '(', ')', '[', ']', '{', or '}'
  /// \endverbatim
  bool ParseCXX11AttributeArgs(IdentifierInfo *AttrName,
                               SourceLocation AttrNameLoc,
                               ParsedAttributes &Attrs, SourceLocation *EndLoc,
                               IdentifierInfo *ScopeName,
                               SourceLocation ScopeLoc,
                               CachedTokens &OpenMPTokens);

  /// Parse the argument to C++23's [[assume()]] attribute. Returns true on
  /// error.
  bool
  ParseCXXAssumeAttributeArg(ParsedAttributes &Attrs, IdentifierInfo *AttrName,
                             SourceLocation AttrNameLoc,
                             IdentifierInfo *ScopeName, SourceLocation ScopeLoc,
                             SourceLocation *EndLoc, ParsedAttr::Form Form);

  /// Try to parse an 'identifier' which appears within an attribute-token.
  ///
  /// \return the parsed identifier on success, and 0 if the next token is not
  /// an attribute-token.
  ///
  /// C++11 [dcl.attr.grammar]p3:
  ///   If a keyword or an alternative token that satisfies the syntactic
  ///   requirements of an identifier is contained in an attribute-token,
  ///   it is considered an identifier.
  IdentifierInfo *TryParseCXX11AttributeIdentifier(
      SourceLocation &Loc,
      SemaCodeCompletion::AttributeCompletion Completion =
          SemaCodeCompletion::AttributeCompletion::None,
      const IdentifierInfo *EnclosingScope = nullptr);

  /// Parse uuid() attribute when it appears in a [] Microsoft attribute.
  void ParseMicrosoftUuidAttributeArgs(ParsedAttributes &Attrs);

  /// ParseMicrosoftAttributes - Parse Microsoft attributes [Attr]
  ///
  /// \verbatim
  /// [MS] ms-attribute:
  ///             '[' token-seq ']'
  ///
  /// [MS] ms-attribute-seq:
  ///             ms-attribute[opt]
  ///             ms-attribute ms-attribute-seq
  /// \endverbatim
  void ParseMicrosoftAttributes(ParsedAttributes &Attrs);

  void ParseMicrosoftInheritanceClassAttributes(ParsedAttributes &attrs);
  void ParseNullabilityClassAttributes(ParsedAttributes &attrs);

  /// ParseDecltypeSpecifier - Parse a C++11 decltype specifier.
  ///
  /// \verbatim
  /// 'decltype' ( expression )
  /// 'decltype' ( 'auto' )      [C++1y]
  /// \endverbatim
  ///
  SourceLocation ParseDecltypeSpecifier(DeclSpec &DS);
  void AnnotateExistingDecltypeSpecifier(const DeclSpec &DS,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc);

  /// isCXX11VirtSpecifier - Determine whether the given token is a C++11
  /// virt-specifier.
  ///
  /// \verbatim
  ///       virt-specifier:
  ///         override
  ///         final
  ///         __final
  /// \endverbatim
  VirtSpecifiers::Specifier isCXX11VirtSpecifier(const Token &Tok) const;
  VirtSpecifiers::Specifier isCXX11VirtSpecifier() const {
    return isCXX11VirtSpecifier(Tok);
  }

  /// ParseOptionalCXX11VirtSpecifierSeq - Parse a virt-specifier-seq.
  ///
  /// \verbatim
  ///       virt-specifier-seq:
  ///         virt-specifier
  ///         virt-specifier-seq virt-specifier
  /// \endverbatim
  void ParseOptionalCXX11VirtSpecifierSeq(VirtSpecifiers &VS, bool IsInterface,
                                          SourceLocation FriendLoc);

  /// isCXX11FinalKeyword - Determine whether the next token is a C++11
  /// 'final' or Microsoft 'sealed' contextual keyword.
  bool isCXX11FinalKeyword() const;

  /// isClassCompatibleKeyword - Determine whether the next token is a C++11
  /// 'final', a C++26 'trivially_relocatable_if_eligible',
  /// 'replaceable_if_eligible', or Microsoft 'sealed' or 'abstract' contextual
  /// keyword.
  bool isClassCompatibleKeyword() const;

  bool MaybeParseTypeTransformTypeSpecifier(DeclSpec &DS);
  DeclSpec::TST TypeTransformTokToDeclSpec();

  void DiagnoseUnexpectedNamespace(NamedDecl *Context);

  /// ParseNamespace - We know that the current token is a namespace keyword.
  /// This may either be a top level namespace or a block-level namespace alias.
  /// If there was an inline keyword, it has already been parsed.
  ///
  /// \verbatim
  ///       namespace-definition: [C++: namespace.def]
  ///         named-namespace-definition
  ///         unnamed-namespace-definition
  ///         nested-namespace-definition
  ///
  ///       named-namespace-definition:
  ///         'inline'[opt] 'namespace' attributes[opt] identifier '{'
  ///         namespace-body '}'
  ///
  ///       unnamed-namespace-definition:
  ///         'inline'[opt] 'namespace' attributes[opt] '{' namespace-body '}'
  ///
  ///       nested-namespace-definition:
  ///         'namespace' enclosing-namespace-specifier '::' 'inline'[opt]
  ///         identifier '{' namespace-body '}'
  ///
  ///       enclosing-namespace-specifier:
  ///         identifier
  ///         enclosing-namespace-specifier '::' 'inline'[opt] identifier
  ///
  ///       namespace-alias-definition:  [C++ 7.3.2: namespace.alias]
  ///         'namespace' identifier '=' qualified-namespace-specifier ';'
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseNamespace(DeclaratorContext Context,
                                SourceLocation &DeclEnd,
                                SourceLocation InlineLoc = SourceLocation());

  struct InnerNamespaceInfo {
    SourceLocation NamespaceLoc;
    SourceLocation InlineLoc;
    SourceLocation IdentLoc;
    IdentifierInfo *Ident;
  };
  using InnerNamespaceInfoList = llvm::SmallVector<InnerNamespaceInfo, 4>;

  /// ParseInnerNamespace - Parse the contents of a namespace.
  void ParseInnerNamespace(const InnerNamespaceInfoList &InnerNSs,
                           unsigned int index, SourceLocation &InlineLoc,
                           ParsedAttributes &attrs,
                           BalancedDelimiterTracker &Tracker);

  /// ParseLinkage - We know that the current token is a string_literal
  /// and just before that, that extern was seen.
  ///
  /// \verbatim
  ///       linkage-specification: [C++ 7.5p2: dcl.link]
  ///         'extern' string-literal '{' declaration-seq[opt] '}'
  ///         'extern' string-literal declaration
  /// \endverbatim
  ///
  Decl *ParseLinkage(ParsingDeclSpec &DS, DeclaratorContext Context);

  /// Parse a standard C++ Modules export-declaration.
  ///
  /// \verbatim
  ///       export-declaration:
  ///         'export' declaration
  ///         'export' '{' declaration-seq[opt] '}'
  /// \endverbatim
  ///
  /// HLSL: Parse export function declaration.
  ///
  /// \verbatim
  ///      export-function-declaration:
  ///         'export' function-declaration
  ///
  ///      export-declaration-group:
  ///         'export' '{' function-declaration-seq[opt] '}'
  /// \endverbatim
  ///
  Decl *ParseExportDeclaration();

  /// ParseUsingDirectiveOrDeclaration - Parse C++ using using-declaration or
  /// using-directive. Assumes that current token is 'using'.
  DeclGroupPtrTy ParseUsingDirectiveOrDeclaration(
      DeclaratorContext Context, const ParsedTemplateInfo &TemplateInfo,
      SourceLocation &DeclEnd, ParsedAttributes &Attrs);

  /// ParseUsingDirective - Parse C++ using-directive, assumes
  /// that current token is 'namespace' and 'using' was already parsed.
  ///
  /// \verbatim
  ///       using-directive: [C++ 7.3.p4: namespace.udir]
  ///        'using' 'namespace' ::[opt] nested-name-specifier[opt]
  ///                 namespace-name ;
  /// [GNU] using-directive:
  ///        'using' 'namespace' ::[opt] nested-name-specifier[opt]
  ///                 namespace-name attributes[opt] ;
  /// \endverbatim
  ///
  Decl *ParseUsingDirective(DeclaratorContext Context, SourceLocation UsingLoc,
                            SourceLocation &DeclEnd, ParsedAttributes &attrs);

  struct UsingDeclarator {
    SourceLocation TypenameLoc;
    CXXScopeSpec SS;
    UnqualifiedId Name;
    SourceLocation EllipsisLoc;

    void clear() {
      TypenameLoc = EllipsisLoc = SourceLocation();
      SS.clear();
      Name.clear();
    }
  };

  /// Parse a using-declarator (or the identifier in a C++11 alias-declaration).
  ///
  /// \verbatim
  ///     using-declarator:
  ///       'typename'[opt] nested-name-specifier unqualified-id
  /// \endverbatim
  ///
  bool ParseUsingDeclarator(DeclaratorContext Context, UsingDeclarator &D);

  /// ParseUsingDeclaration - Parse C++ using-declaration or alias-declaration.
  /// Assumes that 'using' was already seen.
  ///
  /// \verbatim
  ///     using-declaration: [C++ 7.3.p3: namespace.udecl]
  ///       'using' using-declarator-list[opt] ;
  ///
  ///     using-declarator-list: [C++1z]
  ///       using-declarator '...'[opt]
  ///       using-declarator-list ',' using-declarator '...'[opt]
  ///
  ///     using-declarator-list: [C++98-14]
  ///       using-declarator
  ///
  ///     alias-declaration: C++11 [dcl.dcl]p1
  ///       'using' identifier attribute-specifier-seq[opt] = type-id ;
  ///
  ///     using-enum-declaration: [C++20, dcl.enum]
  ///       'using' elaborated-enum-specifier ;
  ///       The terminal name of the elaborated-enum-specifier undergoes
  ///       type-only lookup
  ///
  ///     elaborated-enum-specifier:
  ///       'enum' nested-name-specifier[opt] identifier
  /// \endverbatim
  DeclGroupPtrTy ParseUsingDeclaration(DeclaratorContext Context,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       SourceLocation UsingLoc,
                                       SourceLocation &DeclEnd,
                                       ParsedAttributes &Attrs,
                                       AccessSpecifier AS = AS_none);
  Decl *ParseAliasDeclarationAfterDeclarator(
      const ParsedTemplateInfo &TemplateInfo, SourceLocation UsingLoc,
      UsingDeclarator &D, SourceLocation &DeclEnd, AccessSpecifier AS,
      ParsedAttributes &Attrs, Decl **OwnedType = nullptr);

  /// ParseStaticAssertDeclaration - Parse C++0x or C11
  /// static_assert-declaration.
  ///
  /// \verbatim
  /// [C++0x] static_assert-declaration:
  ///           static_assert ( constant-expression  ,  string-literal  ) ;
  ///
  /// [C11]   static_assert-declaration:
  ///           _Static_assert ( constant-expression  ,  string-literal  ) ;
  /// \endverbatim
  ///
  Decl *ParseStaticAssertDeclaration(SourceLocation &DeclEnd);

  /// ParseNamespaceAlias - Parse the part after the '=' in a namespace
  /// alias definition.
  ///
  Decl *ParseNamespaceAlias(SourceLocation NamespaceLoc,
                            SourceLocation AliasLoc, IdentifierInfo *Alias,
                            SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.

  /// Determine whether the following tokens are valid after a type-specifier
  /// which could be a standalone declaration. This will conservatively return
  /// true if there's any doubt, and is appropriate for insert-';' fixits.
  bool isValidAfterTypeSpecifier(bool CouldBeBitfield);

  /// ParseClassSpecifier - Parse a C++ class-specifier [C++ class] or
  /// elaborated-type-specifier [C++ dcl.type.elab]; we can't tell which
  /// until we reach the start of a definition or see a token that
  /// cannot start a definition.
  ///
  /// \verbatim
  ///       class-specifier: [C++ class]
  ///         class-head '{' member-specification[opt] '}'
  ///         class-head '{' member-specification[opt] '}' attributes[opt]
  ///       class-head:
  ///         class-key identifier[opt] base-clause[opt]
  ///         class-key nested-name-specifier identifier base-clause[opt]
  ///         class-key nested-name-specifier[opt] simple-template-id
  ///                          base-clause[opt]
  /// [GNU]   class-key attributes[opt] identifier[opt] base-clause[opt]
  /// [GNU]   class-key attributes[opt] nested-name-specifier
  ///                          identifier base-clause[opt]
  /// [GNU]   class-key attributes[opt] nested-name-specifier[opt]
  ///                          simple-template-id base-clause[opt]
  ///       class-key:
  ///         'class'
  ///         'struct'
  ///         'union'
  ///
  ///       elaborated-type-specifier: [C++ dcl.type.elab]
  ///         class-key ::[opt] nested-name-specifier[opt] identifier
  ///         class-key ::[opt] nested-name-specifier[opt] 'template'[opt]
  ///                          simple-template-id
  ///
  ///  Note that the C++ class-specifier and elaborated-type-specifier,
  ///  together, subsume the C99 struct-or-union-specifier:
  ///
  ///       struct-or-union-specifier: [C99 6.7.2.1]
  ///         struct-or-union identifier[opt] '{' struct-contents '}'
  ///         struct-or-union identifier
  /// [GNU]   struct-or-union attributes[opt] identifier[opt] '{' struct-contents
  ///                                                         '}' attributes[opt]
  /// [GNU]   struct-or-union attributes[opt] identifier
  ///       struct-or-union:
  ///         'struct'
  ///         'union'
  /// \endverbatim
  void ParseClassSpecifier(tok::TokenKind TagTokKind, SourceLocation TagLoc,
                           DeclSpec &DS, ParsedTemplateInfo &TemplateInfo,
                           AccessSpecifier AS, bool EnteringContext,
                           DeclSpecContext DSC, ParsedAttributes &Attributes);
  void SkipCXXMemberSpecification(SourceLocation StartLoc,
                                  SourceLocation AttrFixitLoc, unsigned TagType,
                                  Decl *TagDecl);

  /// ParseCXXMemberSpecification - Parse the class definition.
  ///
  /// \verbatim
  ///       member-specification:
  ///         member-declaration member-specification[opt]
  ///         access-specifier ':' member-specification[opt]
  /// \endverbatim
  ///
  void ParseCXXMemberSpecification(SourceLocation StartLoc,
                                   SourceLocation AttrFixitLoc,
                                   ParsedAttributes &Attrs, unsigned TagType,
                                   Decl *TagDecl);

  /// ParseCXXMemberInitializer - Parse the brace-or-equal-initializer.
  /// Also detect and reject any attempted defaulted/deleted function
  /// definition. The location of the '=', if any, will be placed in EqualLoc.
  ///
  /// This does not check for a pure-specifier; that's handled elsewhere.
  ///
  /// \verbatim
  ///   brace-or-equal-initializer:
  ///     '=' initializer-expression
  ///     braced-init-list
  ///
  ///   initializer-clause:
  ///     assignment-expression
  ///     braced-init-list
  ///
  ///   defaulted/deleted function-definition:
  ///     '=' 'default'
  ///     '=' 'delete'
  /// \endverbatim
  ///
  /// Prior to C++0x, the assignment-expression in an initializer-clause must
  /// be a constant-expression.
  ExprResult ParseCXXMemberInitializer(Decl *D, bool IsFunction,
                                       SourceLocation &EqualLoc);

  /// Parse a C++ member-declarator up to, but not including, the optional
  /// brace-or-equal-initializer or pure-specifier.
  bool ParseCXXMemberDeclaratorBeforeInitializer(Declarator &DeclaratorInfo,
                                                 VirtSpecifiers &VS,
                                                 ExprResult &BitfieldSize,
                                                 LateParsedAttrList &LateAttrs);

  /// Look for declaration specifiers possibly occurring after C++11
  /// virt-specifier-seq and diagnose them.
  void
  MaybeParseAndDiagnoseDeclSpecAfterCXX11VirtSpecifierSeq(Declarator &D,
                                                          VirtSpecifiers &VS);

  /// ParseCXXClassMemberDeclaration - Parse a C++ class member declaration.
  ///
  /// \verbatim
  ///       member-declaration:
  ///         decl-specifier-seq[opt] member-declarator-list[opt] ';'
  ///         function-definition ';'[opt]
  /// [C++26] friend-type-declaration
  ///         ::[opt] nested-name-specifier template[opt] unqualified-id ';'[TODO]
  ///         using-declaration                                            [TODO]
  /// [C++0x] static_assert-declaration
  ///         template-declaration
  /// [GNU]   '__extension__' member-declaration
  ///
  ///       member-declarator-list:
  ///         member-declarator
  ///         member-declarator-list ',' member-declarator
  ///
  ///       member-declarator:
  ///         declarator virt-specifier-seq[opt] pure-specifier[opt]
  /// [C++2a] declarator requires-clause
  ///         declarator constant-initializer[opt]
  /// [C++11] declarator brace-or-equal-initializer[opt]
  ///         identifier[opt] ':' constant-expression
  ///
  ///       virt-specifier-seq:
  ///         virt-specifier
  ///         virt-specifier-seq virt-specifier
  ///
  ///       virt-specifier:
  ///         override
  ///         final
  /// [MS]    sealed
  ///
  ///       pure-specifier:
  ///         '= 0'
  ///
  ///       constant-initializer:
  ///         '=' constant-expression
  ///
  ///       friend-type-declaration:
  ///         'friend' friend-type-specifier-list ;
  ///
  ///       friend-type-specifier-list:
  ///         friend-type-specifier ...[opt]
  ///         friend-type-specifier-list , friend-type-specifier ...[opt]
  ///
  ///       friend-type-specifier:
  ///         simple-type-specifier
  ///         elaborated-type-specifier
  ///         typename-specifier
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseCXXClassMemberDeclaration(
      AccessSpecifier AS, ParsedAttributes &Attr,
      ParsedTemplateInfo &TemplateInfo,
      ParsingDeclRAIIObject *DiagsFromTParams = nullptr);
  DeclGroupPtrTy
  ParseCXXClassMemberDeclarationWithPragmas(AccessSpecifier &AS,
                                            ParsedAttributes &AccessAttrs,
                                            DeclSpec::TST TagType, Decl *Tag);

  /// ParseConstructorInitializer - Parse a C++ constructor initializer,
  /// which explicitly initializes the members or base classes of a
  /// class (C++ [class.base.init]). For example, the three initializers
  /// after the ':' in the Derived constructor below:
  ///
  /// @code
  /// class Base { };
  /// class Derived : Base {
  ///   int x;
  ///   float f;
  /// public:
  ///   Derived(float f) : Base(), x(17), f(f) { }
  /// };
  /// @endcode
  ///
  /// \verbatim
  /// [C++]  ctor-initializer:
  ///          ':' mem-initializer-list
  ///
  /// [C++]  mem-initializer-list:
  ///          mem-initializer ...[opt]
  ///          mem-initializer ...[opt] , mem-initializer-list
  /// \endverbatim
  void ParseConstructorInitializer(Decl *ConstructorDecl);

  /// ParseMemInitializer - Parse a C++ member initializer, which is
  /// part of a constructor initializer that explicitly initializes one
  /// member or base class (C++ [class.base.init]). See
  /// ParseConstructorInitializer for an example.
  ///
  /// \verbatim
  /// [C++] mem-initializer:
  ///         mem-initializer-id '(' expression-list[opt] ')'
  /// [C++0x] mem-initializer-id braced-init-list
  ///
  /// [C++] mem-initializer-id:
  ///         '::'[opt] nested-name-specifier[opt] class-name
  ///         identifier
  /// \endverbatim
  MemInitResult ParseMemInitializer(Decl *ConstructorDecl);

  /// If the given declarator has any parts for which parsing has to be
  /// delayed, e.g., default arguments or an exception-specification, create a
  /// late-parsed method declaration record to handle the parsing at the end of
  /// the class definition.
  void HandleMemberFunctionDeclDelays(Declarator &DeclaratorInfo,
                                      Decl *ThisDecl);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]

  /// ParseBaseTypeSpecifier - Parse a C++ base-type-specifier which is either a
  /// class name or decltype-specifier. Note that we only check that the result
  /// names a type; semantic analysis will need to verify that the type names a
  /// class. The result is either a type or null, depending on whether a type
  /// name was found.
  ///
  /// \verbatim
  ///       base-type-specifier: [C++11 class.derived]
  ///         class-or-decltype
  ///       class-or-decltype: [C++11 class.derived]
  ///         nested-name-specifier[opt] class-name
  ///         decltype-specifier
  ///       class-name: [C++ class.name]
  ///         identifier
  ///         simple-template-id
  /// \endverbatim
  ///
  /// In C++98, instead of base-type-specifier, we have:
  ///
  /// \verbatim
  ///         ::[opt] nested-name-specifier[opt] class-name
  /// \endverbatim
  TypeResult ParseBaseTypeSpecifier(SourceLocation &BaseLoc,
                                    SourceLocation &EndLocation);

  /// ParseBaseClause - Parse the base-clause of a C++ class [C++
  /// class.derived].
  ///
  /// \verbatim
  ///       base-clause : [C++ class.derived]
  ///         ':' base-specifier-list
  ///       base-specifier-list:
  ///         base-specifier '...'[opt]
  ///         base-specifier-list ',' base-specifier '...'[opt]
  /// \endverbatim
  void ParseBaseClause(Decl *ClassDecl);

  /// ParseBaseSpecifier - Parse a C++ base-specifier. A base-specifier is
  /// one entry in the base class list of a class specifier, for example:
  ///    class foo : public bar, virtual private baz {
  /// 'public bar' and 'virtual private baz' are each base-specifiers.
  ///
  /// \verbatim
  ///       base-specifier: [C++ class.derived]
  ///         attribute-specifier-seq[opt] base-type-specifier
  ///         attribute-specifier-seq[opt] 'virtual' access-specifier[opt]
  ///                 base-type-specifier
  ///         attribute-specifier-seq[opt] access-specifier 'virtual'[opt]
  ///                 base-type-specifier
  /// \endverbatim
  BaseResult ParseBaseSpecifier(Decl *ClassDecl);

  /// getAccessSpecifierIfPresent - Determine whether the next token is
  /// a C++ access-specifier.
  ///
  /// \verbatim
  ///       access-specifier: [C++ class.derived]
  ///         'private'
  ///         'protected'
  ///         'public'
  /// \endverbatim
  AccessSpecifier getAccessSpecifierIfPresent() const;

  bool isCXX2CTriviallyRelocatableKeyword(Token Tok) const;
  bool isCXX2CTriviallyRelocatableKeyword() const;
  void ParseCXX2CTriviallyRelocatableSpecifier(SourceLocation &TRS);

  bool isCXX2CReplaceableKeyword(Token Tok) const;
  bool isCXX2CReplaceableKeyword() const;
  void ParseCXX2CReplaceableSpecifier(SourceLocation &MRS);

  /// 'final', a C++26 'trivially_relocatable_if_eligible',
  /// 'replaceable_if_eligible', or Microsoft 'sealed' or 'abstract' contextual
  /// keyword.
  bool isClassCompatibleKeyword(Token Tok) const;

  void ParseHLSLRootSignatureAttributeArgs(ParsedAttributes &Attrs);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Expressions
  /// Implementations are in ParseExpr.cpp
  ///@{

public:
  friend class OffsetOfStateRAIIObject;

  typedef Sema::FullExprArg FullExprArg;

  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.

  /// Simple precedence-based parser for binary/ternary operators.
  ///
  /// Note: we diverge from the C99 grammar when parsing the
  /// assignment-expression production.  C99 specifies that the LHS of an
  /// assignment operator should be parsed as a unary-expression, but
  /// consistency dictates that it be a conditional-expession.  In practice, the
  /// important thing here is that the LHS of an assignment has to be an
  /// l-value, which productions between unary-expression and
  /// conditional-expression don't produce.  Because we want consistency, we
  /// parse the LHS as a conditional-expression, then check for l-value-ness in
  /// semantic analysis stages.
  ///
  /// \verbatim
  ///       pm-expression: [C++ 5.5]
  ///         cast-expression
  ///         pm-expression '.*' cast-expression
  ///         pm-expression '->*' cast-expression
  ///
  ///       multiplicative-expression: [C99 6.5.5]
  ///     Note: in C++, apply pm-expression instead of cast-expression
  ///         cast-expression
  ///         multiplicative-expression '*' cast-expression
  ///         multiplicative-expression '/' cast-expression
  ///         multiplicative-expression '%' cast-expression
  ///
  ///       additive-expression: [C99 6.5.6]
  ///         multiplicative-expression
  ///         additive-expression '+' multiplicative-expression
  ///         additive-expression '-' multiplicative-expression
  ///
  ///       shift-expression: [C99 6.5.7]
  ///         additive-expression
  ///         shift-expression '<<' additive-expression
  ///         shift-expression '>>' additive-expression
  ///
  ///       compare-expression: [C++20 expr.spaceship]
  ///         shift-expression
  ///         compare-expression '<=>' shift-expression
  ///
  ///       relational-expression: [C99 6.5.8]
  ///         compare-expression
  ///         relational-expression '<' compare-expression
  ///         relational-expression '>' compare-expression
  ///         relational-expression '<=' compare-expression
  ///         relational-expression '>=' compare-expression
  ///
  ///       equality-expression: [C99 6.5.9]
  ///         relational-expression
  ///         equality-expression '==' relational-expression
  ///         equality-expression '!=' relational-expression
  ///
  ///       AND-expression: [C99 6.5.10]
  ///         equality-expression
  ///         AND-expression '&' equality-expression
  ///
  ///       exclusive-OR-expression: [C99 6.5.11]
  ///         AND-expression
  ///         exclusive-OR-expression '^' AND-expression
  ///
  ///       inclusive-OR-expression: [C99 6.5.12]
  ///         exclusive-OR-expression
  ///         inclusive-OR-expression '|' exclusive-OR-expression
  ///
  ///       logical-AND-expression: [C99 6.5.13]
  ///         inclusive-OR-expression
  ///         logical-AND-expression '&&' inclusive-OR-expression
  ///
  ///       logical-OR-expression: [C99 6.5.14]
  ///         logical-AND-expression
  ///         logical-OR-expression '||' logical-AND-expression
  ///
  ///       conditional-expression: [C99 6.5.15]
  ///         logical-OR-expression
  ///         logical-OR-expression '?' expression ':' conditional-expression
  /// [GNU]   logical-OR-expression '?' ':' conditional-expression
  /// [C++] the third operand is an assignment-expression
  ///
  ///       assignment-expression: [C99 6.5.16]
  ///         conditional-expression
  ///         unary-expression assignment-operator assignment-expression
  /// [C++]   throw-expression [C++ 15]
  ///
  ///       assignment-operator: one of
  ///         = *= /= %= += -= <<= >>= &= ^= |=
  ///
  ///       expression: [C99 6.5.17]
  ///         assignment-expression ...[opt]
  ///         expression ',' assignment-expression ...[opt]
  /// \endverbatim
  ExprResult ParseExpression(TypoCorrectionTypeBehavior CorrectionBehavior =
                                 TypoCorrectionTypeBehavior::AllowNonTypes);

  ExprResult ParseConstantExpressionInExprEvalContext(
      TypoCorrectionTypeBehavior CorrectionBehavior =
          TypoCorrectionTypeBehavior::AllowNonTypes);
  ExprResult ParseConstantExpression();
  ExprResult ParseArrayBoundExpression();
  ExprResult ParseCaseExpression(SourceLocation CaseLoc);

  /// Parse a constraint-expression.
  ///
  /// \verbatim
  ///       constraint-expression: C++2a[temp.constr.decl]p1
  ///         logical-or-expression
  /// \endverbatim
  ExprResult ParseConstraintExpression();

  /// \brief Parse a constraint-logical-and-expression.
  ///
  /// \verbatim
  ///       C++2a[temp.constr.decl]p1
  ///       constraint-logical-and-expression:
  ///         primary-expression
  ///         constraint-logical-and-expression '&&' primary-expression
  ///
  /// \endverbatim
  ExprResult ParseConstraintLogicalAndExpression(bool IsTrailingRequiresClause);

  /// \brief Parse a constraint-logical-or-expression.
  ///
  /// \verbatim
  ///       C++2a[temp.constr.decl]p1
  ///       constraint-logical-or-expression:
  ///         constraint-logical-and-expression
  ///         constraint-logical-or-expression '||'
  ///             constraint-logical-and-expression
  ///
  /// \endverbatim
  ExprResult ParseConstraintLogicalOrExpression(bool IsTrailingRequiresClause);

  /// Parse an expr that doesn't include (top-level) commas.
  ExprResult
  ParseAssignmentExpression(TypoCorrectionTypeBehavior CorrectionBehavior =
                                TypoCorrectionTypeBehavior::AllowNonTypes);

  ExprResult ParseConditionalExpression();

  /// ParseStringLiteralExpression - This handles the various token types that
  /// form string literals, and also handles string concatenation [C99 5.1.1.2,
  /// translation phase #6].
  ///
  /// \verbatim
  ///       primary-expression: [C99 6.5.1]
  ///         string-literal
  /// \endverbatim
  ExprResult ParseStringLiteralExpression(bool AllowUserDefinedLiteral = false);
  ExprResult ParseUnevaluatedStringLiteralExpression();

private:
  /// Whether the '>' token acts as an operator or not. This will be
  /// true except when we are parsing an expression within a C++
  /// template argument list, where the '>' closes the template
  /// argument list.
  bool GreaterThanIsOperator;

  // C++ type trait keywords that can be reverted to identifiers and still be
  // used as type traits.
  llvm::SmallDenseMap<IdentifierInfo *, tok::TokenKind> RevertibleTypeTraits;

  OffsetOfKind OffsetOfState = OffsetOfKind::Outside;

  /// The location of the expression statement that is being parsed right now.
  /// Used to determine if an expression that is being parsed is a statement or
  /// just a regular sub-expression.
  SourceLocation ExprStatementTokLoc;

  /// Checks if the \p Level is valid for use in a fold expression.
  bool isFoldOperator(prec::Level Level) const;

  /// Checks if the \p Kind is a valid operator for fold expressions.
  bool isFoldOperator(tok::TokenKind Kind) const;

  /// We have just started parsing the definition of a new class,
  /// so push that class onto our stack of classes that is currently
  /// being parsed.
  Sema::ParsingClassState
  PushParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface);

  /// Deallocate the given parsed class and all of its nested
  /// classes.
  void DeallocateParsedClasses(ParsingClass *Class);

  /// Pop the top class of the stack of classes that are
  /// currently being parsed.
  ///
  /// This routine should be called when we have finished parsing the
  /// definition of a class, but have not yet popped the Scope
  /// associated with the class's definition.
  void PopParsingClass(Sema::ParsingClassState);

  ExprResult ParseStringLiteralExpression(bool AllowUserDefinedLiteral,
                                          bool Unevaluated);

  /// This routine is called when the '@' is seen and consumed.
  /// Current token is an Identifier and is not a 'try'. This
  /// routine is necessary to disambiguate \@try-statement from,
  /// for example, \@encode-expression.
  ///
  ExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  /// This routine is called when a leading '__extension__' is seen and
  /// consumed.  This is necessary because the token gets consumed in the
  /// process of disambiguating between an expression and a declaration.
  ExprResult ParseExpressionWithLeadingExtension(SourceLocation ExtLoc);

  /// Parse a binary expression that starts with \p LHS and has a
  /// precedence of at least \p MinPrec.
  ExprResult ParseRHSOfBinaryExpression(ExprResult LHS, prec::Level MinPrec);

  bool isRevertibleTypeTrait(const IdentifierInfo *Id,
                             clang::tok::TokenKind *Kind = nullptr);

  /// Parse a cast-expression, or, if \pisUnaryExpression is true, parse
  /// a unary-expression.
  ///
  /// \p isAddressOfOperand exists because an id-expression that is the operand
  /// of address-of gets special treatment due to member pointers. NotCastExpr
  /// is set to true if the token is not the start of a cast-expression, and no
  /// diagnostic is emitted in this case and no tokens are consumed.
  ///
  /// \verbatim
  ///       cast-expression: [C99 6.5.4]
  ///         unary-expression
  ///         '(' type-name ')' cast-expression
  ///
  ///       unary-expression:  [C99 6.5.3]
  ///         postfix-expression
  ///         '++' unary-expression
  ///         '--' unary-expression
  /// [Coro]  'co_await' cast-expression
  ///         unary-operator cast-expression
  ///         'sizeof' unary-expression
  ///         'sizeof' '(' type-name ')'
  /// [C++11] 'sizeof' '...' '(' identifier ')'
  /// [GNU]   '__alignof' unary-expression
  /// [GNU]   '__alignof' '(' type-name ')'
  /// [C11]   '_Alignof' '(' type-name ')'
  /// [C++11] 'alignof' '(' type-id ')'
  /// [C2y]   '_Countof' unary-expression
  /// [C2y]   '_Countof' '(' type-name ')'
  /// [GNU]   '&&' identifier
  /// [C++11] 'noexcept' '(' expression ')' [C++11 5.3.7]
  /// [C++]   new-expression
  /// [C++]   delete-expression
  ///
  ///       unary-operator: one of
  ///         '&'  '*'  '+'  '-'  '~'  '!'
  /// [GNU]   '__extension__'  '__real'  '__imag'
  ///
  ///       primary-expression: [C99 6.5.1]
  /// [C99]   identifier
  /// [C++]   id-expression
  ///         constant
  ///         string-literal
  /// [C++]   boolean-literal  [C++ 2.13.5]
  /// [C++11] 'nullptr'        [C++11 2.14.7]
  /// [C++11] user-defined-literal
  ///         '(' expression ')'
  /// [C11]   generic-selection
  /// [C++2a] requires-expression
  ///         '__func__'        [C99 6.4.2.2]
  /// [GNU]   '__FUNCTION__'
  /// [MS]    '__FUNCDNAME__'
  /// [MS]    'L__FUNCTION__'
  /// [MS]    '__FUNCSIG__'
  /// [MS]    'L__FUNCSIG__'
  /// [GNU]   '__PRETTY_FUNCTION__'
  /// [GNU]   '(' compound-statement ')'
  /// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
  /// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
  /// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
  ///                                     assign-expr ')'
  /// [GNU]   '__builtin_FILE' '(' ')'
  /// [CLANG] '__builtin_FILE_NAME' '(' ')'
  /// [GNU]   '__builtin_FUNCTION' '(' ')'
  /// [MS]    '__builtin_FUNCSIG' '(' ')'
  /// [GNU]   '__builtin_LINE' '(' ')'
  /// [CLANG] '__builtin_COLUMN' '(' ')'
  /// [GNU]   '__builtin_source_location' '(' ')'
  /// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
  /// [GNU]   '__null'
  /// [OBJC]  '[' objc-message-expr ']'
  /// [OBJC]  '\@selector' '(' objc-selector-arg ')'
  /// [OBJC]  '\@protocol' '(' identifier ')'
  /// [OBJC]  '\@encode' '(' type-name ')'
  /// [OBJC]  objc-string-literal
  /// [C++]   simple-type-specifier '(' expression-list[opt] ')'      [C++ 5.2.3]
  /// [C++11] simple-type-specifier braced-init-list                  [C++11 5.2.3]
  /// [C++]   typename-specifier '(' expression-list[opt] ')'         [C++ 5.2.3]
  /// [C++11] typename-specifier braced-init-list                     [C++11 5.2.3]
  /// [C++]   'const_cast' '<' type-name '>' '(' expression ')'       [C++ 5.2p1]
  /// [C++]   'dynamic_cast' '<' type-name '>' '(' expression ')'     [C++ 5.2p1]
  /// [C++]   'reinterpret_cast' '<' type-name '>' '(' expression ')' [C++ 5.2p1]
  /// [C++]   'static_cast' '<' type-name '>' '(' expression ')'      [C++ 5.2p1]
  /// [C++]   'typeid' '(' expression ')'                             [C++ 5.2p1]
  /// [C++]   'typeid' '(' type-id ')'                                [C++ 5.2p1]
  /// [C++]   'this'          [C++ 9.3.2]
  /// [G++]   unary-type-trait '(' type-id ')'
  /// [G++]   binary-type-trait '(' type-id ',' type-id ')'           [TODO]
  /// [EMBT]  array-type-trait '(' type-id ',' integer ')'
  /// [clang] '^' block-literal
  ///
  ///       constant: [C99 6.4.4]
  ///         integer-constant
  ///         floating-constant
  ///         enumeration-constant -> identifier
  ///         character-constant
  ///
  ///       id-expression: [C++ 5.1]
  ///                   unqualified-id
  ///                   qualified-id
  ///
  ///       unqualified-id: [C++ 5.1]
  ///                   identifier
  ///                   operator-function-id
  ///                   conversion-function-id
  ///                   '~' class-name
  ///                   template-id
  ///
  ///       new-expression: [C++ 5.3.4]
  ///                   '::'[opt] 'new' new-placement[opt] new-type-id
  ///                                     new-initializer[opt]
  ///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
  ///                                     new-initializer[opt]
  ///
  ///       delete-expression: [C++ 5.3.5]
  ///                   '::'[opt] 'delete' cast-expression
  ///                   '::'[opt] 'delete' '[' ']' cast-expression
  ///
  /// [GNU/Embarcadero] unary-type-trait:
  ///                   '__is_arithmetic'
  ///                   '__is_floating_point'
  ///                   '__is_integral'
  ///                   '__is_lvalue_expr'
  ///                   '__is_rvalue_expr'
  ///                   '__is_complete_type'
  ///                   '__is_void'
  ///                   '__is_array'
  ///                   '__is_function'
  ///                   '__is_reference'
  ///                   '__is_lvalue_reference'
  ///                   '__is_rvalue_reference'
  ///                   '__is_fundamental'
  ///                   '__is_object'
  ///                   '__is_scalar'
  ///                   '__is_compound'
  ///                   '__is_pointer'
  ///                   '__is_member_object_pointer'
  ///                   '__is_member_function_pointer'
  ///                   '__is_member_pointer'
  ///                   '__is_const'
  ///                   '__is_volatile'
  ///                   '__is_trivial'
  ///                   '__is_standard_layout'
  ///                   '__is_signed'
  ///                   '__is_unsigned'
  ///
  /// [GNU] unary-type-trait:
  ///                   '__has_nothrow_assign'
  ///                   '__has_nothrow_copy'
  ///                   '__has_nothrow_constructor'
  ///                   '__has_trivial_assign'                  [TODO]
  ///                   '__has_trivial_copy'                    [TODO]
  ///                   '__has_trivial_constructor'
  ///                   '__has_trivial_destructor'
  ///                   '__has_virtual_destructor'
  ///                   '__is_abstract'                         [TODO]
  ///                   '__is_class'
  ///                   '__is_empty'                            [TODO]
  ///                   '__is_enum'
  ///                   '__is_final'
  ///                   '__is_pod'
  ///                   '__is_polymorphic'
  ///                   '__is_sealed'                           [MS]
  ///                   '__is_trivial'
  ///                   '__is_union'
  ///                   '__has_unique_object_representations'
  ///
  /// [Clang] unary-type-trait:
  ///                   '__is_aggregate'
  ///                   '__trivially_copyable'
  ///
  ///       binary-type-trait:
  /// [GNU]             '__is_base_of'
  /// [MS]              '__is_convertible_to'
  ///                   '__is_convertible'
  ///                   '__is_same'
  ///
  /// [Embarcadero] array-type-trait:
  ///                   '__array_rank'
  ///                   '__array_extent'
  ///
  /// [Embarcadero] expression-trait:
  ///                   '__is_lvalue_expr'
  ///                   '__is_rvalue_expr'
  /// \endverbatim
  ///
  ExprResult ParseCastExpression(CastParseKind ParseKind,
                                 bool isAddressOfOperand, bool &NotCastExpr,
                                 TypoCorrectionTypeBehavior CorrectionBehavior,
                                 bool isVectorLiteral = false,
                                 bool *NotPrimaryExpression = nullptr);
  ExprResult ParseCastExpression(CastParseKind ParseKind,
                                 bool isAddressOfOperand = false,
                                 TypoCorrectionTypeBehavior CorrectionBehavior =
                                     TypoCorrectionTypeBehavior::AllowNonTypes,
                                 bool isVectorLiteral = false,
                                 bool *NotPrimaryExpression = nullptr);

  /// Returns true if the next token cannot start an expression.
  bool isNotExpressionStart();

  /// Returns true if the next token would start a postfix-expression
  /// suffix.
  bool isPostfixExpressionSuffixStart() {
    tok::TokenKind K = Tok.getKind();
    return (K == tok::l_square || K == tok::l_paren || K == tok::period ||
            K == tok::arrow || K == tok::plusplus || K == tok::minusminus);
  }

  /// Once the leading part of a postfix-expression is parsed, this
  /// method parses any suffixes that apply.
  ///
  /// \verbatim
  ///       postfix-expression: [C99 6.5.2]
  ///         primary-expression
  ///         postfix-expression '[' expression ']'
  ///         postfix-expression '[' braced-init-list ']'
  ///         postfix-expression '[' expression-list [opt] ']'  [C++23 12.4.5]
  ///         postfix-expression '(' argument-expression-list[opt] ')'
  ///         postfix-expression '.' identifier
  ///         postfix-expression '->' identifier
  ///         postfix-expression '++'
  ///         postfix-expression '--'
  ///         '(' type-name ')' '{' initializer-list '}'
  ///         '(' type-name ')' '{' initializer-list ',' '}'
  ///
  ///       argument-expression-list: [C99 6.5.2]
  ///         argument-expression ...[opt]
  ///         argument-expression-list ',' assignment-expression ...[opt]
  /// \endverbatim
  ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);

  /// Parse a sizeof or alignof expression.
  ///
  /// \verbatim
  ///       unary-expression:  [C99 6.5.3]
  ///         'sizeof' unary-expression
  ///         'sizeof' '(' type-name ')'
  /// [C++11] 'sizeof' '...' '(' identifier ')'
  /// [Clang] '__datasizeof' unary-expression
  /// [Clang] '__datasizeof' '(' type-name ')'
  /// [GNU]   '__alignof' unary-expression
  /// [GNU]   '__alignof' '(' type-name ')'
  /// [C11]   '_Alignof' '(' type-name ')'
  /// [C++11] 'alignof' '(' type-id ')'
  /// [C2y]   '_Countof' unary-expression
  /// [C2y]   '_Countof' '(' type-name ')'
  /// \endverbatim
  ExprResult ParseUnaryExprOrTypeTraitExpression();

  /// ParseBuiltinPrimaryExpression
  ///
  /// \verbatim
  ///       primary-expression: [C99 6.5.1]
  /// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
  /// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
  /// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
  ///                                     assign-expr ')'
  /// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
  /// [GNU]   '__builtin_FILE' '(' ')'
  /// [CLANG] '__builtin_FILE_NAME' '(' ')'
  /// [GNU]   '__builtin_FUNCTION' '(' ')'
  /// [MS]    '__builtin_FUNCSIG' '(' ')'
  /// [GNU]   '__builtin_LINE' '(' ')'
  /// [CLANG] '__builtin_COLUMN' '(' ')'
  /// [GNU]   '__builtin_source_location' '(' ')'
  /// [OCL]   '__builtin_astype' '(' assignment-expression ',' type-name ')'
  ///
  /// [GNU] offsetof-member-designator:
  /// [GNU]   identifier
  /// [GNU]   offsetof-member-designator '.' identifier
  /// [GNU]   offsetof-member-designator '[' expression ']'
  /// \endverbatim
  ExprResult ParseBuiltinPrimaryExpression();

  /// Parse a __builtin_sycl_unique_stable_name expression.  Accepts a type-id
  /// as a parameter.
  ExprResult ParseSYCLUniqueStableNameExpression();

  /// ParseExprAfterUnaryExprOrTypeTrait - We parsed a typeof/sizeof/alignof/
  /// vec_step and we are at the start of an expression or a parenthesized
  /// type-id. OpTok is the operand token (typeof/sizeof/alignof). Returns the
  /// expression (isCastExpr == false) or the type (isCastExpr == true).
  ///
  /// \verbatim
  ///       unary-expression:  [C99 6.5.3]
  ///         'sizeof' unary-expression
  ///         'sizeof' '(' type-name ')'
  /// [Clang] '__datasizeof' unary-expression
  /// [Clang] '__datasizeof' '(' type-name ')'
  /// [GNU]   '__alignof' unary-expression
  /// [GNU]   '__alignof' '(' type-name ')'
  /// [C11]   '_Alignof' '(' type-name ')'
  /// [C++0x] 'alignof' '(' type-id ')'
  ///
  /// [GNU]   typeof-specifier:
  ///           typeof ( expressions )
  ///           typeof ( type-name )
  /// [GNU/C++] typeof unary-expression
  /// [C23]   typeof-specifier:
  ///           typeof '(' typeof-specifier-argument ')'
  ///           typeof_unqual '(' typeof-specifier-argument ')'
  ///
  ///         typeof-specifier-argument:
  ///           expression
  ///           type-name
  ///
  /// [OpenCL 1.1 6.11.12] vec_step built-in function:
  ///           vec_step ( expressions )
  ///           vec_step ( type-name )
  /// \endverbatim
  ExprResult ParseExprAfterUnaryExprOrTypeTrait(const Token &OpTok,
                                                bool &isCastExpr,
                                                ParsedType &CastTy,
                                                SourceRange &CastRange);

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  ///
  /// \verbatim
  ///       argument-expression-list:
  ///         assignment-expression
  ///         argument-expression-list , assignment-expression
  ///
  /// [C++] expression-list:
  /// [C++]   assignment-expression
  /// [C++]   expression-list , assignment-expression
  ///
  /// [C++0x] expression-list:
  /// [C++0x]   initializer-list
  ///
  /// [C++0x] initializer-list
  /// [C++0x]   initializer-clause ...[opt]
  /// [C++0x]   initializer-list , initializer-clause ...[opt]
  ///
  /// [C++0x] initializer-clause:
  /// [C++0x]   assignment-expression
  /// [C++0x]   braced-init-list
  /// \endverbatim
  bool ParseExpressionList(SmallVectorImpl<Expr *> &Exprs,
                           llvm::function_ref<void()> ExpressionStarts =
                               llvm::function_ref<void()>(),
                           bool FailImmediatelyOnInvalidExpr = false);

  /// ParseSimpleExpressionList - A simple comma-separated list of expressions,
  /// used for misc language extensions.
  ///
  /// \verbatim
  ///       simple-expression-list:
  ///         assignment-expression
  ///         simple-expression-list , assignment-expression
  /// \endverbatim
  bool ParseSimpleExpressionList(SmallVectorImpl<Expr *> &Exprs);

  /// This parses the unit that starts with a '(' token, based on what is
  /// allowed by ExprType. The actual thing parsed is returned in ExprType. If
  /// StopIfCastExpr is true, it will only return the parsed type, not the
  /// parsed cast-expression. If ParenBehavior is ParenExprKind::PartOfOperator,
  /// the initial open paren and its matching close paren are known to be part
  /// of another grammar production and not part of the operand. e.g., the
  /// typeof and typeof_unqual operators in C. Otherwise, the function has to
  /// parse the parens to determine whether they're part of a cast or compound
  /// literal expression rather than a parenthesized type.
  ///
  /// \verbatim
  ///       primary-expression: [C99 6.5.1]
  ///         '(' expression ')'
  /// [GNU]   '(' compound-statement ')'      (if !ParenExprOnly)
  ///       postfix-expression: [C99 6.5.2]
  ///         '(' type-name ')' '{' initializer-list '}'
  ///         '(' type-name ')' '{' initializer-list ',' '}'
  ///       cast-expression: [C99 6.5.4]
  ///         '(' type-name ')' cast-expression
  /// [ARC]   bridged-cast-expression
  /// [ARC] bridged-cast-expression:
  ///         (__bridge type-name) cast-expression
  ///         (__bridge_transfer type-name) cast-expression
  ///         (__bridge_retained type-name) cast-expression
  ///       fold-expression: [C++1z]
  ///         '(' cast-expression fold-operator '...' ')'
  ///         '(' '...' fold-operator cast-expression ')'
  ///         '(' cast-expression fold-operator '...'
  ///                 fold-operator cast-expression ')'
  /// [OPENMP] Array shaping operation
  ///       '(' '[' expression ']' { '[' expression ']' } cast-expression
  /// \endverbatim
  ExprResult ParseParenExpression(ParenParseOption &ExprType,
                                  bool StopIfCastExpr,
                                  ParenExprKind ParenBehavior,
                                  TypoCorrectionTypeBehavior CorrectionBehavior,
                                  ParsedType &CastTy,
                                  SourceLocation &RParenLoc);

  /// ParseCompoundLiteralExpression - We have parsed the parenthesized
  /// type-name and we are at the left brace.
  ///
  /// \verbatim
  ///       postfix-expression: [C99 6.5.2]
  ///         '(' type-name ')' '{' initializer-list '}'
  ///         '(' type-name ')' '{' initializer-list ',' '}'
  /// \endverbatim
  ExprResult ParseCompoundLiteralExpression(ParsedType Ty,
                                            SourceLocation LParenLoc,
                                            SourceLocation RParenLoc);

  /// ParseGenericSelectionExpression - Parse a C11 generic-selection
  /// [C11 6.5.1.1].
  ///
  /// \verbatim
  ///    generic-selection:
  ///           _Generic ( assignment-expression , generic-assoc-list )
  ///    generic-assoc-list:
  ///           generic-association
  ///           generic-assoc-list , generic-association
  ///    generic-association:
  ///           type-name : assignment-expression
  ///           default : assignment-expression
  /// \endverbatim
  ///
  /// As an extension, Clang also accepts:
  /// \verbatim
  ///   generic-selection:
  ///          _Generic ( type-name, generic-assoc-list )
  /// \endverbatim
  ExprResult ParseGenericSelectionExpression();

  /// ParseObjCBoolLiteral - This handles the objective-c Boolean literals.
  ///
  ///         '__objc_yes'
  ///         '__objc_no'
  ExprResult ParseObjCBoolLiteral();

  /// Parse A C++1z fold-expression after the opening paren and optional
  /// left-hand-side expression.
  ///
  /// \verbatim
  ///   fold-expression:
  ///       ( cast-expression fold-operator ... )
  ///       ( ... fold-operator cast-expression )
  ///       ( cast-expression fold-operator ... fold-operator cast-expression )
  /// \endverbatim
  ExprResult ParseFoldExpression(ExprResult LHS, BalancedDelimiterTracker &T);

  void injectEmbedTokens();

  //===--------------------------------------------------------------------===//
  // clang Expressions

  /// ParseBlockLiteralExpression - Parse a block literal, which roughly looks
  /// like ^(int x){ return x+1; }
  ///
  /// \verbatim
  ///         block-literal:
  /// [clang]   '^' block-args[opt] compound-statement
  /// [clang]   '^' block-id compound-statement
  /// [clang] block-args:
  /// [clang]   '(' parameter-list ')'
  /// \endverbatim
  ExprResult ParseBlockLiteralExpression(); // ^{...}

  /// Parse an assignment expression where part of an Objective-C message
  /// send has already been parsed.
  ///
  /// In this case \p LBracLoc indicates the location of the '[' of the message
  /// send, and either \p ReceiverName or \p ReceiverExpr is non-null indicating
  /// the receiver of the message.
  ///
  /// Since this handles full assignment-expression's, it handles postfix
  /// expressions and other binary operators for these expressions as well.
  ExprResult ParseAssignmentExprWithObjCMessageExprStart(
      SourceLocation LBracloc, SourceLocation SuperLoc, ParsedType ReceiverType,
      Expr *ReceiverExpr);

  /// Return true if we know that we are definitely looking at a
  /// decl-specifier, and isn't part of an expression such as a function-style
  /// cast. Return false if it's no a decl-specifier, or we're not sure.
  bool isKnownToBeDeclarationSpecifier() {
    if (getLangOpts().CPlusPlus)
      return isCXXDeclarationSpecifier(ImplicitTypenameContext::No) ==
             TPResult::True;
    return isDeclarationSpecifier(ImplicitTypenameContext::No, true);
  }

  /// Checks whether the current tokens form a type-id or an expression for the
  /// purposes of use as the initial operand to a generic selection expression.
  /// This requires special handling in C++ because it accepts either a type or
  /// an expression, and we need to disambiguate which is which. However, we
  /// cannot use the same logic as we've used for sizeof expressions, because
  /// that logic relies on the operator only accepting a single argument,
  /// whereas _Generic accepts a list of arguments.
  bool isTypeIdForGenericSelection() {
    if (getLangOpts().CPlusPlus) {
      bool isAmbiguous;
      return isCXXTypeId(TentativeCXXTypeIdContext::AsGenericSelectionArgument,
                         isAmbiguous);
    }
    return isTypeSpecifierQualifier();
  }

  /// Checks if the current tokens form type-id or expression.
  /// It is similar to isTypeIdInParens but does not suppose that type-id
  /// is in parenthesis.
  bool isTypeIdUnambiguously() {
    if (getLangOpts().CPlusPlus) {
      bool isAmbiguous;
      return isCXXTypeId(TentativeCXXTypeIdContext::Unambiguous, isAmbiguous);
    }
    return isTypeSpecifierQualifier();
  }

  /// ParseBlockId - Parse a block-id, which roughly looks like int (int x).
  ///
  /// \verbatim
  /// [clang] block-id:
  /// [clang]   specifier-qualifier-list block-declarator
  /// \endverbatim
  void ParseBlockId(SourceLocation CaretLoc);

  /// Parse availability query specification.
  ///
  /// \verbatim
  ///  availability-spec:
  ///     '*'
  ///     identifier version-tuple
  /// \endverbatim
  std::optional<AvailabilitySpec> ParseAvailabilitySpec();
  ExprResult ParseAvailabilityCheckExpr(SourceLocation StartLoc);

  /// Tries to parse cast part of OpenMP array shaping operation:
  /// \verbatim
  /// '[' expression ']' { '[' expression ']' } ')'
  /// \endverbatim
  bool tryParseOpenMPArrayShapingCastPart();

  ExprResult ParseBuiltinPtrauthTypeDiscriminator();

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name C++ Expressions
  /// Implementations are in ParseExprCXX.cpp
  ///@{

public:
  /// Parse a C++ unqualified-id (or a C identifier), which describes the
  /// name of an entity.
  ///
  /// \verbatim
  ///       unqualified-id: [C++ expr.prim.general]
  ///         identifier
  ///         operator-function-id
  ///         conversion-function-id
  /// [C++0x] literal-operator-id [TODO]
  ///         ~ class-name
  ///         template-id
  /// \endverbatim
  ///
  /// \param SS The nested-name-specifier that preceded this unqualified-id. If
  /// non-empty, then we are parsing the unqualified-id of a qualified-id.
  ///
  /// \param ObjectType if this unqualified-id occurs within a member access
  /// expression, the type of the base object whose member is being accessed.
  ///
  /// \param ObjectHadErrors if this unqualified-id occurs within a member
  /// access expression, indicates whether the original subexpressions had any
  /// errors. When true, diagnostics for missing 'template' keyword will be
  /// supressed.
  ///
  /// \param EnteringContext whether we are entering the scope of the
  /// nested-name-specifier.
  ///
  /// \param AllowDestructorName whether we allow parsing of a destructor name.
  ///
  /// \param AllowConstructorName whether we allow parsing a constructor name.
  ///
  /// \param AllowDeductionGuide whether we allow parsing a deduction guide
  /// name.
  ///
  /// \param Result on a successful parse, contains the parsed unqualified-id.
  ///
  /// \returns true if parsing fails, false otherwise.
  bool ParseUnqualifiedId(CXXScopeSpec &SS, ParsedType ObjectType,
                          bool ObjectHadErrors, bool EnteringContext,
                          bool AllowDestructorName, bool AllowConstructorName,
                          bool AllowDeductionGuide,
                          SourceLocation *TemplateKWLoc, UnqualifiedId &Result);

private:
  /// ColonIsSacred - When this is false, we aggressively try to recover from
  /// code like "foo : bar" as if it were a typo for "foo :: bar".  This is not
  /// safe in case statements and a few other things.  This is managed by the
  /// ColonProtectionRAIIObject RAII object.
  bool ColonIsSacred;

  /// ParseCXXAmbiguousParenExpression - We have parsed the left paren of a
  /// parenthesized ambiguous type-id. This uses tentative parsing to
  /// disambiguate based on the context past the parens.
  ExprResult ParseCXXAmbiguousParenExpression(
      ParenParseOption &ExprType, ParsedType &CastTy,
      BalancedDelimiterTracker &Tracker, ColonProtectionRAIIObject &ColonProt);

  //===--------------------------------------------------------------------===//
  // C++ Expressions
  ExprResult tryParseCXXIdExpression(CXXScopeSpec &SS, bool isAddressOfOperand,
                                     Token &Replacement);

  ExprResult tryParseCXXPackIndexingExpression(ExprResult PackIdExpression);
  ExprResult ParseCXXPackIndexingExpression(ExprResult PackIdExpression);

  /// ParseCXXIdExpression - Handle id-expression.
  ///
  /// \verbatim
  ///       id-expression:
  ///         unqualified-id
  ///         qualified-id
  ///
  ///       qualified-id:
  ///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
  ///         '::' identifier
  ///         '::' operator-function-id
  ///         '::' template-id
  ///
  /// NOTE: The standard specifies that, for qualified-id, the parser does not
  /// expect:
  ///
  ///   '::' conversion-function-id
  ///   '::' '~' class-name
  /// \endverbatim
  ///
  /// This may cause a slight inconsistency on diagnostics:
  ///
  /// class C {};
  /// namespace A {}
  /// void f() {
  ///   :: A :: ~ C(); // Some Sema error about using destructor with a
  ///                  // namespace.
  ///   :: ~ C(); // Some Parser error like 'unexpected ~'.
  /// }
  ///
  /// We simplify the parser a bit and make it work like:
  ///
  /// \verbatim
  ///       qualified-id:
  ///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
  ///         '::' unqualified-id
  /// \endverbatim
  ///
  /// That way Sema can handle and report similar errors for namespaces and the
  /// global scope.
  ///
  /// The isAddressOfOperand parameter indicates that this id-expression is a
  /// direct operand of the address-of operator. This is, besides member
  /// contexts, the only place where a qualified-id naming a non-static class
  /// member may appear.
  ///
  ExprResult ParseCXXIdExpression(bool isAddressOfOperand = false);

  // Are the two tokens adjacent in the same source file?
  bool areTokensAdjacent(const Token &A, const Token &B);

  // Check for '<::' which should be '< ::' instead of '[:' when following
  // a template name.
  void CheckForTemplateAndDigraph(Token &Next, ParsedType ObjectTypePtr,
                                  bool EnteringContext, IdentifierInfo &II,
                                  CXXScopeSpec &SS);

  /// Parse global scope or nested-name-specifier if present.
  ///
  /// Parses a C++ global scope specifier ('::') or nested-name-specifier (which
  /// may be preceded by '::'). Note that this routine will not parse ::new or
  /// ::delete; it will just leave them in the token stream.
  ///
  /// \verbatim
  ///       '::'[opt] nested-name-specifier
  ///       '::'
  ///
  ///       nested-name-specifier:
  ///         type-name '::'
  ///         namespace-name '::'
  ///         nested-name-specifier identifier '::'
  ///         nested-name-specifier 'template'[opt] simple-template-id '::'
  /// \endverbatim
  ///
  ///
  /// \param SS the scope specifier that will be set to the parsed
  /// nested-name-specifier (or empty)
  ///
  /// \param ObjectType if this nested-name-specifier is being parsed following
  /// the "." or "->" of a member access expression, this parameter provides the
  /// type of the object whose members are being accessed.
  ///
  /// \param ObjectHadErrors if this unqualified-id occurs within a member
  /// access expression, indicates whether the original subexpressions had any
  /// errors. When true, diagnostics for missing 'template' keyword will be
  /// supressed.
  ///
  /// \param EnteringContext whether we will be entering into the context of
  /// the nested-name-specifier after parsing it.
  ///
  /// \param MayBePseudoDestructor When non-NULL, points to a flag that
  /// indicates whether this nested-name-specifier may be part of a
  /// pseudo-destructor name. In this case, the flag will be set false
  /// if we don't actually end up parsing a destructor name. Moreover,
  /// if we do end up determining that we are parsing a destructor name,
  /// the last component of the nested-name-specifier is not parsed as
  /// part of the scope specifier.
  ///
  /// \param IsTypename If \c true, this nested-name-specifier is known to be
  /// part of a type name. This is used to improve error recovery.
  ///
  /// \param LastII When non-NULL, points to an IdentifierInfo* that will be
  /// filled in with the leading identifier in the last component of the
  /// nested-name-specifier, if any.
  ///
  /// \param OnlyNamespace If true, only considers namespaces in lookup.
  ///
  ///
  /// \returns true if there was an error parsing a scope specifier
  bool ParseOptionalCXXScopeSpecifier(
      CXXScopeSpec &SS, ParsedType ObjectType, bool ObjectHasErrors,
      bool EnteringContext, bool *MayBePseudoDestructor = nullptr,
      bool IsTypename = false, const IdentifierInfo **LastII = nullptr,
      bool OnlyNamespace = false, bool InUsingDeclaration = false,
      bool Disambiguation = false);

  //===--------------------------------------------------------------------===//
  // C++11 5.1.2: Lambda expressions

  /// Result of tentatively parsing a lambda-introducer.
  enum class LambdaIntroducerTentativeParse {
    /// This appears to be a lambda-introducer, which has been fully parsed.
    Success,
    /// This is a lambda-introducer, but has not been fully parsed, and this
    /// function needs to be called again to parse it.
    Incomplete,
    /// This is definitely an Objective-C message send expression, rather than
    /// a lambda-introducer, attribute-specifier, or array designator.
    MessageSend,
    /// This is not a lambda-introducer.
    Invalid,
  };

  /// ParseLambdaExpression - Parse a C++11 lambda expression.
  ///
  /// \verbatim
  ///       lambda-expression:
  ///         lambda-introducer lambda-declarator compound-statement
  ///         lambda-introducer '<' template-parameter-list '>'
  ///             requires-clause[opt] lambda-declarator compound-statement
  ///
  ///       lambda-introducer:
  ///         '[' lambda-capture[opt] ']'
  ///
  ///       lambda-capture:
  ///         capture-default
  ///         capture-list
  ///         capture-default ',' capture-list
  ///
  ///       capture-default:
  ///         '&'
  ///         '='
  ///
  ///       capture-list:
  ///         capture
  ///         capture-list ',' capture
  ///
  ///       capture:
  ///         simple-capture
  ///         init-capture     [C++1y]
  ///
  ///       simple-capture:
  ///         identifier
  ///         '&' identifier
  ///         'this'
  ///
  ///       init-capture:      [C++1y]
  ///         identifier initializer
  ///         '&' identifier initializer
  ///
  ///       lambda-declarator:
  ///         lambda-specifiers     [C++23]
  ///         '(' parameter-declaration-clause ')' lambda-specifiers
  ///             requires-clause[opt]
  ///
  ///       lambda-specifiers:
  ///         decl-specifier-seq[opt] noexcept-specifier[opt]
  ///             attribute-specifier-seq[opt] trailing-return-type[opt]
  /// \endverbatim
  ///
  ExprResult ParseLambdaExpression();

  /// Use lookahead and potentially tentative parsing to determine if we are
  /// looking at a C++11 lambda expression, and parse it if we are.
  ///
  /// If we are not looking at a lambda expression, returns ExprError().
  ExprResult TryParseLambdaExpression();

  /// Parse a lambda introducer.
  /// \param Intro A LambdaIntroducer filled in with information about the
  ///        contents of the lambda-introducer.
  /// \param Tentative If non-null, we are disambiguating between a
  ///        lambda-introducer and some other construct. In this mode, we do not
  ///        produce any diagnostics or take any other irreversible action
  ///        unless we're sure that this is a lambda-expression.
  /// \return \c true if parsing (or disambiguation) failed with a diagnostic
  ///         and the caller should bail out / recover.
  bool
  ParseLambdaIntroducer(LambdaIntroducer &Intro,
                        LambdaIntroducerTentativeParse *Tentative = nullptr);

  /// ParseLambdaExpressionAfterIntroducer - Parse the rest of a lambda
  /// expression.
  ExprResult ParseLambdaExpressionAfterIntroducer(LambdaIntroducer &Intro);

  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Casts

  /// ParseCXXCasts - This handles the various ways to cast expressions to
  /// another type.
  ///
  /// \verbatim
  ///       postfix-expression: [C++ 5.2p1]
  ///         'dynamic_cast' '<' type-name '>' '(' expression ')'
  ///         'static_cast' '<' type-name '>' '(' expression ')'
  ///         'reinterpret_cast' '<' type-name '>' '(' expression ')'
  ///         'const_cast' '<' type-name '>' '(' expression ')'
  /// \endverbatim
  ///
  /// C++ for OpenCL s2.3.1 adds:
  ///         'addrspace_cast' '<' type-name '>' '(' expression ')'
  ExprResult ParseCXXCasts();

  /// Parse a __builtin_bit_cast(T, E), used to implement C++2a std::bit_cast.
  ExprResult ParseBuiltinBitCast();

  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Type Identification

  /// ParseCXXTypeid - This handles the C++ typeid expression.
  ///
  /// \verbatim
  ///       postfix-expression: [C++ 5.2p1]
  ///         'typeid' '(' expression ')'
  ///         'typeid' '(' type-id ')'
  /// \endverbatim
  ///
  ExprResult ParseCXXTypeid();

  //===--------------------------------------------------------------------===//
  //  C++ : Microsoft __uuidof Expression

  /// ParseCXXUuidof - This handles the Microsoft C++ __uuidof expression.
  ///
  /// \verbatim
  ///         '__uuidof' '(' expression ')'
  ///         '__uuidof' '(' type-id ')'
  /// \endverbatim
  ///
  ExprResult ParseCXXUuidof();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.4: C++ Pseudo-Destructor Expressions

  /// Parse a C++ pseudo-destructor expression after the base,
  /// . or -> operator, and nested-name-specifier have already been
  /// parsed. We're handling this fragment of the grammar:
  ///
  /// \verbatim
  ///       postfix-expression: [C++2a expr.post]
  ///         postfix-expression . template[opt] id-expression
  ///         postfix-expression -> template[opt] id-expression
  ///
  ///       id-expression:
  ///         qualified-id
  ///         unqualified-id
  ///
  ///       qualified-id:
  ///         nested-name-specifier template[opt] unqualified-id
  ///
  ///       nested-name-specifier:
  ///         type-name ::
  ///         decltype-specifier ::    FIXME: not implemented, but probably only
  ///                                         allowed in C++ grammar by accident
  ///         nested-name-specifier identifier ::
  ///         nested-name-specifier template[opt] simple-template-id ::
  ///         [...]
  ///
  ///       unqualified-id:
  ///         ~ type-name
  ///         ~ decltype-specifier
  ///         [...]
  /// \endverbatim
  ///
  /// ... where the all but the last component of the nested-name-specifier
  /// has already been parsed, and the base expression is not of a non-dependent
  /// class type.
  ExprResult ParseCXXPseudoDestructor(Expr *Base, SourceLocation OpLoc,
                                      tok::TokenKind OpKind, CXXScopeSpec &SS,
                                      ParsedType ObjectType);

  //===--------------------------------------------------------------------===//
  // C++ 9.3.2: C++ 'this' pointer

  /// ParseCXXThis - This handles the C++ 'this' pointer.
  ///
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.
  ExprResult ParseCXXThis();

  //===--------------------------------------------------------------------===//
  // C++ 15: C++ Throw Expression

  /// ParseThrowExpression - This handles the C++ throw expression.
  ///
  /// \verbatim
  ///       throw-expression: [C++ 15]
  ///         'throw' assignment-expression[opt]
  /// \endverbatim
  ExprResult ParseThrowExpression();

  //===--------------------------------------------------------------------===//
  // C++ 2.13.5: C++ Boolean Literals

  /// ParseCXXBoolLiteral - This handles the C++ Boolean literals.
  ///
  /// \verbatim
  ///       boolean-literal: [C++ 2.13.5]
  ///         'true'
  ///         'false'
  /// \endverbatim
  ExprResult ParseCXXBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.3: Explicit type conversion (functional notation)

  /// ParseCXXTypeConstructExpression - Parse construction of a specified type.
  /// Can be interpreted either as function-style casting ("int(x)")
  /// or class type construction ("ClassType(x,y,z)")
  /// or creation of a value-initialized type ("int()").
  /// See [C++ 5.2.3].
  ///
  /// \verbatim
  ///       postfix-expression: [C++ 5.2p1]
  ///         simple-type-specifier '(' expression-list[opt] ')'
  /// [C++0x] simple-type-specifier braced-init-list
  ///         typename-specifier '(' expression-list[opt] ')'
  /// [C++0x] typename-specifier braced-init-list
  /// \endverbatim
  ///
  /// In C++1z onwards, the type specifier can also be a template-name.
  ExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

  /// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
  /// This should only be called when the current token is known to be part of
  /// simple-type-specifier.
  ///
  /// \verbatim
  ///       simple-type-specifier:
  ///         '::'[opt] nested-name-specifier[opt] type-name
  ///         '::'[opt] nested-name-specifier 'template' simple-template-id [TODO]
  ///         char
  ///         wchar_t
  ///         bool
  ///         short
  ///         int
  ///         long
  ///         signed
  ///         unsigned
  ///         float
  ///         double
  ///         void
  /// [GNU]   typeof-specifier
  /// [C++0x] auto               [TODO]
  ///
  ///       type-name:
  ///         class-name
  ///         enum-name
  ///         typedef-name
  /// \endverbatim
  ///
  void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

  /// ParseCXXTypeSpecifierSeq - Parse a C++ type-specifier-seq (C++
  /// [dcl.name]), which is a non-empty sequence of type-specifiers,
  /// e.g., "const short int". Note that the DeclSpec is *not* finished
  /// by parsing the type-specifier-seq, because these sequences are
  /// typically followed by some form of declarator. Returns true and
  /// emits diagnostics if this is not a type-specifier-seq, false
  /// otherwise.
  ///
  /// \verbatim
  ///   type-specifier-seq: [C++ 8.1]
  ///     type-specifier type-specifier-seq[opt]
  /// \endverbatim
  ///
  bool ParseCXXTypeSpecifierSeq(
      DeclSpec &DS, DeclaratorContext Context = DeclaratorContext::TypeName);

  //===--------------------------------------------------------------------===//
  // C++ 5.3.4 and 5.3.5: C++ new and delete

  /// ParseExpressionListOrTypeId - Parse either an expression-list or a
  /// type-id. This ambiguity appears in the syntax of the C++ new operator.
  ///
  /// \verbatim
  ///        new-expression:
  ///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
  ///                                     new-initializer[opt]
  ///
  ///        new-placement:
  ///                   '(' expression-list ')'
  /// \endverbatim
  ///
  bool ParseExpressionListOrTypeId(SmallVectorImpl<Expr *> &Exprs,
                                   Declarator &D);

  /// ParseDirectNewDeclarator - Parses a direct-new-declarator. Intended to be
  /// passed to ParseDeclaratorInternal.
  ///
  /// \verbatim
  ///        direct-new-declarator:
  ///                   '[' expression[opt] ']'
  ///                   direct-new-declarator '[' constant-expression ']'
  /// \endverbatim
  ///
  void ParseDirectNewDeclarator(Declarator &D);

  /// ParseCXXNewExpression - Parse a C++ new-expression. New is used to
  /// allocate memory in a typesafe manner and call constructors.
  ///
  /// This method is called to parse the new expression after the optional ::
  /// has been already parsed.  If the :: was present, "UseGlobal" is true and
  /// "Start" is its location.  Otherwise, "Start" is the location of the 'new'
  /// token.
  ///
  /// \verbatim
  ///        new-expression:
  ///                   '::'[opt] 'new' new-placement[opt] new-type-id
  ///                                     new-initializer[opt]
  ///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
  ///                                     new-initializer[opt]
  ///
  ///        new-placement:
  ///                   '(' expression-list ')'
  ///
  ///        new-type-id:
  ///                   type-specifier-seq new-declarator[opt]
  /// [GNU]             attributes type-specifier-seq new-declarator[opt]
  ///
  ///        new-declarator:
  ///                   ptr-operator new-declarator[opt]
  ///                   direct-new-declarator
  ///
  ///        new-initializer:
  ///                   '(' expression-list[opt] ')'
  /// [C++0x]           braced-init-list
  /// \endverbatim
  ///
  ExprResult ParseCXXNewExpression(bool UseGlobal, SourceLocation Start);

  /// ParseCXXDeleteExpression - Parse a C++ delete-expression. Delete is used
  /// to free memory allocated by new.
  ///
  /// This method is called to parse the 'delete' expression after the optional
  /// '::' has been already parsed.  If the '::' was present, "UseGlobal" is
  /// true and "Start" is its location.  Otherwise, "Start" is the location of
  /// the 'delete' token.
  ///
  /// \verbatim
  ///        delete-expression:
  ///                   '::'[opt] 'delete' cast-expression
  ///                   '::'[opt] 'delete' '[' ']' cast-expression
  /// \endverbatim
  ExprResult ParseCXXDeleteExpression(bool UseGlobal, SourceLocation Start);

  //===--------------------------------------------------------------------===//
  // C++ if/switch/while/for condition expression.

  /// ParseCXXCondition - if/switch/while condition expression.
  ///
  /// \verbatim
  ///       condition:
  ///         expression
  ///         type-specifier-seq declarator '=' assignment-expression
  /// [C++11] type-specifier-seq declarator '=' initializer-clause
  /// [C++11] type-specifier-seq declarator braced-init-list
  /// [Clang] type-specifier-seq ref-qualifier[opt] '[' identifier-list ']'
  ///             brace-or-equal-initializer
  /// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
  ///             '=' assignment-expression
  /// \endverbatim
  ///
  /// In C++1z, a condition may in some contexts be preceded by an
  /// optional init-statement. This function will parse that too.
  ///
  /// \param InitStmt If non-null, an init-statement is permitted, and if
  /// present will be parsed and stored here.
  ///
  /// \param Loc The location of the start of the statement that requires this
  /// condition, e.g., the "for" in a for loop.
  ///
  /// \param MissingOK Whether an empty condition is acceptable here. Otherwise
  /// it is considered an error to be recovered from.
  ///
  /// \param FRI If non-null, a for range declaration is permitted, and if
  /// present will be parsed and stored here, and a null result will be
  /// returned.
  ///
  /// \param EnterForConditionScope If true, enter a continue/break scope at the
  /// appropriate moment for a 'for' loop.
  ///
  /// \returns The parsed condition.
  Sema::ConditionResult ParseCXXCondition(StmtResult *InitStmt,
                                          SourceLocation Loc,
                                          Sema::ConditionKind CK,
                                          bool MissingOK,
                                          ForRangeInfo *FRI = nullptr,
                                          bool EnterForConditionScope = false);
  DeclGroupPtrTy ParseAliasDeclarationInInitStatement(DeclaratorContext Context,
                                                      ParsedAttributes &Attrs);

  //===--------------------------------------------------------------------===//
  // C++ Coroutines

  /// Parse the C++ Coroutines co_yield expression.
  ///
  /// \verbatim
  ///       co_yield-expression:
  ///         'co_yield' assignment-expression[opt]
  /// \endverbatim
  ExprResult ParseCoyieldExpression();

  //===--------------------------------------------------------------------===//
  // C++ Concepts

  /// ParseRequiresExpression - Parse a C++2a requires-expression.
  /// C++2a [expr.prim.req]p1
  ///     A requires-expression provides a concise way to express requirements
  ///     on template arguments. A requirement is one that can be checked by
  ///     name lookup (6.4) or by checking properties of types and expressions.
  ///
  /// \verbatim
  ///     requires-expression:
  ///         'requires' requirement-parameter-list[opt] requirement-body
  ///
  ///     requirement-parameter-list:
  ///         '(' parameter-declaration-clause[opt] ')'
  ///
  ///     requirement-body:
  ///         '{' requirement-seq '}'
  ///
  ///     requirement-seq:
  ///         requirement
  ///         requirement-seq requirement
  ///
  ///     requirement:
  ///         simple-requirement
  ///         type-requirement
  ///         compound-requirement
  ///         nested-requirement
  /// \endverbatim
  ExprResult ParseRequiresExpression();

  /// isTypeIdInParens - Assumes that a '(' was parsed and now we want to know
  /// whether the parens contain an expression or a type-id.
  /// Returns true for a type-id and false for an expression.
  bool isTypeIdInParens(bool &isAmbiguous) {
    if (getLangOpts().CPlusPlus)
      return isCXXTypeId(TentativeCXXTypeIdContext::InParens, isAmbiguous);
    isAmbiguous = false;
    return isTypeSpecifierQualifier();
  }
  bool isTypeIdInParens() {
    bool isAmbiguous;
    return isTypeIdInParens(isAmbiguous);
  }

  /// Finish parsing a C++ unqualified-id that is a template-id of
  /// some form.
  ///
  /// This routine is invoked when a '<' is encountered after an identifier or
  /// operator-function-id is parsed by \c ParseUnqualifiedId() to determine
  /// whether the unqualified-id is actually a template-id. This routine will
  /// then parse the template arguments and form the appropriate template-id to
  /// return to the caller.
  ///
  /// \param SS the nested-name-specifier that precedes this template-id, if
  /// we're actually parsing a qualified-id.
  ///
  /// \param ObjectType if this unqualified-id occurs within a member access
  /// expression, the type of the base object whose member is being accessed.
  ///
  /// \param ObjectHadErrors this unqualified-id occurs within a member access
  /// expression, indicates whether the original subexpressions had any errors.
  ///
  /// \param Name for constructor and destructor names, this is the actual
  /// identifier that may be a template-name.
  ///
  /// \param NameLoc the location of the class-name in a constructor or
  /// destructor.
  ///
  /// \param EnteringContext whether we're entering the scope of the
  /// nested-name-specifier.
  ///
  /// \param Id as input, describes the template-name or operator-function-id
  /// that precedes the '<'. If template arguments were parsed successfully,
  /// will be updated with the template-id.
  ///
  /// \param AssumeTemplateId When true, this routine will assume that the name
  /// refers to a template without performing name lookup to verify.
  ///
  /// \returns true if a parse error occurred, false otherwise.
  bool ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS, ParsedType ObjectType,
                                    bool ObjectHadErrors,
                                    SourceLocation TemplateKWLoc,
                                    IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    bool EnteringContext, UnqualifiedId &Id,
                                    bool AssumeTemplateId);

  /// Parse an operator-function-id or conversion-function-id as part
  /// of a C++ unqualified-id.
  ///
  /// This routine is responsible only for parsing the operator-function-id or
  /// conversion-function-id; it does not handle template arguments in any way.
  ///
  /// \verbatim
  ///       operator-function-id: [C++ 13.5]
  ///         'operator' operator
  ///
  ///       operator: one of
  ///            new   delete  new[]   delete[]
  ///            +     -    *  /    %  ^    &   |   ~
  ///            !     =    <  >    += -=   *=  /=  %=
  ///            ^=    &=   |= <<   >> >>= <<=  ==  !=
  ///            <=    >=   && ||   ++ --   ,   ->* ->
  ///            ()    []   <=>
  ///
  ///       conversion-function-id: [C++ 12.3.2]
  ///         operator conversion-type-id
  ///
  ///       conversion-type-id:
  ///         type-specifier-seq conversion-declarator[opt]
  ///
  ///       conversion-declarator:
  ///         ptr-operator conversion-declarator[opt]
  /// \endverbatim
  ///
  /// \param SS The nested-name-specifier that preceded this unqualified-id. If
  /// non-empty, then we are parsing the unqualified-id of a qualified-id.
  ///
  /// \param EnteringContext whether we are entering the scope of the
  /// nested-name-specifier.
  ///
  /// \param ObjectType if this unqualified-id occurs within a member access
  /// expression, the type of the base object whose member is being accessed.
  ///
  /// \param Result on a successful parse, contains the parsed unqualified-id.
  ///
  /// \returns true if parsing fails, false otherwise.
  bool ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                  ParsedType ObjectType, UnqualifiedId &Result);

  //===--------------------------------------------------------------------===//
  // C++11/G++: Type Traits [Type-Traits.html in the GCC manual]

  /// Parse the built-in type-trait pseudo-functions that allow
  /// implementation of the TR1/C++11 type traits templates.
  ///
  /// \verbatim
  ///       primary-expression:
  ///          unary-type-trait '(' type-id ')'
  ///          binary-type-trait '(' type-id ',' type-id ')'
  ///          type-trait '(' type-id-seq ')'
  ///
  ///       type-id-seq:
  ///          type-id ...[opt] type-id-seq[opt]
  /// \endverbatim
  ///
  ExprResult ParseTypeTrait();

  //===--------------------------------------------------------------------===//
  // Embarcadero: Arary and Expression Traits

  /// ParseArrayTypeTrait - Parse the built-in array type-trait
  /// pseudo-functions.
  ///
  /// \verbatim
  ///       primary-expression:
  /// [Embarcadero]     '__array_rank' '(' type-id ')'
  /// [Embarcadero]     '__array_extent' '(' type-id ',' expression ')'
  /// \endverbatim
  ///
  ExprResult ParseArrayTypeTrait();

  /// ParseExpressionTrait - Parse built-in expression-trait
  /// pseudo-functions like __is_lvalue_expr( xxx ).
  ///
  /// \verbatim
  ///       primary-expression:
  /// [Embarcadero]     expression-trait '(' expression ')'
  /// \endverbatim
  ///
  ExprResult ParseExpressionTrait();

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name HLSL Constructs
  /// Implementations are in ParseHLSL.cpp
  ///@{

private:
  bool MaybeParseHLSLAnnotations(Declarator &D,
                                 SourceLocation *EndLoc = nullptr,
                                 bool CouldBeBitField = false) {
    assert(getLangOpts().HLSL && "MaybeParseHLSLAnnotations is for HLSL only");
    if (Tok.is(tok::colon)) {
      ParsedAttributes Attrs(AttrFactory);
      ParseHLSLAnnotations(Attrs, EndLoc, CouldBeBitField);
      D.takeAttributes(Attrs);
      return true;
    }
    return false;
  }

  void MaybeParseHLSLAnnotations(ParsedAttributes &Attrs,
                                 SourceLocation *EndLoc = nullptr) {
    assert(getLangOpts().HLSL && "MaybeParseHLSLAnnotations is for HLSL only");
    if (Tok.is(tok::colon))
      ParseHLSLAnnotations(Attrs, EndLoc);
  }

  struct ParsedSemantic {
    StringRef Name = "";
    unsigned Index = 0;
    bool Explicit = false;
  };

  ParsedSemantic ParseHLSLSemantic();

  void ParseHLSLAnnotations(ParsedAttributes &Attrs,
                            SourceLocation *EndLoc = nullptr,
                            bool CouldBeBitField = false);
  Decl *ParseHLSLBuffer(SourceLocation &DeclEnd, ParsedAttributes &Attrs);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Initializers
  /// Implementations are in ParseInit.cpp
  ///@{

private:
  //===--------------------------------------------------------------------===//
  // C99 6.7.8: Initialization.

  /// ParseInitializer
  /// \verbatim
  ///       initializer: [C99 6.7.8]
  ///         assignment-expression
  ///         '{' ...
  /// \endverbatim
  ExprResult ParseInitializer() {
    if (Tok.isNot(tok::l_brace))
      return ParseAssignmentExpression();
    return ParseBraceInitializer();
  }

  /// MayBeDesignationStart - Return true if the current token might be the
  /// start of a designator.  If we can tell it is impossible that it is a
  /// designator, return false.
  bool MayBeDesignationStart();

  /// ParseBraceInitializer - Called when parsing an initializer that has a
  /// leading open brace.
  ///
  /// \verbatim
  ///       initializer: [C99 6.7.8]
  ///         '{' initializer-list '}'
  ///         '{' initializer-list ',' '}'
  /// [C23]   '{' '}'
  ///
  ///       initializer-list:
  ///         designation[opt] initializer ...[opt]
  ///         initializer-list ',' designation[opt] initializer ...[opt]
  /// \endverbatim
  ///
  ExprResult ParseBraceInitializer();

  struct DesignatorCompletionInfo {
    SmallVectorImpl<Expr *> &InitExprs;
    QualType PreferredBaseType;
  };

  /// ParseInitializerWithPotentialDesignator - Parse the 'initializer'
  /// production checking to see if the token stream starts with a designator.
  ///
  /// C99:
  ///
  /// \verbatim
  ///       designation:
  ///         designator-list '='
  /// [GNU]   array-designator
  /// [GNU]   identifier ':'
  ///
  ///       designator-list:
  ///         designator
  ///         designator-list designator
  ///
  ///       designator:
  ///         array-designator
  ///         '.' identifier
  ///
  ///       array-designator:
  ///         '[' constant-expression ']'
  /// [GNU]   '[' constant-expression '...' constant-expression ']'
  /// \endverbatim
  ///
  /// C++20:
  ///
  /// \verbatim
  ///       designated-initializer-list:
  ///         designated-initializer-clause
  ///         designated-initializer-list ',' designated-initializer-clause
  ///
  ///       designated-initializer-clause:
  ///         designator brace-or-equal-initializer
  ///
  ///       designator:
  ///         '.' identifier
  /// \endverbatim
  ///
  /// We allow the C99 syntax extensions in C++20, but do not allow the C++20
  /// extension (a braced-init-list after the designator with no '=') in C99.
  ///
  /// NOTE: [OBC] allows '[ objc-receiver objc-message-args ]' as an
  /// initializer (because it is an expression).  We need to consider this case
  /// when parsing array designators.
  ///
  /// \p CodeCompleteCB is called with Designation parsed so far.
  ExprResult ParseInitializerWithPotentialDesignator(DesignatorCompletionInfo);

  ExprResult createEmbedExpr();

  /// A SmallVector of expressions.
  typedef SmallVector<Expr *, 12> ExprVector;

  // Return true if a comma (or closing brace) is necessary after the
  // __if_exists/if_not_exists statement.
  bool ParseMicrosoftIfExistsBraceInitializer(ExprVector &InitExprs,
                                              bool &InitExprsOk);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Objective-C Constructs
  /// Implementations are in ParseObjc.cpp
  ///@{

public:
  friend class InMessageExpressionRAIIObject;
  friend class ObjCDeclContextSwitch;

  ObjCContainerDecl *getObjCDeclContext() const {
    return Actions.ObjC().getObjCDeclContext();
  }

  /// Retrieve the underscored keyword (_Nonnull, _Nullable) that corresponds
  /// to the given nullability kind.
  IdentifierInfo *getNullabilityKeyword(NullabilityKind nullability) {
    return Actions.getNullabilityKeyword(nullability);
  }

private:
  /// Objective-C contextual keywords.
  IdentifierInfo *Ident_instancetype;

  /// Ident_super - IdentifierInfo for "super", to support fast
  /// comparison.
  IdentifierInfo *Ident_super;

  /// When true, we are directly inside an Objective-C message
  /// send expression.
  ///
  /// This is managed by the \c InMessageExpressionRAIIObject class, and
  /// should not be set directly.
  bool InMessageExpression;

  /// True if we are within an Objective-C container while parsing C-like decls.
  ///
  /// This is necessary because Sema thinks we have left the container
  /// to parse the C-like decls, meaning Actions.ObjC().getObjCDeclContext()
  /// will be NULL.
  bool ParsingInObjCContainer;

  /// Returns true if the current token is the identifier 'instancetype'.
  ///
  /// Should only be used in Objective-C language modes.
  bool isObjCInstancetype() {
    assert(getLangOpts().ObjC);
    if (Tok.isAnnotation())
      return false;
    if (!Ident_instancetype)
      Ident_instancetype = PP.getIdentifierInfo("instancetype");
    return Tok.getIdentifierInfo() == Ident_instancetype;
  }

  /// ObjCDeclContextSwitch - An object used to switch context from
  /// an objective-c decl context to its enclosing decl context and
  /// back.
  class ObjCDeclContextSwitch {
    Parser &P;
    ObjCContainerDecl *DC;
    SaveAndRestore<bool> WithinObjCContainer;

  public:
    explicit ObjCDeclContextSwitch(Parser &p)
        : P(p), DC(p.getObjCDeclContext()),
          WithinObjCContainer(P.ParsingInObjCContainer, DC != nullptr) {
      if (DC)
        P.Actions.ObjC().ActOnObjCTemporaryExitContainerContext(DC);
    }
    ~ObjCDeclContextSwitch() {
      if (DC)
        P.Actions.ObjC().ActOnObjCReenterContainerContext(DC);
    }
  };

  void CheckNestedObjCContexts(SourceLocation AtLoc);

  void ParseLexedObjCMethodDefs(LexedMethod &LM, bool parseMethod);

  // Objective-C External Declarations

  /// Skips attributes after an Objective-C @ directive. Emits a diagnostic.
  void MaybeSkipAttributes(tok::ObjCKeywordKind Kind);

  /// ParseObjCAtDirectives - Handle parts of the external-declaration
  /// production:
  /// \verbatim
  ///       external-declaration: [C99 6.9]
  /// [OBJC]  objc-class-definition
  /// [OBJC]  objc-class-declaration
  /// [OBJC]  objc-alias-declaration
  /// [OBJC]  objc-protocol-definition
  /// [OBJC]  objc-method-definition
  /// [OBJC]  '@' 'end'
  /// \endverbatim
  DeclGroupPtrTy ParseObjCAtDirectives(ParsedAttributes &DeclAttrs,
                                       ParsedAttributes &DeclSpecAttrs);

  ///
  /// \verbatim
  /// objc-class-declaration:
  ///    '@' 'class' objc-class-forward-decl (',' objc-class-forward-decl)* ';'
  ///
  /// objc-class-forward-decl:
  ///   identifier objc-type-parameter-list[opt]
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseObjCAtClassDeclaration(SourceLocation atLoc);

  ///
  /// \verbatim
  ///   objc-interface:
  ///     objc-class-interface-attributes[opt] objc-class-interface
  ///     objc-category-interface
  ///
  ///   objc-class-interface:
  ///     '@' 'interface' identifier objc-type-parameter-list[opt]
  ///       objc-superclass[opt] objc-protocol-refs[opt]
  ///       objc-class-instance-variables[opt]
  ///       objc-interface-decl-list
  ///     @end
  ///
  ///   objc-category-interface:
  ///     '@' 'interface' identifier objc-type-parameter-list[opt]
  ///       '(' identifier[opt] ')' objc-protocol-refs[opt]
  ///       objc-interface-decl-list
  ///     @end
  ///
  ///   objc-superclass:
  ///     ':' identifier objc-type-arguments[opt]
  ///
  ///   objc-class-interface-attributes:
  ///     __attribute__((visibility("default")))
  ///     __attribute__((visibility("hidden")))
  ///     __attribute__((deprecated))
  ///     __attribute__((unavailable))
  ///     __attribute__((objc_exception)) - used by NSException on 64-bit
  ///     __attribute__((objc_root_class))
  /// \endverbatim
  ///
  Decl *ParseObjCAtInterfaceDeclaration(SourceLocation AtLoc,
                                        ParsedAttributes &prefixAttrs);

  /// Class to handle popping type parameters when leaving the scope.
  class ObjCTypeParamListScope;

  /// Parse an objc-type-parameter-list.
  ObjCTypeParamList *parseObjCTypeParamList();

  /// Parse an Objective-C type parameter list, if present, or capture
  /// the locations of the protocol identifiers for a list of protocol
  /// references.
  ///
  /// \verbatim
  ///   objc-type-parameter-list:
  ///     '<' objc-type-parameter (',' objc-type-parameter)* '>'
  ///
  ///   objc-type-parameter:
  ///     objc-type-parameter-variance? identifier objc-type-parameter-bound[opt]
  ///
  ///   objc-type-parameter-bound:
  ///     ':' type-name
  ///
  ///   objc-type-parameter-variance:
  ///     '__covariant'
  ///     '__contravariant'
  /// \endverbatim
  ///
  /// \param lAngleLoc The location of the starting '<'.
  ///
  /// \param protocolIdents Will capture the list of identifiers, if the
  /// angle brackets contain a list of protocol references rather than a
  /// type parameter list.
  ///
  /// \param rAngleLoc The location of the ending '>'.
  ObjCTypeParamList *parseObjCTypeParamListOrProtocolRefs(
      ObjCTypeParamListScope &Scope, SourceLocation &lAngleLoc,
      SmallVectorImpl<IdentifierLoc> &protocolIdents, SourceLocation &rAngleLoc,
      bool mayBeProtocolList = true);

  void HelperActionsForIvarDeclarations(ObjCContainerDecl *interfaceDecl,
                                        SourceLocation atLoc,
                                        BalancedDelimiterTracker &T,
                                        SmallVectorImpl<Decl *> &AllIvarDecls,
                                        bool RBraceMissing);

  /// \verbatim
  ///   objc-class-instance-variables:
  ///     '{' objc-instance-variable-decl-list[opt] '}'
  ///
  ///   objc-instance-variable-decl-list:
  ///     objc-visibility-spec
  ///     objc-instance-variable-decl ';'
  ///     ';'
  ///     objc-instance-variable-decl-list objc-visibility-spec
  ///     objc-instance-variable-decl-list objc-instance-variable-decl ';'
  ///     objc-instance-variable-decl-list static_assert-declaration
  ///     objc-instance-variable-decl-list ';'
  ///
  ///   objc-visibility-spec:
  ///     @private
  ///     @protected
  ///     @public
  ///     @package [OBJC2]
  ///
  ///   objc-instance-variable-decl:
  ///     struct-declaration
  /// \endverbatim
  ///
  void ParseObjCClassInstanceVariables(ObjCContainerDecl *interfaceDecl,
                                       tok::ObjCKeywordKind visibility,
                                       SourceLocation atLoc);

  /// \verbatim
  ///   objc-protocol-refs:
  ///     '<' identifier-list '>'
  /// \endverbatim
  ///
  bool ParseObjCProtocolReferences(
      SmallVectorImpl<Decl *> &P, SmallVectorImpl<SourceLocation> &PLocs,
      bool WarnOnDeclarations, bool ForObjCContainer, SourceLocation &LAngleLoc,
      SourceLocation &EndProtoLoc, bool consumeLastToken);

  /// Parse the first angle-bracket-delimited clause for an
  /// Objective-C object or object pointer type, which may be either
  /// type arguments or protocol qualifiers.
  ///
  /// \verbatim
  ///   objc-type-arguments:
  ///     '<' type-name '...'[opt] (',' type-name '...'[opt])* '>'
  /// \endverbatim
  ///
  void parseObjCTypeArgsOrProtocolQualifiers(
      ParsedType baseType, SourceLocation &typeArgsLAngleLoc,
      SmallVectorImpl<ParsedType> &typeArgs, SourceLocation &typeArgsRAngleLoc,
      SourceLocation &protocolLAngleLoc, SmallVectorImpl<Decl *> &protocols,
      SmallVectorImpl<SourceLocation> &protocolLocs,
      SourceLocation &protocolRAngleLoc, bool consumeLastToken,
      bool warnOnIncompleteProtocols);

  /// Parse either Objective-C type arguments or protocol qualifiers; if the
  /// former, also parse protocol qualifiers afterward.
  void parseObjCTypeArgsAndProtocolQualifiers(
      ParsedType baseType, SourceLocation &typeArgsLAngleLoc,
      SmallVectorImpl<ParsedType> &typeArgs, SourceLocation &typeArgsRAngleLoc,
      SourceLocation &protocolLAngleLoc, SmallVectorImpl<Decl *> &protocols,
      SmallVectorImpl<SourceLocation> &protocolLocs,
      SourceLocation &protocolRAngleLoc, bool consumeLastToken);

  /// Parse a protocol qualifier type such as '<NSCopying>', which is
  /// an anachronistic way of writing 'id<NSCopying>'.
  TypeResult parseObjCProtocolQualifierType(SourceLocation &rAngleLoc);

  /// Parse Objective-C type arguments and protocol qualifiers, extending the
  /// current type with the parsed result.
  TypeResult parseObjCTypeArgsAndProtocolQualifiers(SourceLocation loc,
                                                    ParsedType type,
                                                    bool consumeLastToken,
                                                    SourceLocation &endLoc);

  /// \verbatim
  ///   objc-interface-decl-list:
  ///     empty
  ///     objc-interface-decl-list objc-property-decl [OBJC2]
  ///     objc-interface-decl-list objc-method-requirement [OBJC2]
  ///     objc-interface-decl-list objc-method-proto ';'
  ///     objc-interface-decl-list declaration
  ///     objc-interface-decl-list ';'
  ///
  ///   objc-method-requirement: [OBJC2]
  ///     @required
  ///     @optional
  /// \endverbatim
  ///
  void ParseObjCInterfaceDeclList(tok::ObjCKeywordKind contextKey, Decl *CDecl);

  /// \verbatim
  ///   objc-protocol-declaration:
  ///     objc-protocol-definition
  ///     objc-protocol-forward-reference
  ///
  ///   objc-protocol-definition:
  ///     \@protocol identifier
  ///       objc-protocol-refs[opt]
  ///       objc-interface-decl-list
  ///     \@end
  ///
  ///   objc-protocol-forward-reference:
  ///     \@protocol identifier-list ';'
  /// \endverbatim
  ///
  ///   "\@protocol identifier ;" should be resolved as "\@protocol
  ///   identifier-list ;": objc-interface-decl-list may not start with a
  ///   semicolon in the first alternative if objc-protocol-refs are omitted.
  DeclGroupPtrTy ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
                                                ParsedAttributes &prefixAttrs);

  struct ObjCImplParsingDataRAII {
    Parser &P;
    Decl *Dcl;
    bool HasCFunction;
    typedef SmallVector<LexedMethod *, 8> LateParsedObjCMethodContainer;
    LateParsedObjCMethodContainer LateParsedObjCMethods;

    ObjCImplParsingDataRAII(Parser &parser, Decl *D)
        : P(parser), Dcl(D), HasCFunction(false) {
      P.CurParsedObjCImpl = this;
      Finished = false;
    }
    ~ObjCImplParsingDataRAII();

    void finish(SourceRange AtEnd);
    bool isFinished() const { return Finished; }

  private:
    bool Finished;
  };
  ObjCImplParsingDataRAII *CurParsedObjCImpl;

  /// StashAwayMethodOrFunctionBodyTokens -  Consume the tokens and store them
  /// for later parsing.
  void StashAwayMethodOrFunctionBodyTokens(Decl *MDecl);

  /// \verbatim
  ///   objc-implementation:
  ///     objc-class-implementation-prologue
  ///     objc-category-implementation-prologue
  ///
  ///   objc-class-implementation-prologue:
  ///     @implementation identifier objc-superclass[opt]
  ///       objc-class-instance-variables[opt]
  ///
  ///   objc-category-implementation-prologue:
  ///     @implementation identifier ( identifier )
  /// \endverbatim
  DeclGroupPtrTy ParseObjCAtImplementationDeclaration(SourceLocation AtLoc,
                                                      ParsedAttributes &Attrs);
  DeclGroupPtrTy ParseObjCAtEndDeclaration(SourceRange atEnd);

  /// \verbatim
  ///   compatibility-alias-decl:
  ///     @compatibility_alias alias-name  class-name ';'
  /// \endverbatim
  ///
  Decl *ParseObjCAtAliasDeclaration(SourceLocation atLoc);

  /// \verbatim
  ///   property-synthesis:
  ///     @synthesize property-ivar-list ';'
  ///
  ///   property-ivar-list:
  ///     property-ivar
  ///     property-ivar-list ',' property-ivar
  ///
  ///   property-ivar:
  ///     identifier
  ///     identifier '=' identifier
  /// \endverbatim
  ///
  Decl *ParseObjCPropertySynthesize(SourceLocation atLoc);

  /// \verbatim
  ///   property-dynamic:
  ///     @dynamic  property-list
  ///
  ///   property-list:
  ///     identifier
  ///     property-list ',' identifier
  /// \endverbatim
  ///
  Decl *ParseObjCPropertyDynamic(SourceLocation atLoc);

  /// \verbatim
  ///   objc-selector:
  ///     identifier
  ///     one of
  ///       enum struct union if else while do for switch case default
  ///       break continue return goto asm sizeof typeof __alignof
  ///       unsigned long const short volatile signed restrict _Complex
  ///       in out inout bycopy byref oneway int char float double void _Bool
  /// \endverbatim
  ///
  IdentifierInfo *ParseObjCSelectorPiece(SourceLocation &MethodLocation);

  IdentifierInfo *ObjCTypeQuals[llvm::to_underlying(ObjCTypeQual::NumQuals)];

  /// \verbatim
  ///  objc-for-collection-in: 'in'
  /// \endverbatim
  ///
  bool isTokIdentifier_in() const;

  /// \verbatim
  ///   objc-type-name:
  ///     '(' objc-type-qualifiers[opt] type-name ')'
  ///     '(' objc-type-qualifiers[opt] ')'
  /// \endverbatim
  ///
  ParsedType ParseObjCTypeName(ObjCDeclSpec &DS, DeclaratorContext Ctx,
                               ParsedAttributes *ParamAttrs);

  /// \verbatim
  ///   objc-method-proto:
  ///     objc-instance-method objc-method-decl objc-method-attributes[opt]
  ///     objc-class-method objc-method-decl objc-method-attributes[opt]
  ///
  ///   objc-instance-method: '-'
  ///   objc-class-method: '+'
  ///
  ///   objc-method-attributes:         [OBJC2]
  ///     __attribute__((deprecated))
  /// \endverbatim
  ///
  Decl *ParseObjCMethodPrototype(
      tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
      bool MethodDefinition = true);

  /// \verbatim
  ///   objc-method-decl:
  ///     objc-selector
  ///     objc-keyword-selector objc-parmlist[opt]
  ///     objc-type-name objc-selector
  ///     objc-type-name objc-keyword-selector objc-parmlist[opt]
  ///
  ///   objc-keyword-selector:
  ///     objc-keyword-decl
  ///     objc-keyword-selector objc-keyword-decl
  ///
  ///   objc-keyword-decl:
  ///     objc-selector ':' objc-type-name objc-keyword-attributes[opt] identifier
  ///     objc-selector ':' objc-keyword-attributes[opt] identifier
  ///     ':' objc-type-name objc-keyword-attributes[opt] identifier
  ///     ':' objc-keyword-attributes[opt] identifier
  ///
  ///   objc-parmlist:
  ///     objc-parms objc-ellipsis[opt]
  ///
  ///   objc-parms:
  ///     objc-parms , parameter-declaration
  ///
  ///   objc-ellipsis:
  ///     , ...
  ///
  ///   objc-keyword-attributes:         [OBJC2]
  ///     __attribute__((unused))
  /// \endverbatim
  ///
  Decl *ParseObjCMethodDecl(
      SourceLocation mLoc, tok::TokenKind mType,
      tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
      bool MethodDefinition = true);

  ///   Parse property attribute declarations.
  ///
  /// \verbatim
  ///   property-attr-decl: '(' property-attrlist ')'
  ///   property-attrlist:
  ///     property-attribute
  ///     property-attrlist ',' property-attribute
  ///   property-attribute:
  ///     getter '=' identifier
  ///     setter '=' identifier ':'
  ///     direct
  ///     readonly
  ///     readwrite
  ///     assign
  ///     retain
  ///     copy
  ///     nonatomic
  ///     atomic
  ///     strong
  ///     weak
  ///     unsafe_unretained
  ///     nonnull
  ///     nullable
  ///     null_unspecified
  ///     null_resettable
  ///     class
  /// \endverbatim
  ///
  void ParseObjCPropertyAttribute(ObjCDeclSpec &DS);

  /// \verbatim
  ///   objc-method-def: objc-method-proto ';'[opt] '{' body '}'
  /// \endverbatim
  ///
  Decl *ParseObjCMethodDefinition();

  //===--------------------------------------------------------------------===//
  // Objective-C Expressions
  ExprResult ParseObjCAtExpression(SourceLocation AtLocation);
  ExprResult ParseObjCStringLiteral(SourceLocation AtLoc);

  /// ParseObjCCharacterLiteral -
  /// \verbatim
  /// objc-scalar-literal : '@' character-literal
  ///                        ;
  /// \endverbatim
  ExprResult ParseObjCCharacterLiteral(SourceLocation AtLoc);

  /// ParseObjCNumericLiteral -
  /// \verbatim
  /// objc-scalar-literal : '@' scalar-literal
  ///                        ;
  /// scalar-literal : | numeric-constant			/* any numeric constant. */
  ///                    ;
  /// \endverbatim
  ExprResult ParseObjCNumericLiteral(SourceLocation AtLoc);

  /// ParseObjCBooleanLiteral -
  /// \verbatim
  /// objc-scalar-literal : '@' boolean-keyword
  ///                        ;
  /// boolean-keyword: 'true' | 'false' | '__objc_yes' | '__objc_no'
  ///                        ;
  /// \endverbatim
  ExprResult ParseObjCBooleanLiteral(SourceLocation AtLoc, bool ArgValue);

  ExprResult ParseObjCArrayLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCDictionaryLiteral(SourceLocation AtLoc);

  /// ParseObjCBoxedExpr -
  /// \verbatim
  /// objc-box-expression:
  ///       @( assignment-expression )
  /// \endverbatim
  ExprResult ParseObjCBoxedExpr(SourceLocation AtLoc);

  /// \verbatim
  ///    objc-encode-expression:
  ///      \@encode ( type-name )
  /// \endverbatim
  ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc);

  /// \verbatim
  ///     objc-selector-expression
  ///       @selector '(' '('[opt] objc-keyword-selector ')'[opt] ')'
  /// \endverbatim
  ExprResult ParseObjCSelectorExpression(SourceLocation AtLoc);

  /// \verbatim
  ///     objc-protocol-expression
  ///       \@protocol ( protocol-name )
  /// \endverbatim
  ExprResult ParseObjCProtocolExpression(SourceLocation AtLoc);

  /// Determine whether the parser is currently referring to a an
  /// Objective-C message send, using a simplified heuristic to avoid overhead.
  ///
  /// This routine will only return true for a subset of valid message-send
  /// expressions.
  bool isSimpleObjCMessageExpression();

  /// \verbatim
  ///   objc-message-expr:
  ///     '[' objc-receiver objc-message-args ']'
  ///
  ///   objc-receiver: [C]
  ///     'super'
  ///     expression
  ///     class-name
  ///     type-name
  /// \endverbatim
  ///
  ExprResult ParseObjCMessageExpression();

  /// Parse the remainder of an Objective-C message following the
  /// '[' objc-receiver.
  ///
  /// This routine handles sends to super, class messages (sent to a
  /// class name), and instance messages (sent to an object), and the
  /// target is represented by \p SuperLoc, \p ReceiverType, or \p
  /// ReceiverExpr, respectively. Only one of these parameters may have
  /// a valid value.
  ///
  /// \param LBracLoc The location of the opening '['.
  ///
  /// \param SuperLoc If this is a send to 'super', the location of the
  /// 'super' keyword that indicates a send to the superclass.
  ///
  /// \param ReceiverType If this is a class message, the type of the
  /// class we are sending a message to.
  ///
  /// \param ReceiverExpr If this is an instance message, the expression
  /// used to compute the receiver object.
  ///
  /// \verbatim
  ///   objc-message-args:
  ///     objc-selector
  ///     objc-keywordarg-list
  ///
  ///   objc-keywordarg-list:
  ///     objc-keywordarg
  ///     objc-keywordarg-list objc-keywordarg
  ///
  ///   objc-keywordarg:
  ///     selector-name[opt] ':' objc-keywordexpr
  ///
  ///   objc-keywordexpr:
  ///     nonempty-expr-list
  ///
  ///   nonempty-expr-list:
  ///     assignment-expression
  ///     nonempty-expr-list , assignment-expression
  /// \endverbatim
  ///
  ExprResult ParseObjCMessageExpressionBody(SourceLocation LBracloc,
                                            SourceLocation SuperLoc,
                                            ParsedType ReceiverType,
                                            Expr *ReceiverExpr);

  /// Parse the receiver of an Objective-C++ message send.
  ///
  /// This routine parses the receiver of a message send in
  /// Objective-C++ either as a type or as an expression. Note that this
  /// routine must not be called to parse a send to 'super', since it
  /// has no way to return such a result.
  ///
  /// \param IsExpr Whether the receiver was parsed as an expression.
  ///
  /// \param TypeOrExpr If the receiver was parsed as an expression (\c
  /// IsExpr is true), the parsed expression. If the receiver was parsed
  /// as a type (\c IsExpr is false), the parsed type.
  ///
  /// \returns True if an error occurred during parsing or semantic
  /// analysis, in which case the arguments do not have valid
  /// values. Otherwise, returns false for a successful parse.
  ///
  /// \verbatim
  ///   objc-receiver: [C++]
  ///     'super' [not parsed here]
  ///     expression
  ///     simple-type-specifier
  ///     typename-specifier
  /// \endverbatim
  bool ParseObjCXXMessageReceiver(bool &IsExpr, void *&TypeOrExpr);

  //===--------------------------------------------------------------------===//
  // Objective-C Statements

  enum class ParsedStmtContext;

  StmtResult ParseObjCAtStatement(SourceLocation atLoc,
                                  ParsedStmtContext StmtCtx);

  /// \verbatim
  ///  objc-try-catch-statement:
  ///    @try compound-statement objc-catch-list[opt]
  ///    @try compound-statement objc-catch-list[opt] @finally compound-statement
  ///
  ///  objc-catch-list:
  ///    @catch ( parameter-declaration ) compound-statement
  ///    objc-catch-list @catch ( catch-parameter-declaration ) compound-statement
  ///  catch-parameter-declaration:
  ///     parameter-declaration
  ///     '...' [OBJC2]
  /// \endverbatim
  ///
  StmtResult ParseObjCTryStmt(SourceLocation atLoc);

  /// \verbatim
  ///  objc-throw-statement:
  ///    throw expression[opt];
  /// \endverbatim
  ///
  StmtResult ParseObjCThrowStmt(SourceLocation atLoc);

  /// \verbatim
  /// objc-synchronized-statement:
  ///   @synchronized '(' expression ')' compound-statement
  /// \endverbatim
  ///
  StmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);

  /// \verbatim
  /// objc-autoreleasepool-statement:
  ///   @autoreleasepool compound-statement
  /// \endverbatim
  ///
  StmtResult ParseObjCAutoreleasePoolStmt(SourceLocation atLoc);

  /// ParseObjCTypeQualifierList - This routine parses the objective-c's type
  /// qualifier list and builds their bitmask representation in the input
  /// argument.
  ///
  /// \verbatim
  ///   objc-type-qualifiers:
  ///     objc-type-qualifier
  ///     objc-type-qualifiers objc-type-qualifier
  ///
  ///   objc-type-qualifier:
  ///     'in'
  ///     'out'
  ///     'inout'
  ///     'oneway'
  ///     'bycopy's
  ///     'byref'
  ///     'nonnull'
  ///     'nullable'
  ///     'null_unspecified'
  /// \endverbatim
  ///
  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS, DeclaratorContext Context);

  /// Determine whether we are currently at the start of an Objective-C
  /// class message that appears to be missing the open bracket '['.
  bool isStartOfObjCClassMessageMissingOpenBracket();

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name OpenACC Constructs
  /// Implementations are in ParseOpenACC.cpp
  ///@{

public:
  friend class ParsingOpenACCDirectiveRAII;

  /// Parse OpenACC directive on a declaration.
  ///
  /// Placeholder for now, should just ignore the directives after emitting a
  /// diagnostic. Eventually will be split into a few functions to parse
  /// different situations.
  DeclGroupPtrTy ParseOpenACCDirectiveDecl(AccessSpecifier &AS,
                                           ParsedAttributes &Attrs,
                                           DeclSpec::TST TagType,
                                           Decl *TagDecl);

  // Parse OpenACC Directive on a Statement.
  StmtResult ParseOpenACCDirectiveStmt();

private:
  /// Parsing OpenACC directive mode.
  bool OpenACCDirectiveParsing = false;

  /// Currently parsing a situation where an OpenACC array section could be
  /// legal, such as a 'var-list'.
  bool AllowOpenACCArraySections = false;

  /// RAII object to set reset OpenACC parsing a context where Array Sections
  /// are allowed.
  class OpenACCArraySectionRAII {
    Parser &P;

  public:
    OpenACCArraySectionRAII(Parser &P) : P(P) {
      assert(!P.AllowOpenACCArraySections);
      P.AllowOpenACCArraySections = true;
    }
    ~OpenACCArraySectionRAII() {
      assert(P.AllowOpenACCArraySections);
      P.AllowOpenACCArraySections = false;
    }
  };

  /// A struct to hold the information that got parsed by ParseOpenACCDirective,
  /// so that the callers of it can use that to construct the appropriate AST
  /// nodes.
  struct OpenACCDirectiveParseInfo {
    OpenACCDirectiveKind DirKind;
    SourceLocation StartLoc;
    SourceLocation DirLoc;
    SourceLocation LParenLoc;
    SourceLocation RParenLoc;
    SourceLocation EndLoc;
    SourceLocation MiscLoc;
    OpenACCAtomicKind AtomicKind;
    SmallVector<Expr *> Exprs;
    SmallVector<OpenACCClause *> Clauses;
    // TODO OpenACC: As we implement support for the Atomic, Routine, and Cache
    // constructs, we likely want to put that information in here as well.
  };

  struct OpenACCWaitParseInfo {
    bool Failed = false;
    Expr *DevNumExpr = nullptr;
    SourceLocation QueuesLoc;
    SmallVector<Expr *> QueueIdExprs;

    SmallVector<Expr *> getAllExprs() {
      SmallVector<Expr *> Out;
      Out.push_back(DevNumExpr);
      llvm::append_range(Out, QueueIdExprs);
      return Out;
    }
  };
  struct OpenACCCacheParseInfo {
    bool Failed = false;
    SourceLocation ReadOnlyLoc;
    SmallVector<Expr *> Vars;
  };

  /// Represents the 'error' state of parsing an OpenACC Clause, and stores
  /// whether we can continue parsing, or should give up on the directive.
  enum class OpenACCParseCanContinue { Cannot = 0, Can = 1 };

  /// A type to represent the state of parsing an OpenACC Clause. Situations
  /// that result in an OpenACCClause pointer are a success and can continue
  /// parsing, however some other situations can also continue.
  /// FIXME: This is better represented as a std::expected when we get C++23.
  using OpenACCClauseParseResult =
      llvm::PointerIntPair<OpenACCClause *, 1, OpenACCParseCanContinue>;

  OpenACCClauseParseResult OpenACCCanContinue();
  OpenACCClauseParseResult OpenACCCannotContinue();
  OpenACCClauseParseResult OpenACCSuccess(OpenACCClause *Clause);

  /// Parses the OpenACC directive (the entire pragma) including the clause
  /// list, but does not produce the main AST node.
  OpenACCDirectiveParseInfo ParseOpenACCDirective();
  /// Helper that parses an ID Expression based on the language options.
  ExprResult ParseOpenACCIDExpression();

  /// Parses the variable list for the `cache` construct.
  ///
  /// OpenACC 3.3, section 2.10:
  /// In C and C++, the syntax of the cache directive is:
  ///
  /// #pragma acc cache ([readonly:]var-list) new-line
  OpenACCCacheParseInfo ParseOpenACCCacheVarList();

  /// Tries to parse the 'modifier-list' for a 'copy', 'copyin', 'copyout', or
  /// 'create' clause.
  OpenACCModifierKind tryParseModifierList(OpenACCClauseKind CK);

  using OpenACCVarParseResult = std::pair<ExprResult, OpenACCParseCanContinue>;

  /// Parses a single variable in a variable list for OpenACC.
  ///
  /// OpenACC 3.3, section 1.6:
  /// In this spec, a 'var' (in italics) is one of the following:
  /// - a variable name (a scalar, array, or composite variable name)
  /// - a subarray specification with subscript ranges
  /// - an array element
  /// - a member of a composite variable
  /// - a common block name between slashes (fortran only)
  OpenACCVarParseResult ParseOpenACCVar(OpenACCDirectiveKind DK,
                                        OpenACCClauseKind CK);

  /// Parses the variable list for the variety of places that take a var-list.
  llvm::SmallVector<Expr *> ParseOpenACCVarList(OpenACCDirectiveKind DK,
                                                OpenACCClauseKind CK);

  /// Parses any parameters for an OpenACC Clause, including required/optional
  /// parens.
  ///
  /// The OpenACC Clause List is a comma or space-delimited list of clauses (see
  /// the comment on ParseOpenACCClauseList).  The concept of a 'clause' doesn't
  /// really have its owner grammar and each individual one has its own
  /// definition. However, they all are named with a single-identifier (or
  /// auto/default!) token, followed in some cases by either braces or parens.
  OpenACCClauseParseResult
  ParseOpenACCClauseParams(ArrayRef<const OpenACCClause *> ExistingClauses,
                           OpenACCDirectiveKind DirKind, OpenACCClauseKind Kind,
                           SourceLocation ClauseLoc);

  /// Parses a single clause in a clause-list for OpenACC. Returns nullptr on
  /// error.
  OpenACCClauseParseResult
  ParseOpenACCClause(ArrayRef<const OpenACCClause *> ExistingClauses,
                     OpenACCDirectiveKind DirKind);

  /// Parses the clause-list for an OpenACC directive.
  ///
  /// OpenACC 3.3, section 1.7:
  /// To simplify the specification and convey appropriate constraint
  /// information, a pqr-list is a comma-separated list of pdr items. The one
  /// exception is a clause-list, which is a list of one or more clauses
  /// optionally separated by commas.
  SmallVector<OpenACCClause *>
  ParseOpenACCClauseList(OpenACCDirectiveKind DirKind);

  /// OpenACC 3.3, section 2.16:
  /// In this section and throughout the specification, the term wait-argument
  /// means:
  /// \verbatim
  /// [ devnum : int-expr : ] [ queues : ] async-argument-list
  /// \endverbatim
  OpenACCWaitParseInfo ParseOpenACCWaitArgument(SourceLocation Loc,
                                                bool IsDirective);

  /// Parses the clause of the 'bind' argument, which can be a string literal or
  /// an identifier.
  std::variant<std::monostate, StringLiteral *, IdentifierInfo *>
  ParseOpenACCBindClauseArgument();

  /// A type to represent the state of parsing after an attempt to parse an
  /// OpenACC int-expr. This is useful to determine whether an int-expr list can
  /// continue parsing after a failed int-expr.
  using OpenACCIntExprParseResult =
      std::pair<ExprResult, OpenACCParseCanContinue>;
  /// Parses the clause kind of 'int-expr', which can be any integral
  /// expression.
  OpenACCIntExprParseResult ParseOpenACCIntExpr(OpenACCDirectiveKind DK,
                                                OpenACCClauseKind CK,
                                                SourceLocation Loc);
  /// Parses the argument list for 'num_gangs', which allows up to 3
  /// 'int-expr's.
  bool ParseOpenACCIntExprList(OpenACCDirectiveKind DK, OpenACCClauseKind CK,
                               SourceLocation Loc,
                               llvm::SmallVectorImpl<Expr *> &IntExprs);

  /// Parses the 'device-type-list', which is a list of identifiers.
  ///
  /// OpenACC 3.3 Section 2.4:
  /// The argument to the device_type clause is a comma-separated list of one or
  /// more device architecture name identifiers, or an asterisk.
  ///
  /// The syntax of the device_type clause is
  /// device_type( * )
  /// device_type( device-type-list )
  ///
  /// The device_type clause may be abbreviated to dtype.
  bool ParseOpenACCDeviceTypeList(llvm::SmallVector<IdentifierLoc> &Archs);

  /// Parses the 'async-argument', which is an integral value with two
  /// 'special' values that are likely negative (but come from Macros).
  ///
  /// OpenACC 3.3 section 2.16:
  /// In this section and throughout the specification, the term async-argument
  /// means a nonnegative scalar integer expression (int for C or C++, integer
  /// for Fortran), or one of the special values acc_async_noval or
  /// acc_async_sync, as defined in the C header file and the Fortran openacc
  /// module. The special values are negative values, so as not to conflict with
  /// a user-specified nonnegative async-argument.
  OpenACCIntExprParseResult ParseOpenACCAsyncArgument(OpenACCDirectiveKind DK,
                                                      OpenACCClauseKind CK,
                                                      SourceLocation Loc);

  /// Parses the 'size-expr', which is an integral value, or an asterisk.
  /// Asterisk is represented by a OpenACCAsteriskSizeExpr
  ///
  /// OpenACC 3.3 Section 2.9:
  /// size-expr is one of:
  ///    *
  ///    int-expr
  /// Note that this is specified under 'gang-arg-list', but also applies to
  /// 'tile' via reference.
  ExprResult ParseOpenACCSizeExpr(OpenACCClauseKind CK);

  /// Parses a comma delimited list of 'size-expr's.
  bool ParseOpenACCSizeExprList(OpenACCClauseKind CK,
                                llvm::SmallVectorImpl<Expr *> &SizeExprs);

  /// Parses a 'gang-arg-list', used for the 'gang' clause.
  ///
  /// OpenACC 3.3 Section 2.9:
  ///
  /// where gang-arg is one of:
  /// \verbatim
  /// [num:]int-expr
  /// dim:int-expr
  /// static:size-expr
  /// \endverbatim
  bool ParseOpenACCGangArgList(SourceLocation GangLoc,
                               llvm::SmallVectorImpl<OpenACCGangKind> &GKs,
                               llvm::SmallVectorImpl<Expr *> &IntExprs);

  using OpenACCGangArgRes = std::pair<OpenACCGangKind, ExprResult>;
  /// Parses a 'gang-arg', used for the 'gang' clause. Returns a pair of the
  /// ExprResult (which contains the validity of the expression), plus the gang
  /// kind for the current argument.
  OpenACCGangArgRes ParseOpenACCGangArg(SourceLocation GangLoc);
  /// Parses a 'condition' expr, ensuring it results in a
  ExprResult ParseOpenACCConditionExpr();
  DeclGroupPtrTy
  ParseOpenACCAfterRoutineDecl(AccessSpecifier &AS, ParsedAttributes &Attrs,
                               DeclSpec::TST TagType, Decl *TagDecl,
                               OpenACCDirectiveParseInfo &DirInfo);
  StmtResult ParseOpenACCAfterRoutineStmt(OpenACCDirectiveParseInfo &DirInfo);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name OpenMP Constructs
  /// Implementations are in ParseOpenMP.cpp
  ///@{

private:
  friend class ParsingOpenMPDirectiveRAII;

  /// Parsing OpenMP directive mode.
  bool OpenMPDirectiveParsing = false;

  /// Current kind of OpenMP clause
  OpenMPClauseKind OMPClauseKind = llvm::omp::OMPC_unknown;

  void ReplayOpenMPAttributeTokens(CachedTokens &OpenMPTokens) {
    // If parsing the attributes found an OpenMP directive, emit those tokens
    // to the parse stream now.
    if (!OpenMPTokens.empty()) {
      PP.EnterToken(Tok, /*IsReinject*/ true);
      PP.EnterTokenStream(OpenMPTokens, /*DisableMacroExpansion*/ true,
                          /*IsReinject*/ true);
      ConsumeAnyToken(/*ConsumeCodeCompletionTok*/ true);
    }
  }

  //===--------------------------------------------------------------------===//
  // OpenMP: Directives and clauses.

  /// Parse clauses for '#pragma omp declare simd'.
  DeclGroupPtrTy ParseOMPDeclareSimdClauses(DeclGroupPtrTy Ptr,
                                            CachedTokens &Toks,
                                            SourceLocation Loc);

  /// Parse a property kind into \p TIProperty for the selector set \p Set and
  /// selector \p Selector.
  void parseOMPTraitPropertyKind(OMPTraitProperty &TIProperty,
                                 llvm::omp::TraitSet Set,
                                 llvm::omp::TraitSelector Selector,
                                 llvm::StringMap<SourceLocation> &Seen);

  /// Parse a selector kind into \p TISelector for the selector set \p Set.
  void parseOMPTraitSelectorKind(OMPTraitSelector &TISelector,
                                 llvm::omp::TraitSet Set,
                                 llvm::StringMap<SourceLocation> &Seen);

  /// Parse a selector set kind into \p TISet.
  void parseOMPTraitSetKind(OMPTraitSet &TISet,
                            llvm::StringMap<SourceLocation> &Seen);

  /// Parses an OpenMP context property.
  void parseOMPContextProperty(OMPTraitSelector &TISelector,
                               llvm::omp::TraitSet Set,
                               llvm::StringMap<SourceLocation> &Seen);

  /// Parses an OpenMP context selector.
  ///
  /// \verbatim
  /// <trait-selector-name> ['('[<trait-score>] <trait-property> [, <t-p>]* ')']
  /// \endverbatim
  void parseOMPContextSelector(OMPTraitSelector &TISelector,
                               llvm::omp::TraitSet Set,
                               llvm::StringMap<SourceLocation> &SeenSelectors);

  /// Parses an OpenMP context selector set.
  ///
  /// \verbatim
  /// <trait-set-selector-name> '=' '{' <trait-selector> [, <trait-selector>]* '}'
  /// \endverbatim
  void parseOMPContextSelectorSet(OMPTraitSet &TISet,
                                  llvm::StringMap<SourceLocation> &SeenSets);

  /// Parse OpenMP context selectors:
  ///
  /// \verbatim
  /// <trait-set-selector> [, <trait-set-selector>]*
  /// \endverbatim
  bool parseOMPContextSelectors(SourceLocation Loc, OMPTraitInfo &TI);

  /// Parse an 'append_args' clause for '#pragma omp declare variant'.
  bool parseOpenMPAppendArgs(SmallVectorImpl<OMPInteropInfo> &InteropInfos);

  /// Parse a `match` clause for an '#pragma omp declare variant'. Return true
  /// if there was an error.
  bool parseOMPDeclareVariantMatchClause(SourceLocation Loc, OMPTraitInfo &TI,
                                         OMPTraitInfo *ParentTI);

  /// Parse clauses for '#pragma omp declare variant ( variant-func-id )
  /// clause'.
  void ParseOMPDeclareVariantClauses(DeclGroupPtrTy Ptr, CachedTokens &Toks,
                                     SourceLocation Loc);

  /// Parse 'omp [begin] assume[s]' directive.
  ///
  /// `omp assumes` or `omp begin/end assumes` <clause> [[,]<clause>]...
  /// where
  ///
  /// \verbatim
  ///   clause:
  ///     'ext_IMPL_DEFINED'
  ///     'absent' '(' directive-name [, directive-name]* ')'
  ///     'contains' '(' directive-name [, directive-name]* ')'
  ///     'holds' '(' scalar-expression ')'
  ///     'no_openmp'
  ///     'no_openmp_routines'
  ///     'no_openmp_constructs' (OpenMP 6.0)
  ///     'no_parallelism'
  /// \endverbatim
  ///
  void ParseOpenMPAssumesDirective(OpenMPDirectiveKind DKind,
                                   SourceLocation Loc);

  /// Parse 'omp end assumes' directive.
  void ParseOpenMPEndAssumesDirective(SourceLocation Loc);

  /// Parses clauses for directive.
  ///
  /// \verbatim
  /// <clause> [clause[ [,] clause] ... ]
  ///
  ///  clauses: for error directive
  ///     'at' '(' compilation | execution ')'
  ///     'severity' '(' fatal | warning ')'
  ///     'message' '(' msg-string ')'
  /// ....
  /// \endverbatim
  ///
  /// \param DKind Kind of current directive.
  /// \param clauses for current directive.
  /// \param start location for clauses of current directive
  void ParseOpenMPClauses(OpenMPDirectiveKind DKind,
                          SmallVectorImpl<clang::OMPClause *> &Clauses,
                          SourceLocation Loc);

  /// Parse clauses for '#pragma omp [begin] declare target'.
  void ParseOMPDeclareTargetClauses(SemaOpenMP::DeclareTargetContextInfo &DTCI);

  /// Parse '#pragma omp end declare target'.
  void ParseOMPEndDeclareTargetDirective(OpenMPDirectiveKind BeginDKind,
                                         OpenMPDirectiveKind EndDKind,
                                         SourceLocation Loc);

  /// Skip tokens until a `annot_pragma_openmp_end` was found. Emit a warning if
  /// it is not the current token.
  void skipUntilPragmaOpenMPEnd(OpenMPDirectiveKind DKind);

  /// Check the \p FoundKind against the \p ExpectedKind, if not issue an error
  /// that the "end" matching the "begin" directive of kind \p BeginKind was not
  /// found. Finally, if the expected kind was found or if \p SkipUntilOpenMPEnd
  /// is set, skip ahead using the helper `skipUntilPragmaOpenMPEnd`.
  void parseOMPEndDirective(OpenMPDirectiveKind BeginKind,
                            OpenMPDirectiveKind ExpectedKind,
                            OpenMPDirectiveKind FoundKind,
                            SourceLocation MatchingLoc, SourceLocation FoundLoc,
                            bool SkipUntilOpenMPEnd);

  /// Parses declarative OpenMP directives.
  ///
  /// \verbatim
  ///       threadprivate-directive:
  ///         annot_pragma_openmp 'threadprivate' simple-variable-list
  ///         annot_pragma_openmp_end
  ///
  ///       allocate-directive:
  ///         annot_pragma_openmp 'allocate' simple-variable-list [<clause>]
  ///         annot_pragma_openmp_end
  ///
  ///       declare-reduction-directive:
  ///        annot_pragma_openmp 'declare' 'reduction' [...]
  ///        annot_pragma_openmp_end
  ///
  ///       declare-mapper-directive:
  ///         annot_pragma_openmp 'declare' 'mapper' '(' [<mapper-identifer> ':']
  ///         <type> <var> ')' [<clause>[[,] <clause>] ... ]
  ///         annot_pragma_openmp_end
  ///
  ///       declare-simd-directive:
  ///         annot_pragma_openmp 'declare simd' {<clause> [,]}
  ///         annot_pragma_openmp_end
  ///         <function declaration/definition>
  ///
  ///       requires directive:
  ///         annot_pragma_openmp 'requires' <clause> [[[,] <clause>] ... ]
  ///         annot_pragma_openmp_end
  ///
  ///       assumes directive:
  ///         annot_pragma_openmp 'assumes' <clause> [[[,] <clause>] ... ]
  ///         annot_pragma_openmp_end
  ///       or
  ///         annot_pragma_openmp 'begin assumes' <clause> [[[,] <clause>] ... ]
  ///         annot_pragma_openmp 'end assumes'
  ///         annot_pragma_openmp_end
  /// \endverbatim
  ///
  DeclGroupPtrTy ParseOpenMPDeclarativeDirectiveWithExtDecl(
      AccessSpecifier &AS, ParsedAttributes &Attrs, bool Delayed = false,
      DeclSpec::TST TagType = DeclSpec::TST_unspecified,
      Decl *TagDecl = nullptr);

  /// Parse 'omp declare reduction' construct.
  ///
  /// \verbatim
  ///       declare-reduction-directive:
  ///        annot_pragma_openmp 'declare' 'reduction'
  ///        '(' <reduction_id> ':' <type> {',' <type>} ':' <expression> ')'
  ///        ['initializer' '(' ('omp_priv' '=' <expression>)|<function_call> ')']
  ///        annot_pragma_openmp_end
  /// \endverbatim
  /// <reduction_id> is either a base language identifier or one of the
  /// following operators: '+', '-', '*', '&', '|', '^', '&&' and '||'.
  ///
  DeclGroupPtrTy ParseOpenMPDeclareReductionDirective(AccessSpecifier AS);

  /// Parses initializer for provided omp_priv declaration inside the reduction
  /// initializer.
  void ParseOpenMPReductionInitializerForDecl(VarDecl *OmpPrivParm);

  /// Parses 'omp declare mapper' directive.
  ///
  /// \verbatim
  ///       declare-mapper-directive:
  ///         annot_pragma_openmp 'declare' 'mapper' '(' [<mapper-identifier> ':']
  ///         <type> <var> ')' [<clause>[[,] <clause>] ... ]
  ///         annot_pragma_openmp_end
  /// \endverbatim
  /// <mapper-identifier> and <var> are base language identifiers.
  ///
  DeclGroupPtrTy ParseOpenMPDeclareMapperDirective(AccessSpecifier AS);

  /// Parses variable declaration in 'omp declare mapper' directive.
  TypeResult parseOpenMPDeclareMapperVarDecl(SourceRange &Range,
                                             DeclarationName &Name,
                                             AccessSpecifier AS = AS_none);

  /// Parses simple list of variables.
  ///
  /// \verbatim
  ///   simple-variable-list:
  ///         '(' id-expression {, id-expression} ')'
  /// \endverbatim
  ///
  /// \param Kind Kind of the directive.
  /// \param Callback Callback function to be called for the list elements.
  /// \param AllowScopeSpecifier true, if the variables can have fully
  /// qualified names.
  ///
  bool ParseOpenMPSimpleVarList(
      OpenMPDirectiveKind Kind,
      const llvm::function_ref<void(CXXScopeSpec &, DeclarationNameInfo)>
          &Callback,
      bool AllowScopeSpecifier);

  /// Parses declarative or executable directive.
  ///
  /// \verbatim
  ///       threadprivate-directive:
  ///         annot_pragma_openmp 'threadprivate' simple-variable-list
  ///         annot_pragma_openmp_end
  ///
  ///       allocate-directive:
  ///         annot_pragma_openmp 'allocate' simple-variable-list
  ///         annot_pragma_openmp_end
  ///
  ///       declare-reduction-directive:
  ///         annot_pragma_openmp 'declare' 'reduction' '(' <reduction_id> ':'
  ///         <type> {',' <type>} ':' <expression> ')' ['initializer' '('
  ///         ('omp_priv' '=' <expression>|<function_call>) ')']
  ///         annot_pragma_openmp_end
  ///
  ///       declare-mapper-directive:
  ///         annot_pragma_openmp 'declare' 'mapper' '(' [<mapper-identifer> ':']
  ///         <type> <var> ')' [<clause>[[,] <clause>] ... ]
  ///         annot_pragma_openmp_end
  ///
  ///       executable-directive:
  ///         annot_pragma_openmp 'parallel' | 'simd' | 'for' | 'sections' |
  ///         'section' | 'single' | 'master' | 'critical' [ '(' <name> ')' ] |
  ///         'parallel for' | 'parallel sections' | 'parallel master' | 'task'
  ///         | 'taskyield' | 'barrier' | 'taskwait' | 'flush' | 'ordered' |
  ///         'error' | 'atomic' | 'for simd' | 'parallel for simd' | 'target' |
  ///         'target data' | 'taskgroup' | 'teams' | 'taskloop' | 'taskloop
  ///         simd' | 'master taskloop' | 'master taskloop simd' | 'parallel
  ///         master taskloop' | 'parallel master taskloop simd' | 'distribute'
  ///         | 'target enter data' | 'target exit data' | 'target parallel' |
  ///         'target parallel for' | 'target update' | 'distribute parallel
  ///         for' | 'distribute paralle for simd' | 'distribute simd' | 'target
  ///         parallel for simd' | 'target simd' | 'teams distribute' | 'teams
  ///         distribute simd' | 'teams distribute parallel for simd' | 'teams
  ///         distribute parallel for' | 'target teams' | 'target teams
  ///         distribute' | 'target teams distribute parallel for' | 'target
  ///         teams distribute parallel for simd' | 'target teams distribute
  ///         simd' | 'masked' | 'parallel masked' {clause}
  ///         annot_pragma_openmp_end
  /// \endverbatim
  ///
  ///
  /// \param StmtCtx The context in which we're parsing the directive.
  /// \param ReadDirectiveWithinMetadirective true if directive is within a
  /// metadirective and therefore ends on the closing paren.
  StmtResult ParseOpenMPDeclarativeOrExecutableDirective(
      ParsedStmtContext StmtCtx, bool ReadDirectiveWithinMetadirective = false);

  /// Parses executable directive.
  ///
  /// \param StmtCtx The context in which we're parsing the directive.
  /// \param DKind The kind of the executable directive.
  /// \param Loc Source location of the beginning of the directive.
  /// \param ReadDirectiveWithinMetadirective true if directive is within a
  /// metadirective and therefore ends on the closing paren.
  StmtResult
  ParseOpenMPExecutableDirective(ParsedStmtContext StmtCtx,
                                 OpenMPDirectiveKind DKind, SourceLocation Loc,
                                 bool ReadDirectiveWithinMetadirective);

  /// Parses informational directive.
  ///
  /// \param StmtCtx The context in which we're parsing the directive.
  /// \param DKind The kind of the informational directive.
  /// \param Loc Source location of the beginning of the directive.
  /// \param ReadDirectiveWithinMetadirective true if directive is within a
  /// metadirective and therefore ends on the closing paren.
  StmtResult ParseOpenMPInformationalDirective(
      ParsedStmtContext StmtCtx, OpenMPDirectiveKind DKind, SourceLocation Loc,
      bool ReadDirectiveWithinMetadirective);

  /// Parses clause of kind \a CKind for directive of a kind \a Kind.
  ///
  /// \verbatim
  ///    clause:
  ///       if-clause | final-clause | num_threads-clause | safelen-clause |
  ///       default-clause | private-clause | firstprivate-clause |
  ///       shared-clause | linear-clause | aligned-clause | collapse-clause |
  ///       bind-clause | lastprivate-clause | reduction-clause |
  ///       proc_bind-clause | schedule-clause | copyin-clause |
  ///       copyprivate-clause | untied-clause | mergeable-clause | flush-clause
  ///       | read-clause | write-clause | update-clause | capture-clause |
  ///       seq_cst-clause | device-clause | simdlen-clause | threads-clause |
  ///       simd-clause | num_teams-clause | thread_limit-clause |
  ///       priority-clause | grainsize-clause | nogroup-clause |
  ///       num_tasks-clause | hint-clause | to-clause | from-clause |
  ///       is_device_ptr-clause | task_reduction-clause | in_reduction-clause |
  ///       allocator-clause | allocate-clause | acq_rel-clause | acquire-clause
  ///       | release-clause | relaxed-clause | depobj-clause | destroy-clause |
  ///       detach-clause | inclusive-clause | exclusive-clause |
  ///       uses_allocators-clause | use_device_addr-clause | has_device_addr
  /// \endverbatim
  ///
  /// \param DKind Kind of current directive.
  /// \param CKind Kind of current clause.
  /// \param FirstClause true, if this is the first clause of a kind \a CKind
  /// in current directive.
  ///
  OMPClause *ParseOpenMPClause(OpenMPDirectiveKind DKind,
                               OpenMPClauseKind CKind, bool FirstClause);

  /// Parses clause with a single expression of a kind \a Kind.
  ///
  /// Parsing of OpenMP clauses with single expressions like 'final',
  /// 'collapse', 'safelen', 'num_threads', 'simdlen', 'num_teams',
  /// 'thread_limit', 'simdlen', 'priority', 'grainsize', 'num_tasks', 'hint' or
  /// 'detach'.
  ///
  /// \verbatim
  ///    final-clause:
  ///      'final' '(' expression ')'
  ///
  ///    num_threads-clause:
  ///      'num_threads' '(' expression ')'
  ///
  ///    safelen-clause:
  ///      'safelen' '(' expression ')'
  ///
  ///    simdlen-clause:
  ///      'simdlen' '(' expression ')'
  ///
  ///    collapse-clause:
  ///      'collapse' '(' expression ')'
  ///
  ///    priority-clause:
  ///      'priority' '(' expression ')'
  ///
  ///    grainsize-clause:
  ///      'grainsize' '(' expression ')'
  ///
  ///    num_tasks-clause:
  ///      'num_tasks' '(' expression ')'
  ///
  ///    hint-clause:
  ///      'hint' '(' expression ')'
  ///
  ///    allocator-clause:
  ///      'allocator' '(' expression ')'
  ///
  ///    detach-clause:
  ///      'detach' '(' event-handler-expression ')'
  ///
  ///    align-clause
  ///      'align' '(' positive-integer-constant ')'
  ///
  ///    holds-clause
  ///      'holds' '(' expression ')'
  /// \endverbatim
  ///
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  ///
  OMPClause *ParseOpenMPSingleExprClause(OpenMPClauseKind Kind, bool ParseOnly);
  /// Parses simple clause like 'default' or 'proc_bind' of a kind \a Kind.
  ///
  /// \verbatim
  ///    default-clause:
  ///         'default' '(' 'none' | 'shared' | 'private' | 'firstprivate' ')'
  ///
  ///    proc_bind-clause:
  ///         'proc_bind' '(' 'master' | 'close' | 'spread' ')'
  ///
  ///    bind-clause:
  ///         'bind' '(' 'teams' | 'parallel' | 'thread' ')'
  ///
  ///    update-clause:
  ///         'update' '(' 'in' | 'out' | 'inout' | 'mutexinoutset' |
  ///         'inoutset' ')'
  /// \endverbatim
  ///
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  ///
  OMPClause *ParseOpenMPSimpleClause(OpenMPClauseKind Kind, bool ParseOnly);

  /// Parse indirect clause for '#pragma omp declare target' directive.
  ///  'indirect' '[' '(' invoked-by-fptr ')' ']'
  /// where invoked-by-fptr is a constant boolean expression that evaluates to
  /// true or false at compile time.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// false;
  bool ParseOpenMPIndirectClause(SemaOpenMP::DeclareTargetContextInfo &DTCI,
                                 bool ParseOnly);
  /// Parses clause with a single expression and an additional argument
  /// of a kind \a Kind like 'schedule' or 'dist_schedule'.
  ///
  /// \verbatim
  ///    schedule-clause:
  ///      'schedule' '(' [ modifier [ ',' modifier ] ':' ] kind [',' expression ]
  ///      ')'
  ///
  ///    if-clause:
  ///      'if' '(' [ directive-name-modifier ':' ] expression ')'
  ///
  ///    defaultmap:
  ///      'defaultmap' '(' modifier [ ':' kind ] ')'
  ///
  ///    device-clause:
  ///      'device' '(' [ device-modifier ':' ] expression ')'
  /// \endverbatim
  ///
  /// \param DKind Directive kind.
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  ///
  OMPClause *ParseOpenMPSingleExprWithArgClause(OpenMPDirectiveKind DKind,
                                                OpenMPClauseKind Kind,
                                                bool ParseOnly);

  /// Parses the 'sizes' clause of a '#pragma omp tile' directive.
  OMPClause *ParseOpenMPSizesClause();

  /// Parses the 'permutation' clause of a '#pragma omp interchange' directive.
  OMPClause *ParseOpenMPPermutationClause();

  /// Parses clause without any additional arguments like 'ordered'.
  ///
  /// \verbatim
  ///    ordered-clause:
  ///         'ordered'
  ///
  ///    nowait-clause:
  ///         'nowait'
  ///
  ///    untied-clause:
  ///         'untied'
  ///
  ///    mergeable-clause:
  ///         'mergeable'
  ///
  ///    read-clause:
  ///         'read'
  ///
  ///    threads-clause:
  ///         'threads'
  ///
  ///    simd-clause:
  ///         'simd'
  ///
  ///    nogroup-clause:
  ///         'nogroup'
  /// \endverbatim
  ///
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  ///
  OMPClause *ParseOpenMPClause(OpenMPClauseKind Kind, bool ParseOnly = false);

  /// Parses clause with the list of variables of a kind \a Kind:
  /// 'private', 'firstprivate', 'lastprivate',
  /// 'shared', 'copyin', 'copyprivate', 'flush', 'reduction', 'task_reduction',
  /// 'in_reduction', 'nontemporal', 'exclusive' or 'inclusive'.
  ///
  /// \verbatim
  ///    private-clause:
  ///       'private' '(' list ')'
  ///    firstprivate-clause:
  ///       'firstprivate' '(' list ')'
  ///    lastprivate-clause:
  ///       'lastprivate' '(' list ')'
  ///    shared-clause:
  ///       'shared' '(' list ')'
  ///    linear-clause:
  ///       'linear' '(' linear-list [ ':' linear-step ] ')'
  ///    aligned-clause:
  ///       'aligned' '(' list [ ':' alignment ] ')'
  ///    reduction-clause:
  ///       'reduction' '(' [ modifier ',' ] reduction-identifier ':' list ')'
  ///    task_reduction-clause:
  ///       'task_reduction' '(' reduction-identifier ':' list ')'
  ///    in_reduction-clause:
  ///       'in_reduction' '(' reduction-identifier ':' list ')'
  ///    copyprivate-clause:
  ///       'copyprivate' '(' list ')'
  ///    flush-clause:
  ///       'flush' '(' list ')'
  ///    depend-clause:
  ///       'depend' '(' in | out | inout : list | source ')'
  ///    map-clause:
  ///       'map' '(' [ [ always [,] ] [ close [,] ]
  ///          [ mapper '(' mapper-identifier ')' [,] ]
  ///          to | from | tofrom | alloc | release | delete ':' ] list ')';
  ///    to-clause:
  ///       'to' '(' [ mapper '(' mapper-identifier ')' ':' ] list ')'
  ///    from-clause:
  ///       'from' '(' [ mapper '(' mapper-identifier ')' ':' ] list ')'
  ///    use_device_ptr-clause:
  ///       'use_device_ptr' '(' list ')'
  ///    use_device_addr-clause:
  ///       'use_device_addr' '(' list ')'
  ///    is_device_ptr-clause:
  ///       'is_device_ptr' '(' list ')'
  ///    has_device_addr-clause:
  ///       'has_device_addr' '(' list ')'
  ///    allocate-clause:
  ///       'allocate' '(' [ allocator ':' ] list ')'
  ///       As of OpenMP 5.1 there's also
  ///         'allocate' '(' allocate-modifier: list ')'
  ///         where allocate-modifier is: 'allocator' '(' allocator ')'
  ///    nontemporal-clause:
  ///       'nontemporal' '(' list ')'
  ///    inclusive-clause:
  ///       'inclusive' '(' list ')'
  ///    exclusive-clause:
  ///       'exclusive' '(' list ')'
  /// \endverbatim
  ///
  /// For 'linear' clause linear-list may have the following forms:
  ///  list
  ///  modifier(list)
  /// where modifier is 'val' (C) or 'ref', 'val' or 'uval'(C++).
  ///
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  ///
  OMPClause *ParseOpenMPVarListClause(OpenMPDirectiveKind DKind,
                                      OpenMPClauseKind Kind, bool ParseOnly);

  /// Parses a clause consisting of a list of expressions.
  ///
  /// \param Kind          The clause to parse.
  /// \param ClauseNameLoc [out] The location of the clause name.
  /// \param OpenLoc       [out] The location of '('.
  /// \param CloseLoc      [out] The location of ')'.
  /// \param Exprs         [out] The parsed expressions.
  /// \param ReqIntConst   If true, each expression must be an integer constant.
  ///
  /// \return Whether the clause was parsed successfully.
  bool ParseOpenMPExprListClause(OpenMPClauseKind Kind,
                                 SourceLocation &ClauseNameLoc,
                                 SourceLocation &OpenLoc,
                                 SourceLocation &CloseLoc,
                                 SmallVectorImpl<Expr *> &Exprs,
                                 bool ReqIntConst = false);

  /// Parses simple expression in parens for single-expression clauses of OpenMP
  /// constructs.
  /// \verbatim
  /// <iterators> = 'iterator' '(' { [ <iterator-type> ] identifier =
  /// <range-specification> }+ ')'
  /// \endverbatim
  ExprResult ParseOpenMPIteratorsExpr();

  /// Parses allocators and traits in the context of the uses_allocator clause.
  /// Expected format:
  /// \verbatim
  /// '(' { <allocator> [ '(' <allocator_traits> ')' ] }+ ')'
  /// \endverbatim
  OMPClause *ParseOpenMPUsesAllocatorClause(OpenMPDirectiveKind DKind);

  /// Parses the 'interop' parts of the 'append_args' and 'init' clauses.
  bool ParseOMPInteropInfo(OMPInteropInfo &InteropInfo, OpenMPClauseKind Kind);

  /// Parses clause with an interop variable of kind \a Kind.
  ///
  /// \verbatim
  /// init-clause:
  ///   init([interop-modifier, ]interop-type[[, interop-type] ... ]:interop-var)
  ///
  /// destroy-clause:
  ///   destroy(interop-var)
  ///
  /// use-clause:
  ///   use(interop-var)
  ///
  /// interop-modifier:
  ///   prefer_type(preference-list)
  ///
  /// preference-list:
  ///   foreign-runtime-id [, foreign-runtime-id]...
  ///
  /// foreign-runtime-id:
  ///   <string-literal> | <constant-integral-expression>
  ///
  /// interop-type:
  ///   target | targetsync
  /// \endverbatim
  ///
  /// \param Kind Kind of current clause.
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  //
  OMPClause *ParseOpenMPInteropClause(OpenMPClauseKind Kind, bool ParseOnly);

  /// Parses a ompx_attribute clause
  ///
  /// \param ParseOnly true to skip the clause's semantic actions and return
  /// nullptr.
  //
  OMPClause *ParseOpenMPOMPXAttributesClause(bool ParseOnly);

public:
  /// Parses simple expression in parens for single-expression clauses of OpenMP
  /// constructs.
  /// \param RLoc Returned location of right paren.
  ExprResult ParseOpenMPParensExpr(StringRef ClauseName, SourceLocation &RLoc,
                                   bool IsAddressOfOperand = false);

  /// Parses a reserved locator like 'omp_all_memory'.
  bool ParseOpenMPReservedLocator(OpenMPClauseKind Kind,
                                  SemaOpenMP::OpenMPVarListDataTy &Data,
                                  const LangOptions &LangOpts);
  /// Parses clauses with list.
  bool ParseOpenMPVarList(OpenMPDirectiveKind DKind, OpenMPClauseKind Kind,
                          SmallVectorImpl<Expr *> &Vars,
                          SemaOpenMP::OpenMPVarListDataTy &Data);

  /// Parses the mapper modifier in map, to, and from clauses.
  bool parseMapperModifier(SemaOpenMP::OpenMPVarListDataTy &Data);

  /// Parse map-type-modifiers in map clause.
  /// map([ [map-type-modifier[,] [map-type-modifier[,] ...] [map-type] : ] list)
  /// where, map-type-modifier ::= always | close | mapper(mapper-identifier) |
  /// present
  /// where, map-type ::= alloc | delete | from | release | to | tofrom
  bool parseMapTypeModifiers(SemaOpenMP::OpenMPVarListDataTy &Data);

  /// Parses 'omp begin declare variant' directive.
  /// The syntax is:
  /// \verbatim
  /// { #pragma omp begin declare variant clause }
  /// <function-declaration-or-definition-sequence>
  /// { #pragma omp end declare variant }
  /// \endverbatim
  ///
  bool ParseOpenMPDeclareBeginVariantDirective(SourceLocation Loc);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Pragmas
  /// Implementations are in ParsePragma.cpp
  ///@{

private:
  std::unique_ptr<PragmaHandler> AlignHandler;
  std::unique_ptr<PragmaHandler> GCCVisibilityHandler;
  std::unique_ptr<PragmaHandler> OptionsHandler;
  std::unique_ptr<PragmaHandler> PackHandler;
  std::unique_ptr<PragmaHandler> MSStructHandler;
  std::unique_ptr<PragmaHandler> UnusedHandler;
  std::unique_ptr<PragmaHandler> WeakHandler;
  std::unique_ptr<PragmaHandler> RedefineExtnameHandler;
  std::unique_ptr<PragmaHandler> FPContractHandler;
  std::unique_ptr<PragmaHandler> OpenCLExtensionHandler;
  std::unique_ptr<PragmaHandler> OpenMPHandler;
  std::unique_ptr<PragmaHandler> OpenACCHandler;
  std::unique_ptr<PragmaHandler> PCSectionHandler;
  std::unique_ptr<PragmaHandler> MSCommentHandler;
  std::unique_ptr<PragmaHandler> MSDetectMismatchHandler;
  std::unique_ptr<PragmaHandler> FPEvalMethodHandler;
  std::unique_ptr<PragmaHandler> FloatControlHandler;
  std::unique_ptr<PragmaHandler> MSPointersToMembers;
  std::unique_ptr<PragmaHandler> MSVtorDisp;
  std::unique_ptr<PragmaHandler> MSInitSeg;
  std::unique_ptr<PragmaHandler> MSDataSeg;
  std::unique_ptr<PragmaHandler> MSBSSSeg;
  std::unique_ptr<PragmaHandler> MSConstSeg;
  std::unique_ptr<PragmaHandler> MSCodeSeg;
  std::unique_ptr<PragmaHandler> MSSection;
  std::unique_ptr<PragmaHandler> MSStrictGuardStackCheck;
  std::unique_ptr<PragmaHandler> MSRuntimeChecks;
  std::unique_ptr<PragmaHandler> MSIntrinsic;
  std::unique_ptr<PragmaHandler> MSFunction;
  std::unique_ptr<PragmaHandler> MSOptimize;
  std::unique_ptr<PragmaHandler> MSFenvAccess;
  std::unique_ptr<PragmaHandler> MSAllocText;
  std::unique_ptr<PragmaHandler> CUDAForceHostDeviceHandler;
  std::unique_ptr<PragmaHandler> OptimizeHandler;
  std::unique_ptr<PragmaHandler> LoopHintHandler;
  std::unique_ptr<PragmaHandler> UnrollHintHandler;
  std::unique_ptr<PragmaHandler> NoUnrollHintHandler;
  std::unique_ptr<PragmaHandler> UnrollAndJamHintHandler;
  std::unique_ptr<PragmaHandler> NoUnrollAndJamHintHandler;
  std::unique_ptr<PragmaHandler> FPHandler;
  std::unique_ptr<PragmaHandler> STDCFenvAccessHandler;
  std::unique_ptr<PragmaHandler> STDCFenvRoundHandler;
  std::unique_ptr<PragmaHandler> STDCCXLIMITHandler;
  std::unique_ptr<PragmaHandler> STDCUnknownHandler;
  std::unique_ptr<PragmaHandler> AttributePragmaHandler;
  std::unique_ptr<PragmaHandler> MaxTokensHerePragmaHandler;
  std::unique_ptr<PragmaHandler> MaxTokensTotalPragmaHandler;
  std::unique_ptr<PragmaHandler> RISCVPragmaHandler;

  /// Initialize all pragma handlers.
  void initializePragmaHandlers();

  /// Destroy and reset all pragma handlers.
  void resetPragmaHandlers();

  /// Handle the annotation token produced for #pragma unused(...)
  ///
  /// Each annot_pragma_unused is followed by the argument token so e.g.
  /// "#pragma unused(x,y)" becomes:
  /// annot_pragma_unused 'x' annot_pragma_unused 'y'
  void HandlePragmaUnused();

  /// Handle the annotation token produced for
  /// #pragma GCC visibility...
  void HandlePragmaVisibility();

  /// Handle the annotation token produced for
  /// #pragma pack...
  void HandlePragmaPack();

  /// Handle the annotation token produced for
  /// #pragma ms_struct...
  void HandlePragmaMSStruct();

  void HandlePragmaMSPointersToMembers();

  void HandlePragmaMSVtorDisp();

  void HandlePragmaMSPragma();
  bool HandlePragmaMSSection(StringRef PragmaName,
                             SourceLocation PragmaLocation);
  bool HandlePragmaMSSegment(StringRef PragmaName,
                             SourceLocation PragmaLocation);

  // #pragma init_seg({ compiler | lib | user | "section-name" [, func-name]} )
  bool HandlePragmaMSInitSeg(StringRef PragmaName,
                             SourceLocation PragmaLocation);

  // #pragma strict_gs_check(pop)
  // #pragma strict_gs_check(push, "on" | "off")
  // #pragma strict_gs_check("on" | "off")
  bool HandlePragmaMSStrictGuardStackCheck(StringRef PragmaName,
                                           SourceLocation PragmaLocation);
  bool HandlePragmaMSFunction(StringRef PragmaName,
                              SourceLocation PragmaLocation);
  bool HandlePragmaMSAllocText(StringRef PragmaName,
                               SourceLocation PragmaLocation);

  // #pragma optimize("gsty", on|off)
  bool HandlePragmaMSOptimize(StringRef PragmaName,
                              SourceLocation PragmaLocation);

  // #pragma intrinsic("foo")
  bool HandlePragmaMSIntrinsic(StringRef PragmaName,
                               SourceLocation PragmaLocation);

  /// Handle the annotation token produced for
  /// #pragma align...
  void HandlePragmaAlign();

  /// Handle the annotation token produced for
  /// #pragma clang __debug dump...
  void HandlePragmaDump();

  /// Handle the annotation token produced for
  /// #pragma weak id...
  void HandlePragmaWeak();

  /// Handle the annotation token produced for
  /// #pragma weak id = id...
  void HandlePragmaWeakAlias();

  /// Handle the annotation token produced for
  /// #pragma redefine_extname...
  void HandlePragmaRedefineExtname();

  /// Handle the annotation token produced for
  /// #pragma STDC FP_CONTRACT...
  void HandlePragmaFPContract();

  /// Handle the annotation token produced for
  /// #pragma STDC FENV_ACCESS...
  void HandlePragmaFEnvAccess();

  /// Handle the annotation token produced for
  /// #pragma STDC FENV_ROUND...
  void HandlePragmaFEnvRound();

  /// Handle the annotation token produced for
  /// #pragma STDC CX_LIMITED_RANGE...
  void HandlePragmaCXLimitedRange();

  /// Handle the annotation token produced for
  /// #pragma float_control
  void HandlePragmaFloatControl();

  /// \brief Handle the annotation token produced for
  /// #pragma clang fp ...
  void HandlePragmaFP();

  /// Handle the annotation token produced for
  /// #pragma OPENCL EXTENSION...
  void HandlePragmaOpenCLExtension();

  /// Handle the annotation token produced for
  /// #pragma clang __debug captured
  StmtResult HandlePragmaCaptured();

  /// Handle the annotation token produced for
  /// #pragma clang loop and #pragma unroll.
  bool HandlePragmaLoopHint(LoopHint &Hint);

  bool ParsePragmaAttributeSubjectMatchRuleSet(
      attr::ParsedSubjectMatchRuleSet &SubjectMatchRules,
      SourceLocation &AnyLoc, SourceLocation &LastMatchRuleEndLoc);

  void HandlePragmaAttribute();

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Statements
  /// Implementations are in ParseStmt.cpp
  ///@{

public:
  /// A SmallVector of statements.
  typedef SmallVector<Stmt *, 24> StmtVector;

  /// The location of the first statement inside an else that might
  /// have a missleading indentation. If there is no
  /// MisleadingIndentationChecker on an else active, this location is invalid.
  SourceLocation MisleadingIndentationElseLoc;

  private:

  /// Flags describing a context in which we're parsing a statement.
  enum class ParsedStmtContext {
    /// This context permits declarations in language modes where declarations
    /// are not statements.
    AllowDeclarationsInC = 0x1,
    /// This context permits standalone OpenMP directives.
    AllowStandaloneOpenMPDirectives = 0x2,
    /// This context is at the top level of a GNU statement expression.
    InStmtExpr = 0x4,

    /// The context of a regular substatement.
    SubStmt = 0,
    /// The context of a compound-statement.
    Compound = AllowDeclarationsInC | AllowStandaloneOpenMPDirectives,

    LLVM_MARK_AS_BITMASK_ENUM(InStmtExpr)
  };

  /// Act on an expression statement that might be the last statement in a
  /// GNU statement expression. Checks whether we are actually at the end of
  /// a statement expression and builds a suitable expression statement.
  StmtResult handleExprStmt(ExprResult E, ParsedStmtContext StmtCtx);

  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.

  /// Parse a standalone statement (for instance, as the body of an 'if',
  /// 'while', or 'for').
  StmtResult
  ParseStatement(SourceLocation *TrailingElseLoc = nullptr,
                 ParsedStmtContext StmtCtx = ParsedStmtContext::SubStmt,
                 LabelDecl *PrecedingLabel = nullptr);

  /// ParseStatementOrDeclaration - Read 'statement' or 'declaration'.
  /// \verbatim
  ///       StatementOrDeclaration:
  ///         statement
  ///         declaration
  ///
  ///       statement:
  ///         labeled-statement
  ///         compound-statement
  ///         expression-statement
  ///         selection-statement
  ///         iteration-statement
  ///         jump-statement
  /// [C++]   declaration-statement
  /// [C++]   try-block
  /// [MS]    seh-try-block
  /// [OBC]   objc-throw-statement
  /// [OBC]   objc-try-catch-statement
  /// [OBC]   objc-synchronized-statement
  /// [GNU]   asm-statement
  /// [OMP]   openmp-construct             [TODO]
  ///
  ///       labeled-statement:
  ///         identifier ':' statement
  ///         'case' constant-expression ':' statement
  ///         'default' ':' statement
  ///
  ///       selection-statement:
  ///         if-statement
  ///         switch-statement
  ///
  ///       iteration-statement:
  ///         while-statement
  ///         do-statement
  ///         for-statement
  ///
  ///       expression-statement:
  ///         expression[opt] ';'
  ///
  ///       jump-statement:
  ///         'goto' identifier ';'
  ///         'continue' ';'
  ///         'break' ';'
  ///         'return' expression[opt] ';'
  /// [GNU]   'goto' '*' expression ';'
  ///
  /// [OBC] objc-throw-statement:
  /// [OBC]   '@' 'throw' expression ';'
  /// [OBC]   '@' 'throw' ';'
  /// \endverbatim
  ///
  StmtResult
  ParseStatementOrDeclaration(StmtVector &Stmts, ParsedStmtContext StmtCtx,
                              SourceLocation *TrailingElseLoc = nullptr,
                              LabelDecl *PrecedingLabel = nullptr);

  StmtResult ParseStatementOrDeclarationAfterAttributes(
      StmtVector &Stmts, ParsedStmtContext StmtCtx,
      SourceLocation *TrailingElseLoc, ParsedAttributes &DeclAttrs,
      ParsedAttributes &DeclSpecAttrs, LabelDecl *PrecedingLabel);

  /// Parse an expression statement.
  StmtResult ParseExprStatement(ParsedStmtContext StmtCtx);

  /// ParseLabeledStatement - We have an identifier and a ':' after it.
  ///
  /// \verbatim
  ///       label:
  ///         identifier ':'
  /// [GNU]   identifier ':' attributes[opt]
  ///
  ///       labeled-statement:
  ///         label statement
  /// \endverbatim
  ///
  StmtResult ParseLabeledStatement(ParsedAttributes &Attrs,
                                   ParsedStmtContext StmtCtx);

  /// ParseCaseStatement
  /// \verbatim
  ///       labeled-statement:
  ///         'case' constant-expression ':' statement
  /// [GNU]   'case' constant-expression '...' constant-expression ':' statement
  /// \endverbatim
  ///
  StmtResult ParseCaseStatement(ParsedStmtContext StmtCtx,
                                bool MissingCase = false,
                                ExprResult Expr = ExprResult());

  /// ParseDefaultStatement
  /// \verbatim
  ///       labeled-statement:
  ///         'default' ':' statement
  /// \endverbatim
  /// Note that this does not parse the 'statement' at the end.
  ///
  StmtResult ParseDefaultStatement(ParsedStmtContext StmtCtx);

  StmtResult ParseCompoundStatement(bool isStmtExpr = false);

  /// ParseCompoundStatement - Parse a "{}" block.
  ///
  /// \verbatim
  ///       compound-statement: [C99 6.8.2]
  ///         { block-item-list[opt] }
  /// [GNU]   { label-declarations block-item-list } [TODO]
  ///
  ///       block-item-list:
  ///         block-item
  ///         block-item-list block-item
  ///
  ///       block-item:
  ///         declaration
  /// [GNU]   '__extension__' declaration
  ///         statement
  ///
  /// [GNU] label-declarations:
  /// [GNU]   label-declaration
  /// [GNU]   label-declarations label-declaration
  ///
  /// [GNU] label-declaration:
  /// [GNU]   '__label__' identifier-list ';'
  /// \endverbatim
  ///
  StmtResult ParseCompoundStatement(bool isStmtExpr, unsigned ScopeFlags);

  /// Parse any pragmas at the start of the compound expression. We handle these
  /// separately since some pragmas (FP_CONTRACT) must appear before any C
  /// statement in the compound, but may be intermingled with other pragmas.
  void ParseCompoundStatementLeadingPragmas();

  void DiagnoseLabelAtEndOfCompoundStatement();

  /// Consume any extra semi-colons resulting in null statements,
  /// returning true if any tok::semi were consumed.
  bool ConsumeNullStmt(StmtVector &Stmts);

  /// ParseCompoundStatementBody - Parse a sequence of statements optionally
  /// followed by a label and invoke the ActOnCompoundStmt action.  This expects
  /// the '{' to be the current token, and consume the '}' at the end of the
  /// block.  It does not manipulate the scope stack.
  StmtResult ParseCompoundStatementBody(bool isStmtExpr = false);

  /// ParseParenExprOrCondition:
  /// \verbatim
  /// [C  ]     '(' expression ')'
  /// [C++]     '(' condition ')'
  /// [C++1z]   '(' init-statement[opt] condition ')'
  /// \endverbatim
  ///
  /// This function parses and performs error recovery on the specified
  /// condition or expression (depending on whether we're in C++ or C mode).
  /// This function goes out of its way to recover well.  It returns true if
  /// there was a parser error (the right paren couldn't be found), which
  /// indicates that the caller should try to recover harder.  It returns false
  /// if the condition is successfully parsed.  Note that a successful parse can
  /// still have semantic errors in the condition. Additionally, it will assign
  /// the location of the outer-most '(' and ')', to LParenLoc and RParenLoc,
  /// respectively.
  bool ParseParenExprOrCondition(StmtResult *InitStmt,
                                 Sema::ConditionResult &CondResult,
                                 SourceLocation Loc, Sema::ConditionKind CK,
                                 SourceLocation &LParenLoc,
                                 SourceLocation &RParenLoc);

  /// ParseIfStatement
  /// \verbatim
  ///       if-statement: [C99 6.8.4.1]
  ///         'if' '(' expression ')' statement
  ///         'if' '(' expression ')' statement 'else' statement
  /// [C++]   'if' '(' condition ')' statement
  /// [C++]   'if' '(' condition ')' statement 'else' statement
  /// [C++23] 'if' '!' [opt] consteval compound-statement
  /// [C++23] 'if' '!' [opt] consteval compound-statement 'else' statement
  /// \endverbatim
  ///
  StmtResult ParseIfStatement(SourceLocation *TrailingElseLoc);

  /// ParseSwitchStatement
  /// \verbatim
  ///       switch-statement:
  ///         'switch' '(' expression ')' statement
  /// [C++]   'switch' '(' condition ')' statement
  /// \endverbatim
  StmtResult ParseSwitchStatement(SourceLocation *TrailingElseLoc,
                                  LabelDecl *PrecedingLabel);

  /// ParseWhileStatement
  /// \verbatim
  ///       while-statement: [C99 6.8.5.1]
  ///         'while' '(' expression ')' statement
  /// [C++]   'while' '(' condition ')' statement
  /// \endverbatim
  StmtResult ParseWhileStatement(SourceLocation *TrailingElseLoc,
                                 LabelDecl *PrecedingLabel);

  /// ParseDoStatement
  /// \verbatim
  ///       do-statement: [C99 6.8.5.2]
  ///         'do' statement 'while' '(' expression ')' ';'
  /// \endverbatim
  /// Note: this lets the caller parse the end ';'.
  StmtResult ParseDoStatement(LabelDecl *PrecedingLabel);

  /// ParseForStatement
  /// \verbatim
  ///       for-statement: [C99 6.8.5.3]
  ///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
  ///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
  /// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
  /// [C++]       statement
  /// [C++0x] 'for'
  ///             'co_await'[opt]    [Coroutines]
  ///             '(' for-range-declaration ':' for-range-initializer ')'
  ///             statement
  /// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
  /// [OBJC2] 'for' '(' expr 'in' expr ')' statement
  ///
  /// [C++] for-init-statement:
  /// [C++]   expression-statement
  /// [C++]   simple-declaration
  /// [C++23] alias-declaration
  ///
  /// [C++0x] for-range-declaration:
  /// [C++0x]   attribute-specifier-seq[opt] type-specifier-seq declarator
  /// [C++0x] for-range-initializer:
  /// [C++0x]   expression
  /// [C++0x]   braced-init-list            [TODO]
  /// \endverbatim
  StmtResult ParseForStatement(SourceLocation *TrailingElseLoc,
                               LabelDecl *PrecedingLabel);

  /// ParseGotoStatement
  /// \verbatim
  ///       jump-statement:
  ///         'goto' identifier ';'
  /// [GNU]   'goto' '*' expression ';'
  /// \endverbatim
  ///
  /// Note: this lets the caller parse the end ';'.
  ///
  StmtResult ParseGotoStatement();

  /// ParseContinueStatement
  /// \verbatim
  ///       jump-statement:
  ///         'continue' ';'
  /// [C2y]   'continue' identifier ';'
  /// \endverbatim
  ///
  /// Note: this lets the caller parse the end ';'.
  ///
  StmtResult ParseContinueStatement();

  /// ParseBreakStatement
  /// \verbatim
  ///       jump-statement:
  ///         'break' ';'
  /// [C2y]   'break' identifier ';'
  /// \endverbatim
  ///
  /// Note: this lets the caller parse the end ';'.
  ///
  StmtResult ParseBreakStatement();

  /// ParseReturnStatement
  /// \verbatim
  ///       jump-statement:
  ///         'return' expression[opt] ';'
  ///         'return' braced-init-list ';'
  ///         'co_return' expression[opt] ';'
  ///         'co_return' braced-init-list ';'
  /// \endverbatim
  StmtResult ParseReturnStatement();

  StmtResult ParseBreakOrContinueStatement(bool IsContinue);

  StmtResult ParsePragmaLoopHint(StmtVector &Stmts, ParsedStmtContext StmtCtx,
                                 SourceLocation *TrailingElseLoc,
                                 ParsedAttributes &Attrs,
                                 LabelDecl *PrecedingLabel);

  void ParseMicrosoftIfExistsStatement(StmtVector &Stmts);

  //===--------------------------------------------------------------------===//
  // C++ 6: Statements and Blocks

  /// ParseCXXTryBlock - Parse a C++ try-block.
  ///
  /// \verbatim
  ///       try-block:
  ///         'try' compound-statement handler-seq
  /// \endverbatim
  ///
  StmtResult ParseCXXTryBlock();

  /// ParseCXXTryBlockCommon - Parse the common part of try-block and
  /// function-try-block.
  ///
  /// \verbatim
  ///       try-block:
  ///         'try' compound-statement handler-seq
  ///
  ///       function-try-block:
  ///         'try' ctor-initializer[opt] compound-statement handler-seq
  ///
  ///       handler-seq:
  ///         handler handler-seq[opt]
  ///
  ///       [Borland] try-block:
  ///         'try' compound-statement seh-except-block
  ///         'try' compound-statement seh-finally-block
  /// \endverbatim
  ///
  StmtResult ParseCXXTryBlockCommon(SourceLocation TryLoc, bool FnTry = false);

  /// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the
  /// standard
  ///
  /// \verbatim
  ///   handler:
  ///     'catch' '(' exception-declaration ')' compound-statement
  ///
  ///   exception-declaration:
  ///     attribute-specifier-seq[opt] type-specifier-seq declarator
  ///     attribute-specifier-seq[opt] type-specifier-seq abstract-declarator[opt]
  ///     '...'
  /// \endverbatim
  ///
  StmtResult ParseCXXCatchBlock(bool FnCatch = false);

  //===--------------------------------------------------------------------===//
  // MS: SEH Statements and Blocks

  /// ParseSEHTryBlockCommon
  ///
  /// \verbatim
  /// seh-try-block:
  ///   '__try' compound-statement seh-handler
  ///
  /// seh-handler:
  ///   seh-except-block
  ///   seh-finally-block
  /// \endverbatim
  ///
  StmtResult ParseSEHTryBlock();

  /// ParseSEHExceptBlock - Handle __except
  ///
  /// \verbatim
  /// seh-except-block:
  ///   '__except' '(' seh-filter-expression ')' compound-statement
  /// \endverbatim
  ///
  StmtResult ParseSEHExceptBlock(SourceLocation Loc);

  /// ParseSEHFinallyBlock - Handle __finally
  ///
  /// \verbatim
  /// seh-finally-block:
  ///   '__finally' compound-statement
  /// \endverbatim
  ///
  StmtResult ParseSEHFinallyBlock(SourceLocation Loc);

  StmtResult ParseSEHLeaveStatement();

  Decl *ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope);

  /// ParseFunctionTryBlock - Parse a C++ function-try-block.
  ///
  /// \verbatim
  ///       function-try-block:
  ///         'try' ctor-initializer[opt] compound-statement handler-seq
  /// \endverbatim
  ///
  Decl *ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope);

  /// When in code-completion, skip parsing of the function/method body
  /// unless the body contains the code-completion point.
  ///
  /// \returns true if the function body was skipped.
  bool trySkippingFunctionBody();

  /// isDeclarationStatement - Disambiguates between a declaration or an
  /// expression statement, when parsing function bodies.
  ///
  /// \param DisambiguatingWithExpression - True to indicate that the purpose of
  /// this check is to disambiguate between an expression and a declaration.
  /// Returns true for declaration, false for expression.
  bool isDeclarationStatement(bool DisambiguatingWithExpression = false) {
    if (getLangOpts().CPlusPlus)
      return isCXXDeclarationStatement(DisambiguatingWithExpression);
    return isDeclarationSpecifier(ImplicitTypenameContext::No, true);
  }

  /// isForInitDeclaration - Disambiguates between a declaration or an
  /// expression in the context of the C 'clause-1' or the C++
  // 'for-init-statement' part of a 'for' statement.
  /// Returns true for declaration, false for expression.
  bool isForInitDeclaration() {
    if (getLangOpts().OpenMP)
      Actions.OpenMP().startOpenMPLoop();
    if (getLangOpts().CPlusPlus)
      return Tok.is(tok::kw_using) ||
             isCXXSimpleDeclaration(/*AllowForRangeDecl=*/true);
    return isDeclarationSpecifier(ImplicitTypenameContext::No, true);
  }

  /// Determine whether this is a C++1z for-range-identifier.
  bool isForRangeIdentifier();

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name `inline asm` Statement
  /// Implementations are in ParseStmtAsm.cpp
  ///@{

public:
  /// Parse an identifier in an MS-style inline assembly block.
  ExprResult ParseMSAsmIdentifier(llvm::SmallVectorImpl<Token> &LineToks,
                                  unsigned &NumLineToksConsumed,
                                  bool IsUnevaluated);

private:
  /// ParseAsmStatement - Parse a GNU extended asm statement.
  /// \verbatim
  ///       asm-statement:
  ///         gnu-asm-statement
  ///         ms-asm-statement
  ///
  /// [GNU] gnu-asm-statement:
  ///         'asm' asm-qualifier-list[opt] '(' asm-argument ')' ';'
  ///
  /// [GNU] asm-argument:
  ///         asm-string-literal
  ///         asm-string-literal ':' asm-operands[opt]
  ///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
  ///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
  ///                 ':' asm-clobbers
  ///
  /// [GNU] asm-clobbers:
  ///         asm-string-literal
  ///         asm-clobbers ',' asm-string-literal
  /// \endverbatim 
  ///
  StmtResult ParseAsmStatement(bool &msAsm);

  /// ParseMicrosoftAsmStatement. When -fms-extensions/-fasm-blocks is enabled,
  /// this routine is called to collect the tokens for an MS asm statement.
  ///
  /// \verbatim
  /// [MS]  ms-asm-statement:
  ///         ms-asm-block
  ///         ms-asm-block ms-asm-statement
  ///
  /// [MS]  ms-asm-block:
  ///         '__asm' ms-asm-line '\n'
  ///         '__asm' '{' ms-asm-instruction-block[opt] '}' ';'[opt]
  ///
  /// [MS]  ms-asm-instruction-block
  ///         ms-asm-line
  ///         ms-asm-line '\n' ms-asm-instruction-block
  /// \endverbatim
  ///
  StmtResult ParseMicrosoftAsmStatement(SourceLocation AsmLoc);

  /// ParseAsmOperands - Parse the asm-operands production as used by
  /// asm-statement, assuming the leading ':' token was eaten.
  ///
  /// \verbatim
  /// [GNU] asm-operands:
  ///         asm-operand
  ///         asm-operands ',' asm-operand
  ///
  /// [GNU] asm-operand:
  ///         asm-string-literal '(' expression ')'
  ///         '[' identifier ']' asm-string-literal '(' expression ')'
  /// \endverbatim
  ///
  // FIXME: Avoid unnecessary std::string trashing.
  bool ParseAsmOperandsOpt(SmallVectorImpl<IdentifierInfo *> &Names,
                           SmallVectorImpl<Expr *> &Constraints,
                           SmallVectorImpl<Expr *> &Exprs);

  class GNUAsmQualifiers {
    unsigned Qualifiers = AQ_unspecified;

  public:
    enum AQ {
      AQ_unspecified = 0,
      AQ_volatile = 1,
      AQ_inline = 2,
      AQ_goto = 4,
    };
    static const char *getQualifierName(AQ Qualifier);
    bool setAsmQualifier(AQ Qualifier);
    inline bool isVolatile() const { return Qualifiers & AQ_volatile; };
    inline bool isInline() const { return Qualifiers & AQ_inline; };
    inline bool isGoto() const { return Qualifiers & AQ_goto; }
  };

  // Determine if this is a GCC-style asm statement.
  bool isGCCAsmStatement(const Token &TokAfterAsm) const;

  bool isGNUAsmQualifier(const Token &TokAfterAsm) const;
  GNUAsmQualifiers::AQ getGNUAsmQualifier(const Token &Tok) const;

  /// parseGNUAsmQualifierListOpt - Parse a GNU extended asm qualifier list.
  /// \verbatim
  ///       asm-qualifier:
  ///         volatile
  ///         inline
  ///         goto
  ///
  ///       asm-qualifier-list:
  ///         asm-qualifier
  ///         asm-qualifier-list asm-qualifier
  /// \endverbatim
  bool parseGNUAsmQualifierListOpt(GNUAsmQualifiers &AQ);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name C++ Templates
  /// Implementations are in ParseTemplate.cpp
  ///@{

public:
  typedef SmallVector<TemplateParameterList *, 4> TemplateParameterLists;

  /// Re-enter a possible template scope, creating as many template parameter
  /// scopes as necessary.
  /// \return The number of template parameter scopes entered.
  unsigned ReenterTemplateScopes(MultiParseScope &S, Decl *D);

private:
  /// The "depth" of the template parameters currently being parsed.
  unsigned TemplateParameterDepth;

  /// RAII class that manages the template parameter depth.
  class TemplateParameterDepthRAII {
    unsigned &Depth;
    unsigned AddedLevels;

  public:
    explicit TemplateParameterDepthRAII(unsigned &Depth)
        : Depth(Depth), AddedLevels(0) {}

    ~TemplateParameterDepthRAII() { Depth -= AddedLevels; }

    void operator++() {
      ++Depth;
      ++AddedLevels;
    }
    void addDepth(unsigned D) {
      Depth += D;
      AddedLevels += D;
    }
    void setAddedDepth(unsigned D) {
      Depth = Depth - AddedLevels + D;
      AddedLevels = D;
    }

    unsigned getDepth() const { return Depth; }
    unsigned getOriginalDepth() const { return Depth - AddedLevels; }
  };

  /// Gathers and cleans up TemplateIdAnnotations when parsing of a
  /// top-level declaration is finished.
  SmallVector<TemplateIdAnnotation *, 16> TemplateIds;

  /// Don't destroy template annotations in MaybeDestroyTemplateIds even if
  /// we're at the end of a declaration. Instead, we defer the destruction until
  /// after a top-level declaration.
  /// Use DelayTemplateIdDestructionRAII rather than setting it directly.
  bool DelayTemplateIdDestruction = false;

  void MaybeDestroyTemplateIds() {
    if (DelayTemplateIdDestruction)
      return;
    if (!TemplateIds.empty() &&
        (Tok.is(tok::eof) || !PP.mightHavePendingAnnotationTokens()))
      DestroyTemplateIds();
  }
  void DestroyTemplateIds();

  /// RAII object to destroy TemplateIdAnnotations where possible, from a
  /// likely-good position during parsing.
  struct DestroyTemplateIdAnnotationsRAIIObj {
    Parser &Self;

    DestroyTemplateIdAnnotationsRAIIObj(Parser &Self) : Self(Self) {}
    ~DestroyTemplateIdAnnotationsRAIIObj() { Self.MaybeDestroyTemplateIds(); }
  };

  struct DelayTemplateIdDestructionRAII {
    Parser &Self;
    bool PrevDelayTemplateIdDestruction;

    DelayTemplateIdDestructionRAII(Parser &Self,
                                   bool DelayTemplateIdDestruction) noexcept
        : Self(Self),
          PrevDelayTemplateIdDestruction(Self.DelayTemplateIdDestruction) {
      Self.DelayTemplateIdDestruction = DelayTemplateIdDestruction;
    }

    ~DelayTemplateIdDestructionRAII() noexcept {
      Self.DelayTemplateIdDestruction = PrevDelayTemplateIdDestruction;
    }
  };

  /// Identifiers which have been declared within a tentative parse.
  SmallVector<const IdentifierInfo *, 8> TentativelyDeclaredIdentifiers;

  /// Tracker for '<' tokens that might have been intended to be treated as an
  /// angle bracket instead of a less-than comparison.
  ///
  /// This happens when the user intends to form a template-id, but typoes the
  /// template-name or forgets a 'template' keyword for a dependent template
  /// name.
  ///
  /// We track these locations from the point where we see a '<' with a
  /// name-like expression on its left until we see a '>' or '>>' that might
  /// match it.
  struct AngleBracketTracker {
    /// Flags used to rank candidate template names when there is more than one
    /// '<' in a scope.
    enum Priority : unsigned short {
      /// A non-dependent name that is a potential typo for a template name.
      PotentialTypo = 0x0,
      /// A dependent name that might instantiate to a template-name.
      DependentName = 0x2,

      /// A space appears before the '<' token.
      SpaceBeforeLess = 0x0,
      /// No space before the '<' token
      NoSpaceBeforeLess = 0x1,

      LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ DependentName)
    };

    struct Loc {
      Expr *TemplateName;
      SourceLocation LessLoc;
      AngleBracketTracker::Priority Priority;
      unsigned short ParenCount, BracketCount, BraceCount;

      bool isActive(Parser &P) const {
        return P.ParenCount == ParenCount && P.BracketCount == BracketCount &&
               P.BraceCount == BraceCount;
      }

      bool isActiveOrNested(Parser &P) const {
        return isActive(P) || P.ParenCount > ParenCount ||
               P.BracketCount > BracketCount || P.BraceCount > BraceCount;
      }
    };

    SmallVector<Loc, 8> Locs;

    /// Add an expression that might have been intended to be a template name.
    /// In the case of ambiguity, we arbitrarily select the innermost such
    /// expression, for example in 'foo < bar < baz', 'bar' is the current
    /// candidate. No attempt is made to track that 'foo' is also a candidate
    /// for the case where we see a second suspicious '>' token.
    void add(Parser &P, Expr *TemplateName, SourceLocation LessLoc,
             Priority Prio) {
      if (!Locs.empty() && Locs.back().isActive(P)) {
        if (Locs.back().Priority <= Prio) {
          Locs.back().TemplateName = TemplateName;
          Locs.back().LessLoc = LessLoc;
          Locs.back().Priority = Prio;
        }
      } else {
        Locs.push_back({TemplateName, LessLoc, Prio, P.ParenCount,
                        P.BracketCount, P.BraceCount});
      }
    }

    /// Mark the current potential missing template location as having been
    /// handled (this happens if we pass a "corresponding" '>' or '>>' token
    /// or leave a bracket scope).
    void clear(Parser &P) {
      while (!Locs.empty() && Locs.back().isActiveOrNested(P))
        Locs.pop_back();
    }

    /// Get the current enclosing expression that might hve been intended to be
    /// a template name.
    Loc *getCurrent(Parser &P) {
      if (!Locs.empty() && Locs.back().isActive(P))
        return &Locs.back();
      return nullptr;
    }
  };

  AngleBracketTracker AngleBrackets;

  /// Contains information about any template-specific
  /// information that has been parsed prior to parsing declaration
  /// specifiers.
  struct ParsedTemplateInfo {
    ParsedTemplateInfo()
        : Kind(ParsedTemplateKind::NonTemplate), TemplateParams(nullptr) {}

    ParsedTemplateInfo(TemplateParameterLists *TemplateParams,
                       bool isSpecialization,
                       bool lastParameterListWasEmpty = false)
        : Kind(isSpecialization ? ParsedTemplateKind::ExplicitSpecialization
                                : ParsedTemplateKind::Template),
          TemplateParams(TemplateParams),
          LastParameterListWasEmpty(lastParameterListWasEmpty) {}

    explicit ParsedTemplateInfo(SourceLocation ExternLoc,
                                SourceLocation TemplateLoc)
        : Kind(ParsedTemplateKind::ExplicitInstantiation),
          TemplateParams(nullptr), ExternLoc(ExternLoc),
          TemplateLoc(TemplateLoc), LastParameterListWasEmpty(false) {}

    ParsedTemplateKind Kind;

    /// The template parameter lists, for template declarations
    /// and explicit specializations.
    TemplateParameterLists *TemplateParams;

    /// The location of the 'extern' keyword, if any, for an explicit
    /// instantiation
    SourceLocation ExternLoc;

    /// The location of the 'template' keyword, for an explicit
    /// instantiation.
    SourceLocation TemplateLoc;

    /// Whether the last template parameter list was empty.
    bool LastParameterListWasEmpty;

    SourceRange getSourceRange() const LLVM_READONLY;
  };

  /// Lex a delayed template function for late parsing.
  void LexTemplateFunctionForLateParsing(CachedTokens &Toks);

  /// Late parse a C++ function template in Microsoft mode.
  void ParseLateTemplatedFuncDef(LateParsedTemplate &LPT);

  static void LateTemplateParserCallback(void *P, LateParsedTemplate &LPT);

  /// We've parsed something that could plausibly be intended to be a template
  /// name (\p LHS) followed by a '<' token, and the following code can't
  /// possibly be an expression. Determine if this is likely to be a template-id
  /// and if so, diagnose it.
  bool diagnoseUnknownTemplateId(ExprResult TemplateName, SourceLocation Less);

  void checkPotentialAngleBracket(ExprResult &PotentialTemplateName);
  bool checkPotentialAngleBracketDelimiter(const AngleBracketTracker::Loc &,
                                           const Token &OpToken);
  bool checkPotentialAngleBracketDelimiter(const Token &OpToken) {
    if (auto *Info = AngleBrackets.getCurrent(*this))
      return checkPotentialAngleBracketDelimiter(*Info, OpToken);
    return false;
  }

  //===--------------------------------------------------------------------===//
  // C++ 14: Templates [temp]

  /// Parse a template declaration, explicit instantiation, or
  /// explicit specialization.
  DeclGroupPtrTy
  ParseDeclarationStartingWithTemplate(DeclaratorContext Context,
                                       SourceLocation &DeclEnd,
                                       ParsedAttributes &AccessAttrs);

  /// Parse a template declaration or an explicit specialization.
  ///
  /// Template declarations include one or more template parameter lists
  /// and either the function or class template declaration. Explicit
  /// specializations contain one or more 'template < >' prefixes
  /// followed by a (possibly templated) declaration. Since the
  /// syntactic form of both features is nearly identical, we parse all
  /// of the template headers together and let semantic analysis sort
  /// the declarations from the explicit specializations.
  ///
  /// \verbatim
  ///       template-declaration: [C++ temp]
  ///         'export'[opt] 'template' '<' template-parameter-list '>' declaration
  ///
  ///       template-declaration: [C++2a]
  ///         template-head declaration
  ///         template-head concept-definition
  ///
  ///       TODO: requires-clause
  ///       template-head: [C++2a]
  ///         'template' '<' template-parameter-list '>'
  ///             requires-clause[opt]
  ///
  ///       explicit-specialization: [ C++ temp.expl.spec]
  ///         'template' '<' '>' declaration
  /// \endverbatim
  DeclGroupPtrTy ParseTemplateDeclarationOrSpecialization(
      DeclaratorContext Context, SourceLocation &DeclEnd,
      ParsedAttributes &AccessAttrs, AccessSpecifier AS);

  clang::Parser::DeclGroupPtrTy ParseTemplateDeclarationOrSpecialization(
      DeclaratorContext Context, SourceLocation &DeclEnd, AccessSpecifier AS);

  /// Parse a single declaration that declares a template,
  /// template specialization, or explicit instantiation of a template.
  ///
  /// \param DeclEnd will receive the source location of the last token
  /// within this declaration.
  ///
  /// \param AS the access specifier associated with this
  /// declaration. Will be AS_none for namespace-scope declarations.
  ///
  /// \returns the new declaration.
  DeclGroupPtrTy ParseDeclarationAfterTemplate(
      DeclaratorContext Context, ParsedTemplateInfo &TemplateInfo,
      ParsingDeclRAIIObject &DiagsFromParams, SourceLocation &DeclEnd,
      ParsedAttributes &AccessAttrs, AccessSpecifier AS = AS_none);

  /// ParseTemplateParameters - Parses a template-parameter-list enclosed in
  /// angle brackets. Depth is the depth of this template-parameter-list, which
  /// is the number of template headers directly enclosing this template header.
  /// TemplateParams is the current list of template parameters we're building.
  /// The template parameter we parse will be added to this list. LAngleLoc and
  /// RAngleLoc will receive the positions of the '<' and '>', respectively,
  /// that enclose this template parameter list.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool ParseTemplateParameters(MultiParseScope &TemplateScopes, unsigned Depth,
                               SmallVectorImpl<NamedDecl *> &TemplateParams,
                               SourceLocation &LAngleLoc,
                               SourceLocation &RAngleLoc);

  /// ParseTemplateParameterList - Parse a template parameter list. If
  /// the parsing fails badly (i.e., closing bracket was left out), this
  /// will try to put the token stream in a reasonable position (closing
  /// a statement, etc.) and return false.
  ///
  /// \verbatim
  ///       template-parameter-list:    [C++ temp]
  ///         template-parameter
  ///         template-parameter-list ',' template-parameter
  /// \endverbatim
  bool ParseTemplateParameterList(unsigned Depth,
                                  SmallVectorImpl<NamedDecl *> &TemplateParams);

  enum class TPResult;

  /// Determine whether the parser is at the start of a template
  /// type parameter.
  TPResult isStartOfTemplateTypeParameter();

  /// ParseTemplateParameter - Parse a template-parameter (C++ [temp.param]).
  ///
  /// \verbatim
  ///       template-parameter: [C++ temp.param]
  ///         type-parameter
  ///         parameter-declaration
  ///
  ///       type-parameter: (See below)
  ///         type-parameter-key ...[opt] identifier[opt]
  ///         type-parameter-key identifier[opt] = type-id
  /// (C++2a) type-constraint ...[opt] identifier[opt]
  /// (C++2a) type-constraint identifier[opt] = type-id
  ///         'template' '<' template-parameter-list '>' type-parameter-key
  ///               ...[opt] identifier[opt]
  ///         'template' '<' template-parameter-list '>' type-parameter-key
  ///               identifier[opt] '=' id-expression
  ///
  ///       type-parameter-key:
  ///         class
  ///         typename
  /// \endverbatim
  ///
  NamedDecl *ParseTemplateParameter(unsigned Depth, unsigned Position);

  /// ParseTypeParameter - Parse a template type parameter (C++ [temp.param]).
  /// Other kinds of template parameters are parsed in
  /// ParseTemplateTemplateParameter and ParseNonTypeTemplateParameter.
  ///
  /// \verbatim
  ///       type-parameter:     [C++ temp.param]
  ///         'class' ...[opt][C++0x] identifier[opt]
  ///         'class' identifier[opt] '=' type-id
  ///         'typename' ...[opt][C++0x] identifier[opt]
  ///         'typename' identifier[opt] '=' type-id
  /// \endverbatim
  NamedDecl *ParseTypeParameter(unsigned Depth, unsigned Position);

  /// ParseTemplateTemplateParameter - Handle the parsing of template
  /// template parameters.
  ///
  /// \verbatim
  ///       type-parameter:    [C++ temp.param]
  ///         template-head type-parameter-key ...[opt] identifier[opt]
  ///         template-head type-parameter-key identifier[opt] = id-expression
  ///       type-parameter-key:
  ///         'class'
  ///         'typename'       [C++1z]
  ///       template-head:     [C++2a]
  ///         'template' '<' template-parameter-list '>'
  ///             requires-clause[opt]
  /// \endverbatim
  NamedDecl *ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);

  /// ParseNonTypeTemplateParameter - Handle the parsing of non-type
  /// template parameters (e.g., in "template<int Size> class array;").
  ///
  /// \verbatim
  ///       template-parameter:
  ///         ...
  ///         parameter-declaration
  /// \endverbatim
  NamedDecl *ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);

  /// Check whether the current token is a template-id annotation denoting a
  /// type-constraint.
  bool isTypeConstraintAnnotation();

  /// Try parsing a type-constraint at the current location.
  ///
  /// \verbatim
  ///     type-constraint:
  ///       nested-name-specifier[opt] concept-name
  ///       nested-name-specifier[opt] concept-name
  ///           '<' template-argument-list[opt] '>'[opt]
  /// \endverbatim
  ///
  /// \returns true if an error occurred, and false otherwise.
  bool TryAnnotateTypeConstraint();

  void DiagnoseMisplacedEllipsis(SourceLocation EllipsisLoc,
                                 SourceLocation CorrectLoc,
                                 bool AlreadyHasEllipsis,
                                 bool IdentifierHasName);
  void DiagnoseMisplacedEllipsisInDeclarator(SourceLocation EllipsisLoc,
                                             Declarator &D);
  // C++ 14.3: Template arguments [temp.arg]
  typedef SmallVector<ParsedTemplateArgument, 16> TemplateArgList;

  /// Parses a '>' at the end of a template list.
  ///
  /// If this function encounters '>>', '>>>', '>=', or '>>=', it tries
  /// to determine if these tokens were supposed to be a '>' followed by
  /// '>', '>>', '>=', or '>='. It emits an appropriate diagnostic if necessary.
  ///
  /// \param RAngleLoc the location of the consumed '>'.
  ///
  /// \param ConsumeLastToken if true, the '>' is consumed.
  ///
  /// \param ObjCGenericList if true, this is the '>' closing an Objective-C
  /// type parameter or type argument list, rather than a C++ template parameter
  /// or argument list.
  ///
  /// \returns true, if current token does not start with '>', false otherwise.
  bool ParseGreaterThanInTemplateList(SourceLocation LAngleLoc,
                                      SourceLocation &RAngleLoc,
                                      bool ConsumeLastToken,
                                      bool ObjCGenericList);

  /// Parses a template-id that after the template name has
  /// already been parsed.
  ///
  /// This routine takes care of parsing the enclosed template argument
  /// list ('<' template-parameter-list [opt] '>') and placing the
  /// results into a form that can be transferred to semantic analysis.
  ///
  /// \param ConsumeLastToken if true, then we will consume the last
  /// token that forms the template-id. Otherwise, we will leave the
  /// last token in the stream (e.g., so that it can be replaced with an
  /// annotation token).
  bool ParseTemplateIdAfterTemplateName(bool ConsumeLastToken,
                                        SourceLocation &LAngleLoc,
                                        TemplateArgList &TemplateArgs,
                                        SourceLocation &RAngleLoc,
                                        TemplateTy NameHint = nullptr);

  /// Replace the tokens that form a simple-template-id with an
  /// annotation token containing the complete template-id.
  ///
  /// The first token in the stream must be the name of a template that
  /// is followed by a '<'. This routine will parse the complete
  /// simple-template-id and replace the tokens with a single annotation
  /// token with one of two different kinds: if the template-id names a
  /// type (and \p AllowTypeAnnotation is true), the annotation token is
  /// a type annotation that includes the optional nested-name-specifier
  /// (\p SS). Otherwise, the annotation token is a template-id
  /// annotation that does not include the optional
  /// nested-name-specifier.
  ///
  /// \param Template  the declaration of the template named by the first
  /// token (an identifier), as returned from \c Action::isTemplateName().
  ///
  /// \param TNK the kind of template that \p Template
  /// refers to, as returned from \c Action::isTemplateName().
  ///
  /// \param SS if non-NULL, the nested-name-specifier that precedes
  /// this template name.
  ///
  /// \param TemplateKWLoc if valid, specifies that this template-id
  /// annotation was preceded by the 'template' keyword and gives the
  /// location of that keyword. If invalid (the default), then this
  /// template-id was not preceded by a 'template' keyword.
  ///
  /// \param AllowTypeAnnotation if true (the default), then a
  /// simple-template-id that refers to a class template, template
  /// template parameter, or other template that produces a type will be
  /// replaced with a type annotation token. Otherwise, the
  /// simple-template-id is always replaced with a template-id
  /// annotation token.
  ///
  /// \param TypeConstraint if true, then this is actually a type-constraint,
  /// meaning that the template argument list can be omitted (and the template
  /// in question must be a concept).
  ///
  /// If an unrecoverable parse error occurs and no annotation token can be
  /// formed, this function returns true.
  ///
  bool AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
                               CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
                               UnqualifiedId &TemplateName,
                               bool AllowTypeAnnotation = true,
                               bool TypeConstraint = false);

  /// Replaces a template-id annotation token with a type
  /// annotation token.
  ///
  /// If there was a failure when forming the type from the template-id,
  /// a type annotation token will still be created, but will have a
  /// NULL type pointer to signify an error.
  ///
  /// \param SS The scope specifier appearing before the template-id, if any.
  ///
  /// \param AllowImplicitTypename whether this is a context where T::type
  /// denotes a dependent type.
  /// \param IsClassName Is this template-id appearing in a context where we
  /// know it names a class, such as in an elaborated-type-specifier or
  /// base-specifier? ('typename' and 'template' are unneeded and disallowed
  /// in those contexts.)
  void
  AnnotateTemplateIdTokenAsType(CXXScopeSpec &SS,
                                ImplicitTypenameContext AllowImplicitTypename,
                                bool IsClassName = false);

  /// ParseTemplateArgumentList - Parse a C++ template-argument-list
  /// (C++ [temp.names]). Returns true if there was an error.
  ///
  /// \verbatim
  ///       template-argument-list: [C++ 14.2]
  ///         template-argument
  ///         template-argument-list ',' template-argument
  /// \endverbatim
  ///
  /// \param Template is only used for code completion, and may be null.
  bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs,
                                 TemplateTy Template, SourceLocation OpenLoc);

  /// Parse a C++ template template argument.
  ParsedTemplateArgument ParseTemplateTemplateArgument();

  /// ParseTemplateArgument - Parse a C++ template argument (C++ [temp.names]).
  ///
  /// \verbatim
  ///       template-argument: [C++ 14.2]
  ///         constant-expression
  ///         type-id
  ///         id-expression
  ///         braced-init-list  [C++26, DR]
  /// \endverbatim
  ///
  ParsedTemplateArgument ParseTemplateArgument();

  /// Parse a C++ explicit template instantiation
  /// (C++ [temp.explicit]).
  ///
  /// \verbatim
  ///       explicit-instantiation:
  ///         'extern' [opt] 'template' declaration
  /// \endverbatim
  ///
  /// Note that the 'extern' is a GNU extension and C++11 feature.
  DeclGroupPtrTy ParseExplicitInstantiation(DeclaratorContext Context,
                                            SourceLocation ExternLoc,
                                            SourceLocation TemplateLoc,
                                            SourceLocation &DeclEnd,
                                            ParsedAttributes &AccessAttrs,
                                            AccessSpecifier AS = AS_none);

  /// \brief Parse a single declaration that declares a concept.
  ///
  /// \param DeclEnd will receive the source location of the last token
  /// within this declaration.
  ///
  /// \returns the new declaration.
  Decl *ParseConceptDefinition(const ParsedTemplateInfo &TemplateInfo,
                               SourceLocation &DeclEnd);

  ///@}

  //
  //
  // -------------------------------------------------------------------------
  //
  //

  /// \name Tentative Parsing
  /// Implementations are in ParseTentative.cpp
  ///@{

private:
  /// TentativeParsingAction - An object that is used as a kind of "tentative
  /// parsing transaction". It gets instantiated to mark the token position and
  /// after the token consumption is done, Commit() or Revert() is called to
  /// either "commit the consumed tokens" or revert to the previously marked
  /// token position. Example:
  ///
  ///   TentativeParsingAction TPA(*this);
  ///   ConsumeToken();
  ///   ....
  ///   TPA.Revert();
  ///
  /// If the Unannotated parameter is true, any token annotations created
  /// during the tentative parse are reverted.
  class TentativeParsingAction {
    Parser &P;
    PreferredTypeBuilder PrevPreferredType;
    Token PrevTok;
    size_t PrevTentativelyDeclaredIdentifierCount;
    unsigned short PrevParenCount, PrevBracketCount, PrevBraceCount;
    bool isActive;

  public:
    explicit TentativeParsingAction(Parser &p, bool Unannotated = false)
        : P(p), PrevPreferredType(P.PreferredType) {
      PrevTok = P.Tok;
      PrevTentativelyDeclaredIdentifierCount =
          P.TentativelyDeclaredIdentifiers.size();
      PrevParenCount = P.ParenCount;
      PrevBracketCount = P.BracketCount;
      PrevBraceCount = P.BraceCount;
      P.PP.EnableBacktrackAtThisPos(Unannotated);
      isActive = true;
    }
    void Commit() {
      assert(isActive && "Parsing action was finished!");
      P.TentativelyDeclaredIdentifiers.resize(
          PrevTentativelyDeclaredIdentifierCount);
      P.PP.CommitBacktrackedTokens();
      isActive = false;
    }
    void Revert() {
      assert(isActive && "Parsing action was finished!");
      P.PP.Backtrack();
      P.PreferredType = PrevPreferredType;
      P.Tok = PrevTok;
      P.TentativelyDeclaredIdentifiers.resize(
          PrevTentativelyDeclaredIdentifierCount);
      P.ParenCount = PrevParenCount;
      P.BracketCount = PrevBracketCount;
      P.BraceCount = PrevBraceCount;
      isActive = false;
    }
    ~TentativeParsingAction() {
      assert(!isActive && "Forgot to call Commit or Revert!");
    }
  };

  /// A TentativeParsingAction that automatically reverts in its destructor.
  /// Useful for disambiguation parses that will always be reverted.
  class RevertingTentativeParsingAction
      : private Parser::TentativeParsingAction {
  public:
    using TentativeParsingAction::TentativeParsingAction;

    ~RevertingTentativeParsingAction() { Revert(); }
  };

  /// isCXXDeclarationStatement - C++-specialized function that disambiguates
  /// between a declaration or an expression statement, when parsing function
  /// bodies. Returns true for declaration, false for expression.
  ///
  /// \verbatim
  ///         declaration-statement:
  ///           block-declaration
  ///
  ///         block-declaration:
  ///           simple-declaration
  ///           asm-definition
  ///           namespace-alias-definition
  ///           using-declaration
  ///           using-directive
  /// [C++0x]   static_assert-declaration
  ///
  ///         asm-definition:
  ///           'asm' '(' string-literal ')' ';'
  ///
  ///         namespace-alias-definition:
  ///           'namespace' identifier = qualified-namespace-specifier ';'
  ///
  ///         using-declaration:
  ///           'using' typename[opt] '::'[opt] nested-name-specifier
  ///                 unqualified-id ';'
  ///           'using' '::' unqualified-id ;
  ///
  ///         using-directive:
  ///           'using' 'namespace' '::'[opt] nested-name-specifier[opt]
  ///                 namespace-name ';'
  /// \endverbatim
  ///
  bool isCXXDeclarationStatement(bool DisambiguatingWithExpression = false);

  /// isCXXSimpleDeclaration - C++-specialized function that disambiguates
  /// between a simple-declaration or an expression-statement.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  /// Returns false if the statement is disambiguated as expression.
  ///
  /// \verbatim
  /// simple-declaration:
  ///   decl-specifier-seq init-declarator-list[opt] ';'
  ///   decl-specifier-seq ref-qualifier[opt] '[' identifier-list ']'
  ///                      brace-or-equal-initializer ';'    [C++17]
  /// \endverbatim
  ///
  /// (if AllowForRangeDecl specified)
  /// for ( for-range-declaration : for-range-initializer ) statement
  ///
  /// \verbatim
  /// for-range-declaration:
  ///    decl-specifier-seq declarator
  ///    decl-specifier-seq ref-qualifier[opt] '[' identifier-list ']'
  /// \endverbatim
  ///
  /// In any of the above cases there can be a preceding
  /// attribute-specifier-seq, but the caller is expected to handle that.
  bool isCXXSimpleDeclaration(bool AllowForRangeDecl);

  /// isCXXFunctionDeclarator - Disambiguates between a function declarator or
  /// a constructor-style initializer, when parsing declaration statements.
  /// Returns true for function declarator and false for constructor-style
  /// initializer. Sets 'IsAmbiguous' to true to indicate that this declaration
  /// might be a constructor-style initializer.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  ///
  /// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
  ///         exception-specification[opt]
  ///
  bool isCXXFunctionDeclarator(bool *IsAmbiguous = nullptr,
                               ImplicitTypenameContext AllowImplicitTypename =
                                   ImplicitTypenameContext::No);

  struct ConditionDeclarationOrInitStatementState;
  enum class ConditionOrInitStatement {
    Expression,    ///< Disambiguated as an expression (either kind).
    ConditionDecl, ///< Disambiguated as the declaration form of condition.
    InitStmtDecl,  ///< Disambiguated as a simple-declaration init-statement.
    ForRangeDecl,  ///< Disambiguated as a for-range declaration.
    Error          ///< Can't be any of the above!
  };

  /// Disambiguates between a declaration in a condition, a
  /// simple-declaration in an init-statement, and an expression for
  /// a condition of a if/switch statement.
  ///
  /// \verbatim
  ///       condition:
  ///         expression
  ///         type-specifier-seq declarator '=' assignment-expression
  /// [C++11] type-specifier-seq declarator '=' initializer-clause
  /// [C++11] type-specifier-seq declarator braced-init-list
  /// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
  ///             '=' assignment-expression
  ///       simple-declaration:
  ///         decl-specifier-seq init-declarator-list[opt] ';'
  /// \endverbatim
  ///
  /// Note that, unlike isCXXSimpleDeclaration, we must disambiguate all the way
  /// to the ';' to disambiguate cases like 'int(x))' (an expression) from
  /// 'int(x);' (a simple-declaration in an init-statement).
  ConditionOrInitStatement
  isCXXConditionDeclarationOrInitStatement(bool CanBeInitStmt,
                                           bool CanBeForRangeDecl);

  /// Determine whether the next set of tokens contains a type-id.
  ///
  /// The context parameter states what context we're parsing right
  /// now, which affects how this routine copes with the token
  /// following the type-id. If the context is
  /// TentativeCXXTypeIdContext::InParens, we have already parsed the '(' and we
  /// will cease lookahead when we hit the corresponding ')'. If the context is
  /// TentativeCXXTypeIdContext::AsTemplateArgument, we've already parsed the
  /// '<' or ',' before this template argument, and will cease lookahead when we
  /// hit a
  /// '>', '>>' (in C++0x), or ','; or, in C++0x, an ellipsis immediately
  /// preceding such. Returns true for a type-id and false for an expression.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  ///
  /// \verbatim
  /// type-id:
  ///   type-specifier-seq abstract-declarator[opt]
  /// \endverbatim
  ///
  bool isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous);

  bool isCXXTypeId(TentativeCXXTypeIdContext Context) {
    bool isAmbiguous;
    return isCXXTypeId(Context, isAmbiguous);
  }

  /// TPResult - Used as the result value for functions whose purpose is to
  /// disambiguate C++ constructs by "tentatively parsing" them.
  enum class TPResult { True, False, Ambiguous, Error };

  /// Determine whether we could have an enum-base.
  ///
  /// \p AllowSemi If \c true, then allow a ';' after the enum-base; otherwise
  /// only consider this to be an enum-base if the next token is a '{'.
  ///
  /// \return \c false if this cannot possibly be an enum base; \c true
  /// otherwise.
  bool isEnumBase(bool AllowSemi);

  /// isCXXDeclarationSpecifier - Returns TPResult::True if it is a declaration
  /// specifier, TPResult::False if it is not, TPResult::Ambiguous if it could
  /// be either a decl-specifier or a function-style cast, and TPResult::Error
  /// if a parsing error was found and reported.
  ///
  /// Does not consume tokens.
  ///
  /// If InvalidAsDeclSpec is not null, some cases that would be ill-formed as
  /// declaration specifiers but possibly valid as some other kind of construct
  /// return TPResult::Ambiguous instead of TPResult::False. When this happens,
  /// the intent is to keep trying to disambiguate, on the basis that we might
  /// find a better reason to treat this construct as a declaration later on.
  /// When this happens and the name could possibly be valid in some other
  /// syntactic context, *InvalidAsDeclSpec is set to 'true'. The current cases
  /// that trigger this are:
  ///
  ///   * When parsing X::Y (with no 'typename') where X is dependent
  ///   * When parsing X<Y> where X is undeclared
  ///
  /// \verbatim
  ///         decl-specifier:
  ///           storage-class-specifier
  ///           type-specifier
  ///           function-specifier
  ///           'friend'
  ///           'typedef'
  /// [C++11]   'constexpr'
  /// [C++20]   'consteval'
  /// [GNU]     attributes declaration-specifiers[opt]
  ///
  ///         storage-class-specifier:
  ///           'register'
  ///           'static'
  ///           'extern'
  ///           'mutable'
  ///           'auto'
  /// [GNU]     '__thread'
  /// [C++11]   'thread_local'
  /// [C11]     '_Thread_local'
  ///
  ///         function-specifier:
  ///           'inline'
  ///           'virtual'
  ///           'explicit'
  ///
  ///         typedef-name:
  ///           identifier
  ///
  ///         type-specifier:
  ///           simple-type-specifier
  ///           class-specifier
  ///           enum-specifier
  ///           elaborated-type-specifier
  ///           typename-specifier
  ///           cv-qualifier
  ///
  ///         simple-type-specifier:
  ///           '::'[opt] nested-name-specifier[opt] type-name
  ///           '::'[opt] nested-name-specifier 'template'
  ///                 simple-template-id                              [TODO]
  ///           'char'
  ///           'wchar_t'
  ///           'bool'
  ///           'short'
  ///           'int'
  ///           'long'
  ///           'signed'
  ///           'unsigned'
  ///           'float'
  ///           'double'
  ///           'void'
  /// [GNU]     typeof-specifier
  /// [GNU]     '_Complex'
  /// [C++11]   'auto'
  /// [GNU]     '__auto_type'
  /// [C++11]   'decltype' ( expression )
  /// [C++1y]   'decltype' ( 'auto' )
  ///
  ///         type-name:
  ///           class-name
  ///           enum-name
  ///           typedef-name
  ///
  ///         elaborated-type-specifier:
  ///           class-key '::'[opt] nested-name-specifier[opt] identifier
  ///           class-key '::'[opt] nested-name-specifier[opt] 'template'[opt]
  ///               simple-template-id
  ///           'enum' '::'[opt] nested-name-specifier[opt] identifier
  ///
  ///         enum-name:
  ///           identifier
  ///
  ///         enum-specifier:
  ///           'enum' identifier[opt] '{' enumerator-list[opt] '}'
  ///           'enum' identifier[opt] '{' enumerator-list ',' '}'
  ///
  ///         class-specifier:
  ///           class-head '{' member-specification[opt] '}'
  ///
  ///         class-head:
  ///           class-key identifier[opt] base-clause[opt]
  ///           class-key nested-name-specifier identifier base-clause[opt]
  ///           class-key nested-name-specifier[opt] simple-template-id
  ///               base-clause[opt]
  ///
  ///         class-key:
  ///           'class'
  ///           'struct'
  ///           'union'
  ///
  ///         cv-qualifier:
  ///           'const'
  ///           'volatile'
  /// [GNU]     restrict
  /// \endverbatim
  ///
  TPResult
  isCXXDeclarationSpecifier(ImplicitTypenameContext AllowImplicitTypename,
                            TPResult BracedCastResult = TPResult::False,
                            bool *InvalidAsDeclSpec = nullptr);

  /// Given that isCXXDeclarationSpecifier returns \c TPResult::True or
  /// \c TPResult::Ambiguous, determine whether the decl-specifier would be
  /// a type-specifier other than a cv-qualifier.
  bool isCXXDeclarationSpecifierAType();

  /// Determine whether we might be looking at the '<' template-argument-list
  /// '>' of a template-id or simple-template-id, rather than a less-than
  /// comparison. This will often fail and produce an ambiguity, but should
  /// never be wrong if it returns True or False.
  TPResult isTemplateArgumentList(unsigned TokensToSkip);

  /// Determine whether an '(' after an 'explicit' keyword is part of a C++20
  /// 'explicit(bool)' declaration, in earlier language modes where that is an
  /// extension.
  TPResult isExplicitBool();

  /// Determine whether an identifier has been tentatively declared as a
  /// non-type. Such tentative declarations should not be found to name a type
  /// during a tentative parse, but also should not be annotated as a non-type.
  bool isTentativelyDeclared(IdentifierInfo *II);

  // "Tentative parsing" functions, used for disambiguation. If a parsing error
  // is encountered they will return TPResult::Error.
  // Returning TPResult::True/False indicates that the ambiguity was
  // resolved and tentative parsing may stop. TPResult::Ambiguous indicates
  // that more tentative parsing is necessary for disambiguation.
  // They all consume tokens, so backtracking should be used after calling them.

  /// \verbatim
  /// simple-declaration:
  ///   decl-specifier-seq init-declarator-list[opt] ';'
  ///
  /// (if AllowForRangeDecl specified)
  /// for ( for-range-declaration : for-range-initializer ) statement
  /// for-range-declaration:
  ///    attribute-specifier-seqopt type-specifier-seq declarator
  /// \endverbatim
  ///
  TPResult TryParseSimpleDeclaration(bool AllowForRangeDecl);

  /// \verbatim
  /// [GNU] typeof-specifier:
  ///         'typeof' '(' expressions ')'
  ///         'typeof' '(' type-name ')'
  /// \endverbatim
  ///
  TPResult TryParseTypeofSpecifier();

  /// [ObjC] protocol-qualifiers:
  ///         '<' identifier-list '>'
  TPResult TryParseProtocolQualifiers();

  TPResult TryParsePtrOperatorSeq();

  /// \verbatim
  ///         operator-function-id:
  ///           'operator' operator
  ///
  ///         operator: one of
  ///           new  delete  new[]  delete[]  +  -  *  /  %  ^  [...]
  ///
  ///         conversion-function-id:
  ///           'operator' conversion-type-id
  ///
  ///         conversion-type-id:
  ///           type-specifier-seq conversion-declarator[opt]
  ///
  ///         conversion-declarator:
  ///           ptr-operator conversion-declarator[opt]
  ///
  ///         literal-operator-id:
  ///           'operator' string-literal identifier
  ///           'operator' user-defined-string-literal
  /// \endverbatim
  TPResult TryParseOperatorId();

  /// Tentatively parse an init-declarator-list in order to disambiguate it from
  /// an expression.
  ///
  /// \verbatim
  ///       init-declarator-list:
  ///         init-declarator
  ///         init-declarator-list ',' init-declarator
  ///
  ///       init-declarator:
  ///         declarator initializer[opt]
  /// [GNU]   declarator simple-asm-expr[opt] attributes[opt] initializer[opt]
  ///
  ///       initializer:
  ///         brace-or-equal-initializer
  ///         '(' expression-list ')'
  ///
  ///       brace-or-equal-initializer:
  ///         '=' initializer-clause
  /// [C++11] braced-init-list
  ///
  ///       initializer-clause:
  ///         assignment-expression
  ///         braced-init-list
  ///
  ///       braced-init-list:
  ///         '{' initializer-list ','[opt] '}'
  ///         '{' '}'
  /// \endverbatim
  ///
  TPResult TryParseInitDeclaratorList(bool MayHaveTrailingReturnType = false);

  /// \verbatim
  ///         declarator:
  ///           direct-declarator
  ///           ptr-operator declarator
  ///
  ///         direct-declarator:
  ///           declarator-id
  ///           direct-declarator '(' parameter-declaration-clause ')'
  ///                 cv-qualifier-seq[opt] exception-specification[opt]
  ///           direct-declarator '[' constant-expression[opt] ']'
  ///           '(' declarator ')'
  /// [GNU]     '(' attributes declarator ')'
  ///
  ///         abstract-declarator:
  ///           ptr-operator abstract-declarator[opt]
  ///           direct-abstract-declarator
  ///
  ///         direct-abstract-declarator:
  ///           direct-abstract-declarator[opt]
  ///                 '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
  ///                 exception-specification[opt]
  ///           direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
  ///           '(' abstract-declarator ')'
  /// [C++0x]   ...
  ///
  ///         ptr-operator:
  ///           '*' cv-qualifier-seq[opt]
  ///           '&'
  /// [C++0x]   '&&'                                                        [TODO]
  ///           '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt]
  ///
  ///         cv-qualifier-seq:
  ///           cv-qualifier cv-qualifier-seq[opt]
  ///
  ///         cv-qualifier:
  ///           'const'
  ///           'volatile'
  ///
  ///         declarator-id:
  ///           '...'[opt] id-expression
  ///
  ///         id-expression:
  ///           unqualified-id
  ///           qualified-id                                                [TODO]
  ///
  ///         unqualified-id:
  ///           identifier
  ///           operator-function-id
  ///           conversion-function-id
  ///           literal-operator-id
  ///           '~' class-name                                              [TODO]
  ///           '~' decltype-specifier                                      [TODO]
  ///           template-id                                                 [TODO]
  /// \endverbatim
  ///
  TPResult TryParseDeclarator(bool mayBeAbstract, bool mayHaveIdentifier = true,
                              bool mayHaveDirectInit = false,
                              bool mayHaveTrailingReturnType = false);

  /// \verbatim
  /// parameter-declaration-clause:
  ///   parameter-declaration-list[opt] '...'[opt]
  ///   parameter-declaration-list ',' '...'
  ///
  /// parameter-declaration-list:
  ///   parameter-declaration
  ///   parameter-declaration-list ',' parameter-declaration
  ///
  /// parameter-declaration:
  ///   attribute-specifier-seq[opt] decl-specifier-seq declarator attributes[opt]
  ///   attribute-specifier-seq[opt] decl-specifier-seq declarator attributes[opt]
  ///     '=' assignment-expression
  ///   attribute-specifier-seq[opt] decl-specifier-seq abstract-declarator[opt]
  ///     attributes[opt]
  ///   attribute-specifier-seq[opt] decl-specifier-seq abstract-declarator[opt]
  ///     attributes[opt] '=' assignment-expression
  /// \endverbatim
  ///
  TPResult TryParseParameterDeclarationClause(
      bool *InvalidAsDeclaration = nullptr, bool VersusTemplateArg = false,
      ImplicitTypenameContext AllowImplicitTypename =
          ImplicitTypenameContext::No);

  /// TryParseFunctionDeclarator - We parsed a '(' and we want to try to
  /// continue parsing as a function declarator. If TryParseFunctionDeclarator
  /// fully parsed the function declarator, it will return TPResult::Ambiguous,
  /// otherwise it will return either False() or Error().
  ///
  /// \verbatim
  /// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
  ///         exception-specification[opt]
  ///
  /// exception-specification:
  ///   'throw' '(' type-id-list[opt] ')'
  /// \endverbatim
  ///
  TPResult TryParseFunctionDeclarator(bool MayHaveTrailingReturnType = false);

  // When parsing an identifier after an arrow it may be a member expression,
  // in which case we should not annotate it as an independant expression
  // so we just lookup that name, if it's not a type the construct is not
  // a function declaration.
  bool NameAfterArrowIsNonType();

  /// \verbatim
  /// '[' constant-expression[opt] ']'
  /// \endverbatim
  ///
  TPResult TryParseBracketDeclarator();

  /// Try to consume a token sequence that we've already identified as
  /// (potentially) starting a decl-specifier.
  TPResult TryConsumeDeclarationSpecifier();

  /// Try to skip a possibly empty sequence of 'attribute-specifier's without
  /// full validation of the syntactic structure of attributes.
  bool TrySkipAttributes();

  //===--------------------------------------------------------------------===//
  // C++ 7: Declarations [dcl.dcl]

  /// Returns true if this is a C++11 attribute-specifier. Per
  /// C++11 [dcl.attr.grammar]p6, two consecutive left square bracket tokens
  /// always introduce an attribute. In Objective-C++11, this rule does not
  /// apply if either '[' begins a message-send.
  ///
  /// If Disambiguate is true, we try harder to determine whether a '[[' starts
  /// an attribute-specifier, and return
  /// CXX11AttributeKind::InvalidAttributeSpecifier if not.
  ///
  /// If OuterMightBeMessageSend is true, we assume the outer '[' is either an
  /// Obj-C message send or the start of an attribute. Otherwise, we assume it
  /// is not an Obj-C message send.
  ///
  /// C++11 [dcl.attr.grammar]:
  ///
  /// \verbatim
  ///     attribute-specifier:
  ///         '[' '[' attribute-list ']' ']'
  ///         alignment-specifier
  ///
  ///     attribute-list:
  ///         attribute[opt]
  ///         attribute-list ',' attribute[opt]
  ///         attribute '...'
  ///         attribute-list ',' attribute '...'
  ///
  ///     attribute:
  ///         attribute-token attribute-argument-clause[opt]
  ///
  ///     attribute-token:
  ///         identifier
  ///         identifier '::' identifier
  ///
  ///     attribute-argument-clause:
  ///         '(' balanced-token-seq ')'
  /// \endverbatim
  CXX11AttributeKind
  isCXX11AttributeSpecifier(bool Disambiguate = false,
                            bool OuterMightBeMessageSend = false);

  ///@}
};

} // end namespace clang

#endif
