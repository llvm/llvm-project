//===--- Parser.h - C Language Parser ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Parser interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSER_H
#define LLVM_CLANG_PARSE_PARSER_H

#include "clang/Basic/Specifiers.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/CodeCompletionHandler.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/DeclSpec.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SaveAndRestore.h"
#include <stack>

namespace clang {
  class PragmaHandler;
  class Scope;
  class BalancedDelimiterTracker;
  class CorrectionCandidateCallback;
  class DeclGroupRef;
  class DiagnosticBuilder;
  class Parser;
  class ParsingDeclRAIIObject;
  class ParsingDeclSpec;
  class ParsingDeclarator;
  class ParsingFieldDeclarator;
  class PragmaUnusedHandler;
  class ColonProtectionRAIIObject;
  class InMessageExpressionRAIIObject;
  class PoisonSEHIdentifiersRAIIObject;
  class VersionTuple;

/// PrettyStackTraceParserEntry - If a crash happens while the parser is active,
/// an entry is printed for it.
class PrettyStackTraceParserEntry : public llvm::PrettyStackTraceEntry {
  const Parser &P;
public:
  PrettyStackTraceParserEntry(const Parser &p) : P(p) {}
  virtual void print(raw_ostream &OS) const;
};

/// PrecedenceLevels - These are precedences for the binary/ternary
/// operators in the C99 grammar.  These have been named to relate
/// with the C99 grammar productions.  Low precedences numbers bind
/// more weakly than high numbers.
namespace prec {
  enum Level {
    Unknown         = 0,    // Not binary operator.
    Comma           = 1,    // ,
    Assignment      = 2,    // =, *=, /=, %=, +=, -=, <<=, >>=, &=, ^=, |=
    Conditional     = 3,    // ?
    LogicalOr       = 4,    // ||
    LogicalAnd      = 5,    // &&
    InclusiveOr     = 6,    // |
    ExclusiveOr     = 7,    // ^
    And             = 8,    // &
    Equality        = 9,    // ==, !=
    Relational      = 10,   //  >=, <=, >, <
    Shift           = 11,   // <<, >>
    Additive        = 12,   // -, +
    Multiplicative  = 13,   // *, /, %
    PointerToMember = 14    // .*, ->*
  };
}

/// Parser - This implements a parser for the C family of languages.  After
/// parsing units of the grammar, productions are invoked to handle whatever has
/// been read.
///
class Parser : public CodeCompletionHandler {
  friend class PragmaUnusedHandler;
  friend class ColonProtectionRAIIObject;
  friend class InMessageExpressionRAIIObject;
  friend class PoisonSEHIdentifiersRAIIObject;
  friend class ObjCDeclContextSwitch;
  friend class ParenBraceBracketBalancer;
  friend class BalancedDelimiterTracker;

  Preprocessor &PP;

  /// Tok - The current token we are peeking ahead.  All parsing methods assume
  /// that this is valid.
  Token Tok;

  // PrevTokLocation - The location of the token we previously
  // consumed. This token is used for diagnostics where we expected to
  // see a token following another token (e.g., the ';' at the end of
  // a statement).
  SourceLocation PrevTokLocation;

  unsigned short ParenCount, BracketCount, BraceCount;
  
  /// Actions - These are the callbacks we invoke as we parse various constructs
  /// in the file.
  Sema &Actions;

  DiagnosticsEngine &Diags;

  /// ScopeCache - Cache scopes to reduce malloc traffic.
  enum { ScopeCacheSize = 16 };
  unsigned NumCachedScopes;
  Scope *ScopeCache[ScopeCacheSize];

  /// Identifiers used for SEH handling in Borland. These are only
  /// allowed in particular circumstances
  // __except block
  IdentifierInfo *Ident__exception_code,
                 *Ident___exception_code,
                 *Ident_GetExceptionCode;
  // __except filter expression
  IdentifierInfo *Ident__exception_info,
                 *Ident___exception_info,
                 *Ident_GetExceptionInfo;
  // __finally
  IdentifierInfo *Ident__abnormal_termination,
                 *Ident___abnormal_termination,
                 *Ident_AbnormalTermination;

  /// Contextual keywords for Microsoft extensions.
  IdentifierInfo *Ident__except;

  /// Ident_super - IdentifierInfo for "super", to support fast
  /// comparison.
  IdentifierInfo *Ident_super;
  /// Ident_vector and Ident_pixel - cached IdentifierInfo's for
  /// "vector" and "pixel" fast comparison.  Only present if
  /// AltiVec enabled.
  IdentifierInfo *Ident_vector;
  IdentifierInfo *Ident_pixel;

  /// Objective-C contextual keywords.
  mutable IdentifierInfo *Ident_instancetype;

  /// \brief Identifier for "introduced".
  IdentifierInfo *Ident_introduced;

  /// \brief Identifier for "deprecated".
  IdentifierInfo *Ident_deprecated;

  /// \brief Identifier for "obsoleted".
  IdentifierInfo *Ident_obsoleted;

  /// \brief Identifier for "unavailable".
  IdentifierInfo *Ident_unavailable;
  
  /// \brief Identifier for "message".
  IdentifierInfo *Ident_message;

  /// C++0x contextual keywords.
  mutable IdentifierInfo *Ident_final;
  mutable IdentifierInfo *Ident_override;

  // C++ type trait keywords that have can be reverted to identifiers and
  // still used as type traits.
  llvm::SmallDenseMap<IdentifierInfo *, tok::TokenKind> RevertableTypeTraits;

  OwningPtr<PragmaHandler> AlignHandler;
  OwningPtr<PragmaHandler> GCCVisibilityHandler;
  OwningPtr<PragmaHandler> OptionsHandler;
  OwningPtr<PragmaHandler> PackHandler;
  OwningPtr<PragmaHandler> MSStructHandler;
  OwningPtr<PragmaHandler> UnusedHandler;
  OwningPtr<PragmaHandler> WeakHandler;
  OwningPtr<PragmaHandler> RedefineExtnameHandler;
  OwningPtr<PragmaHandler> FPContractHandler;
  OwningPtr<PragmaHandler> OpenCLExtensionHandler;
  OwningPtr<CommentHandler> CommentSemaHandler;

  /// Whether the '>' token acts as an operator or not. This will be
  /// true except when we are parsing an expression within a C++
  /// template argument list, where the '>' closes the template
  /// argument list.
  bool GreaterThanIsOperator;

  /// ColonIsSacred - When this is false, we aggressively try to recover from
  /// code like "foo : bar" as if it were a typo for "foo :: bar".  This is not
  /// safe in case statements and a few other things.  This is managed by the
  /// ColonProtectionRAIIObject RAII object.
  bool ColonIsSacred;

  /// \brief When true, we are directly inside an Objective-C messsage
  /// send expression.
  ///
  /// This is managed by the \c InMessageExpressionRAIIObject class, and
  /// should not be set directly.
  bool InMessageExpression;

  /// The "depth" of the template parameters currently being parsed.
  unsigned TemplateParameterDepth;

  /// Factory object for creating AttributeList objects.
  AttributeFactory AttrFactory;

  /// \brief Gathers and cleans up TemplateIdAnnotations when parsing of a
  /// top-level declaration is finished.
  SmallVector<TemplateIdAnnotation *, 16> TemplateIds;

  /// \brief Identifiers which have been declared within a tentative parse.
  SmallVector<IdentifierInfo *, 8> TentativelyDeclaredIdentifiers;

  IdentifierInfo *getSEHExceptKeyword();

  /// True if we are within an Objective-C container while parsing C-like decls.
  ///
  /// This is necessary because Sema thinks we have left the container
  /// to parse the C-like decls, meaning Actions.getObjCDeclContext() will
  /// be NULL.
  bool ParsingInObjCContainer;

  bool SkipFunctionBodies;

public:
  Parser(Preprocessor &PP, Sema &Actions, bool SkipFunctionBodies);
  ~Parser();

  const LangOptions &getLangOpts() const { return PP.getLangOpts(); }
  const TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
  Preprocessor &getPreprocessor() const { return PP; }
  Sema &getActions() const { return Actions; }
  AttributeFactory &getAttrFactory() { return AttrFactory; }

  const Token &getCurToken() const { return Tok; }
  Scope *getCurScope() const { return Actions.getCurScope(); }

  Decl  *getObjCDeclContext() const { return Actions.getObjCDeclContext(); }

  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
  typedef OpaquePtr<TemplateName> TemplateTy;

  typedef SmallVector<TemplateParameterList *, 4> TemplateParameterLists;

  typedef clang::ExprResult        ExprResult;
  typedef clang::StmtResult        StmtResult;
  typedef clang::BaseResult        BaseResult;
  typedef clang::MemInitResult     MemInitResult;
  typedef clang::TypeResult        TypeResult;

  typedef Expr *ExprArg;
  typedef llvm::MutableArrayRef<Stmt*> MultiStmtArg;
  typedef Sema::FullExprArg FullExprArg;

  /// Adorns a ExprResult with Actions to make it an ExprResult
  ExprResult Owned(ExprResult res) {
    return ExprResult(res);
  }
  /// Adorns a StmtResult with Actions to make it an StmtResult
  StmtResult Owned(StmtResult res) {
    return StmtResult(res);
  }

  ExprResult ExprError() { return ExprResult(true); }
  StmtResult StmtError() { return StmtResult(true); }

  ExprResult ExprError(const DiagnosticBuilder &) { return ExprError(); }
  StmtResult StmtError(const DiagnosticBuilder &) { return StmtError(); }

  ExprResult ExprEmpty() { return ExprResult(false); }

  // Parsing methods.

  /// ParseTranslationUnit - All in one method that initializes parses, and
  /// shuts down the parser.
  void ParseTranslationUnit();

  /// Initialize - Warm up the parser.
  ///
  void Initialize();

  /// ParseTopLevelDecl - Parse one top-level declaration. Returns true if
  /// the EOF was encountered.
  bool ParseTopLevelDecl(DeclGroupPtrTy &Result);

  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work with all kinds of tokens: strings and specific other
  /// tokens must be consumed with custom methods below.  This returns the
  /// location of the consumed token.
  SourceLocation ConsumeToken() {
    assert(!isTokenStringLiteral() && !isTokenParen() && !isTokenBracket() &&
           !isTokenBrace() &&
           "Should consume special tokens with Consume*Token");

    if (Tok.is(tok::code_completion))
      return handleUnexpectedCodeCompletionToken();

    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

private:
  //===--------------------------------------------------------------------===//
  // Low-Level token peeking and consumption methods.
  //

  /// isTokenParen - Return true if the cur token is '(' or ')'.
  bool isTokenParen() const {
    return Tok.getKind() == tok::l_paren || Tok.getKind() == tok::r_paren;
  }
  /// isTokenBracket - Return true if the cur token is '[' or ']'.
  bool isTokenBracket() const {
    return Tok.getKind() == tok::l_square || Tok.getKind() == tok::r_square;
  }
  /// isTokenBrace - Return true if the cur token is '{' or '}'.
  bool isTokenBrace() const {
    return Tok.getKind() == tok::l_brace || Tok.getKind() == tok::r_brace;
  }

  /// isTokenStringLiteral - True if this token is a string-literal.
  ///
  bool isTokenStringLiteral() const {
    return Tok.getKind() == tok::string_literal ||
           Tok.getKind() == tok::wide_string_literal ||
           Tok.getKind() == tok::utf8_string_literal ||
           Tok.getKind() == tok::utf16_string_literal ||
           Tok.getKind() == tok::utf32_string_literal;
  }

  /// \brief Returns true if the current token is '=' or is a type of '='.
  /// For typos, give a fixit to '='
  bool isTokenEqualOrEqualTypo();

  /// ConsumeAnyToken - Dispatch to the right Consume* method based on the
  /// current token type.  This should only be used in cases where the type of
  /// the token really isn't known, e.g. in error recovery.
  SourceLocation ConsumeAnyToken() {
    if (isTokenParen())
      return ConsumeParen();
    else if (isTokenBracket())
      return ConsumeBracket();
    else if (isTokenBrace())
      return ConsumeBrace();
    else if (isTokenStringLiteral())
      return ConsumeStringToken();
    else
      return ConsumeToken();
  }

  /// ConsumeParen - This consume method keeps the paren count up-to-date.
  ///
  SourceLocation ConsumeParen() {
    assert(isTokenParen() && "wrong consume method");
    if (Tok.getKind() == tok::l_paren)
      ++ParenCount;
    else if (ParenCount)
      --ParenCount;       // Don't let unbalanced )'s drive the count negative.
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
    else if (BracketCount)
      --BracketCount;     // Don't let unbalanced ]'s drive the count negative.

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
    else if (BraceCount)
      --BraceCount;     // Don't let unbalanced }'s drive the count negative.

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

  /// \brief Consume the current code-completion token.
  ///
  /// This routine should be called to consume the code-completion token once
  /// a code-completion action has already been invoked.
  SourceLocation ConsumeCodeCompletionToken() {
    assert(Tok.is(tok::code_completion));
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

  ///\ brief When we are consuming a code-completion token without having
  /// matched specific position in the grammar, provide code-completion results
  /// based on context.
  ///
  /// \returns the source location of the code-completion token.
  SourceLocation handleUnexpectedCodeCompletionToken();

  /// \brief Abruptly cut off parsing; mainly used when we have reached the
  /// code-completion point.
  void cutOffParsing() {
    PP.setCodeCompletionReached();
    // Cut off parsing by acting as if we reached the end-of-file.
    Tok.setKind(tok::eof);
  }

  /// \brief Handle the annotation token produced for #pragma unused(...)
  void HandlePragmaUnused();

  /// \brief Handle the annotation token produced for
  /// #pragma GCC visibility...
  void HandlePragmaVisibility();

  /// \brief Handle the annotation token produced for
  /// #pragma pack...
  void HandlePragmaPack();

  /// \brief Handle the annotation token produced for
  /// #pragma ms_struct...
  void HandlePragmaMSStruct();

  /// \brief Handle the annotation token produced for
  /// #pragma align...
  void HandlePragmaAlign();

  /// \brief Handle the annotation token produced for
  /// #pragma weak id...
  void HandlePragmaWeak();

  /// \brief Handle the annotation token produced for
  /// #pragma weak id = id...
  void HandlePragmaWeakAlias();

  /// \brief Handle the annotation token produced for
  /// #pragma redefine_extname...
  void HandlePragmaRedefineExtname();

  /// \brief Handle the annotation token produced for
  /// #pragma STDC FP_CONTRACT...
  void HandlePragmaFPContract();

  /// \brief Handle the annotation token produced for
  /// #pragma OPENCL EXTENSION...
  void HandlePragmaOpenCLExtension();

  /// GetLookAheadToken - This peeks ahead N tokens and returns that token
  /// without consuming any tokens.  LookAhead(0) returns 'Tok', LookAhead(1)
  /// returns the token after Tok, etc.
  ///
  /// Note that this differs from the Preprocessor's LookAhead method, because
  /// the Parser always has one token lexed that the preprocessor doesn't.
  ///
  const Token &GetLookAheadToken(unsigned N) {
    if (N == 0 || Tok.is(tok::eof)) return Tok;
    return PP.LookAhead(N-1);
  }

public:
  /// NextToken - This peeks ahead one token and returns it without
  /// consuming it.
  const Token &NextToken() {
    return PP.LookAhead(0);
  }

  /// getTypeAnnotation - Read a parsed type out of an annotation token.
  static ParsedType getTypeAnnotation(Token &Tok) {
    return ParsedType::getFromOpaquePtr(Tok.getAnnotationValue());
  }

private:
  static void setTypeAnnotation(Token &Tok, ParsedType T) {
    Tok.setAnnotationValue(T.getAsOpaquePtr());
  }

  /// \brief Read an already-translated primary expression out of an annotation
  /// token.
  static ExprResult getExprAnnotation(Token &Tok) {
    if (Tok.getAnnotationValue())
      return ExprResult((Expr *)Tok.getAnnotationValue());

    return ExprResult(true);
  }

  /// \brief Set the primary expression corresponding to the given annotation
  /// token.
  static void setExprAnnotation(Token &Tok, ExprResult ER) {
    if (ER.isInvalid())
      Tok.setAnnotationValue(0);
    else
      Tok.setAnnotationValue(ER.get());
  }

public:
  // If NeedType is true, then TryAnnotateTypeOrScopeToken will try harder to
  // find a type name by attempting typo correction.
  bool TryAnnotateTypeOrScopeToken(bool EnteringContext = false,
                                   bool NeedType = false);
  bool TryAnnotateTypeOrScopeTokenAfterScopeSpec(bool EnteringContext,
                                                 bool NeedType,
                                                 CXXScopeSpec &SS,
                                                 bool IsNewScope);
  bool TryAnnotateCXXScopeToken(bool EnteringContext = false);

private:
  enum AnnotatedNameKind {
    /// Annotation has failed and emitted an error.
    ANK_Error,
    /// The identifier is a tentatively-declared name.
    ANK_TentativeDecl,
    /// The identifier is a template name. FIXME: Add an annotation for that.
    ANK_TemplateName,
    /// The identifier can't be resolved.
    ANK_Unresolved,
    /// Annotation was successful.
    ANK_Success
  };
  AnnotatedNameKind TryAnnotateName(bool IsAddressOfOperand,
                                    CorrectionCandidateCallback *CCC = 0);

  /// Push a tok::annot_cxxscope token onto the token stream.
  void AnnotateScopeToken(CXXScopeSpec &SS, bool IsNewAnnotation);

  /// TryAltiVecToken - Check for context-sensitive AltiVec identifier tokens,
  /// replacing them with the non-context-sensitive keywords.  This returns
  /// true if the token was replaced.
  bool TryAltiVecToken(DeclSpec &DS, SourceLocation Loc,
                       const char *&PrevSpec, unsigned &DiagID,
                       bool &isInvalid) {
    if (!getLangOpts().AltiVec ||
        (Tok.getIdentifierInfo() != Ident_vector &&
         Tok.getIdentifierInfo() != Ident_pixel))
      return false;

    return TryAltiVecTokenOutOfLine(DS, Loc, PrevSpec, DiagID, isInvalid);
  }

  /// TryAltiVecVectorToken - Check for context-sensitive AltiVec vector
  /// identifier token, replacing it with the non-context-sensitive __vector.
  /// This returns true if the token was replaced.
  bool TryAltiVecVectorToken() {
    if (!getLangOpts().AltiVec ||
        Tok.getIdentifierInfo() != Ident_vector) return false;
    return TryAltiVecVectorTokenOutOfLine();
  }

  bool TryAltiVecVectorTokenOutOfLine();
  bool TryAltiVecTokenOutOfLine(DeclSpec &DS, SourceLocation Loc,
                                const char *&PrevSpec, unsigned &DiagID,
                                bool &isInvalid);

  /// \brief Get the TemplateIdAnnotation from the token.
  TemplateIdAnnotation *takeTemplateIdAnnotation(const Token &tok);

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
  class TentativeParsingAction {
    Parser &P;
    Token PrevTok;
    size_t PrevTentativelyDeclaredIdentifierCount;
    unsigned short PrevParenCount, PrevBracketCount, PrevBraceCount;
    bool isActive;

  public:
    explicit TentativeParsingAction(Parser& p) : P(p) {
      PrevTok = P.Tok;
      PrevTentativelyDeclaredIdentifierCount =
          P.TentativelyDeclaredIdentifiers.size();
      PrevParenCount = P.ParenCount;
      PrevBracketCount = P.BracketCount;
      PrevBraceCount = P.BraceCount;
      P.PP.EnableBacktrackAtThisPos();
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

  /// ObjCDeclContextSwitch - An object used to switch context from
  /// an objective-c decl context to its enclosing decl context and
  /// back.
  class ObjCDeclContextSwitch {
    Parser &P;
    Decl *DC;
    SaveAndRestore<bool> WithinObjCContainer;
  public:
    explicit ObjCDeclContextSwitch(Parser &p)
      : P(p), DC(p.getObjCDeclContext()),
        WithinObjCContainer(P.ParsingInObjCContainer, DC != 0) {
      if (DC)
        P.Actions.ActOnObjCTemporaryExitContainerContext(cast<DeclContext>(DC));
    }
    ~ObjCDeclContextSwitch() {
      if (DC)
        P.Actions.ActOnObjCReenterContainerContext(cast<DeclContext>(DC));
    }
  };

  /// ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
  /// input.  If so, it is consumed and false is returned.
  ///
  /// If the input is malformed, this emits the specified diagnostic.  Next, if
  /// SkipToTok is specified, it calls SkipUntil(SkipToTok).  Finally, true is
  /// returned.
  bool ExpectAndConsume(tok::TokenKind ExpectedTok, unsigned Diag,
                        const char *DiagMsg = "",
                        tok::TokenKind SkipToTok = tok::unknown);

  /// \brief The parser expects a semicolon and, if present, will consume it.
  ///
  /// If the next token is not a semicolon, this emits the specified diagnostic,
  /// or, if there's just some closing-delimiter noise (e.g., ')' or ']') prior
  /// to the semicolon, consumes that extra token.
  bool ExpectAndConsumeSemi(unsigned DiagID);

  /// \brief The kind of extra semi diagnostic to emit.
  enum ExtraSemiKind {
    OutsideFunction = 0,
    InsideStruct = 1,
    InstanceVariableList = 2,
    AfterMemberFunctionDefinition = 3
  };

  /// \brief Consume any extra semi-colons until the end of the line.
  void ConsumeExtraSemi(ExtraSemiKind Kind, unsigned TST = TST_unspecified);

public:
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
    ParseScope(const ParseScope &) LLVM_DELETED_FUNCTION;
    void operator=(const ParseScope &) LLVM_DELETED_FUNCTION;

  public:
    // ParseScope - Construct a new object to manage a scope in the
    // parser Self where the new Scope is created with the flags
    // ScopeFlags, but only when ManageScope is true (the default). If
    // ManageScope is false, this object does nothing.
    ParseScope(Parser *Self, unsigned ScopeFlags, bool ManageScope = true)
      : Self(Self) {
      if (ManageScope)
        Self->EnterScope(ScopeFlags);
      else
        this->Self = 0;
    }

    // Exit - Exit the scope associated with this object now, rather
    // than waiting until the object is destroyed.
    void Exit() {
      if (Self) {
        Self->ExitScope();
        Self = 0;
      }
    }

    ~ParseScope() {
      Exit();
    }
  };

  /// EnterScope - Start a new scope.
  void EnterScope(unsigned ScopeFlags);

  /// ExitScope - Pop a scope off the scope stack.
  void ExitScope();

private:
  /// \brief RAII object used to modify the scope flags for the current scope.
  class ParseScopeFlags {
    Scope *CurScope;
    unsigned OldFlags;
    ParseScopeFlags(const ParseScopeFlags &) LLVM_DELETED_FUNCTION;
    void operator=(const ParseScopeFlags &) LLVM_DELETED_FUNCTION;

  public:
    ParseScopeFlags(Parser *Self, unsigned ScopeFlags, bool ManageFlags = true);
    ~ParseScopeFlags();
  };

  //===--------------------------------------------------------------------===//
  // Diagnostic Emission and Error recovery.

public:
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);
  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID);
  DiagnosticBuilder Diag(unsigned DiagID) {
    return Diag(Tok, DiagID);
  }

private:
  void SuggestParentheses(SourceLocation Loc, unsigned DK,
                          SourceRange ParenRange);
  void CheckNestedObjCContexts(SourceLocation AtLoc);

public:
  /// SkipUntil - Read tokens until we get to the specified token, then consume
  /// it (unless DontConsume is true).  Because we cannot guarantee that the
  /// token will ever occur, this skips to the next token, or to some likely
  /// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
  /// character.
  ///
  /// If SkipUntil finds the specified token, it returns true, otherwise it
  /// returns false.
  bool SkipUntil(tok::TokenKind T, bool StopAtSemi = true,
                 bool DontConsume = false, bool StopAtCodeCompletion = false) {
    return SkipUntil(llvm::makeArrayRef(T), StopAtSemi, DontConsume,
                     StopAtCodeCompletion);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, bool StopAtSemi = true,
                 bool DontConsume = false, bool StopAtCodeCompletion = false) {
    tok::TokenKind TokArray[] = {T1, T2};
    return SkipUntil(TokArray, StopAtSemi, DontConsume,StopAtCodeCompletion);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, tok::TokenKind T3,
                 bool StopAtSemi = true, bool DontConsume = false,
                 bool StopAtCodeCompletion = false) {
    tok::TokenKind TokArray[] = {T1, T2, T3};
    return SkipUntil(TokArray, StopAtSemi, DontConsume,StopAtCodeCompletion);
  }
  bool SkipUntil(ArrayRef<tok::TokenKind> Toks, bool StopAtSemi = true,
                 bool DontConsume = false, bool StopAtCodeCompletion = false);

  /// SkipMalformedDecl - Read tokens until we get to some likely good stopping
  /// point for skipping past a simple-declaration.
  void SkipMalformedDecl();

private:
  //===--------------------------------------------------------------------===//
  // Lexing and parsing of C++ inline methods.

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
  };

  /// Inner node of the LateParsedDeclaration tree that parses
  /// all its members recursively.
  class LateParsedClass : public LateParsedDeclaration {
  public:
    LateParsedClass(Parser *P, ParsingClass *C);
    virtual ~LateParsedClass();

    virtual void ParseLexedMethodDeclarations();
    virtual void ParseLexedMemberInitializers();
    virtual void ParseLexedMethodDefs();
    virtual void ParseLexedAttributes();

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
    SourceLocation AttrNameLoc;
    SmallVector<Decl*, 2> Decls;

    explicit LateParsedAttribute(Parser *P, IdentifierInfo &Name,
                                 SourceLocation Loc)
      : Self(P), AttrName(Name), AttrNameLoc(Loc) {}

    virtual void ParseLexedAttributes();

    void addDecl(Decl *D) { Decls.push_back(D); }
  };

  // A list of late-parsed attributes.  Used by ParseGNUAttributes.
  class LateParsedAttrList: public llvm::SmallVector<LateParsedAttribute*, 2> {
  public:
    LateParsedAttrList(bool PSoon = false) : ParseSoon(PSoon) { }

    bool parseSoon() { return ParseSoon; }

  private:
    bool ParseSoon;  // Are we planning to parse these shortly after creation?
  };

  /// Contains the lexed tokens of a member function definition
  /// which needs to be parsed at the end of the class declaration
  /// after parsing all other member declarations.
  struct LexedMethod : public LateParsedDeclaration {
    Parser *Self;
    Decl *D;
    CachedTokens Toks;

    /// \brief Whether this member function had an associated template
    /// scope. When true, D is a template declaration.
    /// othewise, it is a member function declaration.
    bool TemplateScope;

    explicit LexedMethod(Parser* P, Decl *MD)
      : Self(P), D(MD), TemplateScope(false) {}

    virtual void ParseLexedMethodDefs();
  };

  /// LateParsedDefaultArgument - Keeps track of a parameter that may
  /// have a default argument that cannot be parsed yet because it
  /// occurs within a member function declaration inside the class
  /// (C++ [class.mem]p2).
  struct LateParsedDefaultArgument {
    explicit LateParsedDefaultArgument(Decl *P,
                                       CachedTokens *Toks = 0)
      : Param(P), Toks(Toks) { }

    /// Param - The parameter declaration for this parameter.
    Decl *Param;

    /// Toks - The sequence of tokens that comprises the default
    /// argument expression, not including the '=' or the terminating
    /// ')' or ','. This will be NULL for parameters that have no
    /// default argument.
    CachedTokens *Toks;
  };

  /// LateParsedMethodDeclaration - A method declaration inside a class that
  /// contains at least one entity whose parsing needs to be delayed
  /// until the class itself is completely-defined, such as a default
  /// argument (C++ [class.mem]p2).
  struct LateParsedMethodDeclaration : public LateParsedDeclaration {
    explicit LateParsedMethodDeclaration(Parser *P, Decl *M)
      : Self(P), Method(M), TemplateScope(false), ExceptionSpecTokens(0) { }

    virtual void ParseLexedMethodDeclarations();

    Parser* Self;

    /// Method - The method declaration.
    Decl *Method;

    /// \brief Whether this member function had an associated template
    /// scope. When true, D is a template declaration.
    /// othewise, it is a member function declaration.
    bool TemplateScope;

    /// DefaultArgs - Contains the parameters of the function and
    /// their default arguments. At least one of the parameters will
    /// have a default argument, but all of the parameters of the
    /// method will be stored so that they can be reintroduced into
    /// scope at the appropriate times.
    SmallVector<LateParsedDefaultArgument, 8> DefaultArgs;
  
    /// \brief The set of tokens that make up an exception-specification that
    /// has not yet been parsed.
    CachedTokens *ExceptionSpecTokens;
  };

  /// LateParsedMemberInitializer - An initializer for a non-static class data
  /// member whose parsing must to be delayed until the class is completely
  /// defined (C++11 [class.mem]p2).
  struct LateParsedMemberInitializer : public LateParsedDeclaration {
    LateParsedMemberInitializer(Parser *P, Decl *FD)
      : Self(P), Field(FD) { }

    virtual void ParseLexedMemberInitializers();

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
  typedef SmallVector<LateParsedDeclaration*,2> LateParsedDeclarationsContainer;

  /// \brief Representation of a class that has been parsed, including
  /// any member function declarations or definitions that need to be
  /// parsed after the corresponding top-level class is complete.
  struct ParsingClass {
    ParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface)
      : TopLevelClass(TopLevelClass), TemplateScope(false),
        IsInterface(IsInterface), TagOrTemplate(TagOrTemplate) { }

    /// \brief Whether this is a "top-level" class, meaning that it is
    /// not nested within another class.
    bool TopLevelClass : 1;

    /// \brief Whether this class had an associated template
    /// scope. When true, TagOrTemplate is a template declaration;
    /// othewise, it is a tag declaration.
    bool TemplateScope : 1;

    /// \brief Whether this class is an __interface.
    bool IsInterface : 1;

    /// \brief The class or class template whose definition we are parsing.
    Decl *TagOrTemplate;

    /// LateParsedDeclarations - Method declarations, inline definitions and
    /// nested classes that contain pieces whose parsing will be delayed until
    /// the top-level class is fully defined.
    LateParsedDeclarationsContainer LateParsedDeclarations;
  };

  /// \brief The stack of classes that is currently being
  /// parsed. Nested and local classes will be pushed onto this stack
  /// when they are parsed, and removed afterward.
  std::stack<ParsingClass *> ClassStack;

  ParsingClass &getCurrentClass() {
    assert(!ClassStack.empty() && "No lexed method stacks!");
    return *ClassStack.top();
  }

  /// \brief RAII object used to manage the parsing of a class definition.
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

    /// \brief Pop this class of the stack.
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

  /// \brief Contains information about any template-specific
  /// information that has been parsed prior to parsing declaration
  /// specifiers.
  struct ParsedTemplateInfo {
    ParsedTemplateInfo()
      : Kind(NonTemplate), TemplateParams(0), TemplateLoc() { }

    ParsedTemplateInfo(TemplateParameterLists *TemplateParams,
                       bool isSpecialization,
                       bool lastParameterListWasEmpty = false)
      : Kind(isSpecialization? ExplicitSpecialization : Template),
        TemplateParams(TemplateParams),
        LastParameterListWasEmpty(lastParameterListWasEmpty) { }

    explicit ParsedTemplateInfo(SourceLocation ExternLoc,
                                SourceLocation TemplateLoc)
      : Kind(ExplicitInstantiation), TemplateParams(0),
        ExternLoc(ExternLoc), TemplateLoc(TemplateLoc),
        LastParameterListWasEmpty(false){ }

    /// \brief The kind of template we are parsing.
    enum {
      /// \brief We are not parsing a template at all.
      NonTemplate = 0,
      /// \brief We are parsing a template declaration.
      Template,
      /// \brief We are parsing an explicit specialization.
      ExplicitSpecialization,
      /// \brief We are parsing an explicit instantiation.
      ExplicitInstantiation
    } Kind;

    /// \brief The template parameter lists, for template declarations
    /// and explicit specializations.
    TemplateParameterLists *TemplateParams;

    /// \brief The location of the 'extern' keyword, if any, for an explicit
    /// instantiation
    SourceLocation ExternLoc;

    /// \brief The location of the 'template' keyword, for an explicit
    /// instantiation.
    SourceLocation TemplateLoc;

    /// \brief Whether the last template parameter list was empty.
    bool LastParameterListWasEmpty;

    SourceRange getSourceRange() const LLVM_READONLY;
  };

  /// \brief Contains a late templated function.
  /// Will be parsed at the end of the translation unit.
  struct LateParsedTemplatedFunction {
    explicit LateParsedTemplatedFunction(Decl *MD)
      : D(MD) {}

    CachedTokens Toks;

    /// \brief The template function declaration to be late parsed.
    Decl *D;
  };

  void LexTemplateFunctionForLateParsing(CachedTokens &Toks);
  void ParseLateTemplatedFuncDef(LateParsedTemplatedFunction &LMT);
  typedef llvm::DenseMap<const FunctionDecl*, LateParsedTemplatedFunction*>
    LateParsedTemplateMapT;
  LateParsedTemplateMapT LateParsedTemplateMap;

  static void LateTemplateParserCallback(void *P, const FunctionDecl *FD);
  void LateTemplateParser(const FunctionDecl *FD);

  Sema::ParsingClassState
  PushParsingClass(Decl *TagOrTemplate, bool TopLevelClass, bool IsInterface);
  void DeallocateParsedClasses(ParsingClass *Class);
  void PopParsingClass(Sema::ParsingClassState);

  Decl *ParseCXXInlineMethodDef(AccessSpecifier AS, AttributeList *AccessAttrs,
                                ParsingDeclarator &D,
                                const ParsedTemplateInfo &TemplateInfo,
                                const VirtSpecifiers& VS,
                                FunctionDefinitionKind DefinitionKind,
                                ExprResult& Init);
  void ParseCXXNonStaticMemberInitializer(Decl *VarD);
  void ParseLexedAttributes(ParsingClass &Class);
  void ParseLexedAttributeList(LateParsedAttrList &LAs, Decl *D,
                               bool EnterScope, bool OnDefinition);
  void ParseLexedAttribute(LateParsedAttribute &LA,
                           bool EnterScope, bool OnDefinition);
  void ParseLexedMethodDeclarations(ParsingClass &Class);
  void ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM);
  void ParseLexedMethodDefs(ParsingClass &Class);
  void ParseLexedMethodDef(LexedMethod &LM);
  void ParseLexedMemberInitializers(ParsingClass &Class);
  void ParseLexedMemberInitializer(LateParsedMemberInitializer &MI);
  void ParseLexedObjCMethodDefs(LexedMethod &LM, bool parseMethod);
  bool ConsumeAndStoreFunctionPrologue(CachedTokens &Toks);
  bool ConsumeAndStoreUntil(tok::TokenKind T1,
                            CachedTokens &Toks,
                            bool StopAtSemi = true,
                            bool ConsumeFinalToken = true) {
    return ConsumeAndStoreUntil(T1, T1, Toks, StopAtSemi, ConsumeFinalToken);
  }
  bool ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                            CachedTokens &Toks,
                            bool StopAtSemi = true,
                            bool ConsumeFinalToken = true);

  //===--------------------------------------------------------------------===//
  // C99 6.9: External Definitions.
  struct ParsedAttributesWithRange : ParsedAttributes {
    ParsedAttributesWithRange(AttributeFactory &factory)
      : ParsedAttributes(factory) {}

    SourceRange Range;
  };

  DeclGroupPtrTy ParseExternalDeclaration(ParsedAttributesWithRange &attrs,
                                          ParsingDeclSpec *DS = 0);
  bool isDeclarationAfterDeclarator();
  bool isStartOfFunctionDefinition(const ParsingDeclarator &Declarator);
  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(
                                                  ParsedAttributesWithRange &attrs,
                                                  ParsingDeclSpec *DS = 0,
                                                  AccessSpecifier AS = AS_none);
  DeclGroupPtrTy ParseDeclOrFunctionDefInternal(ParsedAttributesWithRange &attrs,
                                                ParsingDeclSpec &DS,
                                                AccessSpecifier AS);

  Decl *ParseFunctionDefinition(ParsingDeclarator &D,
                 const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                 LateParsedAttrList *LateParsedAttrs = 0);
  void ParseKNRParamDeclarations(Declarator &D);
  // EndLoc, if non-NULL, is filled with the location of the last token of
  // the simple-asm.
  ExprResult ParseSimpleAsm(SourceLocation *EndLoc = 0);
  ExprResult ParseAsmStringLiteral();

  // Objective-C External Declarations
  DeclGroupPtrTy ParseObjCAtDirectives();
  DeclGroupPtrTy ParseObjCAtClassDeclaration(SourceLocation atLoc);
  Decl *ParseObjCAtInterfaceDeclaration(SourceLocation AtLoc,
                                        ParsedAttributes &prefixAttrs);
  void ParseObjCClassInstanceVariables(Decl *interfaceDecl,
                                       tok::ObjCKeywordKind visibility,
                                       SourceLocation atLoc);
  bool ParseObjCProtocolReferences(SmallVectorImpl<Decl *> &P,
                                   SmallVectorImpl<SourceLocation> &PLocs,
                                   bool WarnOnDeclarations,
                                   SourceLocation &LAngleLoc,
                                   SourceLocation &EndProtoLoc);
  bool ParseObjCProtocolQualifiers(DeclSpec &DS);
  void ParseObjCInterfaceDeclList(tok::ObjCKeywordKind contextKey,
                                  Decl *CDecl);
  DeclGroupPtrTy ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
                                                ParsedAttributes &prefixAttrs);

  struct ObjCImplParsingDataRAII {
    Parser &P;
    Decl *Dcl;
    bool HasCFunction;
    typedef SmallVector<LexedMethod*, 8> LateParsedObjCMethodContainer;
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
  void StashAwayMethodOrFunctionBodyTokens(Decl *MDecl);

  DeclGroupPtrTy ParseObjCAtImplementationDeclaration(SourceLocation AtLoc);
  DeclGroupPtrTy ParseObjCAtEndDeclaration(SourceRange atEnd);
  Decl *ParseObjCAtAliasDeclaration(SourceLocation atLoc);
  Decl *ParseObjCPropertySynthesize(SourceLocation atLoc);
  Decl *ParseObjCPropertyDynamic(SourceLocation atLoc);

  IdentifierInfo *ParseObjCSelectorPiece(SourceLocation &MethodLocation);
  // Definitions for Objective-c context sensitive keywords recognition.
  enum ObjCTypeQual {
    objc_in=0, objc_out, objc_inout, objc_oneway, objc_bycopy, objc_byref,
    objc_NumQuals
  };
  IdentifierInfo *ObjCTypeQuals[objc_NumQuals];

  bool isTokIdentifier_in() const;

  ParsedType ParseObjCTypeName(ObjCDeclSpec &DS, Declarator::TheContext Ctx,
                               ParsedAttributes *ParamAttrs);
  void ParseObjCMethodRequirement();
  Decl *ParseObjCMethodPrototype(
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
            bool MethodDefinition = true);
  Decl *ParseObjCMethodDecl(SourceLocation mLoc, tok::TokenKind mType,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword,
            bool MethodDefinition=true);
  void ParseObjCPropertyAttribute(ObjCDeclSpec &DS);

  Decl *ParseObjCMethodDefinition();

public:
  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.

  /// TypeCastState - State whether an expression is or may be a type cast.
  enum TypeCastState {
    NotTypeCast = 0,
    MaybeTypeCast,
    IsTypeCast
  };

  ExprResult ParseExpression(TypeCastState isTypeCast = NotTypeCast);
  ExprResult ParseConstantExpression(TypeCastState isTypeCast = NotTypeCast);
  // Expr that doesn't include commas.
  ExprResult ParseAssignmentExpression(TypeCastState isTypeCast = NotTypeCast);

private:
  ExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  ExprResult ParseExpressionWithLeadingExtension(SourceLocation ExtLoc);

  ExprResult ParseRHSOfBinaryExpression(ExprResult LHS,
                                        prec::Level MinPrec);
  ExprResult ParseCastExpression(bool isUnaryExpression,
                                 bool isAddressOfOperand,
                                 bool &NotCastExpr,
                                 TypeCastState isTypeCast);
  ExprResult ParseCastExpression(bool isUnaryExpression,
                                 bool isAddressOfOperand = false,
                                 TypeCastState isTypeCast = NotTypeCast);

  /// Returns true if the next token cannot start an expression.
  bool isNotExpressionStart();

  /// Returns true if the next token would start a postfix-expression
  /// suffix.
  bool isPostfixExpressionSuffixStart() {
    tok::TokenKind K = Tok.getKind();
    return (K == tok::l_square || K == tok::l_paren ||
            K == tok::period || K == tok::arrow ||
            K == tok::plusplus || K == tok::minusminus);
  }

  ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);
  ExprResult ParseUnaryExprOrTypeTraitExpression();
  ExprResult ParseBuiltinPrimaryExpression();

  ExprResult ParseExprAfterUnaryExprOrTypeTrait(const Token &OpTok,
                                                     bool &isCastExpr,
                                                     ParsedType &CastTy,
                                                     SourceRange &CastRange);

  typedef SmallVector<Expr*, 20> ExprListTy;
  typedef SmallVector<SourceLocation, 20> CommaLocsTy;

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  bool ParseExpressionList(SmallVectorImpl<Expr*> &Exprs,
                           SmallVectorImpl<SourceLocation> &CommaLocs,
                           void (Sema::*Completer)(Scope *S,
                                                   Expr *Data,
                                             llvm::ArrayRef<Expr *> Args) = 0,
                           Expr *Data = 0);

  /// ParenParseOption - Control what ParseParenExpression will parse.
  enum ParenParseOption {
    SimpleExpr,      // Only parse '(' expression ')'
    CompoundStmt,    // Also allow '(' compound-statement ')'
    CompoundLiteral, // Also allow '(' type-name ')' '{' ... '}'
    CastExpr         // Also allow '(' type-name ')' <anything>
  };
  ExprResult ParseParenExpression(ParenParseOption &ExprType,
                                        bool stopIfCastExpr,
                                        bool isTypeCast,
                                        ParsedType &CastTy,
                                        SourceLocation &RParenLoc);

  ExprResult ParseCXXAmbiguousParenExpression(ParenParseOption &ExprType,
                                            ParsedType &CastTy,
                                            BalancedDelimiterTracker &Tracker);
  ExprResult ParseCompoundLiteralExpression(ParsedType Ty,
                                                  SourceLocation LParenLoc,
                                                  SourceLocation RParenLoc);

  ExprResult ParseStringLiteralExpression(bool AllowUserDefinedLiteral = false);

  ExprResult ParseGenericSelectionExpression();
  
  ExprResult ParseObjCBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ Expressions
  ExprResult ParseCXXIdExpression(bool isAddressOfOperand = false);

  bool areTokensAdjacent(const Token &A, const Token &B);

  void CheckForTemplateAndDigraph(Token &Next, ParsedType ObjectTypePtr,
                                  bool EnteringContext, IdentifierInfo &II,
                                  CXXScopeSpec &SS);

  bool ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
                                      ParsedType ObjectType,
                                      bool EnteringContext,
                                      bool *MayBePseudoDestructor = 0,
                                      bool IsTypename = false);

  void CheckForLParenAfterColonColon();

  //===--------------------------------------------------------------------===//
  // C++0x 5.1.2: Lambda expressions

  // [...] () -> type {...}
  ExprResult ParseLambdaExpression();
  ExprResult TryParseLambdaExpression();
  llvm::Optional<unsigned> ParseLambdaIntroducer(LambdaIntroducer &Intro);
  bool TryParseLambdaIntroducer(LambdaIntroducer &Intro);
  ExprResult ParseLambdaExpressionAfterIntroducer(
               LambdaIntroducer &Intro);

  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Casts
  ExprResult ParseCXXCasts();

  //===--------------------------------------------------------------------===//
  // C++ 5.2p1: C++ Type Identification
  ExprResult ParseCXXTypeid();

  //===--------------------------------------------------------------------===//
  //  C++ : Microsoft __uuidof Expression
  ExprResult ParseCXXUuidof();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.4: C++ Pseudo-Destructor Expressions
  ExprResult ParseCXXPseudoDestructor(ExprArg Base, SourceLocation OpLoc,
                                            tok::TokenKind OpKind,
                                            CXXScopeSpec &SS,
                                            ParsedType ObjectType);

  //===--------------------------------------------------------------------===//
  // C++ 9.3.2: C++ 'this' pointer
  ExprResult ParseCXXThis();

  //===--------------------------------------------------------------------===//
  // C++ 15: C++ Throw Expression
  ExprResult ParseThrowExpression();

  ExceptionSpecificationType tryParseExceptionSpecification(
                    SourceRange &SpecificationRange,
                    SmallVectorImpl<ParsedType> &DynamicExceptions,
                    SmallVectorImpl<SourceRange> &DynamicExceptionRanges,
                    ExprResult &NoexceptExpr);

  // EndLoc is filled with the location of the last token of the specification.
  ExceptionSpecificationType ParseDynamicExceptionSpecification(
                                  SourceRange &SpecificationRange,
                                  SmallVectorImpl<ParsedType> &Exceptions,
                                  SmallVectorImpl<SourceRange> &Ranges);

  //===--------------------------------------------------------------------===//
  // C++0x 8: Function declaration trailing-return-type
  TypeResult ParseTrailingReturnType(SourceRange &Range);

  //===--------------------------------------------------------------------===//
  // C++ 2.13.5: C++ Boolean Literals
  ExprResult ParseCXXBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.3: Explicit type conversion (functional notation)
  ExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

  /// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
  /// This should only be called when the current token is known to be part of
  /// simple-type-specifier.
  void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

  bool ParseCXXTypeSpecifierSeq(DeclSpec &DS);

  //===--------------------------------------------------------------------===//
  // C++ 5.3.4 and 5.3.5: C++ new and delete
  bool ParseExpressionListOrTypeId(SmallVectorImpl<Expr*> &Exprs,
                                   Declarator &D);
  void ParseDirectNewDeclarator(Declarator &D);
  ExprResult ParseCXXNewExpression(bool UseGlobal, SourceLocation Start);
  ExprResult ParseCXXDeleteExpression(bool UseGlobal,
                                            SourceLocation Start);

  //===--------------------------------------------------------------------===//
  // C++ if/switch/while condition expression.
  bool ParseCXXCondition(ExprResult &ExprResult, Decl *&DeclResult,
                         SourceLocation Loc, bool ConvertToBoolean);

  //===--------------------------------------------------------------------===//
  // C++ types

  //===--------------------------------------------------------------------===//
  // C99 6.7.8: Initialization.

  /// ParseInitializer
  ///       initializer: [C99 6.7.8]
  ///         assignment-expression
  ///         '{' ...
  ExprResult ParseInitializer() {
    if (Tok.isNot(tok::l_brace))
      return ParseAssignmentExpression();
    return ParseBraceInitializer();
  }
  bool MayBeDesignationStart();
  ExprResult ParseBraceInitializer();
  ExprResult ParseInitializerWithPotentialDesignator();

  //===--------------------------------------------------------------------===//
  // clang Expressions

  ExprResult ParseBlockLiteralExpression();  // ^{...}

  //===--------------------------------------------------------------------===//
  // Objective-C Expressions
  ExprResult ParseObjCAtExpression(SourceLocation AtLocation);
  ExprResult ParseObjCStringLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCCharacterLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCNumericLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCBooleanLiteral(SourceLocation AtLoc, bool ArgValue);
  ExprResult ParseObjCArrayLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCDictionaryLiteral(SourceLocation AtLoc);
  ExprResult ParseObjCBoxedExpr(SourceLocation AtLoc);
  ExprResult ParseObjCEncodeExpression(SourceLocation AtLoc);
  ExprResult ParseObjCSelectorExpression(SourceLocation AtLoc);
  ExprResult ParseObjCProtocolExpression(SourceLocation AtLoc);
  bool isSimpleObjCMessageExpression();
  ExprResult ParseObjCMessageExpression();
  ExprResult ParseObjCMessageExpressionBody(SourceLocation LBracloc,
                                            SourceLocation SuperLoc,
                                            ParsedType ReceiverType,
                                            ExprArg ReceiverExpr);
  ExprResult ParseAssignmentExprWithObjCMessageExprStart(
      SourceLocation LBracloc, SourceLocation SuperLoc,
      ParsedType ReceiverType, ExprArg ReceiverExpr);
  bool ParseObjCXXMessageReceiver(bool &IsExpr, void *&TypeOrExpr);
    
  //===--------------------------------------------------------------------===//
  // C99 6.8: Statements and Blocks.

  /// A SmallVector of statements, with stack size 32 (as that is the only one
  /// used.)
  typedef SmallVector<Stmt*, 32> StmtVector;
  /// A SmallVector of expressions, with stack size 12 (the maximum used.)
  typedef SmallVector<Expr*, 12> ExprVector;
  /// A SmallVector of types.
  typedef SmallVector<ParsedType, 12> TypeVector;

  StmtResult ParseStatement(SourceLocation *TrailingElseLoc = 0) {
    StmtVector Stmts;
    return ParseStatementOrDeclaration(Stmts, true, TrailingElseLoc);
  }
  StmtResult ParseStatementOrDeclaration(StmtVector &Stmts,
                                         bool OnlyStatement,
                                         SourceLocation *TrailingElseLoc = 0);
  StmtResult ParseStatementOrDeclarationAfterAttributes(
                                         StmtVector &Stmts,
                                         bool OnlyStatement,
                                         SourceLocation *TrailingElseLoc,
                                         ParsedAttributesWithRange &Attrs);
  StmtResult ParseExprStatement();
  StmtResult ParseLabeledStatement(ParsedAttributesWithRange &attrs);
  StmtResult ParseCaseStatement(bool MissingCase = false,
                                ExprResult Expr = ExprResult());
  StmtResult ParseDefaultStatement();
  StmtResult ParseCompoundStatement(bool isStmtExpr = false);
  StmtResult ParseCompoundStatement(bool isStmtExpr,
                                    unsigned ScopeFlags);
  void ParseCompoundStatementLeadingPragmas();
  StmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
  bool ParseParenExprOrCondition(ExprResult &ExprResult,
                                 Decl *&DeclResult,
                                 SourceLocation Loc,
                                 bool ConvertToBoolean);
  StmtResult ParseIfStatement(SourceLocation *TrailingElseLoc);
  StmtResult ParseSwitchStatement(SourceLocation *TrailingElseLoc);
  StmtResult ParseWhileStatement(SourceLocation *TrailingElseLoc);
  StmtResult ParseDoStatement();
  StmtResult ParseForStatement(SourceLocation *TrailingElseLoc);
  StmtResult ParseGotoStatement();
  StmtResult ParseContinueStatement();
  StmtResult ParseBreakStatement();
  StmtResult ParseReturnStatement();
  StmtResult ParseAsmStatement(bool &msAsm);
  StmtResult ParseMicrosoftAsmStatement(SourceLocation AsmLoc);

  /// \brief Describes the behavior that should be taken for an __if_exists
  /// block.
  enum IfExistsBehavior {
    /// \brief Parse the block; this code is always used.
    IEB_Parse,
    /// \brief Skip the block entirely; this code is never used.
    IEB_Skip,
    /// \brief Parse the block as a dependent block, which may be used in
    /// some template instantiations but not others.
    IEB_Dependent
  };

  /// \brief Describes the condition of a Microsoft __if_exists or
  /// __if_not_exists block.
  struct IfExistsCondition {
    /// \brief The location of the initial keyword.
    SourceLocation KeywordLoc;
    /// \brief Whether this is an __if_exists block (rather than an
    /// __if_not_exists block).
    bool IsIfExists;

    /// \brief Nested-name-specifier preceding the name.
    CXXScopeSpec SS;

    /// \brief The name we're looking for.
    UnqualifiedId Name;

    /// \brief The behavior of this __if_exists or __if_not_exists block
    /// should.
    IfExistsBehavior Behavior;
  };

  bool ParseMicrosoftIfExistsCondition(IfExistsCondition& Result);
  void ParseMicrosoftIfExistsStatement(StmtVector &Stmts);
  void ParseMicrosoftIfExistsExternalDeclaration();
  void ParseMicrosoftIfExistsClassDeclaration(DeclSpec::TST TagType,
                                              AccessSpecifier& CurAS);
  bool ParseMicrosoftIfExistsBraceInitializer(ExprVector &InitExprs,
                                              bool &InitExprsOk);
  bool ParseAsmOperandsOpt(SmallVectorImpl<IdentifierInfo *> &Names,
                           SmallVectorImpl<Expr *> &Constraints,
                           SmallVectorImpl<Expr *> &Exprs);

  //===--------------------------------------------------------------------===//
  // C++ 6: Statements and Blocks

  StmtResult ParseCXXTryBlock();
  StmtResult ParseCXXTryBlockCommon(SourceLocation TryLoc, bool FnTry = false);
  StmtResult ParseCXXCatchBlock(bool FnCatch = false);

  //===--------------------------------------------------------------------===//
  // MS: SEH Statements and Blocks

  StmtResult ParseSEHTryBlock();
  StmtResult ParseSEHTryBlockCommon(SourceLocation Loc);
  StmtResult ParseSEHExceptBlock(SourceLocation Loc);
  StmtResult ParseSEHFinallyBlock(SourceLocation Loc);

  //===--------------------------------------------------------------------===//
  // Objective-C Statements

  StmtResult ParseObjCAtStatement(SourceLocation atLoc);
  StmtResult ParseObjCTryStmt(SourceLocation atLoc);
  StmtResult ParseObjCThrowStmt(SourceLocation atLoc);
  StmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);
  StmtResult ParseObjCAutoreleasePoolStmt(SourceLocation atLoc);


  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.

  /// A context for parsing declaration specifiers.  TODO: flesh this
  /// out, there are other significant restrictions on specifiers than
  /// would be best implemented in the parser.
  enum DeclSpecContext {
    DSC_normal, // normal context
    DSC_class,  // class context, enables 'friend'
    DSC_type_specifier, // C++ type-specifier-seq or C specifier-qualifier-list
    DSC_trailing, // C++11 trailing-type-specifier in a trailing return type
    DSC_top_level // top-level/namespace declaration context
  };

  /// Information on a C++0x for-range-initializer found while parsing a
  /// declaration which turns out to be a for-range-declaration.
  struct ForRangeInit {
    SourceLocation ColonLoc;
    ExprResult RangeExpr;

    bool ParsedForRangeDecl() { return !ColonLoc.isInvalid(); }
  };

  DeclGroupPtrTy ParseDeclaration(StmtVector &Stmts,
                                  unsigned Context, SourceLocation &DeclEnd,
                                  ParsedAttributesWithRange &attrs);
  DeclGroupPtrTy ParseSimpleDeclaration(StmtVector &Stmts,
                                        unsigned Context,
                                        SourceLocation &DeclEnd,
                                        ParsedAttributesWithRange &attrs,
                                        bool RequireSemi,
                                        ForRangeInit *FRI = 0);
  bool MightBeDeclarator(unsigned Context);
  DeclGroupPtrTy ParseDeclGroup(ParsingDeclSpec &DS, unsigned Context,
                                bool AllowFunctionDefinitions,
                                SourceLocation *DeclEnd = 0,
                                ForRangeInit *FRI = 0);
  Decl *ParseDeclarationAfterDeclarator(Declarator &D,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  bool ParseAsmAttributesAfterDeclarator(Declarator &D);
  Decl *ParseDeclarationAfterDeclaratorAndAttributes(Declarator &D,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  Decl *ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope);
  Decl *ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope);

  /// \brief When in code-completion, skip parsing of the function/method body
  /// unless the body contains the code-completion point.
  ///
  /// \returns true if the function body was skipped.
  bool trySkippingFunctionBody();

  bool ParseImplicitInt(DeclSpec &DS, CXXScopeSpec *SS,
                        const ParsedTemplateInfo &TemplateInfo,
                        AccessSpecifier AS, DeclSpecContext DSC);
  DeclSpecContext getDeclSpecContextFromDeclaratorContext(unsigned Context);
  void ParseDeclarationSpecifiers(DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                  AccessSpecifier AS = AS_none,
                                  DeclSpecContext DSC = DSC_normal,
                                  LateParsedAttrList *LateAttrs = 0);

  void ParseSpecifierQualifierList(DeclSpec &DS, AccessSpecifier AS = AS_none,
                                   DeclSpecContext DSC = DSC_normal);

  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS,
                                  Declarator::TheContext Context);

  void ParseEnumSpecifier(SourceLocation TagLoc, DeclSpec &DS,
                          const ParsedTemplateInfo &TemplateInfo,
                          AccessSpecifier AS, DeclSpecContext DSC);
  void ParseEnumBody(SourceLocation StartLoc, Decl *TagDecl);
  void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
                            Decl *TagDecl);

  struct FieldCallback {
    virtual void invoke(ParsingFieldDeclarator &Field) = 0;
    virtual ~FieldCallback() {}

  private:
    virtual void _anchor();
  };
  struct ObjCPropertyCallback;

  void ParseStructDeclaration(ParsingDeclSpec &DS, FieldCallback &Callback);

  bool isDeclarationSpecifier(bool DisambiguatingWithExpression = false);
  bool isTypeSpecifierQualifier();
  bool isTypeQualifier() const;

  /// isKnownToBeTypeSpecifier - Return true if we know that the specified token
  /// is definitely a type-specifier.  Return false if it isn't part of a type
  /// specifier or if we're not sure.
  bool isKnownToBeTypeSpecifier(const Token &Tok) const;

  /// \brief Return true if we know that we are definitely looking at a
  /// decl-specifier, and isn't part of an expression such as a function-style
  /// cast. Return false if it's no a decl-specifier, or we're not sure.
  bool isKnownToBeDeclarationSpecifier() {
    if (getLangOpts().CPlusPlus)
      return isCXXDeclarationSpecifier() == TPResult::True();
    return isDeclarationSpecifier(true);
  }

  /// isDeclarationStatement - Disambiguates between a declaration or an
  /// expression statement, when parsing function bodies.
  /// Returns true for declaration, false for expression.
  bool isDeclarationStatement() {
    if (getLangOpts().CPlusPlus)
      return isCXXDeclarationStatement();
    return isDeclarationSpecifier(true);
  }

  /// isForInitDeclaration - Disambiguates between a declaration or an
  /// expression in the context of the C 'clause-1' or the C++
  // 'for-init-statement' part of a 'for' statement.
  /// Returns true for declaration, false for expression.
  bool isForInitDeclaration() {
    if (getLangOpts().CPlusPlus)
      return isCXXSimpleDeclaration(/*AllowForRangeDecl=*/true);
    return isDeclarationSpecifier(true);
  }

  /// \brief Determine whether we are currently at the start of an Objective-C
  /// class message that appears to be missing the open bracket '['.
  bool isStartOfObjCClassMessageMissingOpenBracket();

  /// \brief Starting with a scope specifier, identifier, or
  /// template-id that refers to the current class, determine whether
  /// this is a constructor declarator.
  bool isConstructorDeclarator();

  /// \brief Specifies the context in which type-id/expression
  /// disambiguation will occur.
  enum TentativeCXXTypeIdContext {
    TypeIdInParens,
    TypeIdAsTemplateArgument
  };


  /// isTypeIdInParens - Assumes that a '(' was parsed and now we want to know
  /// whether the parens contain an expression or a type-id.
  /// Returns true for a type-id and false for an expression.
  bool isTypeIdInParens(bool &isAmbiguous) {
    if (getLangOpts().CPlusPlus)
      return isCXXTypeId(TypeIdInParens, isAmbiguous);
    isAmbiguous = false;
    return isTypeSpecifierQualifier();
  }
  bool isTypeIdInParens() {
    bool isAmbiguous;
    return isTypeIdInParens(isAmbiguous);
  }

  /// isCXXDeclarationStatement - C++-specialized function that disambiguates
  /// between a declaration or an expression statement, when parsing function
  /// bodies. Returns true for declaration, false for expression.
  bool isCXXDeclarationStatement();

  /// isCXXSimpleDeclaration - C++-specialized function that disambiguates
  /// between a simple-declaration or an expression-statement.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  /// Returns false if the statement is disambiguated as expression.
  bool isCXXSimpleDeclaration(bool AllowForRangeDecl);

  /// isCXXFunctionDeclarator - Disambiguates between a function declarator or
  /// a constructor-style initializer, when parsing declaration statements.
  /// Returns true for function declarator and false for constructor-style
  /// initializer. Sets 'IsAmbiguous' to true to indicate that this declaration 
  /// might be a constructor-style initializer.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXFunctionDeclarator(bool *IsAmbiguous = 0);

  /// isCXXConditionDeclaration - Disambiguates between a declaration or an
  /// expression for a condition of a if/switch/while/for statement.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXConditionDeclaration();

  bool isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous);
  bool isCXXTypeId(TentativeCXXTypeIdContext Context) {
    bool isAmbiguous;
    return isCXXTypeId(Context, isAmbiguous);
  }

  /// TPResult - Used as the result value for functions whose purpose is to
  /// disambiguate C++ constructs by "tentatively parsing" them.
  /// This is a class instead of a simple enum because the implicit enum-to-bool
  /// conversions may cause subtle bugs.
  class TPResult {
    enum Result {
      TPR_true,
      TPR_false,
      TPR_ambiguous,
      TPR_error
    };
    Result Res;
    TPResult(Result result) : Res(result) {}
  public:
    static TPResult True() { return TPR_true; }
    static TPResult False() { return TPR_false; }
    static TPResult Ambiguous() { return TPR_ambiguous; }
    static TPResult Error() { return TPR_error; }

    bool operator==(const TPResult &RHS) const { return Res == RHS.Res; }
    bool operator!=(const TPResult &RHS) const { return Res != RHS.Res; }
  };

  /// \brief Based only on the given token kind, determine whether we know that
  /// we're at the start of an expression or a type-specifier-seq (which may
  /// be an expression, in C++).
  ///
  /// This routine does not attempt to resolve any of the trick cases, e.g.,
  /// those involving lookup of identifiers.
  ///
  /// \returns \c TPR_true if this token starts an expression, \c TPR_false if
  /// this token starts a type-specifier-seq, or \c TPR_ambiguous if it cannot
  /// tell.
  TPResult isExpressionOrTypeSpecifierSimple(tok::TokenKind Kind);

  /// isCXXDeclarationSpecifier - Returns TPResult::True() if it is a
  /// declaration specifier, TPResult::False() if it is not,
  /// TPResult::Ambiguous() if it could be either a decl-specifier or a
  /// function-style cast, and TPResult::Error() if a parsing error was
  /// encountered. If it could be a braced C++11 function-style cast, returns
  /// BracedCastResult.
  /// Doesn't consume tokens.
  TPResult
  isCXXDeclarationSpecifier(TPResult BracedCastResult = TPResult::False(),
                            bool *HasMissingTypename = 0);

  /// \brief Determine whether an identifier has been tentatively declared as a
  /// non-type. Such tentative declarations should not be found to name a type
  /// during a tentative parse, but also should not be annotated as a non-type.
  bool isTentativelyDeclared(IdentifierInfo *II);

  // "Tentative parsing" functions, used for disambiguation. If a parsing error
  // is encountered they will return TPResult::Error().
  // Returning TPResult::True()/False() indicates that the ambiguity was
  // resolved and tentative parsing may stop. TPResult::Ambiguous() indicates
  // that more tentative parsing is necessary for disambiguation.
  // They all consume tokens, so backtracking should be used after calling them.

  TPResult TryParseDeclarationSpecifier(bool *HasMissingTypename = 0);
  TPResult TryParseSimpleDeclaration(bool AllowForRangeDecl);
  TPResult TryParseTypeofSpecifier();
  TPResult TryParseProtocolQualifiers();
  TPResult TryParseInitDeclaratorList();
  TPResult TryParseDeclarator(bool mayBeAbstract, bool mayHaveIdentifier=true);
  TPResult TryParseParameterDeclarationClause(bool *InvalidAsDeclaration = 0);
  TPResult TryParseFunctionDeclarator();
  TPResult TryParseBracketDeclarator();

public:
  TypeResult ParseTypeName(SourceRange *Range = 0,
                           Declarator::TheContext Context
                             = Declarator::TypeNameContext,
                           AccessSpecifier AS = AS_none,
                           Decl **OwnedType = 0);

private:
  void ParseBlockId(SourceLocation CaretLoc);

  // Check for the start of a C++11 attribute-specifier-seq in a context where
  // an attribute is not allowed.
  bool CheckProhibitedCXX11Attribute() {
    assert(Tok.is(tok::l_square));
    if (!getLangOpts().CPlusPlus0x || NextToken().isNot(tok::l_square))
      return false;
    return DiagnoseProhibitedCXX11Attribute();
  }
  bool DiagnoseProhibitedCXX11Attribute();

  void ProhibitAttributes(ParsedAttributesWithRange &attrs) {
    if (!attrs.Range.isValid()) return;
    DiagnoseProhibitedAttributes(attrs);
    attrs.clear();
  }
  void DiagnoseProhibitedAttributes(ParsedAttributesWithRange &attrs);

  // Forbid C++11 attributes that appear on certain syntactic 
  // locations which standard permits but we don't supported yet, 
  // for example, attributes appertain to decl specifiers.
  void ProhibitCXX11Attributes(ParsedAttributesWithRange &attrs);

  void MaybeParseGNUAttributes(Declarator &D,
                               LateParsedAttrList *LateAttrs = 0) {
    if (Tok.is(tok::kw___attribute)) {
      ParsedAttributes attrs(AttrFactory);
      SourceLocation endLoc;
      ParseGNUAttributes(attrs, &endLoc, LateAttrs);
      D.takeAttributes(attrs, endLoc);
    }
  }
  void MaybeParseGNUAttributes(ParsedAttributes &attrs,
                               SourceLocation *endLoc = 0,
                               LateParsedAttrList *LateAttrs = 0) {
    if (Tok.is(tok::kw___attribute))
      ParseGNUAttributes(attrs, endLoc, LateAttrs);
  }
  void ParseGNUAttributes(ParsedAttributes &attrs,
                          SourceLocation *endLoc = 0,
                          LateParsedAttrList *LateAttrs = 0);
  void ParseGNUAttributeArgs(IdentifierInfo *AttrName,
                             SourceLocation AttrNameLoc,
                             ParsedAttributes &Attrs,
                             SourceLocation *EndLoc,
                             IdentifierInfo *ScopeName,
                             SourceLocation ScopeLoc,
                             AttributeList::Syntax Syntax);

  void MaybeParseCXX0XAttributes(Declarator &D) {
    if (getLangOpts().CPlusPlus0x && isCXX11AttributeSpecifier()) {
      ParsedAttributesWithRange attrs(AttrFactory);
      SourceLocation endLoc;
      ParseCXX11Attributes(attrs, &endLoc);
      D.takeAttributes(attrs, endLoc);
    }
  }
  void MaybeParseCXX0XAttributes(ParsedAttributes &attrs,
                                 SourceLocation *endLoc = 0) {
    if (getLangOpts().CPlusPlus0x && isCXX11AttributeSpecifier()) {
      ParsedAttributesWithRange attrsWithRange(AttrFactory);
      ParseCXX11Attributes(attrsWithRange, endLoc);
      attrs.takeAllFrom(attrsWithRange);
    }
  }
  void MaybeParseCXX0XAttributes(ParsedAttributesWithRange &attrs,
                                 SourceLocation *endLoc = 0,
                                 bool OuterMightBeMessageSend = false) {
    if (getLangOpts().CPlusPlus0x &&
        isCXX11AttributeSpecifier(false, OuterMightBeMessageSend))
      ParseCXX11Attributes(attrs, endLoc);
  }

  void ParseCXX11AttributeSpecifier(ParsedAttributes &attrs,
                                    SourceLocation *EndLoc = 0);
  void ParseCXX11Attributes(ParsedAttributesWithRange &attrs,
                            SourceLocation *EndLoc = 0);

  IdentifierInfo *TryParseCXX11AttributeIdentifier(SourceLocation &Loc);

  void MaybeParseMicrosoftAttributes(ParsedAttributes &attrs,
                                     SourceLocation *endLoc = 0) {
    if (getLangOpts().MicrosoftExt && Tok.is(tok::l_square))
      ParseMicrosoftAttributes(attrs, endLoc);
  }
  void ParseMicrosoftAttributes(ParsedAttributes &attrs,
                                SourceLocation *endLoc = 0);
  void ParseMicrosoftDeclSpec(ParsedAttributes &Attrs);
  bool IsSimpleMicrosoftDeclSpec(IdentifierInfo *Ident);
  void ParseComplexMicrosoftDeclSpec(IdentifierInfo *Ident, 
                                     SourceLocation Loc,
                                     ParsedAttributes &Attrs);
  void ParseMicrosoftDeclSpecWithSingleArg(IdentifierInfo *AttrName, 
                                           SourceLocation AttrNameLoc, 
                                           ParsedAttributes &Attrs);
  void ParseMicrosoftTypeAttributes(ParsedAttributes &attrs);
  void ParseMicrosoftInheritanceClassAttributes(ParsedAttributes &attrs);
  void ParseBorlandTypeAttributes(ParsedAttributes &attrs);
  void ParseOpenCLAttributes(ParsedAttributes &attrs);
  void ParseOpenCLQualifiers(DeclSpec &DS);

  VersionTuple ParseVersionTuple(SourceRange &Range);
  void ParseAvailabilityAttribute(IdentifierInfo &Availability,
                                  SourceLocation AvailabilityLoc,
                                  ParsedAttributes &attrs,
                                  SourceLocation *endLoc);

  bool IsThreadSafetyAttribute(llvm::StringRef AttrName);
  void ParseThreadSafetyAttribute(IdentifierInfo &AttrName,
                                  SourceLocation AttrNameLoc,
                                  ParsedAttributes &Attrs,
                                  SourceLocation *EndLoc);

  void ParseTypeTagForDatatypeAttribute(IdentifierInfo &AttrName,
                                        SourceLocation AttrNameLoc,
                                        ParsedAttributes &Attrs,
                                        SourceLocation *EndLoc);

  void ParseTypeofSpecifier(DeclSpec &DS);
  SourceLocation ParseDecltypeSpecifier(DeclSpec &DS);
  void AnnotateExistingDecltypeSpecifier(const DeclSpec &DS,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc);
  void ParseUnderlyingTypeSpecifier(DeclSpec &DS);
  void ParseAtomicSpecifier(DeclSpec &DS);

  ExprResult ParseAlignArgument(SourceLocation Start,
                                SourceLocation &EllipsisLoc);
  void ParseAlignmentSpecifier(ParsedAttributes &Attrs,
                               SourceLocation *endLoc = 0);

  VirtSpecifiers::Specifier isCXX0XVirtSpecifier(const Token &Tok) const;
  VirtSpecifiers::Specifier isCXX0XVirtSpecifier() const {
    return isCXX0XVirtSpecifier(Tok);
  }
  void ParseOptionalCXX0XVirtSpecifierSeq(VirtSpecifiers &VS, bool IsInterface);

  bool isCXX0XFinalKeyword() const;

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
  typedef void (Parser::*DirectDeclParseFunction)(Declarator&);
  void ParseDeclaratorInternal(Declarator &D,
                               DirectDeclParseFunction DirectDeclParser);

  void ParseTypeQualifierListOpt(DeclSpec &DS, bool GNUAttributesAllowed = true,
                                 bool CXX0XAttributesAllowed = true);
  void ParseDirectDeclarator(Declarator &D);
  void ParseParenDeclarator(Declarator &D);
  void ParseFunctionDeclarator(Declarator &D,
                               ParsedAttributes &attrs,
                               BalancedDelimiterTracker &Tracker,
                               bool IsAmbiguous,
                               bool RequiresArg = false);
  bool isFunctionDeclaratorIdentifierList();
  void ParseFunctionDeclaratorIdentifierList(
         Declarator &D,
         SmallVector<DeclaratorChunk::ParamInfo, 16> &ParamInfo);
  void ParseParameterDeclarationClause(
         Declarator &D,
         ParsedAttributes &attrs,
         SmallVector<DeclaratorChunk::ParamInfo, 16> &ParamInfo,
         SourceLocation &EllipsisLoc);
  void ParseBracketDeclarator(Declarator &D);

  //===--------------------------------------------------------------------===//
  // C++ 7: Declarations [dcl.dcl]

  /// The kind of attribute specifier we have found.
  enum CXX11AttributeKind {
    /// This is not an attribute specifier.
    CAK_NotAttributeSpecifier,
    /// This should be treated as an attribute-specifier.
    CAK_AttributeSpecifier,
    /// The next tokens are '[[', but this is not an attribute-specifier. This
    /// is ill-formed by C++11 [dcl.attr.grammar]p6.
    CAK_InvalidAttributeSpecifier
  };
  CXX11AttributeKind
  isCXX11AttributeSpecifier(bool Disambiguate = false,
                            bool OuterMightBeMessageSend = false);

  Decl *ParseNamespace(unsigned Context, SourceLocation &DeclEnd,
                       SourceLocation InlineLoc = SourceLocation());
  void ParseInnerNamespace(std::vector<SourceLocation>& IdentLoc,
                           std::vector<IdentifierInfo*>& Ident,
                           std::vector<SourceLocation>& NamespaceLoc,
                           unsigned int index, SourceLocation& InlineLoc,
                           ParsedAttributes& attrs,
                           BalancedDelimiterTracker &Tracker);
  Decl *ParseLinkage(ParsingDeclSpec &DS, unsigned Context);
  Decl *ParseUsingDirectiveOrDeclaration(unsigned Context,
                                         const ParsedTemplateInfo &TemplateInfo,
                                         SourceLocation &DeclEnd,
                                         ParsedAttributesWithRange &attrs,
                                         Decl **OwnedType = 0);
  Decl *ParseUsingDirective(unsigned Context,
                            SourceLocation UsingLoc,
                            SourceLocation &DeclEnd,
                            ParsedAttributes &attrs);
  Decl *ParseUsingDeclaration(unsigned Context,
                              const ParsedTemplateInfo &TemplateInfo,
                              SourceLocation UsingLoc,
                              SourceLocation &DeclEnd,
                              AccessSpecifier AS = AS_none,
                              Decl **OwnedType = 0);
  Decl *ParseStaticAssertDeclaration(SourceLocation &DeclEnd);
  Decl *ParseNamespaceAlias(SourceLocation NamespaceLoc,
                            SourceLocation AliasLoc, IdentifierInfo *Alias,
                            SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.
  bool isValidAfterTypeSpecifier(bool CouldBeBitfield);
  void ParseClassSpecifier(tok::TokenKind TagTokKind, SourceLocation TagLoc,
                           DeclSpec &DS, const ParsedTemplateInfo &TemplateInfo,
                           AccessSpecifier AS, bool EnteringContext,
                           DeclSpecContext DSC);
  void ParseCXXMemberSpecification(SourceLocation StartLoc, unsigned TagType,
                                   Decl *TagDecl);
  ExprResult ParseCXXMemberInitializer(Decl *D, bool IsFunction,
                                       SourceLocation &EqualLoc);
  void ParseCXXClassMemberDeclaration(AccessSpecifier AS, AttributeList *Attr,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                 ParsingDeclRAIIObject *DiagsFromTParams = 0);
  void ParseConstructorInitializer(Decl *ConstructorDecl);
  MemInitResult ParseMemInitializer(Decl *ConstructorDecl);
  void HandleMemberFunctionDeclDelays(Declarator& DeclaratorInfo,
                                      Decl *ThisDecl);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]
  TypeResult ParseBaseTypeSpecifier(SourceLocation &BaseLoc,
                                    SourceLocation &EndLocation);
  void ParseBaseClause(Decl *ClassDecl);
  BaseResult ParseBaseSpecifier(Decl *ClassDecl);
  AccessSpecifier getAccessSpecifierIfPresent() const;

  bool ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS,
                                    SourceLocation TemplateKWLoc,
                                    IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    bool EnteringContext,
                                    ParsedType ObjectType,
                                    UnqualifiedId &Id,
                                    bool AssumeTemplateId);
  bool ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                  ParsedType ObjectType,
                                  UnqualifiedId &Result);

public:
  bool ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
                          bool AllowDestructorName,
                          bool AllowConstructorName,
                          ParsedType ObjectType,
                          SourceLocation& TemplateKWLoc,
                          UnqualifiedId &Result);

private:
  //===--------------------------------------------------------------------===//
  // C++ 14: Templates [temp]

  // C++ 14.1: Template Parameters [temp.param]
  Decl *ParseDeclarationStartingWithTemplate(unsigned Context,
                                             SourceLocation &DeclEnd,
                                             AccessSpecifier AS = AS_none,
                                             AttributeList *AccessAttrs = 0);
  Decl *ParseTemplateDeclarationOrSpecialization(unsigned Context,
                                                 SourceLocation &DeclEnd,
                                                 AccessSpecifier AS,
                                                 AttributeList *AccessAttrs);
  Decl *ParseSingleDeclarationAfterTemplate(
                                       unsigned Context,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       ParsingDeclRAIIObject &DiagsFromParams,
                                       SourceLocation &DeclEnd,
                                       AccessSpecifier AS=AS_none,
                                       AttributeList *AccessAttrs = 0);
  bool ParseTemplateParameters(unsigned Depth,
                               SmallVectorImpl<Decl*> &TemplateParams,
                               SourceLocation &LAngleLoc,
                               SourceLocation &RAngleLoc);
  bool ParseTemplateParameterList(unsigned Depth,
                                  SmallVectorImpl<Decl*> &TemplateParams);
  bool isStartOfTemplateTypeParameter();
  Decl *ParseTemplateParameter(unsigned Depth, unsigned Position);
  Decl *ParseTypeParameter(unsigned Depth, unsigned Position);
  Decl *ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);
  Decl *ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);
  // C++ 14.3: Template arguments [temp.arg]
  typedef SmallVector<ParsedTemplateArgument, 16> TemplateArgList;

  bool ParseTemplateIdAfterTemplateName(TemplateTy Template,
                                        SourceLocation TemplateNameLoc,
                                        const CXXScopeSpec &SS,
                                        bool ConsumeLastToken,
                                        SourceLocation &LAngleLoc,
                                        TemplateArgList &TemplateArgs,
                                        SourceLocation &RAngleLoc);

  bool AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
                               CXXScopeSpec &SS,
                               SourceLocation TemplateKWLoc,
                               UnqualifiedId &TemplateName,
                               bool AllowTypeAnnotation = true);
  void AnnotateTemplateIdTokenAsType();
  bool IsTemplateArgumentList(unsigned Skip = 0);
  bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs);
  ParsedTemplateArgument ParseTemplateTemplateArgument();
  ParsedTemplateArgument ParseTemplateArgument();
  Decl *ParseExplicitInstantiation(unsigned Context,
                                   SourceLocation ExternLoc,
                                   SourceLocation TemplateLoc,
                                   SourceLocation &DeclEnd,
                                   AccessSpecifier AS = AS_none);

  //===--------------------------------------------------------------------===//
  // Modules
  DeclGroupPtrTy ParseModuleImport(SourceLocation AtLoc);

  //===--------------------------------------------------------------------===//
  // GNU G++: Type Traits [Type-Traits.html in the GCC manual]
  ExprResult ParseUnaryTypeTrait();
  ExprResult ParseBinaryTypeTrait();
  ExprResult ParseTypeTrait();
  
  //===--------------------------------------------------------------------===//
  // Embarcadero: Arary and Expression Traits
  ExprResult ParseArrayTypeTrait();
  ExprResult ParseExpressionTrait();

  //===--------------------------------------------------------------------===//
  // Preprocessor code-completion pass-through
  virtual void CodeCompleteDirective(bool InConditional);
  virtual void CodeCompleteInConditionalExclusion();
  virtual void CodeCompleteMacroName(bool IsDefinition);
  virtual void CodeCompletePreprocessorExpression();
  virtual void CodeCompleteMacroArgument(IdentifierInfo *Macro,
                                         MacroInfo *MacroInfo,
                                         unsigned ArgumentIndex);
  virtual void CodeCompleteNaturalLanguage();
};

}  // end namespace clang

#endif
