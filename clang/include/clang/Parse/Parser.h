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
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/ADT/OwningPtr.h"
#include <stack>
#include <list>

namespace clang {
  class PragmaHandler;
  class Scope;
  class DeclGroupRef;
  class DiagnosticBuilder;
  class Parser;
  class PragmaUnusedHandler;
  class ColonProtectionRAIIObject;
  class InMessageExpressionRAIIObject;
  
/// PrettyStackTraceParserEntry - If a crash happens while the parser is active,
/// an entry is printed for it.
class PrettyStackTraceParserEntry : public llvm::PrettyStackTraceEntry {
  const Parser &P;
public:
  PrettyStackTraceParserEntry(const Parser &p) : P(p) {}
  virtual void print(llvm::raw_ostream &OS) const;
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
  friend class ParenBraceBracketBalancer;
  PrettyStackTraceParserEntry CrashInfo;

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

  Diagnostic &Diags;

  /// ScopeCache - Cache scopes to reduce malloc traffic.
  enum { ScopeCacheSize = 16 };
  unsigned NumCachedScopes;
  Scope *ScopeCache[ScopeCacheSize];

  /// Ident_super - IdentifierInfo for "super", to support fast
  /// comparison.
  IdentifierInfo *Ident_super;
  /// Ident_vector and Ident_pixel - cached IdentifierInfo's for
  /// "vector" and "pixel" fast comparison.  Only present if
  /// AltiVec enabled.
  IdentifierInfo *Ident_vector;
  IdentifierInfo *Ident_pixel;

  /// C++0x contextual keywords. 
  mutable IdentifierInfo *Ident_final;
  mutable IdentifierInfo *Ident_override;

  llvm::OwningPtr<PragmaHandler> AlignHandler;
  llvm::OwningPtr<PragmaHandler> GCCVisibilityHandler;
  llvm::OwningPtr<PragmaHandler> OptionsHandler;
  llvm::OwningPtr<PragmaHandler> PackHandler;
  llvm::OwningPtr<PragmaHandler> UnusedHandler;
  llvm::OwningPtr<PragmaHandler> WeakHandler;
  llvm::OwningPtr<PragmaHandler> FPContractHandler;
  llvm::OwningPtr<PragmaHandler> OpenCLExtensionHandler;

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

  /// \brief When true, we are directly inside an Ojective-C messsage 
  /// send expression.
  ///
  /// This is managed by the \c InMessageExpressionRAIIObject class, and
  /// should not be set directly.
  bool InMessageExpression;
  
  /// The "depth" of the template parameters currently being parsed.
  unsigned TemplateParameterDepth;
  
  /// Factory object for creating AttributeList objects.
  AttributeList::Factory AttrFactory;

public:
  Parser(Preprocessor &PP, Sema &Actions);
  ~Parser();

  const LangOptions &getLang() const { return PP.getLangOptions(); }
  const TargetInfo &getTargetInfo() const { return PP.getTargetInfo(); }
  Preprocessor &getPreprocessor() const { return PP; }
  Sema &getActions() const { return Actions; }

  const Token &getCurToken() const { return Tok; }
  Scope *getCurScope() const { return Actions.getCurScope(); }
  
  // Type forwarding.  All of these are statically 'void*', but they may all be
  // different actual classes based on the actions in place.
  typedef Expr ExprTy;
  typedef Stmt StmtTy;
  typedef OpaquePtr<DeclGroupRef> DeclGroupPtrTy;
  typedef CXXBaseSpecifier BaseTy;
  typedef CXXCtorInitializer MemInitTy;
  typedef NestedNameSpecifier CXXScopeTy;
  typedef TemplateParameterList TemplateParamsTy;
  typedef OpaquePtr<TemplateName> TemplateTy;

  typedef llvm::SmallVector<TemplateParameterList *, 4> TemplateParameterLists;

  typedef clang::ExprResult        ExprResult;
  typedef clang::StmtResult        StmtResult;
  typedef clang::BaseResult        BaseResult;
  typedef clang::MemInitResult     MemInitResult;
  typedef clang::TypeResult        TypeResult;

  typedef Expr *ExprArg;
  typedef ASTMultiPtr<Stmt*> MultiStmtArg;
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

  DeclGroupPtrTy FinishPendingObjCActions();

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
           Tok.getKind() == tok::wide_string_literal;
  }

  /// \brief Returns true if the current token is a '=' or '==' and
  /// false otherwise. If it's '==', we assume that it's a typo and we emit
  /// DiagID and a fixit hint to turn '==' -> '='.
  bool isTokenEqualOrMistypedEqualEqual(unsigned DiagID);

  /// ConsumeToken - Consume the current 'peek token' and lex the next one.
  /// This does not work with all kinds of tokens: strings and specific other
  /// tokens must be consumed with custom methods below.  This returns the
  /// location of the consumed token.
  SourceLocation ConsumeToken() {
    assert(!isTokenStringLiteral() && !isTokenParen() && !isTokenBracket() &&
           !isTokenBrace() &&
           "Should consume special tokens with Consume*Token");
    if (Tok.is(tok::code_completion)) {
      CodeCompletionRecovery();
      return ConsumeCodeCompletionToken();
    }
    
    PrevTokLocation = Tok.getLocation();
    PP.Lex(Tok);
    return PrevTokLocation;
  }

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
  
  ///\ brief When we are consuming a code-completion token within having 
  /// matched specific position in the grammar, provide code-completion results
  /// based on context.
  void CodeCompletionRecovery();

  /// \brief Handle the annotation token produced for #pragma unused(...)
  void HandlePragmaUnused();

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

  /// NextToken - This peeks ahead one token and returns it without
  /// consuming it.
  const Token &NextToken() {
    return PP.LookAhead(0);
  }

  /// getTypeAnnotation - Read a parsed type out of an annotation token.
  static ParsedType getTypeAnnotation(Token &Tok) {
    return ParsedType::getFromOpaquePtr(Tok.getAnnotationValue());
  }

  static void setTypeAnnotation(Token &Tok, ParsedType T) {
    Tok.setAnnotationValue(T.getAsOpaquePtr());
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
  ///
  /// This returns true if the token was annotated.
  bool TryAnnotateTypeOrScopeToken(bool EnteringContext = false);

  /// TryAnnotateCXXScopeToken - Like TryAnnotateTypeOrScopeToken but
  /// only annotates C++ scope specifiers.  This returns true if there
  /// was an unrecoverable error.
  bool TryAnnotateCXXScopeToken(bool EnteringContext = false);

  /// TryAltiVecToken - Check for context-sensitive AltiVec identifier tokens,
  /// replacing them with the non-context-sensitive keywords.  This returns
  /// true if the token was replaced.
  bool TryAltiVecToken(DeclSpec &DS, SourceLocation Loc,
                       const char *&PrevSpec, unsigned &DiagID,
                       bool &isInvalid) {
    if (!getLang().AltiVec ||
        (Tok.getIdentifierInfo() != Ident_vector &&
         Tok.getIdentifierInfo() != Ident_pixel))
      return false;
    
    return TryAltiVecTokenOutOfLine(DS, Loc, PrevSpec, DiagID, isInvalid);
  }

  /// TryAltiVecVectorToken - Check for context-sensitive AltiVec vector
  /// identifier token, replacing it with the non-context-sensitive __vector.
  /// This returns true if the token was replaced.
  bool TryAltiVecVectorToken() {
    if (!getLang().AltiVec ||
        Tok.getIdentifierInfo() != Ident_vector) return false;
    return TryAltiVecVectorTokenOutOfLine();
  }
  
  bool TryAltiVecVectorTokenOutOfLine();
  bool TryAltiVecTokenOutOfLine(DeclSpec &DS, SourceLocation Loc,
                                const char *&PrevSpec, unsigned &DiagID,
                                bool &isInvalid);
    
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
    bool isActive;

  public:
    explicit TentativeParsingAction(Parser& p) : P(p) {
      PrevTok = P.Tok;
      P.PP.EnableBacktrackAtThisPos();
      isActive = true;
    }
    void Commit() {
      assert(isActive && "Parsing action was finished!");
      P.PP.CommitBacktrackedTokens();
      isActive = false;
    }
    void Revert() {
      assert(isActive && "Parsing action was finished!");
      P.PP.Backtrack();
      P.Tok = PrevTok;
      isActive = false;
    }
    ~TentativeParsingAction() {
      assert(!isActive && "Forgot to call Commit or Revert!");
    }
  };


  /// MatchRHSPunctuation - For punctuation with a LHS and RHS (e.g. '['/']'),
  /// this helper function matches and consumes the specified RHS token if
  /// present.  If not present, it emits the specified diagnostic indicating
  /// that the parser failed to match the RHS of the token at LHSLoc.  LHSName
  /// should be the name of the unmatched LHS token.  This returns the location
  /// of the consumed token.
  SourceLocation MatchRHSPunctuation(tok::TokenKind RHSTok,
                                     SourceLocation LHSLoc);

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
    ParseScope(const ParseScope&); // do not implement
    ParseScope& operator=(const ParseScope&); // do not implement

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

  //===--------------------------------------------------------------------===//
  // Diagnostic Emission and Error recovery.

public:
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);
  DiagnosticBuilder Diag(const Token &Tok, unsigned DiagID);

private:
  void SuggestParentheses(SourceLocation Loc, unsigned DK,
                          SourceRange ParenRange);

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
    return SkipUntil(&T, 1, StopAtSemi, DontConsume, StopAtCodeCompletion);
  }
  bool SkipUntil(tok::TokenKind T1, tok::TokenKind T2, bool StopAtSemi = true,
                 bool DontConsume = false, bool StopAtCodeCompletion = false) {
    tok::TokenKind TokArray[] = {T1, T2};
    return SkipUntil(TokArray, 2, StopAtSemi, DontConsume,StopAtCodeCompletion);
  }
  bool SkipUntil(const tok::TokenKind *Toks, unsigned NumToks,
                 bool StopAtSemi = true, bool DontConsume = false,
                 bool StopAtCodeCompletion = false);

  //===--------------------------------------------------------------------===//
  // Lexing and parsing of C++ inline methods.

  struct ParsingClass;

  /// [class.mem]p1: "... the class is regarded as complete within
  /// - function bodies
  /// - default arguments
  /// - exception-specifications (TODO: C++0x)
  /// - and brace-or-equal-initializers (TODO: C++0x)
  /// for non-static data members (including such things in nested classes)."
  /// LateParsedDeclarations build the tree of those elements so they can
  /// be parsed after parsing the top-level class.
  class LateParsedDeclaration {
  public:
    virtual ~LateParsedDeclaration();

    virtual void ParseLexedMethodDeclarations();
    virtual void ParseLexedMethodDefs();
  };

  /// Inner node of the LateParsedDeclaration tree that parses
  /// all its members recursively.
  class LateParsedClass : public LateParsedDeclaration {
  public:
    LateParsedClass(Parser *P, ParsingClass *C);
    virtual ~LateParsedClass();

    virtual void ParseLexedMethodDeclarations();
    virtual void ParseLexedMethodDefs();

  private:
    Parser *Self;
    ParsingClass *Class;
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
      : Self(P), Method(M), TemplateScope(false) { }

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
    llvm::SmallVector<LateParsedDefaultArgument, 8> DefaultArgs;
  };

  /// LateParsedDeclarationsContainer - During parsing of a top (non-nested)
  /// C++ class, its method declarations that contain parts that won't be
  /// parsed until after the definiton is completed (C++ [class.mem]p2),
  /// the method declarations and possibly attached inline definitions
  /// will be stored here with the tokens that will be parsed to create those entities.
  typedef llvm::SmallVector<LateParsedDeclaration*, 2> LateParsedDeclarationsContainer;

  /// \brief Representation of a class that has been parsed, including
  /// any member function declarations or definitions that need to be
  /// parsed after the corresponding top-level class is complete.
  struct ParsingClass {
    ParsingClass(Decl *TagOrTemplate, bool TopLevelClass)
      : TopLevelClass(TopLevelClass), TemplateScope(false),
        TagOrTemplate(TagOrTemplate) { }

    /// \brief Whether this is a "top-level" class, meaning that it is
    /// not nested within another class.
    bool TopLevelClass : 1;

    /// \brief Whether this class had an associated template
    /// scope. When true, TagOrTemplate is a template declaration;
    /// othewise, it is a tag declaration.
    bool TemplateScope : 1;

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

  /// \brief RAII object used to inform the actions that we're
  /// currently parsing a declaration.  This is active when parsing a
  /// variable's initializer, but not when parsing the body of a
  /// class or function definition.
  class ParsingDeclRAIIObject {
    Sema &Actions;
    Sema::ParsingDeclState State;
    bool Popped;

  public:
    ParsingDeclRAIIObject(Parser &P) : Actions(P.Actions) {
      push();
    }

    ParsingDeclRAIIObject(Parser &P, ParsingDeclRAIIObject *Other)
        : Actions(P.Actions) {
      if (Other) steal(*Other);
      else push();
    }

    /// Creates a RAII object which steals the state from a different
    /// object instead of pushing.
    ParsingDeclRAIIObject(ParsingDeclRAIIObject &Other)
        : Actions(Other.Actions) {
      steal(Other);
    }

    ~ParsingDeclRAIIObject() {
      abort();
    }

    /// Resets the RAII object for a new declaration.
    void reset() {
      abort();
      push();
    }

    /// Signals that the context was completed without an appropriate
    /// declaration being parsed.
    void abort() {
      pop(0);
    }

    void complete(Decl *D) {
      assert(!Popped && "ParsingDeclaration has already been popped!");
      pop(D);
    }

  private:
    void steal(ParsingDeclRAIIObject &Other) {
      State = Other.State;
      Popped = Other.Popped;
      Other.Popped = true;
    }
    
    void push() {
      State = Actions.PushParsingDeclaration();
      Popped = false;
    }

    void pop(Decl *D) {
      if (!Popped) {
        Actions.PopParsingDeclaration(State, D);
        Popped = true;
      }
    }
  };

  /// A class for parsing a DeclSpec.
  class ParsingDeclSpec : public DeclSpec {
    ParsingDeclRAIIObject ParsingRAII;

  public:
    ParsingDeclSpec(Parser &P) : ParsingRAII(P) {}
    ParsingDeclSpec(ParsingDeclRAIIObject &RAII) : ParsingRAII(RAII) {}
    ParsingDeclSpec(Parser &P, ParsingDeclRAIIObject *RAII)
      : ParsingRAII(P, RAII) {}

    void complete(Decl *D) {
      ParsingRAII.complete(D);
    }

    void abort() {
      ParsingRAII.abort();
    }
  };

  /// A class for parsing a declarator.
  class ParsingDeclarator : public Declarator {
    ParsingDeclRAIIObject ParsingRAII;

  public:
    ParsingDeclarator(Parser &P, const ParsingDeclSpec &DS, TheContext C)
      : Declarator(DS, C), ParsingRAII(P) {
    }

    const ParsingDeclSpec &getDeclSpec() const {
      return static_cast<const ParsingDeclSpec&>(Declarator::getDeclSpec());
    }

    ParsingDeclSpec &getMutableDeclSpec() const {
      return const_cast<ParsingDeclSpec&>(getDeclSpec());
    }

    void clear() {
      Declarator::clear();
      ParsingRAII.reset();
    }

    void complete(Decl *D) {
      ParsingRAII.complete(D);
    }
  };

  /// \brief RAII object used to
  class ParsingClassDefinition {
    Parser &P;
    bool Popped;
    Sema::ParsingClassState State;

  public:
    ParsingClassDefinition(Parser &P, Decl *TagOrTemplate, bool TopLevelClass)
      : P(P), Popped(false),
        State(P.PushParsingClass(TagOrTemplate, TopLevelClass)) {
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

    SourceRange getSourceRange() const;
  };

  Sema::ParsingClassState
  PushParsingClass(Decl *TagOrTemplate, bool TopLevelClass);
  void DeallocateParsedClasses(ParsingClass *Class);
  void PopParsingClass(Sema::ParsingClassState);

  Decl *ParseCXXInlineMethodDef(AccessSpecifier AS, ParsingDeclarator &D,
                                const ParsedTemplateInfo &TemplateInfo,
                                const VirtSpecifiers& VS);
  void ParseLexedMethodDeclarations(ParsingClass &Class);
  void ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM);
  void ParseLexedMethodDefs(ParsingClass &Class);
  void ParseLexedMethodDef(LexedMethod &LM);
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
    SourceRange Range;
  };

  DeclGroupPtrTy ParseExternalDeclaration(ParsedAttributesWithRange &attrs,
                                          ParsingDeclSpec *DS = 0);
  bool isDeclarationAfterDeclarator() const;
  bool isStartOfFunctionDefinition(const ParsingDeclarator &Declarator);
  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(ParsedAttributes &attrs,
                                                  AccessSpecifier AS = AS_none);
  DeclGroupPtrTy ParseDeclarationOrFunctionDefinition(ParsingDeclSpec &DS,
                                                  AccessSpecifier AS = AS_none);
  
  Decl *ParseFunctionDefinition(ParsingDeclarator &D,
                 const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  void ParseKNRParamDeclarations(Declarator &D);
  // EndLoc, if non-NULL, is filled with the location of the last token of
  // the simple-asm.
  ExprResult ParseSimpleAsm(SourceLocation *EndLoc = 0);
  ExprResult ParseAsmStringLiteral();

  // Objective-C External Declarations
  Decl *ParseObjCAtDirectives();
  Decl *ParseObjCAtClassDeclaration(SourceLocation atLoc);
  Decl *ParseObjCAtInterfaceDeclaration(SourceLocation atLoc,
                                        ParsedAttributes &prefixAttrs);
  void ParseObjCClassInstanceVariables(Decl *interfaceDecl,
                                       tok::ObjCKeywordKind visibility,
                                       SourceLocation atLoc);
  bool ParseObjCProtocolReferences(llvm::SmallVectorImpl<Decl *> &P,
                                   llvm::SmallVectorImpl<SourceLocation> &PLocs,
                                   bool WarnOnDeclarations,
                                   SourceLocation &LAngleLoc,
                                   SourceLocation &EndProtoLoc);
  bool ParseObjCProtocolQualifiers(DeclSpec &DS);
  void ParseObjCInterfaceDeclList(Decl *interfaceDecl,
                                  tok::ObjCKeywordKind contextKey);
  Decl *ParseObjCAtProtocolDeclaration(SourceLocation atLoc,
                                       ParsedAttributes &prefixAttrs);

  Decl *ObjCImpDecl;
  llvm::SmallVector<Decl *, 4> PendingObjCImpDecl;

  Decl *ParseObjCAtImplementationDeclaration(SourceLocation atLoc);
  Decl *ParseObjCAtEndDeclaration(SourceRange atEnd);
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

  ParsedType ParseObjCTypeName(ObjCDeclSpec &DS, bool IsParameter);
  void ParseObjCMethodRequirement();
  Decl *ParseObjCMethodPrototype(Decl *classOrCat,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  Decl *ParseObjCMethodDecl(SourceLocation mLoc, tok::TokenKind mType,
                                Decl *classDecl,
            tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword);
  void ParseObjCPropertyAttribute(ObjCDeclSpec &DS, Decl *ClassDecl);

  Decl *ParseObjCMethodDefinition();

  //===--------------------------------------------------------------------===//
  // C99 6.5: Expressions.
  
  ExprResult ParseExpression();
  ExprResult ParseConstantExpression();
  // Expr that doesn't include commas.
  ExprResult ParseAssignmentExpression();

  ExprResult ParseExpressionWithLeadingAt(SourceLocation AtLoc);

  ExprResult ParseExpressionWithLeadingExtension(SourceLocation ExtLoc);

  ExprResult ParseRHSOfBinaryExpression(ExprResult LHS,
                                              prec::Level MinPrec);
  ExprResult ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand,
                                       bool &NotCastExpr,
                                       ParsedType TypeOfCast);
  ExprResult ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand = false,
                                       ParsedType TypeOfCast = ParsedType());

  /// Returns true if the next token would start a postfix-expression
  /// suffix.
  bool isPostfixExpressionSuffixStart() {
    tok::TokenKind K = Tok.getKind();
    return (K == tok::l_square || K == tok::l_paren ||
            K == tok::period || K == tok::arrow ||
            K == tok::plusplus || K == tok::minusminus);
  }

  ExprResult ParsePostfixExpressionSuffix(ExprResult LHS);
  ExprResult ParseSizeofAlignofExpression();
  ExprResult ParseBuiltinPrimaryExpression();

  ExprResult ParseExprAfterTypeofSizeofAlignof(const Token &OpTok,
                                                     bool &isCastExpr,
                                                     ParsedType &CastTy,
                                                     SourceRange &CastRange);

  typedef llvm::SmallVector<Expr*, 20> ExprListTy;
  typedef llvm::SmallVector<SourceLocation, 20> CommaLocsTy;

  /// ParseExpressionList - Used for C/C++ (argument-)expression-list.
  bool ParseExpressionList(llvm::SmallVectorImpl<Expr*> &Exprs,
                           llvm::SmallVectorImpl<SourceLocation> &CommaLocs,
                           void (Sema::*Completer)(Scope *S,
                                                   Expr *Data,
                                                   Expr **Args,
                                                   unsigned NumArgs) = 0,
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
                                        ParsedType TypeOfCast,
                                        ParsedType &CastTy,
                                        SourceLocation &RParenLoc);

  ExprResult ParseCXXAmbiguousParenExpression(ParenParseOption &ExprType,
                                                    ParsedType &CastTy,
                                                    SourceLocation LParenLoc,
                                                    SourceLocation &RParenLoc);

  ExprResult ParseCompoundLiteralExpression(ParsedType Ty,
                                                  SourceLocation LParenLoc,
                                                  SourceLocation RParenLoc);

  ExprResult ParseStringLiteralExpression();

  //===--------------------------------------------------------------------===//
  // C++ Expressions
  ExprResult ParseCXXIdExpression(bool isAddressOfOperand = false);

  bool ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
                                      ParsedType ObjectType,
                                      bool EnteringContext,
                                      bool *MayBePseudoDestructor = 0);

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
  // EndLoc is filled with the location of the last token of the specification.
  bool ParseExceptionSpecification(SourceLocation &EndLoc,
                                   llvm::SmallVectorImpl<ParsedType> &Exns,
                                   llvm::SmallVectorImpl<SourceRange> &Ranges,
                                   bool &hasAnyExceptionSpec);

  //===--------------------------------------------------------------------===//
  // C++0x 8: Function declaration trailing-return-type
  TypeResult ParseTrailingReturnType();

  //===--------------------------------------------------------------------===//
  // C++ 2.13.5: C++ Boolean Literals
  ExprResult ParseCXXBoolLiteral();

  //===--------------------------------------------------------------------===//
  // C++ 5.2.3: Explicit type conversion (functional notation)
  ExprResult ParseCXXTypeConstructExpression(const DeclSpec &DS);

  bool isCXXSimpleTypeSpecifier() const;

  /// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
  /// This should only be called when the current token is known to be part of
  /// simple-type-specifier.
  void ParseCXXSimpleTypeSpecifier(DeclSpec &DS);

  bool ParseCXXTypeSpecifierSeq(DeclSpec &DS);

  //===--------------------------------------------------------------------===//
  // C++ 5.3.4 and 5.3.5: C++ new and delete
  bool ParseExpressionListOrTypeId(llvm::SmallVectorImpl<Expr*> &Exprs,
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
  ExprResult ParseBraceInitializer();
  ExprResult ParseInitializerWithPotentialDesignator();

  //===--------------------------------------------------------------------===//
  // clang Expressions

  ExprResult ParseBlockLiteralExpression();  // ^{...}

  //===--------------------------------------------------------------------===//
  // Objective-C Expressions
  ExprResult ParseObjCAtExpression(SourceLocation AtLocation);
  ExprResult ParseObjCStringLiteral(SourceLocation AtLoc);
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

  StmtResult ParseStatement() {
    StmtVector Stmts(Actions);
    return ParseStatementOrDeclaration(Stmts, true);
  }
  StmtResult ParseStatementOrDeclaration(StmtVector& Stmts,
                                         bool OnlyStatement = false);
  StmtResult ParseLabeledStatement(ParsedAttributes &Attr);
  StmtResult ParseCaseStatement(ParsedAttributes &Attr);
  StmtResult ParseDefaultStatement(ParsedAttributes &Attr);
  StmtResult ParseCompoundStatement(ParsedAttributes &Attr,
                                          bool isStmtExpr = false);
  StmtResult ParseCompoundStatementBody(bool isStmtExpr = false);
  bool ParseParenExprOrCondition(ExprResult &ExprResult,
                                 Decl *&DeclResult,
                                 SourceLocation Loc,
                                 bool ConvertToBoolean);
  StmtResult ParseIfStatement(ParsedAttributes &Attr);
  StmtResult ParseSwitchStatement(ParsedAttributes &Attr);
  StmtResult ParseWhileStatement(ParsedAttributes &Attr);
  StmtResult ParseDoStatement(ParsedAttributes &Attr);
  StmtResult ParseForStatement(ParsedAttributes &Attr);
  StmtResult ParseGotoStatement(ParsedAttributes &Attr);
  StmtResult ParseContinueStatement(ParsedAttributes &Attr);
  StmtResult ParseBreakStatement(ParsedAttributes &Attr);
  StmtResult ParseReturnStatement(ParsedAttributes &Attr);
  StmtResult ParseAsmStatement(bool &msAsm);
  StmtResult FuzzyParseMicrosoftAsmStatement(SourceLocation AsmLoc);
  bool ParseAsmOperandsOpt(llvm::SmallVectorImpl<IdentifierInfo *> &Names,
                           llvm::SmallVectorImpl<ExprTy *> &Constraints,
                           llvm::SmallVectorImpl<ExprTy *> &Exprs);

  //===--------------------------------------------------------------------===//
  // C++ 6: Statements and Blocks

  StmtResult ParseCXXTryBlock(ParsedAttributes &Attr);
  StmtResult ParseCXXTryBlockCommon(SourceLocation TryLoc);
  StmtResult ParseCXXCatchBlock();

  //===--------------------------------------------------------------------===//
  // Objective-C Statements

  StmtResult ParseObjCAtStatement(SourceLocation atLoc);
  StmtResult ParseObjCTryStmt(SourceLocation atLoc);
  StmtResult ParseObjCThrowStmt(SourceLocation atLoc);
  StmtResult ParseObjCSynchronizedStmt(SourceLocation atLoc);


  //===--------------------------------------------------------------------===//
  // C99 6.7: Declarations.

  /// A context for parsing declaration specifiers.  TODO: flesh this
  /// out, there are other significant restrictions on specifiers than
  /// would be best implemented in the parser.
  enum DeclSpecContext {
    DSC_normal, // normal context
    DSC_class,  // class context, enables 'friend'
    DSC_top_level // top-level/namespace declaration context
  };

  DeclGroupPtrTy ParseDeclaration(StmtVector &Stmts,
                                  unsigned Context, SourceLocation &DeclEnd,
                                  ParsedAttributesWithRange &attrs);
  DeclGroupPtrTy ParseSimpleDeclaration(StmtVector &Stmts,
                                        unsigned Context,
                                        SourceLocation &DeclEnd,
                                        ParsedAttributes &attrs,
                                        bool RequireSemi);
  DeclGroupPtrTy ParseDeclGroup(ParsingDeclSpec &DS, unsigned Context,
                                bool AllowFunctionDefinitions,
                                SourceLocation *DeclEnd = 0);
  Decl *ParseDeclarationAfterDeclarator(Declarator &D,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo());
  Decl *ParseFunctionStatementBody(Decl *Decl);
  Decl *ParseFunctionTryBlock(Decl *Decl);

  /// \brief When in code-completion, skip parsing of the function/method body
  /// unless the body contains the code-completion point.
  ///
  /// \returns true if the function body was skipped.
  bool trySkippingFunctionBodyForCodeCompletion();

  bool ParseImplicitInt(DeclSpec &DS, CXXScopeSpec *SS,
                        const ParsedTemplateInfo &TemplateInfo,
                        AccessSpecifier AS);
  DeclSpecContext getDeclSpecContextFromDeclaratorContext(unsigned Context);
  void ParseDeclarationSpecifiers(DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                  AccessSpecifier AS = AS_none,
                                  DeclSpecContext DSC = DSC_normal);
  bool ParseOptionalTypeSpecifier(DeclSpec &DS, bool &isInvalid,
                                  const char *&PrevSpec,
                                  unsigned &DiagID,
               const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                  bool SuppressDeclarations = false);

  void ParseSpecifierQualifierList(DeclSpec &DS);

  void ParseObjCTypeQualifierList(ObjCDeclSpec &DS, bool IsParameter);

  void ParseEnumSpecifier(SourceLocation TagLoc, DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                AccessSpecifier AS = AS_none);
  void ParseEnumBody(SourceLocation StartLoc, Decl *TagDecl);
  void ParseStructUnionBody(SourceLocation StartLoc, unsigned TagType,
                            Decl *TagDecl);

  struct FieldCallback {
    virtual Decl *invoke(FieldDeclarator &Field) = 0;
    virtual ~FieldCallback() {}

  private:
    virtual void _anchor();
  };
  struct ObjCPropertyCallback;

  void ParseStructDeclaration(DeclSpec &DS, FieldCallback &Callback);

  bool isDeclarationSpecifier(bool DisambiguatingWithExpression = false);
  bool isTypeSpecifierQualifier();
  bool isTypeQualifier() const;
  
  /// isKnownToBeTypeSpecifier - Return true if we know that the specified token
  /// is definitely a type-specifier.  Return false if it isn't part of a type
  /// specifier or if we're not sure.
  bool isKnownToBeTypeSpecifier(const Token &Tok) const;

  /// isDeclarationStatement - Disambiguates between a declaration or an
  /// expression statement, when parsing function bodies.
  /// Returns true for declaration, false for expression.
  bool isDeclarationStatement() {
    if (getLang().CPlusPlus)
      return isCXXDeclarationStatement();
    return isDeclarationSpecifier(true);
  }

  /// isSimpleDeclaration - Disambiguates between a declaration or an
  /// expression, mainly used for the C 'clause-1' or the C++
  // 'for-init-statement' part of a 'for' statement.
  /// Returns true for declaration, false for expression.
  bool isSimpleDeclaration() {
    if (getLang().CPlusPlus)
      return isCXXSimpleDeclaration();
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
    if (getLang().CPlusPlus)
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
  bool isCXXSimpleDeclaration();

  /// isCXXFunctionDeclarator - Disambiguates between a function declarator or
  /// a constructor-style initializer, when parsing declaration statements.
  /// Returns true for function declarator and false for constructor-style
  /// initializer. If 'warnIfAmbiguous' is true a warning will be emitted to
  /// indicate that the parens were disambiguated as function declarator.
  /// If during the disambiguation process a parsing error is encountered,
  /// the function returns true to let the declaration parsing code handle it.
  bool isCXXFunctionDeclarator(bool warnIfAmbiguous);

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
  /// encountered.
  /// Doesn't consume tokens.
  TPResult isCXXDeclarationSpecifier();
  
  // "Tentative parsing" functions, used for disambiguation. If a parsing error
  // is encountered they will return TPResult::Error().
  // Returning TPResult::True()/False() indicates that the ambiguity was
  // resolved and tentative parsing may stop. TPResult::Ambiguous() indicates
  // that more tentative parsing is necessary for disambiguation.
  // They all consume tokens, so backtracking should be used after calling them.

  TPResult TryParseDeclarationSpecifier();
  TPResult TryParseSimpleDeclaration();
  TPResult TryParseTypeofSpecifier();
  TPResult TryParseProtocolQualifiers();
  TPResult TryParseInitDeclaratorList();
  TPResult TryParseDeclarator(bool mayBeAbstract, bool mayHaveIdentifier=true);
  TPResult TryParseParameterDeclarationClause();
  TPResult TryParseFunctionDeclarator();
  TPResult TryParseBracketDeclarator();

  TypeResult ParseTypeName(SourceRange *Range = 0,
                           Declarator::TheContext Context
                                                 = Declarator::TypeNameContext);
  void ParseBlockId();

  void ProhibitAttributes(ParsedAttributesWithRange &attrs) {
    if (!attrs.Range.isValid()) return;
    DiagnoseProhibitedAttributes(attrs);
  }
  void DiagnoseProhibitedAttributes(ParsedAttributesWithRange &attrs);

  void MaybeParseGNUAttributes(Declarator &D) {
    if (Tok.is(tok::kw___attribute)) {
      ParsedAttributes attrs;
      SourceLocation endLoc;
      ParseGNUAttributes(attrs, &endLoc);
      D.addAttributes(attrs.getList(), endLoc);
    }
  }
  void MaybeParseGNUAttributes(ParsedAttributes &attrs,
                               SourceLocation *endLoc = 0) {
    if (Tok.is(tok::kw___attribute))
      ParseGNUAttributes(attrs, endLoc);
  }
  void ParseGNUAttributes(ParsedAttributes &attrs,
                          SourceLocation *endLoc = 0);

  void MaybeParseCXX0XAttributes(Declarator &D) {
    if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier()) {
      ParsedAttributesWithRange attrs;
      SourceLocation endLoc;
      ParseCXX0XAttributes(attrs, &endLoc);
      D.addAttributes(attrs.getList(), endLoc);
    }
  }
  void MaybeParseCXX0XAttributes(ParsedAttributes &attrs,
                                 SourceLocation *endLoc = 0) {
    if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier()) {
      ParsedAttributesWithRange attrsWithRange;
      ParseCXX0XAttributes(attrsWithRange, endLoc);
      attrs.append(attrsWithRange.getList());
    }
  }
  void MaybeParseCXX0XAttributes(ParsedAttributesWithRange &attrs,
                                 SourceLocation *endLoc = 0) {
    if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier())
      ParseCXX0XAttributes(attrs, endLoc);
  }
  void ParseCXX0XAttributes(ParsedAttributesWithRange &attrs,
                            SourceLocation *EndLoc = 0);

  void MaybeParseMicrosoftAttributes(ParsedAttributes &attrs,
                                     SourceLocation *endLoc = 0) {
    if (getLang().Microsoft && Tok.is(tok::l_square))
      ParseMicrosoftAttributes(attrs, endLoc);
  }
  void ParseMicrosoftAttributes(ParsedAttributes &attrs,
                                SourceLocation *endLoc = 0);
  void ParseMicrosoftDeclSpec(ParsedAttributes &attrs);
  void ParseMicrosoftTypeAttributes(ParsedAttributes &attrs);
  void ParseBorlandTypeAttributes(ParsedAttributes &attrs);
  void ParseOpenCLAttributes(ParsedAttributes &attrs);

  void ParseTypeofSpecifier(DeclSpec &DS);
  void ParseDecltypeSpecifier(DeclSpec &DS);
  
  ExprResult ParseCXX0XAlignArgument(SourceLocation Start);

  VirtSpecifiers::Specifier isCXX0XVirtSpecifier() const;
  void ParseOptionalCXX0XVirtSpecifierSeq(VirtSpecifiers &VS);

  ClassVirtSpecifiers::Specifier isCXX0XClassVirtSpecifier() const;
  void ParseOptionalCXX0XClassVirtSpecifierSeq(ClassVirtSpecifiers &CVS);

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
  void ParseFunctionDeclarator(SourceLocation LParenLoc, Declarator &D,
                               ParsedAttributes &attrs,
                               bool RequiresArg = false);
  void ParseFunctionDeclaratorIdentifierList(SourceLocation LParenLoc,
                                             IdentifierInfo *FirstIdent,
                                             SourceLocation FirstIdentLoc,
                                             Declarator &D);
  void ParseBracketDeclarator(Declarator &D);

  //===--------------------------------------------------------------------===//
  // C++ 7: Declarations [dcl.dcl]

  bool isCXX0XAttributeSpecifier(bool FullLookahead = false, 
                                 tok::TokenKind *After = 0);
  
  Decl *ParseNamespace(unsigned Context, SourceLocation &DeclEnd,
                       SourceLocation InlineLoc = SourceLocation());
  Decl *ParseLinkage(ParsingDeclSpec &DS, unsigned Context);
  Decl *ParseUsingDirectiveOrDeclaration(unsigned Context,
                                         const ParsedTemplateInfo &TemplateInfo,
                                         SourceLocation &DeclEnd,
                                         ParsedAttributesWithRange &attrs);
  Decl *ParseUsingDirective(unsigned Context,
                            SourceLocation UsingLoc,
                            SourceLocation &DeclEnd,
                            ParsedAttributes &attrs);
  Decl *ParseUsingDeclaration(unsigned Context,
                              const ParsedTemplateInfo &TemplateInfo,
                              SourceLocation UsingLoc,
                              SourceLocation &DeclEnd,
                              AccessSpecifier AS = AS_none);
  Decl *ParseStaticAssertDeclaration(SourceLocation &DeclEnd);
  Decl *ParseNamespaceAlias(SourceLocation NamespaceLoc,
                            SourceLocation AliasLoc, IdentifierInfo *Alias,
                            SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // C++ 9: classes [class] and C structs/unions.
  TypeResult ParseClassName(SourceLocation &EndLocation, CXXScopeSpec &SS);
  void ParseClassSpecifier(tok::TokenKind TagTokKind, SourceLocation TagLoc,
                           DeclSpec &DS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                           AccessSpecifier AS = AS_none,
                           bool SuppressDeclarations = false);
  void ParseCXXMemberSpecification(SourceLocation StartLoc, unsigned TagType,
                                   Decl *TagDecl);
  void ParseCXXClassMemberDeclaration(AccessSpecifier AS,
                const ParsedTemplateInfo &TemplateInfo = ParsedTemplateInfo(),
                                 ParsingDeclRAIIObject *DiagsFromTParams = 0);
  void ParseConstructorInitializer(Decl *ConstructorDecl);
  MemInitResult ParseMemInitializer(Decl *ConstructorDecl);
  void HandleMemberFunctionDefaultArgs(Declarator& DeclaratorInfo,
                                       Decl *ThisDecl);

  //===--------------------------------------------------------------------===//
  // C++ 10: Derived classes [class.derived]
  void ParseBaseClause(Decl *ClassDecl);
  BaseResult ParseBaseSpecifier(Decl *ClassDecl);
  AccessSpecifier getAccessSpecifierIfPresent() const;

  bool ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS, 
                                    IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    bool EnteringContext,
                                    ParsedType ObjectType,
                                    UnqualifiedId &Id,
                                    bool AssumeTemplateId,
                                    SourceLocation TemplateKWLoc);
  bool ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                  ParsedType ObjectType,
                                  UnqualifiedId &Result);
  bool ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
                          bool AllowDestructorName,
                          bool AllowConstructorName,
                          ParsedType ObjectType,
                          UnqualifiedId &Result);
    
  //===--------------------------------------------------------------------===//
  // C++ 14: Templates [temp]

  // C++ 14.1: Template Parameters [temp.param]
  Decl *ParseDeclarationStartingWithTemplate(unsigned Context,
                                                 SourceLocation &DeclEnd,
                                                 AccessSpecifier AS = AS_none);
  Decl *ParseTemplateDeclarationOrSpecialization(unsigned Context,
                                                     SourceLocation &DeclEnd,
                                                     AccessSpecifier AS);
  Decl *ParseSingleDeclarationAfterTemplate(
                                       unsigned Context,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       ParsingDeclRAIIObject &DiagsFromParams,
                                       SourceLocation &DeclEnd,
                                       AccessSpecifier AS=AS_none);
  bool ParseTemplateParameters(unsigned Depth,
                               llvm::SmallVectorImpl<Decl*> &TemplateParams,
                               SourceLocation &LAngleLoc,
                               SourceLocation &RAngleLoc);
  bool ParseTemplateParameterList(unsigned Depth,
                                  llvm::SmallVectorImpl<Decl*> &TemplateParams);
  bool isStartOfTemplateTypeParameter();
  Decl *ParseTemplateParameter(unsigned Depth, unsigned Position);
  Decl *ParseTypeParameter(unsigned Depth, unsigned Position);
  Decl *ParseTemplateTemplateParameter(unsigned Depth, unsigned Position);
  Decl *ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position);
  // C++ 14.3: Template arguments [temp.arg]
  typedef llvm::SmallVector<ParsedTemplateArgument, 16> TemplateArgList;

  bool ParseTemplateIdAfterTemplateName(TemplateTy Template,
                                        SourceLocation TemplateNameLoc,
                                        const CXXScopeSpec &SS,
                                        bool ConsumeLastToken,
                                        SourceLocation &LAngleLoc,
                                        TemplateArgList &TemplateArgs,
                                        SourceLocation &RAngleLoc);

  bool AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
                               CXXScopeSpec &SS,
                               UnqualifiedId &TemplateName,
                               SourceLocation TemplateKWLoc = SourceLocation(),
                               bool AllowTypeAnnotation = true);
  void AnnotateTemplateIdTokenAsType();
  bool IsTemplateArgumentList(unsigned Skip = 0);
  bool ParseTemplateArgumentList(TemplateArgList &TemplateArgs);
  ParsedTemplateArgument ParseTemplateTemplateArgument();
  ParsedTemplateArgument ParseTemplateArgument();
  Decl *ParseExplicitInstantiation(SourceLocation ExternLoc,
                                        SourceLocation TemplateLoc,
                                        SourceLocation &DeclEnd);

  //===--------------------------------------------------------------------===//
  // GNU G++: Type Traits [Type-Traits.html in the GCC manual]
  ExprResult ParseUnaryTypeTrait();
  ExprResult ParseBinaryTypeTrait();

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
