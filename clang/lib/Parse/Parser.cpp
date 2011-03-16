//===--- Parser.cpp - C Language Family Parser ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ParsedTemplate.h"
#include "llvm/Support/raw_ostream.h"
#include "RAIIObjectsForParser.h"
#include "ParsePragma.h"
using namespace clang;

Parser::Parser(Preprocessor &pp, Sema &actions)
  : CrashInfo(*this), PP(pp), Actions(actions), Diags(PP.getDiagnostics()),
    GreaterThanIsOperator(true), ColonIsSacred(false), 
    InMessageExpression(false), TemplateParameterDepth(0) {
  Tok.setKind(tok::eof);
  Actions.CurScope = 0;
  NumCachedScopes = 0;
  ParenCount = BracketCount = BraceCount = 0;
  ObjCImpDecl = 0;

  // Add #pragma handlers. These are removed and destroyed in the
  // destructor.
  AlignHandler.reset(new PragmaAlignHandler(actions));
  PP.AddPragmaHandler(AlignHandler.get());

  GCCVisibilityHandler.reset(new PragmaGCCVisibilityHandler(actions));
  PP.AddPragmaHandler("GCC", GCCVisibilityHandler.get());

  OptionsHandler.reset(new PragmaOptionsHandler(actions));
  PP.AddPragmaHandler(OptionsHandler.get());

  PackHandler.reset(new PragmaPackHandler(actions));
  PP.AddPragmaHandler(PackHandler.get());

  UnusedHandler.reset(new PragmaUnusedHandler(actions, *this));
  PP.AddPragmaHandler(UnusedHandler.get());

  WeakHandler.reset(new PragmaWeakHandler(actions));
  PP.AddPragmaHandler(WeakHandler.get());

  FPContractHandler.reset(new PragmaFPContractHandler(actions, *this));
  PP.AddPragmaHandler("STDC", FPContractHandler.get());

  if (getLang().OpenCL) {
    OpenCLExtensionHandler.reset(
                  new PragmaOpenCLExtensionHandler(actions, *this));
    PP.AddPragmaHandler("OPENCL", OpenCLExtensionHandler.get());

    PP.AddPragmaHandler("OPENCL", FPContractHandler.get());
  }
      
  PP.setCodeCompletionHandler(*this);
}

/// If a crash happens while the parser is active, print out a line indicating
/// what the current token is.
void PrettyStackTraceParserEntry::print(llvm::raw_ostream &OS) const {
  const Token &Tok = P.getCurToken();
  if (Tok.is(tok::eof)) {
    OS << "<eof> parser at end of file\n";
    return;
  }

  if (Tok.getLocation().isInvalid()) {
    OS << "<unknown> parser at unknown location\n";
    return;
  }

  const Preprocessor &PP = P.getPreprocessor();
  Tok.getLocation().print(OS, PP.getSourceManager());
  if (Tok.isAnnotation())
    OS << ": at annotation token \n";
  else
    OS << ": current parser token '" << PP.getSpelling(Tok) << "'\n";
}


DiagnosticBuilder Parser::Diag(SourceLocation Loc, unsigned DiagID) {
  return Diags.Report(Loc, DiagID);
}

DiagnosticBuilder Parser::Diag(const Token &Tok, unsigned DiagID) {
  return Diag(Tok.getLocation(), DiagID);
}

/// \brief Emits a diagnostic suggesting parentheses surrounding a
/// given range.
///
/// \param Loc The location where we'll emit the diagnostic.
/// \param Loc The kind of diagnostic to emit.
/// \param ParenRange Source range enclosing code that should be parenthesized.
void Parser::SuggestParentheses(SourceLocation Loc, unsigned DK,
                                SourceRange ParenRange) {
  SourceLocation EndLoc = PP.getLocForEndOfToken(ParenRange.getEnd());
  if (!ParenRange.getEnd().isFileID() || EndLoc.isInvalid()) {
    // We can't display the parentheses, so just dig the
    // warning/error and return.
    Diag(Loc, DK);
    return;
  }

  Diag(Loc, DK)
    << FixItHint::CreateInsertion(ParenRange.getBegin(), "(")
    << FixItHint::CreateInsertion(EndLoc, ")");
}

/// MatchRHSPunctuation - For punctuation with a LHS and RHS (e.g. '['/']'),
/// this helper function matches and consumes the specified RHS token if
/// present.  If not present, it emits the specified diagnostic indicating
/// that the parser failed to match the RHS of the token at LHSLoc.  LHSName
/// should be the name of the unmatched LHS token.
SourceLocation Parser::MatchRHSPunctuation(tok::TokenKind RHSTok,
                                           SourceLocation LHSLoc) {

  if (Tok.is(RHSTok))
    return ConsumeAnyToken();

  SourceLocation R = Tok.getLocation();
  const char *LHSName = "unknown";
  diag::kind DID = diag::err_parse_error;
  switch (RHSTok) {
  default: break;
  case tok::r_paren : LHSName = "("; DID = diag::err_expected_rparen; break;
  case tok::r_brace : LHSName = "{"; DID = diag::err_expected_rbrace; break;
  case tok::r_square: LHSName = "["; DID = diag::err_expected_rsquare; break;
  case tok::greater:  LHSName = "<"; DID = diag::err_expected_greater; break;
  case tok::greatergreatergreater:
                      LHSName = "<<<"; DID = diag::err_expected_ggg; break;
  }
  Diag(Tok, DID);
  Diag(LHSLoc, diag::note_matching) << LHSName;
  SkipUntil(RHSTok);
  return R;
}

static bool IsCommonTypo(tok::TokenKind ExpectedTok, const Token &Tok) {
  switch (ExpectedTok) {
  case tok::semi: return Tok.is(tok::colon); // : for ;
  default: return false;
  }
}

/// ExpectAndConsume - The parser expects that 'ExpectedTok' is next in the
/// input.  If so, it is consumed and false is returned.
///
/// If the input is malformed, this emits the specified diagnostic.  Next, if
/// SkipToTok is specified, it calls SkipUntil(SkipToTok).  Finally, true is
/// returned.
bool Parser::ExpectAndConsume(tok::TokenKind ExpectedTok, unsigned DiagID,
                              const char *Msg, tok::TokenKind SkipToTok) {
  if (Tok.is(ExpectedTok) || Tok.is(tok::code_completion)) {
    ConsumeAnyToken();
    return false;
  }

  // Detect common single-character typos and resume.
  if (IsCommonTypo(ExpectedTok, Tok)) {
    SourceLocation Loc = Tok.getLocation();
    Diag(Loc, DiagID)
      << Msg
      << FixItHint::CreateReplacement(SourceRange(Loc),
                                      getTokenSimpleSpelling(ExpectedTok));
    ConsumeAnyToken();

    // Pretend there wasn't a problem.
    return false;
  }

  const char *Spelling = 0;
  SourceLocation EndLoc = PP.getLocForEndOfToken(PrevTokLocation);
  if (EndLoc.isValid() &&
      (Spelling = tok::getTokenSimpleSpelling(ExpectedTok))) {
    // Show what code to insert to fix this problem.
    Diag(EndLoc, DiagID)
      << Msg
      << FixItHint::CreateInsertion(EndLoc, Spelling);
  } else
    Diag(Tok, DiagID) << Msg;

  if (SkipToTok != tok::unknown)
    SkipUntil(SkipToTok);
  return true;
}

bool Parser::ExpectAndConsumeSemi(unsigned DiagID) {
  if (Tok.is(tok::semi) || Tok.is(tok::code_completion)) {
    ConsumeAnyToken();
    return false;
  }
  
  if ((Tok.is(tok::r_paren) || Tok.is(tok::r_square)) && 
      NextToken().is(tok::semi)) {
    Diag(Tok, diag::err_extraneous_token_before_semi)
      << PP.getSpelling(Tok)
      << FixItHint::CreateRemoval(Tok.getLocation());
    ConsumeAnyToken(); // The ')' or ']'.
    ConsumeToken(); // The ';'.
    return false;
  }
  
  return ExpectAndConsume(tok::semi, DiagID);
}

//===----------------------------------------------------------------------===//
// Error recovery.
//===----------------------------------------------------------------------===//

/// SkipUntil - Read tokens until we get to the specified token, then consume
/// it (unless DontConsume is true).  Because we cannot guarantee that the
/// token will ever occur, this skips to the next token, or to some likely
/// good stopping point.  If StopAtSemi is true, skipping will stop at a ';'
/// character.
///
/// If SkipUntil finds the specified token, it returns true, otherwise it
/// returns false.
bool Parser::SkipUntil(const tok::TokenKind *Toks, unsigned NumToks,
                       bool StopAtSemi, bool DontConsume,
                       bool StopAtCodeCompletion) {
  // We always want this function to skip at least one token if the first token
  // isn't T and if not at EOF.
  bool isFirstTokenSkipped = true;
  while (1) {
    // If we found one of the tokens, stop and return true.
    for (unsigned i = 0; i != NumToks; ++i) {
      if (Tok.is(Toks[i])) {
        if (DontConsume) {
          // Noop, don't consume the token.
        } else {
          ConsumeAnyToken();
        }
        return true;
      }
    }

    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;
        
    case tok::code_completion:
      if (!StopAtCodeCompletion)
        ConsumeToken();
      return false;
        
    case tok::l_paren:
      // Recursively skip properly-nested parens.
      ConsumeParen();
      SkipUntil(tok::r_paren, false, false, StopAtCodeCompletion);
      break;
    case tok::l_square:
      // Recursively skip properly-nested square brackets.
      ConsumeBracket();
      SkipUntil(tok::r_square, false, false, StopAtCodeCompletion);
      break;
    case tok::l_brace:
      // Recursively skip properly-nested braces.
      ConsumeBrace();
      SkipUntil(tok::r_brace, false, false, StopAtCodeCompletion);
      break;

    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount && !isFirstTokenSkipped)
        return false;  // Matches something.
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount && !isFirstTokenSkipped)
        return false;  // Matches something.
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount && !isFirstTokenSkipped)
        return false;  // Matches something.
      ConsumeBrace();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
      ConsumeStringToken();
      break;
        
    case tok::at:
      return false;
      
    case tok::semi:
      if (StopAtSemi)
        return false;
      // FALL THROUGH.
    default:
      // Skip this token.
      ConsumeToken();
      break;
    }
    isFirstTokenSkipped = false;
  }
}

//===----------------------------------------------------------------------===//
// Scope manipulation
//===----------------------------------------------------------------------===//

/// EnterScope - Start a new scope.
void Parser::EnterScope(unsigned ScopeFlags) {
  if (NumCachedScopes) {
    Scope *N = ScopeCache[--NumCachedScopes];
    N->Init(getCurScope(), ScopeFlags);
    Actions.CurScope = N;
  } else {
    Actions.CurScope = new Scope(getCurScope(), ScopeFlags, Diags);
  }
}

/// ExitScope - Pop a scope off the scope stack.
void Parser::ExitScope() {
  assert(getCurScope() && "Scope imbalance!");

  // Inform the actions module that this scope is going away if there are any
  // decls in it.
  if (!getCurScope()->decl_empty())
    Actions.ActOnPopScope(Tok.getLocation(), getCurScope());

  Scope *OldScope = getCurScope();
  Actions.CurScope = OldScope->getParent();

  if (NumCachedScopes == ScopeCacheSize)
    delete OldScope;
  else
    ScopeCache[NumCachedScopes++] = OldScope;
}




//===----------------------------------------------------------------------===//
// C99 6.9: External Definitions.
//===----------------------------------------------------------------------===//

Parser::~Parser() {
  // If we still have scopes active, delete the scope tree.
  delete getCurScope();
  Actions.CurScope = 0;
  
  // Free the scope cache.
  for (unsigned i = 0, e = NumCachedScopes; i != e; ++i)
    delete ScopeCache[i];

  // Remove the pragma handlers we installed.
  PP.RemovePragmaHandler(AlignHandler.get());
  AlignHandler.reset();
  PP.RemovePragmaHandler("GCC", GCCVisibilityHandler.get());
  GCCVisibilityHandler.reset();
  PP.RemovePragmaHandler(OptionsHandler.get());
  OptionsHandler.reset();
  PP.RemovePragmaHandler(PackHandler.get());
  PackHandler.reset();
  PP.RemovePragmaHandler(UnusedHandler.get());
  UnusedHandler.reset();
  PP.RemovePragmaHandler(WeakHandler.get());
  WeakHandler.reset();

  if (getLang().OpenCL) {
    PP.RemovePragmaHandler("OPENCL", OpenCLExtensionHandler.get());
    OpenCLExtensionHandler.reset();
    PP.RemovePragmaHandler("OPENCL", FPContractHandler.get());
  }

  PP.RemovePragmaHandler("STDC", FPContractHandler.get());
  FPContractHandler.reset();
  PP.clearCodeCompletionHandler();
}

/// Initialize - Warm up the parser.
///
void Parser::Initialize() {
  // Create the translation unit scope.  Install it as the current scope.
  assert(getCurScope() == 0 && "A scope is already active?");
  EnterScope(Scope::DeclScope);
  Actions.ActOnTranslationUnitScope(getCurScope());

  // Prime the lexer look-ahead.
  ConsumeToken();

  if (Tok.is(tok::eof) &&
      !getLang().CPlusPlus)  // Empty source file is an extension in C
    Diag(Tok, diag::ext_empty_source_file);

  // Initialization for Objective-C context sensitive keywords recognition.
  // Referenced in Parser::ParseObjCTypeQualifierList.
  if (getLang().ObjC1) {
    ObjCTypeQuals[objc_in] = &PP.getIdentifierTable().get("in");
    ObjCTypeQuals[objc_out] = &PP.getIdentifierTable().get("out");
    ObjCTypeQuals[objc_inout] = &PP.getIdentifierTable().get("inout");
    ObjCTypeQuals[objc_oneway] = &PP.getIdentifierTable().get("oneway");
    ObjCTypeQuals[objc_bycopy] = &PP.getIdentifierTable().get("bycopy");
    ObjCTypeQuals[objc_byref] = &PP.getIdentifierTable().get("byref");
  }

  Ident_final = 0;
  Ident_override = 0;

  Ident_super = &PP.getIdentifierTable().get("super");

  if (getLang().AltiVec) {
    Ident_vector = &PP.getIdentifierTable().get("vector");
    Ident_pixel = &PP.getIdentifierTable().get("pixel");
  }
}

/// ParseTopLevelDecl - Parse one top-level declaration, return whatever the
/// action tells us to.  This returns true if the EOF was encountered.
bool Parser::ParseTopLevelDecl(DeclGroupPtrTy &Result) {

  while (Tok.is(tok::annot_pragma_unused))
    HandlePragmaUnused();

  Result = DeclGroupPtrTy();
  if (Tok.is(tok::eof)) {
    Actions.ActOnEndOfTranslationUnit();
    return true;
  }

  ParsedAttributesWithRange attrs;
  MaybeParseCXX0XAttributes(attrs);
  MaybeParseMicrosoftAttributes(attrs);
  
  Result = ParseExternalDeclaration(attrs);
  return false;
}

/// ParseTranslationUnit:
///       translation-unit: [C99 6.9]
///         external-declaration
///         translation-unit external-declaration
void Parser::ParseTranslationUnit() {
  Initialize();

  DeclGroupPtrTy Res;
  while (!ParseTopLevelDecl(Res))
    /*parse them all*/;

  ExitScope();
  assert(getCurScope() == 0 && "Scope imbalance!");
}

/// ParseExternalDeclaration:
///
///       external-declaration: [C99 6.9], declaration: [C++ dcl.dcl]
///         function-definition
///         declaration
/// [C++0x] empty-declaration
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
///
/// [C++0x] empty-declaration:
///           ';'
///
/// [C++0x/GNU] 'extern' 'template' declaration
Parser::DeclGroupPtrTy
Parser::ParseExternalDeclaration(ParsedAttributesWithRange &attrs,
                                 ParsingDeclSpec *DS) {
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  
  Decl *SingleDecl = 0;
  switch (Tok.getKind()) {
  case tok::semi:
    if (!getLang().CPlusPlus0x)
      Diag(Tok, diag::ext_top_level_semi)
        << FixItHint::CreateRemoval(Tok.getLocation());

    ConsumeToken();
    // TODO: Invoke action for top-level semicolon.
    return DeclGroupPtrTy();
  case tok::r_brace:
    Diag(Tok, diag::err_expected_external_declaration);
    ConsumeBrace();
    return DeclGroupPtrTy();
  case tok::eof:
    Diag(Tok, diag::err_expected_external_declaration);
    return DeclGroupPtrTy();
  case tok::kw___extension__: {
    // __extension__ silences extension warnings in the subexpression.
    ExtensionRAIIObject O(Diags);  // Use RAII to do this.
    ConsumeToken();
    return ParseExternalDeclaration(attrs);
  }
  case tok::kw_asm: {
    ProhibitAttributes(attrs);

    SourceLocation StartLoc = Tok.getLocation();
    SourceLocation EndLoc;
    ExprResult Result(ParseSimpleAsm(&EndLoc));

    ExpectAndConsume(tok::semi, diag::err_expected_semi_after,
                     "top-level asm block");

    if (Result.isInvalid())
      return DeclGroupPtrTy();
    SingleDecl = Actions.ActOnFileScopeAsmDecl(Result.get(), StartLoc, EndLoc);
    break;
  }
  case tok::at:
    // @ is not a legal token unless objc is enabled, no need to check for ObjC.
    /// FIXME: ParseObjCAtDirectives should return a DeclGroup for things like
    /// @class foo, bar;
    SingleDecl = ParseObjCAtDirectives();
    break;
  case tok::minus:
  case tok::plus:
    if (!getLang().ObjC1) {
      Diag(Tok, diag::err_expected_external_declaration);
      ConsumeToken();
      return DeclGroupPtrTy();
    }
    SingleDecl = ParseObjCMethodDefinition();
    break;
  case tok::code_completion:
      Actions.CodeCompleteOrdinaryName(getCurScope(), 
                                   ObjCImpDecl? Sema::PCC_ObjCImplementation
                                              : Sema::PCC_Namespace);
    ConsumeCodeCompletionToken();
    return ParseExternalDeclaration(attrs);
  case tok::kw_using:
  case tok::kw_namespace:
  case tok::kw_typedef:
  case tok::kw_template:
  case tok::kw_export:    // As in 'export template'
  case tok::kw_static_assert:
    // A function definition cannot start with a these keywords.
    {
      SourceLocation DeclEnd;
      StmtVector Stmts(Actions);
      return ParseDeclaration(Stmts, Declarator::FileContext, DeclEnd, attrs);
    }

  case tok::kw_static:
    // Parse (then ignore) 'static' prior to a template instantiation. This is
    // a GCC extension that we intentionally do not support.
    if (getLang().CPlusPlus && NextToken().is(tok::kw_template)) {
      Diag(ConsumeToken(), diag::warn_static_inline_explicit_inst_ignored)
        << 0;
      SourceLocation DeclEnd;
      StmtVector Stmts(Actions);
      return ParseDeclaration(Stmts, Declarator::FileContext, DeclEnd, attrs);  
    }
    goto dont_know;
      
  case tok::kw_inline:
    if (getLang().CPlusPlus) {
      tok::TokenKind NextKind = NextToken().getKind();
      
      // Inline namespaces. Allowed as an extension even in C++03.
      if (NextKind == tok::kw_namespace) {
        SourceLocation DeclEnd;
        StmtVector Stmts(Actions);
        return ParseDeclaration(Stmts, Declarator::FileContext, DeclEnd, attrs);
      }
      
      // Parse (then ignore) 'inline' prior to a template instantiation. This is
      // a GCC extension that we intentionally do not support.
      if (NextKind == tok::kw_template) {
        Diag(ConsumeToken(), diag::warn_static_inline_explicit_inst_ignored)
          << 1;
        SourceLocation DeclEnd;
        StmtVector Stmts(Actions);
        return ParseDeclaration(Stmts, Declarator::FileContext, DeclEnd, attrs);  
      }
    }
    goto dont_know;

  case tok::kw_extern:
    if (getLang().CPlusPlus && NextToken().is(tok::kw_template)) {
      // Extern templates
      SourceLocation ExternLoc = ConsumeToken();
      SourceLocation TemplateLoc = ConsumeToken();
      SourceLocation DeclEnd;
      return Actions.ConvertDeclToDeclGroup(
                  ParseExplicitInstantiation(ExternLoc, TemplateLoc, DeclEnd));
    }
    // FIXME: Detect C++ linkage specifications here?
    goto dont_know;

  default:
  dont_know:
    // We can't tell whether this is a function-definition or declaration yet.
    if (DS) {
      DS->takeAttributesFrom(attrs);
      return ParseDeclarationOrFunctionDefinition(*DS);
    } else {
      return ParseDeclarationOrFunctionDefinition(attrs);
    }
  }

  // This routine returns a DeclGroup, if the thing we parsed only contains a
  // single decl, convert it now.
  return Actions.ConvertDeclToDeclGroup(SingleDecl);
}

/// \brief Determine whether the current token, if it occurs after a
/// declarator, continues a declaration or declaration list.
bool Parser::isDeclarationAfterDeclarator() const {
  return Tok.is(tok::equal) ||      // int X()=  -> not a function def
    Tok.is(tok::comma) ||           // int X(),  -> not a function def
    Tok.is(tok::semi)  ||           // int X();  -> not a function def
    Tok.is(tok::kw_asm) ||          // int X() __asm__ -> not a function def
    Tok.is(tok::kw___attribute) ||  // int X() __attr__ -> not a function def
    (getLang().CPlusPlus &&
     Tok.is(tok::l_paren));         // int X(0) -> not a function def [C++]
}

/// \brief Determine whether the current token, if it occurs after a
/// declarator, indicates the start of a function definition.
bool Parser::isStartOfFunctionDefinition(const ParsingDeclarator &Declarator) {
  assert(Declarator.isFunctionDeclarator() && "Isn't a function declarator");
  if (Tok.is(tok::l_brace))   // int X() {}
    return true;
  
  // Handle K&R C argument lists: int X(f) int f; {}
  if (!getLang().CPlusPlus &&
      Declarator.getFunctionTypeInfo().isKNRPrototype()) 
    return isDeclarationSpecifier();
  
  return Tok.is(tok::colon) ||         // X() : Base() {} (used for ctors)
         Tok.is(tok::kw_try);          // X() try { ... }
}

/// ParseDeclarationOrFunctionDefinition - Parse either a function-definition or
/// a declaration.  We can't tell which we have until we read up to the
/// compound-statement in function-definition. TemplateParams, if
/// non-NULL, provides the template parameters when we're parsing a
/// C++ template-declaration.
///
///       function-definition: [C99 6.9.1]
///         decl-specs      declarator declaration-list[opt] compound-statement
/// [C90] function-definition: [C99 6.7.1] - implicit int result
/// [C90]   decl-specs[opt] declarator declaration-list[opt] compound-statement
///
///       declaration: [C99 6.7]
///         declaration-specifiers init-declarator-list[opt] ';'
/// [!C99]  init-declarator-list ';'                   [TODO: warn in c99 mode]
/// [OMP]   threadprivate-directive                              [TODO]
///
Parser::DeclGroupPtrTy
Parser::ParseDeclarationOrFunctionDefinition(ParsingDeclSpec &DS,
                                             AccessSpecifier AS) {
  // Parse the common declaration-specifiers piece.
  ParseDeclarationSpecifiers(DS, ParsedTemplateInfo(), AS, DSC_top_level);

  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  // declaration-specifiers init-declarator-list[opt] ';'
  if (Tok.is(tok::semi)) {
    ConsumeToken();
    Decl *TheDecl = Actions.ParsedFreeStandingDeclSpec(getCurScope(), AS, DS);
    DS.complete(TheDecl);
    return Actions.ConvertDeclToDeclGroup(TheDecl);
  }

  // ObjC2 allows prefix attributes on class interfaces and protocols.
  // FIXME: This still needs better diagnostics. We should only accept
  // attributes here, no types, etc.
  if (getLang().ObjC2 && Tok.is(tok::at)) {
    SourceLocation AtLoc = ConsumeToken(); // the "@"
    if (!Tok.isObjCAtKeyword(tok::objc_interface) &&
        !Tok.isObjCAtKeyword(tok::objc_protocol)) {
      Diag(Tok, diag::err_objc_unexpected_attr);
      SkipUntil(tok::semi); // FIXME: better skip?
      return DeclGroupPtrTy();
    }

    DS.abort();

    const char *PrevSpec = 0;
    unsigned DiagID;
    if (DS.SetTypeSpecType(DeclSpec::TST_unspecified, AtLoc, PrevSpec, DiagID))
      Diag(AtLoc, DiagID) << PrevSpec;

    Decl *TheDecl = 0;
    if (Tok.isObjCAtKeyword(tok::objc_protocol))
      TheDecl = ParseObjCAtProtocolDeclaration(AtLoc, DS.getAttributes());
    else
      TheDecl = ParseObjCAtInterfaceDeclaration(AtLoc, DS.getAttributes());
    return Actions.ConvertDeclToDeclGroup(TheDecl);
  }

  // If the declspec consisted only of 'extern' and we have a string
  // literal following it, this must be a C++ linkage specifier like
  // 'extern "C"'.
  if (Tok.is(tok::string_literal) && getLang().CPlusPlus &&
      DS.getStorageClassSpec() == DeclSpec::SCS_extern &&
      DS.getParsedSpecifiers() == DeclSpec::PQ_StorageClassSpecifier) {
    Decl *TheDecl = ParseLinkage(DS, Declarator::FileContext);
    return Actions.ConvertDeclToDeclGroup(TheDecl);
  }

  return ParseDeclGroup(DS, Declarator::FileContext, true);
}

Parser::DeclGroupPtrTy
Parser::ParseDeclarationOrFunctionDefinition(ParsedAttributes &attrs,
                                             AccessSpecifier AS) {
  ParsingDeclSpec DS(*this);
  DS.takeAttributesFrom(attrs);
  return ParseDeclarationOrFunctionDefinition(DS, AS);
}

/// ParseFunctionDefinition - We parsed and verified that the specified
/// Declarator is well formed.  If this is a K&R-style function, read the
/// parameters declaration-list, then start the compound-statement.
///
///       function-definition: [C99 6.9.1]
///         decl-specs      declarator declaration-list[opt] compound-statement
/// [C90] function-definition: [C99 6.7.1] - implicit int result
/// [C90]   decl-specs[opt] declarator declaration-list[opt] compound-statement
/// [C++] function-definition: [C++ 8.4]
///         decl-specifier-seq[opt] declarator ctor-initializer[opt]
///         function-body
/// [C++] function-definition: [C++ 8.4]
///         decl-specifier-seq[opt] declarator function-try-block
///
Decl *Parser::ParseFunctionDefinition(ParsingDeclarator &D,
                                      const ParsedTemplateInfo &TemplateInfo) {
  const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();

  // If this is C90 and the declspecs were completely missing, fudge in an
  // implicit int.  We do this here because this is the only place where
  // declaration-specifiers are completely optional in the grammar.
  if (getLang().ImplicitInt && D.getDeclSpec().isEmpty()) {
    const char *PrevSpec;
    unsigned DiagID;
    D.getMutableDeclSpec().SetTypeSpecType(DeclSpec::TST_int,
                                           D.getIdentifierLoc(),
                                           PrevSpec, DiagID);
    D.SetRangeBegin(D.getDeclSpec().getSourceRange().getBegin());
  }

  // If this declaration was formed with a K&R-style identifier list for the
  // arguments, parse declarations for all of the args next.
  // int foo(a,b) int a; float b; {}
  if (FTI.isKNRPrototype())
    ParseKNRParamDeclarations(D);

  // We should have either an opening brace or, in a C++ constructor,
  // we may have a colon.
  if (Tok.isNot(tok::l_brace) && 
      (!getLang().CPlusPlus ||
       (Tok.isNot(tok::colon) && Tok.isNot(tok::kw_try)))) {
    Diag(Tok, diag::err_expected_fn_body);

    // Skip over garbage, until we get to '{'.  Don't eat the '{'.
    SkipUntil(tok::l_brace, true, true);

    // If we didn't find the '{', bail out.
    if (Tok.isNot(tok::l_brace))
      return 0;
  }

  // Enter a scope for the function body.
  ParseScope BodyScope(this, Scope::FnScope|Scope::DeclScope);

  // Tell the actions module that we have entered a function definition with the
  // specified Declarator for the function.
  Decl *Res = TemplateInfo.TemplateParams?
      Actions.ActOnStartOfFunctionTemplateDef(getCurScope(),
                              MultiTemplateParamsArg(Actions,
                                          TemplateInfo.TemplateParams->data(),
                                         TemplateInfo.TemplateParams->size()),
                                              D)
    : Actions.ActOnStartOfFunctionDef(getCurScope(), D);

  // Break out of the ParsingDeclarator context before we parse the body.
  D.complete(Res);
  
  // Break out of the ParsingDeclSpec context, too.  This const_cast is
  // safe because we're always the sole owner.
  D.getMutableDeclSpec().abort();

  if (Tok.is(tok::kw_try))
    return ParseFunctionTryBlock(Res, BodyScope);

  // If we have a colon, then we're probably parsing a C++
  // ctor-initializer.
  if (Tok.is(tok::colon)) {
    ParseConstructorInitializer(Res);

    // Recover from error.
    if (!Tok.is(tok::l_brace)) {
      BodyScope.Exit();
      Actions.ActOnFinishFunctionBody(Res, 0);
      return Res;
    }
  } else
    Actions.ActOnDefaultCtorInitializers(Res);

  return ParseFunctionStatementBody(Res, BodyScope);
}

/// ParseKNRParamDeclarations - Parse 'declaration-list[opt]' which provides
/// types for a function with a K&R-style identifier list for arguments.
void Parser::ParseKNRParamDeclarations(Declarator &D) {
  // We know that the top-level of this declarator is a function.
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();

  // Enter function-declaration scope, limiting any declarators to the
  // function prototype scope, including parameter declarators.
  ParseScope PrototypeScope(this, Scope::FunctionPrototypeScope|Scope::DeclScope);

  // Read all the argument declarations.
  while (isDeclarationSpecifier()) {
    SourceLocation DSStart = Tok.getLocation();

    // Parse the common declaration-specifiers piece.
    DeclSpec DS;
    ParseDeclarationSpecifiers(DS);

    // C99 6.9.1p6: 'each declaration in the declaration list shall have at
    // least one declarator'.
    // NOTE: GCC just makes this an ext-warn.  It's not clear what it does with
    // the declarations though.  It's trivial to ignore them, really hard to do
    // anything else with them.
    if (Tok.is(tok::semi)) {
      Diag(DSStart, diag::err_declaration_does_not_declare_param);
      ConsumeToken();
      continue;
    }

    // C99 6.9.1p6: Declarations shall contain no storage-class specifiers other
    // than register.
    if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified &&
        DS.getStorageClassSpec() != DeclSpec::SCS_register) {
      Diag(DS.getStorageClassSpecLoc(),
           diag::err_invalid_storage_class_in_func_decl);
      DS.ClearStorageClassSpecs();
    }
    if (DS.isThreadSpecified()) {
      Diag(DS.getThreadSpecLoc(),
           diag::err_invalid_storage_class_in_func_decl);
      DS.ClearStorageClassSpecs();
    }

    // Parse the first declarator attached to this declspec.
    Declarator ParmDeclarator(DS, Declarator::KNRTypeListContext);
    ParseDeclarator(ParmDeclarator);

    // Handle the full declarator list.
    while (1) {
      // If attributes are present, parse them.
      MaybeParseGNUAttributes(ParmDeclarator);

      // Ask the actions module to compute the type for this declarator.
      Decl *Param =
        Actions.ActOnParamDeclarator(getCurScope(), ParmDeclarator);

      if (Param &&
          // A missing identifier has already been diagnosed.
          ParmDeclarator.getIdentifier()) {

        // Scan the argument list looking for the correct param to apply this
        // type.
        for (unsigned i = 0; ; ++i) {
          // C99 6.9.1p6: those declarators shall declare only identifiers from
          // the identifier list.
          if (i == FTI.NumArgs) {
            Diag(ParmDeclarator.getIdentifierLoc(), diag::err_no_matching_param)
              << ParmDeclarator.getIdentifier();
            break;
          }

          if (FTI.ArgInfo[i].Ident == ParmDeclarator.getIdentifier()) {
            // Reject redefinitions of parameters.
            if (FTI.ArgInfo[i].Param) {
              Diag(ParmDeclarator.getIdentifierLoc(),
                   diag::err_param_redefinition)
                 << ParmDeclarator.getIdentifier();
            } else {
              FTI.ArgInfo[i].Param = Param;
            }
            break;
          }
        }
      }

      // If we don't have a comma, it is either the end of the list (a ';') or
      // an error, bail out.
      if (Tok.isNot(tok::comma))
        break;

      // Consume the comma.
      ConsumeToken();

      // Parse the next declarator.
      ParmDeclarator.clear();
      ParseDeclarator(ParmDeclarator);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_parse_error);
      // Skip to end of block or statement
      SkipUntil(tok::semi, true);
      if (Tok.is(tok::semi))
        ConsumeToken();
    }
  }

  // The actions module must verify that all arguments were declared.
  Actions.ActOnFinishKNRParamDeclarations(getCurScope(), D, Tok.getLocation());
}


/// ParseAsmStringLiteral - This is just a normal string-literal, but is not
/// allowed to be a wide string, and is not subject to character translation.
///
/// [GNU] asm-string-literal:
///         string-literal
///
Parser::ExprResult Parser::ParseAsmStringLiteral() {
  if (!isTokenStringLiteral()) {
    Diag(Tok, diag::err_expected_string_literal);
    return ExprError();
  }

  ExprResult Res(ParseStringLiteralExpression());
  if (Res.isInvalid()) return move(Res);

  // TODO: Diagnose: wide string literal in 'asm'

  return move(Res);
}

/// ParseSimpleAsm
///
/// [GNU] simple-asm-expr:
///         'asm' '(' asm-string-literal ')'
///
Parser::ExprResult Parser::ParseSimpleAsm(SourceLocation *EndLoc) {
  assert(Tok.is(tok::kw_asm) && "Not an asm!");
  SourceLocation Loc = ConsumeToken();

  if (Tok.is(tok::kw_volatile)) {
    // Remove from the end of 'asm' to the end of 'volatile'.
    SourceRange RemovalRange(PP.getLocForEndOfToken(Loc),
                             PP.getLocForEndOfToken(Tok.getLocation()));

    Diag(Tok, diag::warn_file_asm_volatile)
      << FixItHint::CreateRemoval(RemovalRange);
    ConsumeToken();
  }

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "asm";
    return ExprError();
  }

  Loc = ConsumeParen();

  ExprResult Result(ParseAsmStringLiteral());

  if (Result.isInvalid()) {
    SkipUntil(tok::r_paren, true, true);
    if (EndLoc)
      *EndLoc = Tok.getLocation();
    ConsumeAnyToken();
  } else {
    Loc = MatchRHSPunctuation(tok::r_paren, Loc);
    if (EndLoc)
      *EndLoc = Loc;
  }

  return move(Result);
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
/// Actions.getTypeName will not be needed to be called again (e.g. getTypeName
/// will not be called twice, once to check whether we have a declaration
/// specifier, and another one to get the actual type inside
/// ParseDeclarationSpecifiers).
///
/// This returns true if an error occurred.
///
/// Note that this routine emits an error if you call it with ::new or ::delete
/// as the current tokens, so only call it in contexts where these are invalid.
bool Parser::TryAnnotateTypeOrScopeToken(bool EnteringContext) {
  assert((Tok.is(tok::identifier) || Tok.is(tok::coloncolon)
          || Tok.is(tok::kw_typename) || Tok.is(tok::annot_cxxscope)) &&
         "Cannot be a type or scope token!");

  if (Tok.is(tok::kw_typename)) {
    // Parse a C++ typename-specifier, e.g., "typename T::type".
    //
    //   typename-specifier:
    //     'typename' '::' [opt] nested-name-specifier identifier
    //     'typename' '::' [opt] nested-name-specifier template [opt]
    //            simple-template-id
    SourceLocation TypenameLoc = ConsumeToken();
    CXXScopeSpec SS;
    if (ParseOptionalCXXScopeSpecifier(SS, /*ObjectType=*/ParsedType(), false))
      return true;
    if (!SS.isSet()) {
      Diag(Tok.getLocation(), diag::err_expected_qualified_after_typename);
      return true;
    }

    TypeResult Ty;
    if (Tok.is(tok::identifier)) {
      // FIXME: check whether the next token is '<', first!
      Ty = Actions.ActOnTypenameType(getCurScope(), TypenameLoc, SS, 
                                     *Tok.getIdentifierInfo(),
                                     Tok.getLocation());
    } else if (Tok.is(tok::annot_template_id)) {
      TemplateIdAnnotation *TemplateId
        = static_cast<TemplateIdAnnotation *>(Tok.getAnnotationValue());
      if (TemplateId->Kind == TNK_Function_template) {
        Diag(Tok, diag::err_typename_refers_to_non_type_template)
          << Tok.getAnnotationRange();
        return true;
      }

      ASTTemplateArgsPtr TemplateArgsPtr(Actions,
                                         TemplateId->getTemplateArgs(),
                                         TemplateId->NumArgs);
      
      Ty = Actions.ActOnTypenameType(getCurScope(), TypenameLoc, SS,
                                     /*FIXME:*/SourceLocation(),
                                     TemplateId->Template,
                                     TemplateId->TemplateNameLoc,
                                     TemplateId->LAngleLoc,
                                     TemplateArgsPtr, 
                                     TemplateId->RAngleLoc);
      TemplateId->Destroy();
    } else {
      Diag(Tok, diag::err_expected_type_name_after_typename)
        << SS.getRange();
      return true;
    }

    SourceLocation EndLoc = Tok.getLastLoc();
    Tok.setKind(tok::annot_typename);
    setTypeAnnotation(Tok, Ty.isInvalid() ? ParsedType() : Ty.get());
    Tok.setAnnotationEndLoc(EndLoc);
    Tok.setLocation(TypenameLoc);
    PP.AnnotateCachedTokens(Tok);
    return false;
  }

  // Remembers whether the token was originally a scope annotation.
  bool wasScopeAnnotation = Tok.is(tok::annot_cxxscope);

  CXXScopeSpec SS;
  if (getLang().CPlusPlus)
    if (ParseOptionalCXXScopeSpecifier(SS, ParsedType(), EnteringContext))
      return true;

  if (Tok.is(tok::identifier)) {
    // Determine whether the identifier is a type name.
    if (ParsedType Ty = Actions.getTypeName(*Tok.getIdentifierInfo(),
                                            Tok.getLocation(), getCurScope(),
                                            &SS, false, 
                                            NextToken().is(tok::period),
                                            ParsedType(),
                                            /*NonTrivialTypeSourceInfo*/true)) {
      // This is a typename. Replace the current token in-place with an
      // annotation type token.
      Tok.setKind(tok::annot_typename);
      setTypeAnnotation(Tok, Ty);
      Tok.setAnnotationEndLoc(Tok.getLocation());
      if (SS.isNotEmpty()) // it was a C++ qualified type name.
        Tok.setLocation(SS.getBeginLoc());

      // In case the tokens were cached, have Preprocessor replace
      // them with the annotation token.
      PP.AnnotateCachedTokens(Tok);
      return false;
    }

    if (!getLang().CPlusPlus) {
      // If we're in C, we can't have :: tokens at all (the lexer won't return
      // them).  If the identifier is not a type, then it can't be scope either,
      // just early exit.
      return false;
    }

    // If this is a template-id, annotate with a template-id or type token.
    if (NextToken().is(tok::less)) {
      TemplateTy Template;
      UnqualifiedId TemplateName;
      TemplateName.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
      bool MemberOfUnknownSpecialization;
      if (TemplateNameKind TNK
          = Actions.isTemplateName(getCurScope(), SS,
                                   /*hasTemplateKeyword=*/false, TemplateName,
                                   /*ObjectType=*/ ParsedType(),
                                   EnteringContext,
                                   Template, MemberOfUnknownSpecialization)) {
        // Consume the identifier.
        ConsumeToken();
        if (AnnotateTemplateIdToken(Template, TNK, SS, TemplateName)) {
          // If an unrecoverable error occurred, we need to return true here,
          // because the token stream is in a damaged state.  We may not return
          // a valid identifier.
          return true;
        }
      }
    }

    // The current token, which is either an identifier or a
    // template-id, is not part of the annotation. Fall through to
    // push that token back into the stream and complete the C++ scope
    // specifier annotation.
  }

  if (Tok.is(tok::annot_template_id)) {
    TemplateIdAnnotation *TemplateId
      = static_cast<TemplateIdAnnotation *>(Tok.getAnnotationValue());
    if (TemplateId->Kind == TNK_Type_template) {
      // A template-id that refers to a type was parsed into a
      // template-id annotation in a context where we weren't allowed
      // to produce a type annotation token. Update the template-id
      // annotation token to a type annotation token now.
      AnnotateTemplateIdTokenAsType();
      return false;
    }
  }

  if (SS.isEmpty())
    return false;

  // A C++ scope specifier that isn't followed by a typename.
  // Push the current token back into the token stream (or revert it if it is
  // cached) and use an annotation scope token for current token.
  if (PP.isBacktrackEnabled())
    PP.RevertCachedTokens(1);
  else
    PP.EnterToken(Tok);
  Tok.setKind(tok::annot_cxxscope);
  Tok.setAnnotationValue(Actions.SaveNestedNameSpecifierAnnotation(SS));
  Tok.setAnnotationRange(SS.getRange());

  // In case the tokens were cached, have Preprocessor replace them
  // with the annotation token.  We don't need to do this if we've
  // just reverted back to the state we were in before being called.
  if (!wasScopeAnnotation)
    PP.AnnotateCachedTokens(Tok);
  return false;
}

/// TryAnnotateScopeToken - Like TryAnnotateTypeOrScopeToken but only
/// annotates C++ scope specifiers and template-ids.  This returns
/// true if the token was annotated or there was an error that could not be
/// recovered from.
///
/// Note that this routine emits an error if you call it with ::new or ::delete
/// as the current tokens, so only call it in contexts where these are invalid.
bool Parser::TryAnnotateCXXScopeToken(bool EnteringContext) {
  assert(getLang().CPlusPlus &&
         "Call sites of this function should be guarded by checking for C++");
  assert((Tok.is(tok::identifier) || Tok.is(tok::coloncolon)) &&
         "Cannot be a type or scope token!");

  CXXScopeSpec SS;
  if (ParseOptionalCXXScopeSpecifier(SS, ParsedType(), EnteringContext))
    return true;
  if (SS.isEmpty())
    return false;

  // Push the current token back into the token stream (or revert it if it is
  // cached) and use an annotation scope token for current token.
  if (PP.isBacktrackEnabled())
    PP.RevertCachedTokens(1);
  else
    PP.EnterToken(Tok);
  Tok.setKind(tok::annot_cxxscope);
  Tok.setAnnotationValue(Actions.SaveNestedNameSpecifierAnnotation(SS));
  Tok.setAnnotationRange(SS.getRange());

  // In case the tokens were cached, have Preprocessor replace them with the
  // annotation token.
  PP.AnnotateCachedTokens(Tok);
  return false;
}

bool Parser::isTokenEqualOrMistypedEqualEqual(unsigned DiagID) {
  if (Tok.is(tok::equalequal)) {
    // We have '==' in a context that we would expect a '='.
    // The user probably made a typo, intending to type '='. Emit diagnostic,
    // fixit hint to turn '==' -> '=' and continue as if the user typed '='.
    Diag(Tok, DiagID)
      << FixItHint::CreateReplacement(SourceRange(Tok.getLocation()),
                                      getTokenSimpleSpelling(tok::equal));
    return true;
  }

  return Tok.is(tok::equal);
}

void Parser::CodeCompletionRecovery() {
  for (Scope *S = getCurScope(); S; S = S->getParent()) {
    if (S->getFlags() & Scope::FnScope) {
      Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_RecoveryInFunction);
      return;
    }
    
    if (S->getFlags() & Scope::ClassScope) {
      Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Class);
      return;
    }
  }
  
  Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Namespace);
}

// Anchor the Parser::FieldCallback vtable to this translation unit.
// We use a spurious method instead of the destructor because
// destroying FieldCallbacks can actually be slightly
// performance-sensitive.
void Parser::FieldCallback::_anchor() {
}

// Code-completion pass-through functions

void Parser::CodeCompleteDirective(bool InConditional) {
  Actions.CodeCompletePreprocessorDirective(InConditional);
}

void Parser::CodeCompleteInConditionalExclusion() {
  Actions.CodeCompleteInPreprocessorConditionalExclusion(getCurScope());
}

void Parser::CodeCompleteMacroName(bool IsDefinition) {
  Actions.CodeCompletePreprocessorMacroName(IsDefinition);
}

void Parser::CodeCompletePreprocessorExpression() { 
  Actions.CodeCompletePreprocessorExpression();
}

void Parser::CodeCompleteMacroArgument(IdentifierInfo *Macro,
                                       MacroInfo *MacroInfo,
                                       unsigned ArgumentIndex) {
  Actions.CodeCompletePreprocessorMacroArgument(getCurScope(), Macro, MacroInfo, 
                                                ArgumentIndex);
}

void Parser::CodeCompleteNaturalLanguage() {
  Actions.CodeCompleteNaturalLanguage();
}
