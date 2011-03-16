//===--- ParseCXXInlineMethods.cpp - C++ class inline methods parsing------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements parsing for C++ class inline methods.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
using namespace clang;

/// ParseCXXInlineMethodDef - We parsed and verified that the specified
/// Declarator is a well formed C++ inline method definition. Now lex its body
/// and store its tokens for parsing after the C++ class is complete.
Decl *Parser::ParseCXXInlineMethodDef(AccessSpecifier AS, ParsingDeclarator &D,
                                const ParsedTemplateInfo &TemplateInfo,
                                const VirtSpecifiers& VS) {
  assert(D.isFunctionDeclarator() && "This isn't a function declarator!");
  assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try)) &&
         "Current token not a '{', ':' or 'try'!");

  MultiTemplateParamsArg TemplateParams(Actions,
          TemplateInfo.TemplateParams ? TemplateInfo.TemplateParams->data() : 0,
          TemplateInfo.TemplateParams ? TemplateInfo.TemplateParams->size() : 0);

  Decl *FnD;
  if (D.getDeclSpec().isFriendSpecified())
    // FIXME: Friend templates
    FnD = Actions.ActOnFriendFunctionDecl(getCurScope(), D, true,
                                          move(TemplateParams));
  else { // FIXME: pass template information through
    if (VS.isOverrideSpecified())
      Diag(VS.getOverrideLoc(), diag::ext_override_inline) << "override";
    if (VS.isFinalSpecified())
      Diag(VS.getFinalLoc(), diag::ext_override_inline) << "final";
    if (VS.isNewSpecified())
      Diag(VS.getNewLoc(), diag::ext_override_inline) << "new";

    FnD = Actions.ActOnCXXMemberDeclarator(getCurScope(), AS, D,
                                           move(TemplateParams), 0, 
                                           VS, 0, /*IsDefinition*/true);
  }

  HandleMemberFunctionDefaultArgs(D, FnD);

  D.complete(FnD);

  // Consume the tokens and store them for later parsing.

  LexedMethod* LM = new LexedMethod(this, FnD);
  getCurrentClass().LateParsedDeclarations.push_back(LM);
  LM->TemplateScope = getCurScope()->isTemplateParamScope();
  CachedTokens &Toks = LM->Toks;

  tok::TokenKind kind = Tok.getKind();
  // We may have a constructor initializer or function-try-block here.
  if (kind == tok::colon || kind == tok::kw_try) {
    // Consume everything up to (and including) the left brace.
    if (!ConsumeAndStoreUntil(tok::l_brace, Toks)) {
      // We didn't find the left-brace we expected after the
      // constructor initializer.
      if (Tok.is(tok::semi)) {
        // We found a semicolon; complain, consume the semicolon, and
        // don't try to parse this method later.
        Diag(Tok.getLocation(), diag::err_expected_lbrace);
        ConsumeAnyToken();
        delete getCurrentClass().LateParsedDeclarations.back();
        getCurrentClass().LateParsedDeclarations.pop_back();
        return FnD;
      }
    }

  } else {
    // Begin by storing the '{' token.
    Toks.push_back(Tok);
    ConsumeBrace();
  }
  // Consume everything up to (and including) the matching right brace.
  ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);

  // If we're in a function-try-block, we need to store all the catch blocks.
  if (kind == tok::kw_try) {
    while (Tok.is(tok::kw_catch)) {
      ConsumeAndStoreUntil(tok::l_brace, Toks, /*StopAtSemi=*/false);
      ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
    }
  }

  return FnD;
}

Parser::LateParsedDeclaration::~LateParsedDeclaration() {}
void Parser::LateParsedDeclaration::ParseLexedMethodDeclarations() {}
void Parser::LateParsedDeclaration::ParseLexedMethodDefs() {}

Parser::LateParsedClass::LateParsedClass(Parser *P, ParsingClass *C)
  : Self(P), Class(C) {}

Parser::LateParsedClass::~LateParsedClass() {
  Self->DeallocateParsedClasses(Class);
}

void Parser::LateParsedClass::ParseLexedMethodDeclarations() {
  Self->ParseLexedMethodDeclarations(*Class);
}

void Parser::LateParsedClass::ParseLexedMethodDefs() {
  Self->ParseLexedMethodDefs(*Class);
}

void Parser::LateParsedMethodDeclaration::ParseLexedMethodDeclarations() {
  Self->ParseLexedMethodDeclaration(*this);
}

void Parser::LexedMethod::ParseLexedMethodDefs() {
  Self->ParseLexedMethodDef(*this);
}

/// ParseLexedMethodDeclarations - We finished parsing the member
/// specification of a top (non-nested) C++ class. Now go over the
/// stack of method declarations with some parts for which parsing was
/// delayed (such as default arguments) and parse them.
void Parser::ParseLexedMethodDeclarations(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope ClassTemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  if (HasTemplateScope)
    Actions.ActOnReenterTemplateScope(getCurScope(), Class.TagOrTemplate);

  // The current scope is still active if we're the top-level class.
  // Otherwise we'll need to push and enter a new scope.
  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);
  if (HasClassScope)
    Actions.ActOnStartDelayedMemberDeclarations(getCurScope(), Class.TagOrTemplate);

  for (size_t i = 0; i < Class.LateParsedDeclarations.size(); ++i) {
    Class.LateParsedDeclarations[i]->ParseLexedMethodDeclarations();
  }

  if (HasClassScope)
    Actions.ActOnFinishDelayedMemberDeclarations(getCurScope(), Class.TagOrTemplate);
}

void Parser::ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM) {
  // If this is a member template, introduce the template parameter scope.
  ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
  if (LM.TemplateScope)
    Actions.ActOnReenterTemplateScope(getCurScope(), LM.Method);

  // Start the delayed C++ method declaration
  Actions.ActOnStartDelayedCXXMethodDeclaration(getCurScope(), LM.Method);

  // Introduce the parameters into scope and parse their default
  // arguments.
  ParseScope PrototypeScope(this,
                            Scope::FunctionPrototypeScope|Scope::DeclScope);
  for (unsigned I = 0, N = LM.DefaultArgs.size(); I != N; ++I) {
    // Introduce the parameter into scope.
    Actions.ActOnDelayedCXXMethodParameter(getCurScope(), LM.DefaultArgs[I].Param);

    if (CachedTokens *Toks = LM.DefaultArgs[I].Toks) {
      // Save the current token position.
      SourceLocation origLoc = Tok.getLocation();

      // Parse the default argument from its saved token stream.
      Toks->push_back(Tok); // So that the current token doesn't get lost
      PP.EnterTokenStream(&Toks->front(), Toks->size(), true, false);

      // Consume the previously-pushed token.
      ConsumeAnyToken();

      // Consume the '='.
      assert(Tok.is(tok::equal) && "Default argument not starting with '='");
      SourceLocation EqualLoc = ConsumeToken();

      // The argument isn't actually potentially evaluated unless it is
      // used.
      EnterExpressionEvaluationContext Eval(Actions,
                                            Sema::PotentiallyEvaluatedIfUsed);

      ExprResult DefArgResult(ParseAssignmentExpression());
      if (DefArgResult.isInvalid())
        Actions.ActOnParamDefaultArgumentError(LM.DefaultArgs[I].Param);
      else {
        if (Tok.is(tok::cxx_defaultarg_end))
          ConsumeToken();
        else
          Diag(Tok.getLocation(), diag::err_default_arg_unparsed);
        Actions.ActOnParamDefaultArgument(LM.DefaultArgs[I].Param, EqualLoc,
                                          DefArgResult.take());
      }

      assert(!PP.getSourceManager().isBeforeInTranslationUnit(origLoc,
                                                         Tok.getLocation()) &&
             "ParseAssignmentExpression went over the default arg tokens!");
      // There could be leftover tokens (e.g. because of an error).
      // Skip through until we reach the original token position.
      while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();

      delete Toks;
      LM.DefaultArgs[I].Toks = 0;
    }
  }
  PrototypeScope.Exit();

  // Finish the delayed C++ method declaration.
  Actions.ActOnFinishDelayedCXXMethodDeclaration(getCurScope(), LM.Method);
}

/// ParseLexedMethodDefs - We finished parsing the member specification of a top
/// (non-nested) C++ class. Now go over the stack of lexed methods that were
/// collected during its parsing and parse them all.
void Parser::ParseLexedMethodDefs(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope ClassTemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  if (HasTemplateScope)
    Actions.ActOnReenterTemplateScope(getCurScope(), Class.TagOrTemplate);

  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);

  for (size_t i = 0; i < Class.LateParsedDeclarations.size(); ++i) {
    Class.LateParsedDeclarations[i]->ParseLexedMethodDefs();
  }
}

void Parser::ParseLexedMethodDef(LexedMethod &LM) {
  // If this is a member template, introduce the template parameter scope.
  ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
  if (LM.TemplateScope)
    Actions.ActOnReenterTemplateScope(getCurScope(), LM.D);

  // Save the current token position.
  SourceLocation origLoc = Tok.getLocation();

  assert(!LM.Toks.empty() && "Empty body!");
  // Append the current token at the end of the new token stream so that it
  // doesn't get lost.
  LM.Toks.push_back(Tok);
  PP.EnterTokenStream(LM.Toks.data(), LM.Toks.size(), true, false);

  // Consume the previously pushed token.
  ConsumeAnyToken();
  assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try))
         && "Inline method not starting with '{', ':' or 'try'");

  // Parse the method body. Function body parsing code is similar enough
  // to be re-used for method bodies as well.
  ParseScope FnScope(this, Scope::FnScope|Scope::DeclScope);
  Actions.ActOnStartOfFunctionDef(getCurScope(), LM.D);

  if (Tok.is(tok::kw_try)) {
    ParseFunctionTryBlock(LM.D, FnScope);
    assert(!PP.getSourceManager().isBeforeInTranslationUnit(origLoc,
                                                         Tok.getLocation()) &&
           "ParseFunctionTryBlock went over the cached tokens!");
    // There could be leftover tokens (e.g. because of an error).
    // Skip through until we reach the original token position.
    while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
      ConsumeAnyToken();
    return;
  }
  if (Tok.is(tok::colon)) {
    ParseConstructorInitializer(LM.D);

    // Error recovery.
    if (!Tok.is(tok::l_brace)) {
      FnScope.Exit();
      Actions.ActOnFinishFunctionBody(LM.D, 0);
      return;
    }
  } else
    Actions.ActOnDefaultCtorInitializers(LM.D);

  ParseFunctionStatementBody(LM.D, FnScope);

  if (Tok.getLocation() != origLoc) {
    // Due to parsing error, we either went over the cached tokens or
    // there are still cached tokens left. If it's the latter case skip the
    // leftover tokens.
    // Since this is an uncommon situation that should be avoided, use the
    // expensive isBeforeInTranslationUnit call.
    if (PP.getSourceManager().isBeforeInTranslationUnit(Tok.getLocation(),
                                                        origLoc))
      while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();

  }
}

/// ConsumeAndStoreUntil - Consume and store the token at the passed token
/// container until the token 'T' is reached (which gets
/// consumed/stored too, if ConsumeFinalToken).
/// If StopAtSemi is true, then we will stop early at a ';' character.
/// Returns true if token 'T1' or 'T2' was found.
/// NOTE: This is a specialized version of Parser::SkipUntil.
bool Parser::ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                                  CachedTokens &Toks,
                                  bool StopAtSemi, bool ConsumeFinalToken) {
  // We always want this function to consume at least one token if the first
  // token isn't T and if not at EOF.
  bool isFirstTokenConsumed = true;
  while (1) {
    // If we found one of the tokens, stop and return true.
    if (Tok.is(T1) || Tok.is(T2)) {
      if (ConsumeFinalToken) {
        Toks.push_back(Tok);
        ConsumeAnyToken();
      }
      return true;
    }

    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;

    case tok::l_paren:
      // Recursively consume properly-nested parens.
      Toks.push_back(Tok);
      ConsumeParen();
      ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_square:
      // Recursively consume properly-nested square brackets.
      Toks.push_back(Tok);
      ConsumeBracket();
      ConsumeAndStoreUntil(tok::r_square, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_brace:
      // Recursively consume properly-nested braces.
      Toks.push_back(Tok);
      ConsumeBrace();
      ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
      break;

    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBrace();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
      Toks.push_back(Tok);
      ConsumeStringToken();
      break;
    case tok::semi:
      if (StopAtSemi)
        return false;
      // FALL THROUGH.
    default:
      // consume this token.
      Toks.push_back(Tok);
      ConsumeToken();
      break;
    }
    isFirstTokenConsumed = false;
  }
}
