//===--- ParseExprCXX.cpp - C++ Expression Parsing ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expression parsing implementation for C++.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ParsedTemplate.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

static int SelectDigraphErrorMessage(tok::TokenKind Kind) {
  switch (Kind) {
    case tok::kw_template:         return 0;
    case tok::kw_const_cast:       return 1;
    case tok::kw_dynamic_cast:     return 2;
    case tok::kw_reinterpret_cast: return 3;
    case tok::kw_static_cast:      return 4;
    default:
      llvm_unreachable("Unknown type for digraph error message.");
  }
}

// Are the two tokens adjacent in the same source file?
static bool AreTokensAdjacent(Preprocessor &PP, Token &First, Token &Second) {
  SourceManager &SM = PP.getSourceManager();
  SourceLocation FirstLoc = SM.getSpellingLoc(First.getLocation());
  SourceLocation FirstEnd = FirstLoc.getLocWithOffset(First.getLength());
  return FirstEnd == SM.getSpellingLoc(Second.getLocation());
}

// Suggest fixit for "<::" after a cast.
static void FixDigraph(Parser &P, Preprocessor &PP, Token &DigraphToken,
                       Token &ColonToken, tok::TokenKind Kind, bool AtDigraph) {
  // Pull '<:' and ':' off token stream.
  if (!AtDigraph)
    PP.Lex(DigraphToken);
  PP.Lex(ColonToken);

  SourceRange Range;
  Range.setBegin(DigraphToken.getLocation());
  Range.setEnd(ColonToken.getLocation());
  P.Diag(DigraphToken.getLocation(), diag::err_missing_whitespace_digraph)
      << SelectDigraphErrorMessage(Kind)
      << FixItHint::CreateReplacement(Range, "< ::");

  // Update token information to reflect their change in token type.
  ColonToken.setKind(tok::coloncolon);
  ColonToken.setLocation(ColonToken.getLocation().getLocWithOffset(-1));
  ColonToken.setLength(2);
  DigraphToken.setKind(tok::less);
  DigraphToken.setLength(1);

  // Push new tokens back to token stream.
  PP.EnterToken(ColonToken);
  if (!AtDigraph)
    PP.EnterToken(DigraphToken);
}

// Check for '<::' which should be '< ::' instead of '[:' when following
// a template name.
void Parser::CheckForTemplateAndDigraph(Token &Next, ParsedType ObjectType,
                                        bool EnteringContext,
                                        IdentifierInfo &II, CXXScopeSpec &SS) {
  if (!Next.is(tok::l_square) || Next.getLength() != 2)
    return;

  Token SecondToken = GetLookAheadToken(2);
  if (!SecondToken.is(tok::colon) || !AreTokensAdjacent(PP, Next, SecondToken))
    return;

  TemplateTy Template;
  UnqualifiedId TemplateName;
  TemplateName.setIdentifier(&II, Tok.getLocation());
  bool MemberOfUnknownSpecialization;
  if (!Actions.isTemplateName(getCurScope(), SS, /*hasTemplateKeyword=*/false,
                              TemplateName, ObjectType, EnteringContext,
                              Template, MemberOfUnknownSpecialization))
    return;

  FixDigraph(*this, PP, Next, SecondToken, tok::kw_template,
             /*AtDigraph*/false);
}

/// \brief Parse global scope or nested-name-specifier if present.
///
/// Parses a C++ global scope specifier ('::') or nested-name-specifier (which
/// may be preceded by '::'). Note that this routine will not parse ::new or
/// ::delete; it will just leave them in the token stream.
///
///       '::'[opt] nested-name-specifier
///       '::'
///
///       nested-name-specifier:
///         type-name '::'
///         namespace-name '::'
///         nested-name-specifier identifier '::'
///         nested-name-specifier 'template'[opt] simple-template-id '::'
///
///
/// \param SS the scope specifier that will be set to the parsed
/// nested-name-specifier (or empty)
///
/// \param ObjectType if this nested-name-specifier is being parsed following
/// the "." or "->" of a member access expression, this parameter provides the
/// type of the object whose members are being accessed.
///
/// \param EnteringContext whether we will be entering into the context of
/// the nested-name-specifier after parsing it.
///
/// \param MayBePseudoDestructor When non-NULL, points to a flag that
/// indicates whether this nested-name-specifier may be part of a
/// pseudo-destructor name. In this case, the flag will be set false
/// if we don't actually end up parsing a destructor name. Moreorover,
/// if we do end up determining that we are parsing a destructor name,
/// the last component of the nested-name-specifier is not parsed as
/// part of the scope specifier.

/// member access expression, e.g., the \p T:: in \p p->T::m.
///
/// \returns true if there was an error parsing a scope specifier
bool Parser::ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
                                            ParsedType ObjectType,
                                            bool EnteringContext,
                                            bool *MayBePseudoDestructor,
                                            bool IsTypename) {
  assert(getLangOpts().CPlusPlus &&
         "Call sites of this function should be guarded by checking for C++");

  if (Tok.is(tok::annot_cxxscope)) {
    Actions.RestoreNestedNameSpecifierAnnotation(Tok.getAnnotationValue(),
                                                 Tok.getAnnotationRange(),
                                                 SS);
    ConsumeToken();
    return false;
  }

  bool HasScopeSpecifier = false;

  if (Tok.is(tok::coloncolon)) {
    // ::new and ::delete aren't nested-name-specifiers.
    tok::TokenKind NextKind = NextToken().getKind();
    if (NextKind == tok::kw_new || NextKind == tok::kw_delete)
      return false;

    // '::' - Global scope qualifier.
    if (Actions.ActOnCXXGlobalScopeSpecifier(getCurScope(), ConsumeToken(), SS))
      return true;
    
    HasScopeSpecifier = true;
  }

  bool CheckForDestructor = false;
  if (MayBePseudoDestructor && *MayBePseudoDestructor) {
    CheckForDestructor = true;
    *MayBePseudoDestructor = false;
  }

  if (Tok.is(tok::kw_decltype) || Tok.is(tok::annot_decltype)) {
    DeclSpec DS(AttrFactory);
    SourceLocation DeclLoc = Tok.getLocation();
    SourceLocation EndLoc  = ParseDecltypeSpecifier(DS);
    if (Tok.isNot(tok::coloncolon)) {
      AnnotateExistingDecltypeSpecifier(DS, DeclLoc, EndLoc);
      return false;
    }
    
    SourceLocation CCLoc = ConsumeToken();
    if (Actions.ActOnCXXNestedNameSpecifierDecltype(SS, DS, CCLoc))
      SS.SetInvalid(SourceRange(DeclLoc, CCLoc));

    HasScopeSpecifier = true;
  }

  while (true) {
    if (HasScopeSpecifier) {
      // C++ [basic.lookup.classref]p5:
      //   If the qualified-id has the form
      //
      //       ::class-name-or-namespace-name::...
      //
      //   the class-name-or-namespace-name is looked up in global scope as a
      //   class-name or namespace-name.
      //
      // To implement this, we clear out the object type as soon as we've
      // seen a leading '::' or part of a nested-name-specifier.
      ObjectType = ParsedType();
      
      if (Tok.is(tok::code_completion)) {
        // Code completion for a nested-name-specifier, where the code
        // code completion token follows the '::'.
        Actions.CodeCompleteQualifiedId(getCurScope(), SS, EnteringContext);
        // Include code completion token into the range of the scope otherwise
        // when we try to annotate the scope tokens the dangling code completion
        // token will cause assertion in
        // Preprocessor::AnnotatePreviousCachedTokens.
        SS.setEndLoc(Tok.getLocation());
        cutOffParsing();
        return true;
      }
    }

    // nested-name-specifier:
    //   nested-name-specifier 'template'[opt] simple-template-id '::'

    // Parse the optional 'template' keyword, then make sure we have
    // 'identifier <' after it.
    if (Tok.is(tok::kw_template)) {
      // If we don't have a scope specifier or an object type, this isn't a
      // nested-name-specifier, since they aren't allowed to start with
      // 'template'.
      if (!HasScopeSpecifier && !ObjectType)
        break;

      TentativeParsingAction TPA(*this);
      SourceLocation TemplateKWLoc = ConsumeToken();
      
      UnqualifiedId TemplateName;
      if (Tok.is(tok::identifier)) {
        // Consume the identifier.
        TemplateName.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
        ConsumeToken();
      } else if (Tok.is(tok::kw_operator)) {
        if (ParseUnqualifiedIdOperator(SS, EnteringContext, ObjectType, 
                                       TemplateName)) {
          TPA.Commit();
          break;
        }
        
        if (TemplateName.getKind() != UnqualifiedId::IK_OperatorFunctionId &&
            TemplateName.getKind() != UnqualifiedId::IK_LiteralOperatorId) {
          Diag(TemplateName.getSourceRange().getBegin(),
               diag::err_id_after_template_in_nested_name_spec)
            << TemplateName.getSourceRange();
          TPA.Commit();
          break;
        }
      } else {
        TPA.Revert();
        break;
      }

      // If the next token is not '<', we have a qualified-id that refers
      // to a template name, such as T::template apply, but is not a 
      // template-id.
      if (Tok.isNot(tok::less)) {
        TPA.Revert();
        break;
      }        
      
      // Commit to parsing the template-id.
      TPA.Commit();
      TemplateTy Template;
      if (TemplateNameKind TNK
          = Actions.ActOnDependentTemplateName(getCurScope(),
                                               SS, TemplateKWLoc, TemplateName,
                                               ObjectType, EnteringContext,
                                               Template)) {
        if (AnnotateTemplateIdToken(Template, TNK, SS, TemplateKWLoc,
                                    TemplateName, false))
          return true;
      } else
        return true;

      continue;
    }

    if (Tok.is(tok::annot_template_id) && NextToken().is(tok::coloncolon)) {
      // We have
      //
      //   simple-template-id '::'
      //
      // So we need to check whether the simple-template-id is of the
      // right kind (it should name a type or be dependent), and then
      // convert it into a type within the nested-name-specifier.
      TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);
      if (CheckForDestructor && GetLookAheadToken(2).is(tok::tilde)) {
        *MayBePseudoDestructor = true;
        return false;
      }

      // Consume the template-id token.
      ConsumeToken();
      
      assert(Tok.is(tok::coloncolon) && "NextToken() not working properly!");
      SourceLocation CCLoc = ConsumeToken();

      HasScopeSpecifier = true;
      
      ASTTemplateArgsPtr TemplateArgsPtr(Actions,
                                         TemplateId->getTemplateArgs(),
                                         TemplateId->NumArgs);
      
      if (Actions.ActOnCXXNestedNameSpecifier(getCurScope(),
                                              SS,
                                              TemplateId->TemplateKWLoc,
                                              TemplateId->Template,
                                              TemplateId->TemplateNameLoc,
                                              TemplateId->LAngleLoc,
                                              TemplateArgsPtr,
                                              TemplateId->RAngleLoc,
                                              CCLoc,
                                              EnteringContext)) {
        SourceLocation StartLoc 
          = SS.getBeginLoc().isValid()? SS.getBeginLoc()
                                      : TemplateId->TemplateNameLoc;
        SS.SetInvalid(SourceRange(StartLoc, CCLoc));
      }

      continue;
    }


    // The rest of the nested-name-specifier possibilities start with
    // tok::identifier.
    if (Tok.isNot(tok::identifier))
      break;

    IdentifierInfo &II = *Tok.getIdentifierInfo();

    // nested-name-specifier:
    //   type-name '::'
    //   namespace-name '::'
    //   nested-name-specifier identifier '::'
    Token Next = NextToken();
    
    // If we get foo:bar, this is almost certainly a typo for foo::bar.  Recover
    // and emit a fixit hint for it.
    if (Next.is(tok::colon) && !ColonIsSacred) {
      if (Actions.IsInvalidUnlessNestedName(getCurScope(), SS, II, 
                                            Tok.getLocation(), 
                                            Next.getLocation(), ObjectType,
                                            EnteringContext) &&
          // If the token after the colon isn't an identifier, it's still an
          // error, but they probably meant something else strange so don't
          // recover like this.
          PP.LookAhead(1).is(tok::identifier)) {
        Diag(Next, diag::err_unexected_colon_in_nested_name_spec)
          << FixItHint::CreateReplacement(Next.getLocation(), "::");
        
        // Recover as if the user wrote '::'.
        Next.setKind(tok::coloncolon);
      }
    }
    
    if (Next.is(tok::coloncolon)) {
      if (CheckForDestructor && GetLookAheadToken(2).is(tok::tilde) &&
          !Actions.isNonTypeNestedNameSpecifier(getCurScope(), SS, Tok.getLocation(),
                                                II, ObjectType)) {
        *MayBePseudoDestructor = true;
        return false;
      }

      // We have an identifier followed by a '::'. Lookup this name
      // as the name in a nested-name-specifier.
      SourceLocation IdLoc = ConsumeToken();
      assert((Tok.is(tok::coloncolon) || Tok.is(tok::colon)) &&
             "NextToken() not working properly!");
      SourceLocation CCLoc = ConsumeToken();

      HasScopeSpecifier = true;
      if (Actions.ActOnCXXNestedNameSpecifier(getCurScope(), II, IdLoc, CCLoc,
                                              ObjectType, EnteringContext, SS))
        SS.SetInvalid(SourceRange(IdLoc, CCLoc));
      
      continue;
    }

    CheckForTemplateAndDigraph(Next, ObjectType, EnteringContext, II, SS);

    // nested-name-specifier:
    //   type-name '<'
    if (Next.is(tok::less)) {
      TemplateTy Template;
      UnqualifiedId TemplateName;
      TemplateName.setIdentifier(&II, Tok.getLocation());
      bool MemberOfUnknownSpecialization;
      if (TemplateNameKind TNK = Actions.isTemplateName(getCurScope(), SS, 
                                              /*hasTemplateKeyword=*/false,
                                                        TemplateName,
                                                        ObjectType,
                                                        EnteringContext,
                                                        Template,
                                              MemberOfUnknownSpecialization)) {
        // We have found a template name, so annotate this token
        // with a template-id annotation. We do not permit the
        // template-id to be translated into a type annotation,
        // because some clients (e.g., the parsing of class template
        // specializations) still want to see the original template-id
        // token.
        ConsumeToken();
        if (AnnotateTemplateIdToken(Template, TNK, SS, SourceLocation(),
                                    TemplateName, false))
          return true;
        continue;
      } 
      
      if (MemberOfUnknownSpecialization && (ObjectType || SS.isSet()) && 
          (IsTypename || IsTemplateArgumentList(1))) {
        // We have something like t::getAs<T>, where getAs is a 
        // member of an unknown specialization. However, this will only
        // parse correctly as a template, so suggest the keyword 'template'
        // before 'getAs' and treat this as a dependent template name.
        unsigned DiagID = diag::err_missing_dependent_template_keyword;
        if (getLangOpts().MicrosoftExt)
          DiagID = diag::warn_missing_dependent_template_keyword;
        
        Diag(Tok.getLocation(), DiagID)
          << II.getName()
          << FixItHint::CreateInsertion(Tok.getLocation(), "template ");
        
        if (TemplateNameKind TNK 
              = Actions.ActOnDependentTemplateName(getCurScope(), 
                                                   SS, SourceLocation(),
                                                   TemplateName, ObjectType,
                                                   EnteringContext, Template)) {
          // Consume the identifier.
          ConsumeToken();
          if (AnnotateTemplateIdToken(Template, TNK, SS, SourceLocation(),
                                      TemplateName, false))
            return true;
        }
        else
          return true;     
                
        continue;        
      }
    }

    // We don't have any tokens that form the beginning of a
    // nested-name-specifier, so we're done.
    break;
  }

  // Even if we didn't see any pieces of a nested-name-specifier, we
  // still check whether there is a tilde in this position, which
  // indicates a potential pseudo-destructor.
  if (CheckForDestructor && Tok.is(tok::tilde))
    *MayBePseudoDestructor = true;

  return false;
}

/// ParseCXXIdExpression - Handle id-expression.
///
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
///       qualified-id:
///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
///         '::' unqualified-id
///
/// That way Sema can handle and report similar errors for namespaces and the
/// global scope.
///
/// The isAddressOfOperand parameter indicates that this id-expression is a
/// direct operand of the address-of operator. This is, besides member contexts,
/// the only place where a qualified-id naming a non-static class member may
/// appear.
///
ExprResult Parser::ParseCXXIdExpression(bool isAddressOfOperand) {
  // qualified-id:
  //   '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
  //   '::' unqualified-id
  //
  CXXScopeSpec SS;
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);

  SourceLocation TemplateKWLoc;
  UnqualifiedId Name;
  if (ParseUnqualifiedId(SS,
                         /*EnteringContext=*/false,
                         /*AllowDestructorName=*/false,
                         /*AllowConstructorName=*/false,
                         /*ObjectType=*/ ParsedType(),
                         TemplateKWLoc,
                         Name))
    return ExprError();

  // This is only the direct operand of an & operator if it is not
  // followed by a postfix-expression suffix.
  if (isAddressOfOperand && isPostfixExpressionSuffixStart())
    isAddressOfOperand = false;

  return Actions.ActOnIdExpression(getCurScope(), SS, TemplateKWLoc, Name,
                                   Tok.is(tok::l_paren), isAddressOfOperand);
}

/// ParseLambdaExpression - Parse a C++0x lambda expression.
///
///       lambda-expression:
///         lambda-introducer lambda-declarator[opt] compound-statement
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
///         identifier
///         '&' identifier
///         'this'
///
///       lambda-declarator:
///         '(' parameter-declaration-clause ')' attribute-specifier[opt]
///           'mutable'[opt] exception-specification[opt]
///           trailing-return-type[opt]
///
ExprResult Parser::ParseLambdaExpression() {
  // Parse lambda-introducer.
  LambdaIntroducer Intro;

  llvm::Optional<unsigned> DiagID(ParseLambdaIntroducer(Intro));
  if (DiagID) {
    Diag(Tok, DiagID.getValue());
    SkipUntil(tok::r_square);
    SkipUntil(tok::l_brace);
    SkipUntil(tok::r_brace);
    return ExprError();
  }

  return ParseLambdaExpressionAfterIntroducer(Intro);
}

/// TryParseLambdaExpression - Use lookahead and potentially tentative
/// parsing to determine if we are looking at a C++0x lambda expression, and parse
/// it if we are.
///
/// If we are not looking at a lambda expression, returns ExprError().
ExprResult Parser::TryParseLambdaExpression() {
  assert(getLangOpts().CPlusPlus0x
         && Tok.is(tok::l_square)
         && "Not at the start of a possible lambda expression.");

  const Token Next = NextToken(), After = GetLookAheadToken(2);

  // If lookahead indicates this is a lambda...
  if (Next.is(tok::r_square) ||     // []
      Next.is(tok::equal) ||        // [=
      (Next.is(tok::amp) &&         // [&] or [&,
       (After.is(tok::r_square) ||
        After.is(tok::comma))) ||
      (Next.is(tok::identifier) &&  // [identifier]
       After.is(tok::r_square))) {
    return ParseLambdaExpression();
  }

  // If lookahead indicates an ObjC message send...
  // [identifier identifier
  if (Next.is(tok::identifier) && After.is(tok::identifier)) {
    return ExprEmpty();
  }

  // Here, we're stuck: lambda introducers and Objective-C message sends are
  // unambiguous, but it requires arbitrary lookhead.  [a,b,c,d,e,f,g] is a
  // lambda, and [a,b,c,d,e,f,g h] is a Objective-C message send.  Instead of
  // writing two routines to parse a lambda introducer, just try to parse
  // a lambda introducer first, and fall back if that fails.
  // (TryParseLambdaIntroducer never produces any diagnostic output.)
  LambdaIntroducer Intro;
  if (TryParseLambdaIntroducer(Intro))
    return ExprEmpty();
  return ParseLambdaExpressionAfterIntroducer(Intro);
}

/// ParseLambdaExpression - Parse a lambda introducer.
///
/// Returns a DiagnosticID if it hit something unexpected.
llvm::Optional<unsigned> Parser::ParseLambdaIntroducer(LambdaIntroducer &Intro){
  typedef llvm::Optional<unsigned> DiagResult;

  assert(Tok.is(tok::l_square) && "Lambda expressions begin with '['.");
  BalancedDelimiterTracker T(*this, tok::l_square);
  T.consumeOpen();

  Intro.Range.setBegin(T.getOpenLocation());

  bool first = true;

  // Parse capture-default.
  if (Tok.is(tok::amp) &&
      (NextToken().is(tok::comma) || NextToken().is(tok::r_square))) {
    Intro.Default = LCD_ByRef;
    Intro.DefaultLoc = ConsumeToken();
    first = false;
  } else if (Tok.is(tok::equal)) {
    Intro.Default = LCD_ByCopy;
    Intro.DefaultLoc = ConsumeToken();
    first = false;
  }

  while (Tok.isNot(tok::r_square)) {
    if (!first) {
      if (Tok.isNot(tok::comma)) {
        if (Tok.is(tok::code_completion)) {
          Actions.CodeCompleteLambdaIntroducer(getCurScope(), Intro, 
                                               /*AfterAmpersand=*/false);
          ConsumeCodeCompletionToken();
          break;
        }

        return DiagResult(diag::err_expected_comma_or_rsquare);
      }
      ConsumeToken();
    }

    if (Tok.is(tok::code_completion)) {
      // If we're in Objective-C++ and we have a bare '[', then this is more
      // likely to be a message receiver.
      if (getLangOpts().ObjC1 && first)
        Actions.CodeCompleteObjCMessageReceiver(getCurScope());
      else
        Actions.CodeCompleteLambdaIntroducer(getCurScope(), Intro, 
                                             /*AfterAmpersand=*/false);
      ConsumeCodeCompletionToken();
      break;
    }

    first = false;
    
    // Parse capture.
    LambdaCaptureKind Kind = LCK_ByCopy;
    SourceLocation Loc;
    IdentifierInfo* Id = 0;
    SourceLocation EllipsisLoc;
    
    if (Tok.is(tok::kw_this)) {
      Kind = LCK_This;
      Loc = ConsumeToken();
    } else {
      if (Tok.is(tok::amp)) {
        Kind = LCK_ByRef;
        ConsumeToken();

        if (Tok.is(tok::code_completion)) {
          Actions.CodeCompleteLambdaIntroducer(getCurScope(), Intro, 
                                               /*AfterAmpersand=*/true);
          ConsumeCodeCompletionToken();
          break;
        }
      }

      if (Tok.is(tok::identifier)) {
        Id = Tok.getIdentifierInfo();
        Loc = ConsumeToken();
        
        if (Tok.is(tok::ellipsis))
          EllipsisLoc = ConsumeToken();
      } else if (Tok.is(tok::kw_this)) {
        // FIXME: If we want to suggest a fixit here, will need to return more
        // than just DiagnosticID. Perhaps full DiagnosticBuilder that can be
        // Clear()ed to prevent emission in case of tentative parsing?
        return DiagResult(diag::err_this_captured_by_reference);
      } else {
        return DiagResult(diag::err_expected_capture);
      }
    }

    Intro.addCapture(Kind, Loc, Id, EllipsisLoc);
  }

  T.consumeClose();
  Intro.Range.setEnd(T.getCloseLocation());

  return DiagResult();
}

/// TryParseLambdaIntroducer - Tentatively parse a lambda introducer.
///
/// Returns true if it hit something unexpected.
bool Parser::TryParseLambdaIntroducer(LambdaIntroducer &Intro) {
  TentativeParsingAction PA(*this);

  llvm::Optional<unsigned> DiagID(ParseLambdaIntroducer(Intro));

  if (DiagID) {
    PA.Revert();
    return true;
  }

  PA.Commit();
  return false;
}

/// ParseLambdaExpressionAfterIntroducer - Parse the rest of a lambda
/// expression.
ExprResult Parser::ParseLambdaExpressionAfterIntroducer(
                     LambdaIntroducer &Intro) {
  SourceLocation LambdaBeginLoc = Intro.Range.getBegin();
  Diag(LambdaBeginLoc, diag::warn_cxx98_compat_lambda);

  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(), LambdaBeginLoc,
                                "lambda expression parsing");

  // Parse lambda-declarator[opt].
  DeclSpec DS(AttrFactory);
  Declarator D(DS, Declarator::LambdaExprContext);

  if (Tok.is(tok::l_paren)) {
    ParseScope PrototypeScope(this,
                              Scope::FunctionPrototypeScope |
                              Scope::DeclScope);

    SourceLocation DeclLoc, DeclEndLoc;
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    DeclLoc = T.getOpenLocation();

    // Parse parameter-declaration-clause.
    ParsedAttributes Attr(AttrFactory);
    llvm::SmallVector<DeclaratorChunk::ParamInfo, 16> ParamInfo;
    SourceLocation EllipsisLoc;

    if (Tok.isNot(tok::r_paren))
      ParseParameterDeclarationClause(D, Attr, ParamInfo, EllipsisLoc);

    T.consumeClose();
    DeclEndLoc = T.getCloseLocation();

    // Parse 'mutable'[opt].
    SourceLocation MutableLoc;
    if (Tok.is(tok::kw_mutable)) {
      MutableLoc = ConsumeToken();
      DeclEndLoc = MutableLoc;
    }

    // Parse exception-specification[opt].
    ExceptionSpecificationType ESpecType = EST_None;
    SourceRange ESpecRange;
    llvm::SmallVector<ParsedType, 2> DynamicExceptions;
    llvm::SmallVector<SourceRange, 2> DynamicExceptionRanges;
    ExprResult NoexceptExpr;
    ESpecType = tryParseExceptionSpecification(ESpecRange,
                                               DynamicExceptions,
                                               DynamicExceptionRanges,
                                               NoexceptExpr);

    if (ESpecType != EST_None)
      DeclEndLoc = ESpecRange.getEnd();

    // Parse attribute-specifier[opt].
    MaybeParseCXX0XAttributes(Attr, &DeclEndLoc);

    // Parse trailing-return-type[opt].
    ParsedType TrailingReturnType;
    if (Tok.is(tok::arrow)) {
      SourceRange Range;
      TrailingReturnType = ParseTrailingReturnType(Range).get();
      if (Range.getEnd().isValid())
        DeclEndLoc = Range.getEnd();
    }

    PrototypeScope.Exit();

    D.AddTypeInfo(DeclaratorChunk::getFunction(/*hasProto=*/true,
                                           /*isVariadic=*/EllipsisLoc.isValid(),
                                           EllipsisLoc,
                                           ParamInfo.data(), ParamInfo.size(),
                                           DS.getTypeQualifiers(),
                                           /*RefQualifierIsLValueRef=*/true,
                                           /*RefQualifierLoc=*/SourceLocation(),
                                         /*ConstQualifierLoc=*/SourceLocation(),
                                      /*VolatileQualifierLoc=*/SourceLocation(),
                                           MutableLoc,
                                           ESpecType, ESpecRange.getBegin(),
                                           DynamicExceptions.data(),
                                           DynamicExceptionRanges.data(),
                                           DynamicExceptions.size(),
                                           NoexceptExpr.isUsable() ?
                                             NoexceptExpr.get() : 0,
                                           DeclLoc, DeclEndLoc, D,
                                           TrailingReturnType),
                  Attr, DeclEndLoc);
  } else if (Tok.is(tok::kw_mutable) || Tok.is(tok::arrow)) {
    // It's common to forget that one needs '()' before 'mutable' or the 
    // result type. Deal with this.
    Diag(Tok, diag::err_lambda_missing_parens)
      << Tok.is(tok::arrow)
      << FixItHint::CreateInsertion(Tok.getLocation(), "() ");
    SourceLocation DeclLoc = Tok.getLocation();
    SourceLocation DeclEndLoc = DeclLoc;
    
    // Parse 'mutable', if it's there.
    SourceLocation MutableLoc;
    if (Tok.is(tok::kw_mutable)) {
      MutableLoc = ConsumeToken();
      DeclEndLoc = MutableLoc;
    }
    
    // Parse the return type, if there is one.
    ParsedType TrailingReturnType;
    if (Tok.is(tok::arrow)) {
      SourceRange Range;
      TrailingReturnType = ParseTrailingReturnType(Range).get();
      if (Range.getEnd().isValid())
        DeclEndLoc = Range.getEnd();      
    }

    ParsedAttributes Attr(AttrFactory);
    D.AddTypeInfo(DeclaratorChunk::getFunction(/*hasProto=*/true,
                     /*isVariadic=*/false,
                     /*EllipsisLoc=*/SourceLocation(),
                     /*Params=*/0, /*NumParams=*/0,
                     /*TypeQuals=*/0,
                     /*RefQualifierIsLValueRef=*/true,
                     /*RefQualifierLoc=*/SourceLocation(),
                     /*ConstQualifierLoc=*/SourceLocation(),
                     /*VolatileQualifierLoc=*/SourceLocation(),
                     MutableLoc,
                     EST_None, 
                     /*ESpecLoc=*/SourceLocation(),
                     /*Exceptions=*/0,
                     /*ExceptionRanges=*/0,
                     /*NumExceptions=*/0,
                     /*NoexceptExpr=*/0,
                     DeclLoc, DeclEndLoc, D,
                     TrailingReturnType),
                  Attr, DeclEndLoc);
  }
  

  // FIXME: Rename BlockScope -> ClosureScope if we decide to continue using
  // it.
  unsigned ScopeFlags = Scope::BlockScope | Scope::FnScope | Scope::DeclScope;
  ParseScope BodyScope(this, ScopeFlags);

  Actions.ActOnStartOfLambdaDefinition(Intro, D, getCurScope());

  // Parse compound-statement.
  if (!Tok.is(tok::l_brace)) {
    Diag(Tok, diag::err_expected_lambda_body);
    Actions.ActOnLambdaError(LambdaBeginLoc, getCurScope());
    return ExprError();
  }

  StmtResult Stmt(ParseCompoundStatementBody());
  BodyScope.Exit();

  if (!Stmt.isInvalid())
    return Actions.ActOnLambdaExpr(LambdaBeginLoc, Stmt.take(), getCurScope());
 
  Actions.ActOnLambdaError(LambdaBeginLoc, getCurScope());
  return ExprError();
}

/// ParseCXXCasts - This handles the various ways to cast expressions to another
/// type.
///
///       postfix-expression: [C++ 5.2p1]
///         'dynamic_cast' '<' type-name '>' '(' expression ')'
///         'static_cast' '<' type-name '>' '(' expression ')'
///         'reinterpret_cast' '<' type-name '>' '(' expression ')'
///         'const_cast' '<' type-name '>' '(' expression ')'
///
ExprResult Parser::ParseCXXCasts() {
  tok::TokenKind Kind = Tok.getKind();
  const char *CastName = 0;     // For error messages

  switch (Kind) {
  default: llvm_unreachable("Unknown C++ cast!");
  case tok::kw_const_cast:       CastName = "const_cast";       break;
  case tok::kw_dynamic_cast:     CastName = "dynamic_cast";     break;
  case tok::kw_reinterpret_cast: CastName = "reinterpret_cast"; break;
  case tok::kw_static_cast:      CastName = "static_cast";      break;
  }

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LAngleBracketLoc = Tok.getLocation();

  // Check for "<::" which is parsed as "[:".  If found, fix token stream,
  // diagnose error, suggest fix, and recover parsing.
  Token Next = NextToken();
  if (Tok.is(tok::l_square) && Tok.getLength() == 2 && Next.is(tok::colon) &&
      AreTokensAdjacent(PP, Tok, Next))
    FixDigraph(*this, PP, Tok, Next, Kind, /*AtDigraph*/true);

  if (ExpectAndConsume(tok::less, diag::err_expected_less_after, CastName))
    return ExprError();

  // Parse the common declaration-specifiers piece.
  DeclSpec DS(AttrFactory);
  ParseSpecifierQualifierList(DS);

  // Parse the abstract-declarator, if present.
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParseDeclarator(DeclaratorInfo);

  SourceLocation RAngleBracketLoc = Tok.getLocation();

  if (ExpectAndConsume(tok::greater, diag::err_expected_greater))
    return ExprError(Diag(LAngleBracketLoc, diag::note_matching) << "<");

  SourceLocation LParenLoc, RParenLoc;
  BalancedDelimiterTracker T(*this, tok::l_paren);

  if (T.expectAndConsume(diag::err_expected_lparen_after, CastName))
    return ExprError();

  ExprResult Result = ParseExpression();

  // Match the ')'.
  T.consumeClose();

  if (!Result.isInvalid() && !DeclaratorInfo.isInvalidType())
    Result = Actions.ActOnCXXNamedCast(OpLoc, Kind,
                                       LAngleBracketLoc, DeclaratorInfo,
                                       RAngleBracketLoc,
                                       T.getOpenLocation(), Result.take(), 
                                       T.getCloseLocation());

  return move(Result);
}

/// ParseCXXTypeid - This handles the C++ typeid expression.
///
///       postfix-expression: [C++ 5.2p1]
///         'typeid' '(' expression ')'
///         'typeid' '(' type-id ')'
///
ExprResult Parser::ParseCXXTypeid() {
  assert(Tok.is(tok::kw_typeid) && "Not 'typeid'!");

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LParenLoc, RParenLoc;
  BalancedDelimiterTracker T(*this, tok::l_paren);

  // typeid expressions are always parenthesized.
  if (T.expectAndConsume(diag::err_expected_lparen_after, "typeid"))
    return ExprError();
  LParenLoc = T.getOpenLocation();

  ExprResult Result;

  if (isTypeIdInParens()) {
    TypeResult Ty = ParseTypeName();

    // Match the ')'.
    T.consumeClose();
    RParenLoc = T.getCloseLocation();
    if (Ty.isInvalid() || RParenLoc.isInvalid())
      return ExprError();

    Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/true,
                                    Ty.get().getAsOpaquePtr(), RParenLoc);
  } else {
    // C++0x [expr.typeid]p3:
    //   When typeid is applied to an expression other than an lvalue of a
    //   polymorphic class type [...] The expression is an unevaluated
    //   operand (Clause 5).
    //
    // Note that we can't tell whether the expression is an lvalue of a
    // polymorphic class type until after we've parsed the expression; we
    // speculatively assume the subexpression is unevaluated, and fix it up
    // later.
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);
    Result = ParseExpression();

    // Match the ')'.
    if (Result.isInvalid())
      SkipUntil(tok::r_paren);
    else {
      T.consumeClose();
      RParenLoc = T.getCloseLocation();
      if (RParenLoc.isInvalid())
        return ExprError();

      Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/false,
                                      Result.release(), RParenLoc);
    }
  }

  return move(Result);
}

/// ParseCXXUuidof - This handles the Microsoft C++ __uuidof expression.
///
///         '__uuidof' '(' expression ')'
///         '__uuidof' '(' type-id ')'
///
ExprResult Parser::ParseCXXUuidof() {
  assert(Tok.is(tok::kw___uuidof) && "Not '__uuidof'!");

  SourceLocation OpLoc = ConsumeToken();
  BalancedDelimiterTracker T(*this, tok::l_paren);

  // __uuidof expressions are always parenthesized.
  if (T.expectAndConsume(diag::err_expected_lparen_after, "__uuidof"))
    return ExprError();

  ExprResult Result;

  if (isTypeIdInParens()) {
    TypeResult Ty = ParseTypeName();

    // Match the ')'.
    T.consumeClose();

    if (Ty.isInvalid())
      return ExprError();

    Result = Actions.ActOnCXXUuidof(OpLoc, T.getOpenLocation(), /*isType=*/true,
                                    Ty.get().getAsOpaquePtr(), 
                                    T.getCloseLocation());
  } else {
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);
    Result = ParseExpression();

    // Match the ')'.
    if (Result.isInvalid())
      SkipUntil(tok::r_paren);
    else {
      T.consumeClose();

      Result = Actions.ActOnCXXUuidof(OpLoc, T.getOpenLocation(),
                                      /*isType=*/false,
                                      Result.release(), T.getCloseLocation());
    }
  }

  return move(Result);
}

/// \brief Parse a C++ pseudo-destructor expression after the base,
/// . or -> operator, and nested-name-specifier have already been
/// parsed.
///
///       postfix-expression: [C++ 5.2]
///         postfix-expression . pseudo-destructor-name
///         postfix-expression -> pseudo-destructor-name
///
///       pseudo-destructor-name: 
///         ::[opt] nested-name-specifier[opt] type-name :: ~type-name 
///         ::[opt] nested-name-specifier template simple-template-id :: 
///                 ~type-name 
///         ::[opt] nested-name-specifier[opt] ~type-name
///       
ExprResult 
Parser::ParseCXXPseudoDestructor(ExprArg Base, SourceLocation OpLoc,
                                 tok::TokenKind OpKind,
                                 CXXScopeSpec &SS,
                                 ParsedType ObjectType) {
  // We're parsing either a pseudo-destructor-name or a dependent
  // member access that has the same form as a
  // pseudo-destructor-name. We parse both in the same way and let
  // the action model sort them out.
  //
  // Note that the ::[opt] nested-name-specifier[opt] has already
  // been parsed, and if there was a simple-template-id, it has
  // been coalesced into a template-id annotation token.
  UnqualifiedId FirstTypeName;
  SourceLocation CCLoc;
  if (Tok.is(tok::identifier)) {
    FirstTypeName.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
    ConsumeToken();
    assert(Tok.is(tok::coloncolon) &&"ParseOptionalCXXScopeSpecifier fail");
    CCLoc = ConsumeToken();
  } else if (Tok.is(tok::annot_template_id)) {
    // FIXME: retrieve TemplateKWLoc from template-id annotation and
    // store it in the pseudo-dtor node (to be used when instantiating it).
    FirstTypeName.setTemplateId(
                              (TemplateIdAnnotation *)Tok.getAnnotationValue());
    ConsumeToken();
    assert(Tok.is(tok::coloncolon) &&"ParseOptionalCXXScopeSpecifier fail");
    CCLoc = ConsumeToken();
  } else {
    FirstTypeName.setIdentifier(0, SourceLocation());
  }

  // Parse the tilde.
  assert(Tok.is(tok::tilde) && "ParseOptionalCXXScopeSpecifier fail");
  SourceLocation TildeLoc = ConsumeToken();

  if (Tok.is(tok::kw_decltype) && !FirstTypeName.isValid() && SS.isEmpty()) {
    DeclSpec DS(AttrFactory);
    ParseDecltypeSpecifier(DS);
    if (DS.getTypeSpecType() == TST_error)
      return ExprError();
    return Actions.ActOnPseudoDestructorExpr(getCurScope(), Base, OpLoc, 
                                             OpKind, TildeLoc, DS, 
                                             Tok.is(tok::l_paren));
  }

  if (!Tok.is(tok::identifier)) {
    Diag(Tok, diag::err_destructor_tilde_identifier);
    return ExprError();
  }
  
  // Parse the second type.
  UnqualifiedId SecondTypeName;
  IdentifierInfo *Name = Tok.getIdentifierInfo();
  SourceLocation NameLoc = ConsumeToken();
  SecondTypeName.setIdentifier(Name, NameLoc);
  
  // If there is a '<', the second type name is a template-id. Parse
  // it as such.
  if (Tok.is(tok::less) &&
      ParseUnqualifiedIdTemplateId(SS, SourceLocation(),
                                   Name, NameLoc,
                                   false, ObjectType, SecondTypeName,
                                   /*AssumeTemplateName=*/true))
    return ExprError();

  return Actions.ActOnPseudoDestructorExpr(getCurScope(), Base,
                                           OpLoc, OpKind,
                                           SS, FirstTypeName, CCLoc,
                                           TildeLoc, SecondTypeName,
                                           Tok.is(tok::l_paren));
}

/// ParseCXXBoolLiteral - This handles the C++ Boolean literals.
///
///       boolean-literal: [C++ 2.13.5]
///         'true'
///         'false'
ExprResult Parser::ParseCXXBoolLiteral() {
  tok::TokenKind Kind = Tok.getKind();
  return Actions.ActOnCXXBoolLiteral(ConsumeToken(), Kind);
}

/// ParseThrowExpression - This handles the C++ throw expression.
///
///       throw-expression: [C++ 15]
///         'throw' assignment-expression[opt]
ExprResult Parser::ParseThrowExpression() {
  assert(Tok.is(tok::kw_throw) && "Not throw!");
  SourceLocation ThrowLoc = ConsumeToken();           // Eat the throw token.

  // If the current token isn't the start of an assignment-expression,
  // then the expression is not present.  This handles things like:
  //   "C ? throw : (void)42", which is crazy but legal.
  switch (Tok.getKind()) {  // FIXME: move this predicate somewhere common.
  case tok::semi:
  case tok::r_paren:
  case tok::r_square:
  case tok::r_brace:
  case tok::colon:
  case tok::comma:
    return Actions.ActOnCXXThrow(getCurScope(), ThrowLoc, 0);

  default:
    ExprResult Expr(ParseAssignmentExpression());
    if (Expr.isInvalid()) return move(Expr);
    return Actions.ActOnCXXThrow(getCurScope(), ThrowLoc, Expr.take());
  }
}

/// ParseCXXThis - This handles the C++ 'this' pointer.
///
/// C++ 9.3.2: In the body of a non-static member function, the keyword this is
/// a non-lvalue expression whose value is the address of the object for which
/// the function is called.
ExprResult Parser::ParseCXXThis() {
  assert(Tok.is(tok::kw_this) && "Not 'this'!");
  SourceLocation ThisLoc = ConsumeToken();
  return Actions.ActOnCXXThis(ThisLoc);
}

/// ParseCXXTypeConstructExpression - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
/// See [C++ 5.2.3].
///
///       postfix-expression: [C++ 5.2p1]
///         simple-type-specifier '(' expression-list[opt] ')'
/// [C++0x] simple-type-specifier braced-init-list
///         typename-specifier '(' expression-list[opt] ')'
/// [C++0x] typename-specifier braced-init-list
///
ExprResult
Parser::ParseCXXTypeConstructExpression(const DeclSpec &DS) {
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParsedType TypeRep = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo).get();

  assert((Tok.is(tok::l_paren) ||
          (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)))
         && "Expected '(' or '{'!");

  if (Tok.is(tok::l_brace)) {
    ExprResult Init = ParseBraceInitializer();
    if (Init.isInvalid())
      return Init;
    Expr *InitList = Init.take();
    return Actions.ActOnCXXTypeConstructExpr(TypeRep, SourceLocation(),
                                             MultiExprArg(&InitList, 1),
                                             SourceLocation());
  } else {
    GreaterThanIsOperatorScope G(GreaterThanIsOperator, true);

    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();

    ExprVector Exprs(Actions);
    CommaLocsTy CommaLocs;

    if (Tok.isNot(tok::r_paren)) {
      if (ParseExpressionList(Exprs, CommaLocs)) {
        SkipUntil(tok::r_paren);
        return ExprError();
      }
    }

    // Match the ')'.
    T.consumeClose();

    // TypeRep could be null, if it references an invalid typedef.
    if (!TypeRep)
      return ExprError();

    assert((Exprs.size() == 0 || Exprs.size()-1 == CommaLocs.size())&&
           "Unexpected number of commas!");
    return Actions.ActOnCXXTypeConstructExpr(TypeRep, T.getOpenLocation(), 
                                             move_arg(Exprs),
                                             T.getCloseLocation());
  }
}

/// ParseCXXCondition - if/switch/while condition expression.
///
///       condition:
///         expression
///         type-specifier-seq declarator '=' assignment-expression
/// [C++11] type-specifier-seq declarator '=' initializer-clause
/// [C++11] type-specifier-seq declarator braced-init-list
/// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
///             '=' assignment-expression
///
/// \param ExprResult if the condition was parsed as an expression, the
/// parsed expression.
///
/// \param DeclResult if the condition was parsed as a declaration, the
/// parsed declaration.
///
/// \param Loc The location of the start of the statement that requires this
/// condition, e.g., the "for" in a for loop.
///
/// \param ConvertToBoolean Whether the condition expression should be
/// converted to a boolean value.
///
/// \returns true if there was a parsing, false otherwise.
bool Parser::ParseCXXCondition(ExprResult &ExprOut,
                               Decl *&DeclOut,
                               SourceLocation Loc,
                               bool ConvertToBoolean) {
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Condition);
    cutOffParsing();
    return true;
  }

  if (!isCXXConditionDeclaration()) {
    // Parse the expression.
    ExprOut = ParseExpression(); // expression
    DeclOut = 0;
    if (ExprOut.isInvalid())
      return true;

    // If required, convert to a boolean value.
    if (ConvertToBoolean)
      ExprOut
        = Actions.ActOnBooleanCondition(getCurScope(), Loc, ExprOut.get());
    return ExprOut.isInvalid();
  }

  // type-specifier-seq
  DeclSpec DS(AttrFactory);
  ParseSpecifierQualifierList(DS);

  // declarator
  Declarator DeclaratorInfo(DS, Declarator::ConditionContext);
  ParseDeclarator(DeclaratorInfo);

  // simple-asm-expr[opt]
  if (Tok.is(tok::kw_asm)) {
    SourceLocation Loc;
    ExprResult AsmLabel(ParseSimpleAsm(&Loc));
    if (AsmLabel.isInvalid()) {
      SkipUntil(tok::semi);
      return true;
    }
    DeclaratorInfo.setAsmLabel(AsmLabel.release());
    DeclaratorInfo.SetRangeEnd(Loc);
  }

  // If attributes are present, parse them.
  MaybeParseGNUAttributes(DeclaratorInfo);

  // Type-check the declaration itself.
  DeclResult Dcl = Actions.ActOnCXXConditionDeclaration(getCurScope(), 
                                                        DeclaratorInfo);
  DeclOut = Dcl.get();
  ExprOut = ExprError();

  // '=' assignment-expression
  // If a '==' or '+=' is found, suggest a fixit to '='.
  bool CopyInitialization = isTokenEqualOrEqualTypo();
  if (CopyInitialization)
    ConsumeToken();

  ExprResult InitExpr = ExprError();
  if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)) {
    Diag(Tok.getLocation(),
         diag::warn_cxx98_compat_generalized_initializer_lists);
    InitExpr = ParseBraceInitializer();
  } else if (CopyInitialization) {
    InitExpr = ParseAssignmentExpression();
  } else if (Tok.is(tok::l_paren)) {
    // This was probably an attempt to initialize the variable.
    SourceLocation LParen = ConsumeParen(), RParen = LParen;
    if (SkipUntil(tok::r_paren, true, /*DontConsume=*/true))
      RParen = ConsumeParen();
    Diag(DeclOut ? DeclOut->getLocation() : LParen,
         diag::err_expected_init_in_condition_lparen)
      << SourceRange(LParen, RParen);
  } else {
    Diag(DeclOut ? DeclOut->getLocation() : Tok.getLocation(),
         diag::err_expected_init_in_condition);
  }

  if (!InitExpr.isInvalid())
    Actions.AddInitializerToDecl(DeclOut, InitExpr.take(), !CopyInitialization,
                                 DS.getTypeSpecType() == DeclSpec::TST_auto);

  // FIXME: Build a reference to this declaration? Convert it to bool?
  // (This is currently handled by Sema).

  Actions.FinalizeDeclaration(DeclOut);
  
  return false;
}

/// \brief Determine whether the current token starts a C++
/// simple-type-specifier.
bool Parser::isCXXSimpleTypeSpecifier() const {
  switch (Tok.getKind()) {
  case tok::annot_typename:
  case tok::kw_short:
  case tok::kw_long:
  case tok::kw___int64:
  case tok::kw___int128:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw_void:
  case tok::kw_char:
  case tok::kw_int:
  case tok::kw_half:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_wchar_t:
  case tok::kw_char16_t:
  case tok::kw_char32_t:
  case tok::kw_bool:
  case tok::kw_decltype:
  case tok::kw_typeof:
  case tok::kw___underlying_type:
    return true;

  default:
    break;
  }

  return false;
}

/// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
/// This should only be called when the current token is known to be part of
/// simple-type-specifier.
///
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
///
void Parser::ParseCXXSimpleTypeSpecifier(DeclSpec &DS) {
  DS.SetRangeStart(Tok.getLocation());
  const char *PrevSpec;
  unsigned DiagID;
  SourceLocation Loc = Tok.getLocation();

  switch (Tok.getKind()) {
  case tok::identifier:   // foo::bar
  case tok::coloncolon:   // ::foo::bar
    llvm_unreachable("Annotation token should already be formed!");
  default:
    llvm_unreachable("Not a simple-type-specifier token!");

  // type-name
  case tok::annot_typename: {
    if (getTypeAnnotation(Tok))
      DS.SetTypeSpecType(DeclSpec::TST_typename, Loc, PrevSpec, DiagID,
                         getTypeAnnotation(Tok));
    else
      DS.SetTypeSpecError();
    
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
    ConsumeToken();
    
    // Objective-C supports syntax of the form 'id<proto1,proto2>' where 'id'
    // is a specific typedef and 'itf<proto1,proto2>' where 'itf' is an
    // Objective-C interface.  If we don't have Objective-C or a '<', this is
    // just a normal reference to a typedef name.
    if (Tok.is(tok::less) && getLangOpts().ObjC1)
      ParseObjCProtocolQualifiers(DS);
    
    DS.Finish(Diags, PP);
    return;
  }

  // builtin types
  case tok::kw_short:
    DS.SetTypeSpecWidth(DeclSpec::TSW_short, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_long:
    DS.SetTypeSpecWidth(DeclSpec::TSW_long, Loc, PrevSpec, DiagID);
    break;
  case tok::kw___int64:
    DS.SetTypeSpecWidth(DeclSpec::TSW_longlong, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_signed:
    DS.SetTypeSpecSign(DeclSpec::TSS_signed, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_unsigned:
    DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_void:
    DS.SetTypeSpecType(DeclSpec::TST_void, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char:
    DS.SetTypeSpecType(DeclSpec::TST_char, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_int:
    DS.SetTypeSpecType(DeclSpec::TST_int, Loc, PrevSpec, DiagID);
    break;
  case tok::kw___int128:
    DS.SetTypeSpecType(DeclSpec::TST_int128, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_half:
    DS.SetTypeSpecType(DeclSpec::TST_half, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_float:
    DS.SetTypeSpecType(DeclSpec::TST_float, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_double:
    DS.SetTypeSpecType(DeclSpec::TST_double, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_wchar_t:
    DS.SetTypeSpecType(DeclSpec::TST_wchar, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char16_t:
    DS.SetTypeSpecType(DeclSpec::TST_char16, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char32_t:
    DS.SetTypeSpecType(DeclSpec::TST_char32, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_bool:
    DS.SetTypeSpecType(DeclSpec::TST_bool, Loc, PrevSpec, DiagID);
    break;
  case tok::annot_decltype:
  case tok::kw_decltype:
    DS.SetRangeEnd(ParseDecltypeSpecifier(DS));
    return DS.Finish(Diags, PP);

  // GNU typeof support.
  case tok::kw_typeof:
    ParseTypeofSpecifier(DS);
    DS.Finish(Diags, PP);
    return;
  }
  if (Tok.is(tok::annot_typename))
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
  else
    DS.SetRangeEnd(Tok.getLocation());
  ConsumeToken();
  DS.Finish(Diags, PP);
}

/// ParseCXXTypeSpecifierSeq - Parse a C++ type-specifier-seq (C++
/// [dcl.name]), which is a non-empty sequence of type-specifiers,
/// e.g., "const short int". Note that the DeclSpec is *not* finished
/// by parsing the type-specifier-seq, because these sequences are
/// typically followed by some form of declarator. Returns true and
/// emits diagnostics if this is not a type-specifier-seq, false
/// otherwise.
///
///   type-specifier-seq: [C++ 8.1]
///     type-specifier type-specifier-seq[opt]
///
bool Parser::ParseCXXTypeSpecifierSeq(DeclSpec &DS) {
  ParseSpecifierQualifierList(DS, AS_none, DSC_type_specifier);
  DS.Finish(Diags, PP);
  return false;
}

/// \brief Finish parsing a C++ unqualified-id that is a template-id of
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
/// \param Name for constructor and destructor names, this is the actual
/// identifier that may be a template-name.
///
/// \param NameLoc the location of the class-name in a constructor or 
/// destructor.
///
/// \param EnteringContext whether we're entering the scope of the 
/// nested-name-specifier.
///
/// \param ObjectType if this unqualified-id occurs within a member access
/// expression, the type of the base object whose member is being accessed.
///
/// \param Id as input, describes the template-name or operator-function-id
/// that precedes the '<'. If template arguments were parsed successfully,
/// will be updated with the template-id.
/// 
/// \param AssumeTemplateId When true, this routine will assume that the name
/// refers to a template without performing name lookup to verify. 
///
/// \returns true if a parse error occurred, false otherwise.
bool Parser::ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS,
                                          SourceLocation TemplateKWLoc,
                                          IdentifierInfo *Name,
                                          SourceLocation NameLoc,
                                          bool EnteringContext,
                                          ParsedType ObjectType,
                                          UnqualifiedId &Id,
                                          bool AssumeTemplateId) {
  assert((AssumeTemplateId || Tok.is(tok::less)) &&
         "Expected '<' to finish parsing a template-id");
  
  TemplateTy Template;
  TemplateNameKind TNK = TNK_Non_template;
  switch (Id.getKind()) {
  case UnqualifiedId::IK_Identifier:
  case UnqualifiedId::IK_OperatorFunctionId:
  case UnqualifiedId::IK_LiteralOperatorId:
    if (AssumeTemplateId) {
      TNK = Actions.ActOnDependentTemplateName(getCurScope(), SS, TemplateKWLoc,
                                               Id, ObjectType, EnteringContext,
                                               Template);
      if (TNK == TNK_Non_template)
        return true;
    } else {
      bool MemberOfUnknownSpecialization;
      TNK = Actions.isTemplateName(getCurScope(), SS,
                                   TemplateKWLoc.isValid(), Id,
                                   ObjectType, EnteringContext, Template,
                                   MemberOfUnknownSpecialization);
      
      if (TNK == TNK_Non_template && MemberOfUnknownSpecialization &&
          ObjectType && IsTemplateArgumentList()) {
        // We have something like t->getAs<T>(), where getAs is a 
        // member of an unknown specialization. However, this will only
        // parse correctly as a template, so suggest the keyword 'template'
        // before 'getAs' and treat this as a dependent template name.
        std::string Name;
        if (Id.getKind() == UnqualifiedId::IK_Identifier)
          Name = Id.Identifier->getName();
        else {
          Name = "operator ";
          if (Id.getKind() == UnqualifiedId::IK_OperatorFunctionId)
            Name += getOperatorSpelling(Id.OperatorFunctionId.Operator);
          else
            Name += Id.Identifier->getName();
        }
        Diag(Id.StartLocation, diag::err_missing_dependent_template_keyword)
          << Name
          << FixItHint::CreateInsertion(Id.StartLocation, "template ");
        TNK = Actions.ActOnDependentTemplateName(getCurScope(),
                                                 SS, TemplateKWLoc, Id,
                                                 ObjectType, EnteringContext,
                                                 Template);
        if (TNK == TNK_Non_template)
          return true;              
      }
    }
    break;
      
  case UnqualifiedId::IK_ConstructorName: {
    UnqualifiedId TemplateName;
    bool MemberOfUnknownSpecialization;
    TemplateName.setIdentifier(Name, NameLoc);
    TNK = Actions.isTemplateName(getCurScope(), SS, TemplateKWLoc.isValid(),
                                 TemplateName, ObjectType, 
                                 EnteringContext, Template,
                                 MemberOfUnknownSpecialization);
    break;
  }
      
  case UnqualifiedId::IK_DestructorName: {
    UnqualifiedId TemplateName;
    bool MemberOfUnknownSpecialization;
    TemplateName.setIdentifier(Name, NameLoc);
    if (ObjectType) {
      TNK = Actions.ActOnDependentTemplateName(getCurScope(),
                                               SS, TemplateKWLoc, TemplateName,
                                               ObjectType, EnteringContext,
                                               Template);
      if (TNK == TNK_Non_template)
        return true;
    } else {
      TNK = Actions.isTemplateName(getCurScope(), SS, TemplateKWLoc.isValid(),
                                   TemplateName, ObjectType, 
                                   EnteringContext, Template,
                                   MemberOfUnknownSpecialization);
      
      if (TNK == TNK_Non_template && !Id.DestructorName.get()) {
        Diag(NameLoc, diag::err_destructor_template_id)
          << Name << SS.getRange();
        return true;        
      }
    }
    break;
  }
      
  default:
    return false;
  }
  
  if (TNK == TNK_Non_template)
    return false;
  
  // Parse the enclosed template argument list.
  SourceLocation LAngleLoc, RAngleLoc;
  TemplateArgList TemplateArgs;
  if (Tok.is(tok::less) &&
      ParseTemplateIdAfterTemplateName(Template, Id.StartLocation,
                                       SS, true, LAngleLoc,
                                       TemplateArgs,
                                       RAngleLoc))
    return true;
  
  if (Id.getKind() == UnqualifiedId::IK_Identifier ||
      Id.getKind() == UnqualifiedId::IK_OperatorFunctionId ||
      Id.getKind() == UnqualifiedId::IK_LiteralOperatorId) {
    // Form a parsed representation of the template-id to be stored in the
    // UnqualifiedId.
    TemplateIdAnnotation *TemplateId
      = TemplateIdAnnotation::Allocate(TemplateArgs.size(), TemplateIds);

    if (Id.getKind() == UnqualifiedId::IK_Identifier) {
      TemplateId->Name = Id.Identifier;
      TemplateId->Operator = OO_None;
      TemplateId->TemplateNameLoc = Id.StartLocation;
    } else {
      TemplateId->Name = 0;
      TemplateId->Operator = Id.OperatorFunctionId.Operator;
      TemplateId->TemplateNameLoc = Id.StartLocation;
    }

    TemplateId->SS = SS;
    TemplateId->TemplateKWLoc = TemplateKWLoc;
    TemplateId->Template = Template;
    TemplateId->Kind = TNK;
    TemplateId->LAngleLoc = LAngleLoc;
    TemplateId->RAngleLoc = RAngleLoc;
    ParsedTemplateArgument *Args = TemplateId->getTemplateArgs();
    for (unsigned Arg = 0, ArgEnd = TemplateArgs.size(); 
         Arg != ArgEnd; ++Arg)
      Args[Arg] = TemplateArgs[Arg];
    
    Id.setTemplateId(TemplateId);
    return false;
  }

  // Bundle the template arguments together.
  ASTTemplateArgsPtr TemplateArgsPtr(Actions, TemplateArgs.data(),
                                     TemplateArgs.size());

  // Constructor and destructor names.
  TypeResult Type
    = Actions.ActOnTemplateIdType(SS, TemplateKWLoc,
                                  Template, NameLoc,
                                  LAngleLoc, TemplateArgsPtr, RAngleLoc,
                                  /*IsCtorOrDtorName=*/true);
  if (Type.isInvalid())
    return true;
  
  if (Id.getKind() == UnqualifiedId::IK_ConstructorName)
    Id.setConstructorName(Type.get(), NameLoc, RAngleLoc);
  else
    Id.setDestructorName(Id.StartLocation, Type.get(), RAngleLoc);
  
  return false;
}

/// \brief Parse an operator-function-id or conversion-function-id as part
/// of a C++ unqualified-id.
///
/// This routine is responsible only for parsing the operator-function-id or
/// conversion-function-id; it does not handle template arguments in any way.
///
/// \code
///       operator-function-id: [C++ 13.5]
///         'operator' operator
///
///       operator: one of
///            new   delete  new[]   delete[]
///            +     -    *  /    %  ^    &   |   ~
///            !     =    <  >    += -=   *=  /=  %=
///            ^=    &=   |= <<   >> >>= <<=  ==  !=
///            <=    >=   && ||   ++ --   ,   ->* ->
///            ()    []
///
///       conversion-function-id: [C++ 12.3.2]
///         operator conversion-type-id
///
///       conversion-type-id:
///         type-specifier-seq conversion-declarator[opt]
///
///       conversion-declarator:
///         ptr-operator conversion-declarator[opt]
/// \endcode
///
/// \param The nested-name-specifier that preceded this unqualified-id. If
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
bool Parser::ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                        ParsedType ObjectType,
                                        UnqualifiedId &Result) {
  assert(Tok.is(tok::kw_operator) && "Expected 'operator' keyword");
  
  // Consume the 'operator' keyword.
  SourceLocation KeywordLoc = ConsumeToken();
  
  // Determine what kind of operator name we have.
  unsigned SymbolIdx = 0;
  SourceLocation SymbolLocations[3];
  OverloadedOperatorKind Op = OO_None;
  switch (Tok.getKind()) {
    case tok::kw_new:
    case tok::kw_delete: {
      bool isNew = Tok.getKind() == tok::kw_new;
      // Consume the 'new' or 'delete'.
      SymbolLocations[SymbolIdx++] = ConsumeToken();
      // Check for array new/delete.
      if (Tok.is(tok::l_square) &&
          (!getLangOpts().CPlusPlus0x || NextToken().isNot(tok::l_square))) {
        // Consume the '[' and ']'.
        BalancedDelimiterTracker T(*this, tok::l_square);
        T.consumeOpen();
        T.consumeClose();
        if (T.getCloseLocation().isInvalid())
          return true;
        
        SymbolLocations[SymbolIdx++] = T.getOpenLocation();
        SymbolLocations[SymbolIdx++] = T.getCloseLocation();
        Op = isNew? OO_Array_New : OO_Array_Delete;
      } else {
        Op = isNew? OO_New : OO_Delete;
      }
      break;
    }
      
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
    case tok::Token:                                                     \
      SymbolLocations[SymbolIdx++] = ConsumeToken();                     \
      Op = OO_##Name;                                                    \
      break;
#define OVERLOADED_OPERATOR_MULTI(Name,Spelling,Unary,Binary,MemberOnly)
#include "clang/Basic/OperatorKinds.def"
      
    case tok::l_paren: {
      // Consume the '(' and ')'.
      BalancedDelimiterTracker T(*this, tok::l_paren);
      T.consumeOpen();
      T.consumeClose();
      if (T.getCloseLocation().isInvalid())
        return true;
      
      SymbolLocations[SymbolIdx++] = T.getOpenLocation();
      SymbolLocations[SymbolIdx++] = T.getCloseLocation();
      Op = OO_Call;
      break;
    }
      
    case tok::l_square: {
      // Consume the '[' and ']'.
      BalancedDelimiterTracker T(*this, tok::l_square);
      T.consumeOpen();
      T.consumeClose();
      if (T.getCloseLocation().isInvalid())
        return true;
      
      SymbolLocations[SymbolIdx++] = T.getOpenLocation();
      SymbolLocations[SymbolIdx++] = T.getCloseLocation();
      Op = OO_Subscript;
      break;
    }
      
    case tok::code_completion: {
      // Code completion for the operator name.
      Actions.CodeCompleteOperatorName(getCurScope());
      cutOffParsing();      
      // Don't try to parse any further.
      return true;
    }
      
    default:
      break;
  }
  
  if (Op != OO_None) {
    // We have parsed an operator-function-id.
    Result.setOperatorFunctionId(KeywordLoc, Op, SymbolLocations);
    return false;
  }

  // Parse a literal-operator-id.
  //
  //   literal-operator-id: [C++0x 13.5.8]
  //     operator "" identifier

  if (getLangOpts().CPlusPlus0x && isTokenStringLiteral()) {
    Diag(Tok.getLocation(), diag::warn_cxx98_compat_literal_operator);

    SourceLocation DiagLoc;
    unsigned DiagId = 0;

    // We're past translation phase 6, so perform string literal concatenation
    // before checking for "".
    llvm::SmallVector<Token, 4> Toks;
    llvm::SmallVector<SourceLocation, 4> TokLocs;
    while (isTokenStringLiteral()) {
      if (!Tok.is(tok::string_literal) && !DiagId) {
        DiagLoc = Tok.getLocation();
        DiagId = diag::err_literal_operator_string_prefix;
      }
      Toks.push_back(Tok);
      TokLocs.push_back(ConsumeStringToken());
    }

    StringLiteralParser Literal(Toks.data(), Toks.size(), PP);
    if (Literal.hadError)
      return true;

    // Grab the literal operator's suffix, which will be either the next token
    // or a ud-suffix from the string literal.
    IdentifierInfo *II = 0;
    SourceLocation SuffixLoc;
    if (!Literal.getUDSuffix().empty()) {
      II = &PP.getIdentifierTable().get(Literal.getUDSuffix());
      SuffixLoc =
        Lexer::AdvanceToTokenCharacter(TokLocs[Literal.getUDSuffixToken()],
                                       Literal.getUDSuffixOffset(),
                                       PP.getSourceManager(), getLangOpts());
      // This form is not permitted by the standard (yet).
      DiagLoc = SuffixLoc;
      DiagId = diag::err_literal_operator_missing_space;
    } else if (Tok.is(tok::identifier)) {
      II = Tok.getIdentifierInfo();
      SuffixLoc = ConsumeToken();
      TokLocs.push_back(SuffixLoc);
    } else {
      Diag(Tok.getLocation(), diag::err_expected_ident);
      return true;
    }

    // The string literal must be empty.
    if (!Literal.GetString().empty() || Literal.Pascal) {
      DiagLoc = TokLocs.front();
      DiagId = diag::err_literal_operator_string_not_empty;
    }

    if (DiagId) {
      // This isn't a valid literal-operator-id, but we think we know
      // what the user meant. Tell them what they should have written.
      llvm::SmallString<32> Str;
      Str += "\"\" ";
      Str += II->getName();
      Diag(DiagLoc, DiagId) << FixItHint::CreateReplacement(
          SourceRange(TokLocs.front(), TokLocs.back()), Str);
    }

    Result.setLiteralOperatorId(II, KeywordLoc, SuffixLoc);
    return false;
  }
  
  // Parse a conversion-function-id.
  //
  //   conversion-function-id: [C++ 12.3.2]
  //     operator conversion-type-id
  //
  //   conversion-type-id:
  //     type-specifier-seq conversion-declarator[opt]
  //
  //   conversion-declarator:
  //     ptr-operator conversion-declarator[opt]
  
  // Parse the type-specifier-seq.
  DeclSpec DS(AttrFactory);
  if (ParseCXXTypeSpecifierSeq(DS)) // FIXME: ObjectType?
    return true;
  
  // Parse the conversion-declarator, which is merely a sequence of
  // ptr-operators.
  Declarator D(DS, Declarator::TypeNameContext);
  ParseDeclaratorInternal(D, /*DirectDeclParser=*/0);
  
  // Finish up the type.
  TypeResult Ty = Actions.ActOnTypeName(getCurScope(), D);
  if (Ty.isInvalid())
    return true;
  
  // Note that this is a conversion-function-id.
  Result.setConversionFunctionId(KeywordLoc, Ty.get(), 
                                 D.getSourceRange().getEnd());
  return false;  
}

/// \brief Parse a C++ unqualified-id (or a C identifier), which describes the
/// name of an entity.
///
/// \code
///       unqualified-id: [C++ expr.prim.general]
///         identifier
///         operator-function-id
///         conversion-function-id
/// [C++0x] literal-operator-id [TODO]
///         ~ class-name
///         template-id
///
/// \endcode
///
/// \param The nested-name-specifier that preceded this unqualified-id. If
/// non-empty, then we are parsing the unqualified-id of a qualified-id.
///
/// \param EnteringContext whether we are entering the scope of the 
/// nested-name-specifier.
///
/// \param AllowDestructorName whether we allow parsing of a destructor name.
///
/// \param AllowConstructorName whether we allow parsing a constructor name.
///
/// \param ObjectType if this unqualified-id occurs within a member access
/// expression, the type of the base object whose member is being accessed.
///
/// \param Result on a successful parse, contains the parsed unqualified-id.
///
/// \returns true if parsing fails, false otherwise.
bool Parser::ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
                                bool AllowDestructorName,
                                bool AllowConstructorName,
                                ParsedType ObjectType,
                                SourceLocation& TemplateKWLoc,
                                UnqualifiedId &Result) {

  // Handle 'A::template B'. This is for template-ids which have not
  // already been annotated by ParseOptionalCXXScopeSpecifier().
  bool TemplateSpecified = false;
  if (getLangOpts().CPlusPlus && Tok.is(tok::kw_template) &&
      (ObjectType || SS.isSet())) {
    TemplateSpecified = true;
    TemplateKWLoc = ConsumeToken();
  }

  // unqualified-id:
  //   identifier
  //   template-id (when it hasn't already been annotated)
  if (Tok.is(tok::identifier)) {
    // Consume the identifier.
    IdentifierInfo *Id = Tok.getIdentifierInfo();
    SourceLocation IdLoc = ConsumeToken();

    if (!getLangOpts().CPlusPlus) {
      // If we're not in C++, only identifiers matter. Record the
      // identifier and return.
      Result.setIdentifier(Id, IdLoc);
      return false;
    }

    if (AllowConstructorName && 
        Actions.isCurrentClassName(*Id, getCurScope(), &SS)) {
      // We have parsed a constructor name.
      ParsedType Ty = Actions.getTypeName(*Id, IdLoc, getCurScope(),
                                          &SS, false, false,
                                          ParsedType(),
                                          /*IsCtorOrDtorName=*/true,
                                          /*NonTrivialTypeSourceInfo=*/true);
      Result.setConstructorName(Ty, IdLoc, IdLoc);
    } else {
      // We have parsed an identifier.
      Result.setIdentifier(Id, IdLoc);      
    }

    // If the next token is a '<', we may have a template.
    if (TemplateSpecified || Tok.is(tok::less))
      return ParseUnqualifiedIdTemplateId(SS, TemplateKWLoc, Id, IdLoc,
                                          EnteringContext, ObjectType,
                                          Result, TemplateSpecified);
    
    return false;
  }
  
  // unqualified-id:
  //   template-id (already parsed and annotated)
  if (Tok.is(tok::annot_template_id)) {
    TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);

    // If the template-name names the current class, then this is a constructor 
    if (AllowConstructorName && TemplateId->Name &&
        Actions.isCurrentClassName(*TemplateId->Name, getCurScope(), &SS)) {
      if (SS.isSet()) {
        // C++ [class.qual]p2 specifies that a qualified template-name
        // is taken as the constructor name where a constructor can be
        // declared. Thus, the template arguments are extraneous, so
        // complain about them and remove them entirely.
        Diag(TemplateId->TemplateNameLoc, 
             diag::err_out_of_line_constructor_template_id)
          << TemplateId->Name
          << FixItHint::CreateRemoval(
                    SourceRange(TemplateId->LAngleLoc, TemplateId->RAngleLoc));
        ParsedType Ty = Actions.getTypeName(*TemplateId->Name,
                                            TemplateId->TemplateNameLoc,
                                            getCurScope(),
                                            &SS, false, false,
                                            ParsedType(),
                                            /*IsCtorOrDtorName=*/true,
                                            /*NontrivialTypeSourceInfo=*/true);
        Result.setConstructorName(Ty, TemplateId->TemplateNameLoc,
                                  TemplateId->RAngleLoc);
        ConsumeToken();
        return false;
      }

      Result.setConstructorTemplateId(TemplateId);
      ConsumeToken();
      return false;
    }

    // We have already parsed a template-id; consume the annotation token as
    // our unqualified-id.
    Result.setTemplateId(TemplateId);
    TemplateKWLoc = TemplateId->TemplateKWLoc;
    ConsumeToken();
    return false;
  }
  
  // unqualified-id:
  //   operator-function-id
  //   conversion-function-id
  if (Tok.is(tok::kw_operator)) {
    if (ParseUnqualifiedIdOperator(SS, EnteringContext, ObjectType, Result))
      return true;
    
    // If we have an operator-function-id or a literal-operator-id and the next
    // token is a '<', we may have a
    // 
    //   template-id:
    //     operator-function-id < template-argument-list[opt] >
    if ((Result.getKind() == UnqualifiedId::IK_OperatorFunctionId ||
         Result.getKind() == UnqualifiedId::IK_LiteralOperatorId) &&
        (TemplateSpecified || Tok.is(tok::less)))
      return ParseUnqualifiedIdTemplateId(SS, TemplateKWLoc,
                                          0, SourceLocation(),
                                          EnteringContext, ObjectType,
                                          Result, TemplateSpecified);
    
    return false;
  }
  
  if (getLangOpts().CPlusPlus && 
      (AllowDestructorName || SS.isSet()) && Tok.is(tok::tilde)) {
    // C++ [expr.unary.op]p10:
    //   There is an ambiguity in the unary-expression ~X(), where X is a 
    //   class-name. The ambiguity is resolved in favor of treating ~ as a 
    //    unary complement rather than treating ~X as referring to a destructor.
    
    // Parse the '~'.
    SourceLocation TildeLoc = ConsumeToken();

    if (SS.isEmpty() && Tok.is(tok::kw_decltype)) {
      DeclSpec DS(AttrFactory);
      SourceLocation EndLoc = ParseDecltypeSpecifier(DS);
      if (ParsedType Type = Actions.getDestructorType(DS, ObjectType)) {
        Result.setDestructorName(TildeLoc, Type, EndLoc);
        return false;
      }
      return true;
    }
    
    // Parse the class-name.
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_destructor_tilde_identifier);
      return true;
    }

    // Parse the class-name (or template-name in a simple-template-id).
    IdentifierInfo *ClassName = Tok.getIdentifierInfo();
    SourceLocation ClassNameLoc = ConsumeToken();
    
    if (TemplateSpecified || Tok.is(tok::less)) {
      Result.setDestructorName(TildeLoc, ParsedType(), ClassNameLoc);
      return ParseUnqualifiedIdTemplateId(SS, TemplateKWLoc,
                                          ClassName, ClassNameLoc,
                                          EnteringContext, ObjectType,
                                          Result, TemplateSpecified);
    }
    
    // Note that this is a destructor name.
    ParsedType Ty = Actions.getDestructorName(TildeLoc, *ClassName, 
                                              ClassNameLoc, getCurScope(),
                                              SS, ObjectType,
                                              EnteringContext);
    if (!Ty)
      return true;

    Result.setDestructorName(TildeLoc, Ty, ClassNameLoc);
    return false;
  }
  
  Diag(Tok, diag::err_expected_unqualified_id)
    << getLangOpts().CPlusPlus;
  return true;
}

/// ParseCXXNewExpression - Parse a C++ new-expression. New is used to allocate
/// memory in a typesafe manner and call constructors.
///
/// This method is called to parse the new expression after the optional :: has
/// been already parsed.  If the :: was present, "UseGlobal" is true and "Start"
/// is its location.  Otherwise, "Start" is the location of the 'new' token.
///
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
///
ExprResult
Parser::ParseCXXNewExpression(bool UseGlobal, SourceLocation Start) {
  assert(Tok.is(tok::kw_new) && "expected 'new' token");
  ConsumeToken();   // Consume 'new'

  // A '(' now can be a new-placement or the '(' wrapping the type-id in the
  // second form of new-expression. It can't be a new-type-id.

  ExprVector PlacementArgs(Actions);
  SourceLocation PlacementLParen, PlacementRParen;

  SourceRange TypeIdParens;
  DeclSpec DS(AttrFactory);
  Declarator DeclaratorInfo(DS, Declarator::CXXNewContext);
  if (Tok.is(tok::l_paren)) {
    // If it turns out to be a placement, we change the type location.
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    PlacementLParen = T.getOpenLocation();
    if (ParseExpressionListOrTypeId(PlacementArgs, DeclaratorInfo)) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }

    T.consumeClose();
    PlacementRParen = T.getCloseLocation();
    if (PlacementRParen.isInvalid()) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }

    if (PlacementArgs.empty()) {
      // Reset the placement locations. There was no placement.
      TypeIdParens = T.getRange();
      PlacementLParen = PlacementRParen = SourceLocation();
    } else {
      // We still need the type.
      if (Tok.is(tok::l_paren)) {
        BalancedDelimiterTracker T(*this, tok::l_paren);
        T.consumeOpen();
        MaybeParseGNUAttributes(DeclaratorInfo);
        ParseSpecifierQualifierList(DS);
        DeclaratorInfo.SetSourceRange(DS.getSourceRange());
        ParseDeclarator(DeclaratorInfo);
        T.consumeClose();
        TypeIdParens = T.getRange();
      } else {
        MaybeParseGNUAttributes(DeclaratorInfo);
        if (ParseCXXTypeSpecifierSeq(DS))
          DeclaratorInfo.setInvalidType(true);
        else {
          DeclaratorInfo.SetSourceRange(DS.getSourceRange());
          ParseDeclaratorInternal(DeclaratorInfo,
                                  &Parser::ParseDirectNewDeclarator);
        }
      }
    }
  } else {
    // A new-type-id is a simplified type-id, where essentially the
    // direct-declarator is replaced by a direct-new-declarator.
    MaybeParseGNUAttributes(DeclaratorInfo);
    if (ParseCXXTypeSpecifierSeq(DS))
      DeclaratorInfo.setInvalidType(true);
    else {
      DeclaratorInfo.SetSourceRange(DS.getSourceRange());
      ParseDeclaratorInternal(DeclaratorInfo,
                              &Parser::ParseDirectNewDeclarator);
    }
  }
  if (DeclaratorInfo.isInvalidType()) {
    SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
    return ExprError();
  }

  ExprResult Initializer;

  if (Tok.is(tok::l_paren)) {
    SourceLocation ConstructorLParen, ConstructorRParen;
    ExprVector ConstructorArgs(Actions);
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    ConstructorLParen = T.getOpenLocation();
    if (Tok.isNot(tok::r_paren)) {
      CommaLocsTy CommaLocs;
      if (ParseExpressionList(ConstructorArgs, CommaLocs)) {
        SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
        return ExprError();
      }
    }
    T.consumeClose();
    ConstructorRParen = T.getCloseLocation();
    if (ConstructorRParen.isInvalid()) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }
    Initializer = Actions.ActOnParenListExpr(ConstructorLParen,
                                             ConstructorRParen,
                                             move_arg(ConstructorArgs));
  } else if (Tok.is(tok::l_brace) && getLangOpts().CPlusPlus0x) {
    Diag(Tok.getLocation(),
         diag::warn_cxx98_compat_generalized_initializer_lists);
    Initializer = ParseBraceInitializer();
  }
  if (Initializer.isInvalid())
    return Initializer;

  return Actions.ActOnCXXNew(Start, UseGlobal, PlacementLParen,
                             move_arg(PlacementArgs), PlacementRParen,
                             TypeIdParens, DeclaratorInfo, Initializer.take());
}

/// ParseDirectNewDeclarator - Parses a direct-new-declarator. Intended to be
/// passed to ParseDeclaratorInternal.
///
///        direct-new-declarator:
///                   '[' expression ']'
///                   direct-new-declarator '[' constant-expression ']'
///
void Parser::ParseDirectNewDeclarator(Declarator &D) {
  // Parse the array dimensions.
  bool first = true;
  while (Tok.is(tok::l_square)) {
    // An array-size expression can't start with a lambda.
    if (CheckProhibitedCXX11Attribute())
      continue;

    BalancedDelimiterTracker T(*this, tok::l_square);
    T.consumeOpen();

    ExprResult Size(first ? ParseExpression()
                                : ParseConstantExpression());
    if (Size.isInvalid()) {
      // Recover
      SkipUntil(tok::r_square);
      return;
    }
    first = false;

    T.consumeClose();

    // Attributes here appertain to the array type. C++11 [expr.new]p5.
    ParsedAttributes Attrs(AttrFactory);
    MaybeParseCXX0XAttributes(Attrs);

    D.AddTypeInfo(DeclaratorChunk::getArray(0,
                                            /*static=*/false, /*star=*/false,
                                            Size.release(),
                                            T.getOpenLocation(),
                                            T.getCloseLocation()),
                  Attrs, T.getCloseLocation());

    if (T.getCloseLocation().isInvalid())
      return;
  }
}

/// ParseExpressionListOrTypeId - Parse either an expression-list or a type-id.
/// This ambiguity appears in the syntax of the C++ new operator.
///
///        new-expression:
///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
///                                     new-initializer[opt]
///
///        new-placement:
///                   '(' expression-list ')'
///
bool Parser::ParseExpressionListOrTypeId(
                                   SmallVectorImpl<Expr*> &PlacementArgs,
                                         Declarator &D) {
  // The '(' was already consumed.
  if (isTypeIdInParens()) {
    ParseSpecifierQualifierList(D.getMutableDeclSpec());
    D.SetSourceRange(D.getDeclSpec().getSourceRange());
    ParseDeclarator(D);
    return D.isInvalidType();
  }

  // It's not a type, it has to be an expression list.
  // Discard the comma locations - ActOnCXXNew has enough parameters.
  CommaLocsTy CommaLocs;
  return ParseExpressionList(PlacementArgs, CommaLocs);
}

/// ParseCXXDeleteExpression - Parse a C++ delete-expression. Delete is used
/// to free memory allocated by new.
///
/// This method is called to parse the 'delete' expression after the optional
/// '::' has been already parsed.  If the '::' was present, "UseGlobal" is true
/// and "Start" is its location.  Otherwise, "Start" is the location of the
/// 'delete' token.
///
///        delete-expression:
///                   '::'[opt] 'delete' cast-expression
///                   '::'[opt] 'delete' '[' ']' cast-expression
ExprResult
Parser::ParseCXXDeleteExpression(bool UseGlobal, SourceLocation Start) {
  assert(Tok.is(tok::kw_delete) && "Expected 'delete' keyword");
  ConsumeToken(); // Consume 'delete'

  // Array delete?
  bool ArrayDelete = false;
  if (Tok.is(tok::l_square) && NextToken().is(tok::r_square)) {
    // FIXME: This could be the start of a lambda-expression. We should
    // disambiguate this, but that will require arbitrary lookahead if
    // the next token is '(':
    //   delete [](int*){ /* ... */
    ArrayDelete = true;
    BalancedDelimiterTracker T(*this, tok::l_square);

    T.consumeOpen();
    T.consumeClose();
    if (T.getCloseLocation().isInvalid())
      return ExprError();
  }

  ExprResult Operand(ParseCastExpression(false));
  if (Operand.isInvalid())
    return move(Operand);

  return Actions.ActOnCXXDelete(Start, UseGlobal, ArrayDelete, Operand.take());
}

static UnaryTypeTrait UnaryTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known unary type trait.");
  case tok::kw___has_nothrow_assign:      return UTT_HasNothrowAssign;
  case tok::kw___has_nothrow_constructor: return UTT_HasNothrowConstructor;
  case tok::kw___has_nothrow_copy:           return UTT_HasNothrowCopy;
  case tok::kw___has_trivial_assign:      return UTT_HasTrivialAssign;
  case tok::kw___has_trivial_constructor:
                                    return UTT_HasTrivialDefaultConstructor;
  case tok::kw___has_trivial_copy:           return UTT_HasTrivialCopy;
  case tok::kw___has_trivial_destructor:  return UTT_HasTrivialDestructor;
  case tok::kw___has_virtual_destructor:  return UTT_HasVirtualDestructor;
  case tok::kw___is_abstract:             return UTT_IsAbstract;
  case tok::kw___is_arithmetic:              return UTT_IsArithmetic;
  case tok::kw___is_array:                   return UTT_IsArray;
  case tok::kw___is_class:                return UTT_IsClass;
  case tok::kw___is_complete_type:           return UTT_IsCompleteType;
  case tok::kw___is_compound:                return UTT_IsCompound;
  case tok::kw___is_const:                   return UTT_IsConst;
  case tok::kw___is_empty:                return UTT_IsEmpty;
  case tok::kw___is_enum:                 return UTT_IsEnum;
  case tok::kw___is_final:                 return UTT_IsFinal;
  case tok::kw___is_floating_point:          return UTT_IsFloatingPoint;
  case tok::kw___is_function:                return UTT_IsFunction;
  case tok::kw___is_fundamental:             return UTT_IsFundamental;
  case tok::kw___is_integral:                return UTT_IsIntegral;
  case tok::kw___is_lvalue_reference:        return UTT_IsLvalueReference;
  case tok::kw___is_member_function_pointer: return UTT_IsMemberFunctionPointer;
  case tok::kw___is_member_object_pointer:   return UTT_IsMemberObjectPointer;
  case tok::kw___is_member_pointer:          return UTT_IsMemberPointer;
  case tok::kw___is_object:                  return UTT_IsObject;
  case tok::kw___is_literal:              return UTT_IsLiteral;
  case tok::kw___is_literal_type:         return UTT_IsLiteral;
  case tok::kw___is_pod:                  return UTT_IsPOD;
  case tok::kw___is_pointer:                 return UTT_IsPointer;
  case tok::kw___is_polymorphic:          return UTT_IsPolymorphic;
  case tok::kw___is_reference:               return UTT_IsReference;
  case tok::kw___is_rvalue_reference:        return UTT_IsRvalueReference;
  case tok::kw___is_scalar:                  return UTT_IsScalar;
  case tok::kw___is_signed:                  return UTT_IsSigned;
  case tok::kw___is_standard_layout:         return UTT_IsStandardLayout;
  case tok::kw___is_trivial:                 return UTT_IsTrivial;
  case tok::kw___is_trivially_copyable:      return UTT_IsTriviallyCopyable;
  case tok::kw___is_union:                return UTT_IsUnion;
  case tok::kw___is_unsigned:                return UTT_IsUnsigned;
  case tok::kw___is_void:                    return UTT_IsVoid;
  case tok::kw___is_volatile:                return UTT_IsVolatile;
  }
}

static BinaryTypeTrait BinaryTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known binary type trait");
  case tok::kw___is_base_of:                 return BTT_IsBaseOf;
  case tok::kw___is_convertible:             return BTT_IsConvertible;
  case tok::kw___is_same:                    return BTT_IsSame;
  case tok::kw___builtin_types_compatible_p: return BTT_TypeCompatible;
  case tok::kw___is_convertible_to:          return BTT_IsConvertibleTo;
  case tok::kw___is_trivially_assignable:    return BTT_IsTriviallyAssignable;
  }
}

static TypeTrait TypeTraitFromTokKind(tok::TokenKind kind) {
  switch (kind) {
  default: llvm_unreachable("Not a known type trait");
  case tok::kw___is_trivially_constructible: 
    return TT_IsTriviallyConstructible;
  }
}

static ArrayTypeTrait ArrayTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known binary type trait");
  case tok::kw___array_rank:                 return ATT_ArrayRank;
  case tok::kw___array_extent:               return ATT_ArrayExtent;
  }
}

static ExpressionTrait ExpressionTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known unary expression trait.");
  case tok::kw___is_lvalue_expr:             return ET_IsLValueExpr;
  case tok::kw___is_rvalue_expr:             return ET_IsRValueExpr;
  }
}

/// ParseUnaryTypeTrait - Parse the built-in unary type-trait
/// pseudo-functions that allow implementation of the TR1/C++0x type traits
/// templates.
///
///       primary-expression:
/// [GNU]             unary-type-trait '(' type-id ')'
///
ExprResult Parser::ParseUnaryTypeTrait() {
  UnaryTypeTrait UTT = UnaryTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  // FIXME: Error reporting absolutely sucks! If the this fails to parse a type
  // there will be cryptic errors about mismatched parentheses and missing
  // specifiers.
  TypeResult Ty = ParseTypeName();

  T.consumeClose();

  if (Ty.isInvalid())
    return ExprError();

  return Actions.ActOnUnaryTypeTrait(UTT, Loc, Ty.get(), T.getCloseLocation());
}

/// ParseBinaryTypeTrait - Parse the built-in binary type-trait
/// pseudo-functions that allow implementation of the TR1/C++0x type traits
/// templates.
///
///       primary-expression:
/// [GNU]             binary-type-trait '(' type-id ',' type-id ')'
///
ExprResult Parser::ParseBinaryTypeTrait() {
  BinaryTypeTrait BTT = BinaryTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  TypeResult LhsTy = ParseTypeName();
  if (LhsTy.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  if (ExpectAndConsume(tok::comma, diag::err_expected_comma)) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  TypeResult RhsTy = ParseTypeName();
  if (RhsTy.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  T.consumeClose();

  return Actions.ActOnBinaryTypeTrait(BTT, Loc, LhsTy.get(), RhsTy.get(),
                                      T.getCloseLocation());
}

/// \brief Parse the built-in type-trait pseudo-functions that allow 
/// implementation of the TR1/C++11 type traits templates.
///
///       primary-expression:
///          type-trait '(' type-id-seq ')'
///
///       type-id-seq:
///          type-id ...[opt] type-id-seq[opt]
///
ExprResult Parser::ParseTypeTrait() {
  TypeTrait Kind = TypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();
  
  BalancedDelimiterTracker Parens(*this, tok::l_paren);
  if (Parens.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  llvm::SmallVector<ParsedType, 2> Args;
  do {
    // Parse the next type.
    TypeResult Ty = ParseTypeName();
    if (Ty.isInvalid()) {
      Parens.skipToEnd();
      return ExprError();
    }

    // Parse the ellipsis, if present.
    if (Tok.is(tok::ellipsis)) {
      Ty = Actions.ActOnPackExpansion(Ty.get(), ConsumeToken());
      if (Ty.isInvalid()) {
        Parens.skipToEnd();
        return ExprError();
      }
    }
    
    // Add this type to the list of arguments.
    Args.push_back(Ty.get());
    
    if (Tok.is(tok::comma)) {
      ConsumeToken();
      continue;
    }
    
    break;
  } while (true);
  
  if (Parens.consumeClose())
    return ExprError();
  
  return Actions.ActOnTypeTrait(Kind, Loc, Args, Parens.getCloseLocation());
}

/// ParseArrayTypeTrait - Parse the built-in array type-trait
/// pseudo-functions.
///
///       primary-expression:
/// [Embarcadero]     '__array_rank' '(' type-id ')'
/// [Embarcadero]     '__array_extent' '(' type-id ',' expression ')'
///
ExprResult Parser::ParseArrayTypeTrait() {
  ArrayTypeTrait ATT = ArrayTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  TypeResult Ty = ParseTypeName();
  if (Ty.isInvalid()) {
    SkipUntil(tok::comma);
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  switch (ATT) {
  case ATT_ArrayRank: {
    T.consumeClose();
    return Actions.ActOnArrayTypeTrait(ATT, Loc, Ty.get(), NULL,
                                       T.getCloseLocation());
  }
  case ATT_ArrayExtent: {
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma)) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }

    ExprResult DimExpr = ParseExpression();
    T.consumeClose();

    return Actions.ActOnArrayTypeTrait(ATT, Loc, Ty.get(), DimExpr.get(),
                                       T.getCloseLocation());
  }
  }
  llvm_unreachable("Invalid ArrayTypeTrait!");
}

/// ParseExpressionTrait - Parse built-in expression-trait
/// pseudo-functions like __is_lvalue_expr( xxx ).
///
///       primary-expression:
/// [Embarcadero]     expression-trait '(' expression ')'
///
ExprResult Parser::ParseExpressionTrait() {
  ExpressionTrait ET = ExpressionTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  ExprResult Expr = ParseExpression();

  T.consumeClose();

  return Actions.ActOnExpressionTrait(ET, Loc, Expr.get(),
                                      T.getCloseLocation());
}


/// ParseCXXAmbiguousParenExpression - We have parsed the left paren of a
/// parenthesized ambiguous type-id. This uses tentative parsing to disambiguate
/// based on the context past the parens.
ExprResult
Parser::ParseCXXAmbiguousParenExpression(ParenParseOption &ExprType,
                                         ParsedType &CastTy,
                                         BalancedDelimiterTracker &Tracker) {
  assert(getLangOpts().CPlusPlus && "Should only be called for C++!");
  assert(ExprType == CastExpr && "Compound literals are not ambiguous!");
  assert(isTypeIdInParens() && "Not a type-id!");

  ExprResult Result(true);
  CastTy = ParsedType();

  // We need to disambiguate a very ugly part of the C++ syntax:
  //
  // (T())x;  - type-id
  // (T())*x; - type-id
  // (T())/x; - expression
  // (T());   - expression
  //
  // The bad news is that we cannot use the specialized tentative parser, since
  // it can only verify that the thing inside the parens can be parsed as
  // type-id, it is not useful for determining the context past the parens.
  //
  // The good news is that the parser can disambiguate this part without
  // making any unnecessary Action calls.
  //
  // It uses a scheme similar to parsing inline methods. The parenthesized
  // tokens are cached, the context that follows is determined (possibly by
  // parsing a cast-expression), and then we re-introduce the cached tokens
  // into the token stream and parse them appropriately.

  ParenParseOption ParseAs;
  CachedTokens Toks;

  // Store the tokens of the parentheses. We will parse them after we determine
  // the context that follows them.
  if (!ConsumeAndStoreUntil(tok::r_paren, Toks)) {
    // We didn't find the ')' we expected.
    Tracker.consumeClose();
    return ExprError();
  }

  if (Tok.is(tok::l_brace)) {
    ParseAs = CompoundLiteral;
  } else {
    bool NotCastExpr;
    // FIXME: Special-case ++ and --: "(S())++;" is not a cast-expression
    if (Tok.is(tok::l_paren) && NextToken().is(tok::r_paren)) {
      NotCastExpr = true;
    } else {
      // Try parsing the cast-expression that may follow.
      // If it is not a cast-expression, NotCastExpr will be true and no token
      // will be consumed.
      Result = ParseCastExpression(false/*isUnaryExpression*/,
                                   false/*isAddressofOperand*/,
                                   NotCastExpr,
                                   // type-id has priority.
                                   IsTypeCast);
    }

    // If we parsed a cast-expression, it's really a type-id, otherwise it's
    // an expression.
    ParseAs = NotCastExpr ? SimpleExpr : CastExpr;
  }

  // The current token should go after the cached tokens.
  Toks.push_back(Tok);
  // Re-enter the stored parenthesized tokens into the token stream, so we may
  // parse them now.
  PP.EnterTokenStream(Toks.data(), Toks.size(),
                      true/*DisableMacroExpansion*/, false/*OwnsTokens*/);
  // Drop the current token and bring the first cached one. It's the same token
  // as when we entered this function.
  ConsumeAnyToken();

  if (ParseAs >= CompoundLiteral) {
    // Parse the type declarator.
    DeclSpec DS(AttrFactory);
    ParseSpecifierQualifierList(DS);
    Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
    ParseDeclarator(DeclaratorInfo);

    // Match the ')'.
    Tracker.consumeClose();

    if (ParseAs == CompoundLiteral) {
      ExprType = CompoundLiteral;
      TypeResult Ty = ParseTypeName();
       return ParseCompoundLiteralExpression(Ty.get(),
                                            Tracker.getOpenLocation(),
                                            Tracker.getCloseLocation());
    }

    // We parsed '(' type-id ')' and the thing after it wasn't a '{'.
    assert(ParseAs == CastExpr);

    if (DeclaratorInfo.isInvalidType())
      return ExprError();

    // Result is what ParseCastExpression returned earlier.
    if (!Result.isInvalid())
      Result = Actions.ActOnCastExpr(getCurScope(), Tracker.getOpenLocation(),
                                    DeclaratorInfo, CastTy,
                                    Tracker.getCloseLocation(), Result.take());
    return move(Result);
  }

  // Not a compound literal, and not followed by a cast-expression.
  assert(ParseAs == SimpleExpr);

  ExprType = SimpleExpr;
  Result = ParseExpression();
  if (!Result.isInvalid() && Tok.is(tok::r_paren))
    Result = Actions.ActOnParenExpr(Tracker.getOpenLocation(), 
                                    Tok.getLocation(), Result.take());

  // Match the ')'.
  if (Result.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  Tracker.consumeClose();
  return move(Result);
}
