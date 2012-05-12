//===--- ParseDeclCXX.cpp - C++ Declaration Parsing -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the C++ Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OperatorKinds.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "llvm/ADT/SmallString.h"
#include "RAIIObjectsForParser.h"
using namespace clang;

/// ParseNamespace - We know that the current token is a namespace keyword. This
/// may either be a top level namespace or a block-level namespace alias. If
/// there was an inline keyword, it has already been parsed.
///
///       namespace-definition: [C++ 7.3: basic.namespace]
///         named-namespace-definition
///         unnamed-namespace-definition
///
///       unnamed-namespace-definition:
///         'inline'[opt] 'namespace' attributes[opt] '{' namespace-body '}'
///
///       named-namespace-definition:
///         original-namespace-definition
///         extension-namespace-definition
///
///       original-namespace-definition:
///         'inline'[opt] 'namespace' identifier attributes[opt]
///             '{' namespace-body '}'
///
///       extension-namespace-definition:
///         'inline'[opt] 'namespace' original-namespace-name
///             '{' namespace-body '}'
///
///       namespace-alias-definition:  [C++ 7.3.2: namespace.alias]
///         'namespace' identifier '=' qualified-namespace-specifier ';'
///
Decl *Parser::ParseNamespace(unsigned Context,
                             SourceLocation &DeclEnd,
                             SourceLocation InlineLoc) {
  assert(Tok.is(tok::kw_namespace) && "Not a namespace!");
  SourceLocation NamespaceLoc = ConsumeToken();  // eat the 'namespace'.
  ObjCDeclContextSwitch ObjCDC(*this);
    
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteNamespaceDecl(getCurScope());
    cutOffParsing();
    return 0;
  }

  SourceLocation IdentLoc;
  IdentifierInfo *Ident = 0;
  std::vector<SourceLocation> ExtraIdentLoc;
  std::vector<IdentifierInfo*> ExtraIdent;
  std::vector<SourceLocation> ExtraNamespaceLoc;

  Token attrTok;

  if (Tok.is(tok::identifier)) {
    Ident = Tok.getIdentifierInfo();
    IdentLoc = ConsumeToken();  // eat the identifier.
    while (Tok.is(tok::coloncolon) && NextToken().is(tok::identifier)) {
      ExtraNamespaceLoc.push_back(ConsumeToken());
      ExtraIdent.push_back(Tok.getIdentifierInfo());
      ExtraIdentLoc.push_back(ConsumeToken());
    }
  }

  // Read label attributes, if present.
  ParsedAttributes attrs(AttrFactory);
  if (Tok.is(tok::kw___attribute)) {
    attrTok = Tok;
    ParseGNUAttributes(attrs);
  }

  if (Tok.is(tok::equal)) {
    if (!attrs.empty())
      Diag(attrTok, diag::err_unexpected_namespace_attributes_alias);
    if (InlineLoc.isValid())
      Diag(InlineLoc, diag::err_inline_namespace_alias)
          << FixItHint::CreateRemoval(InlineLoc);
    return ParseNamespaceAlias(NamespaceLoc, IdentLoc, Ident, DeclEnd);
  }


  BalancedDelimiterTracker T(*this, tok::l_brace);
  if (T.consumeOpen()) {
    if (!ExtraIdent.empty()) {
      Diag(ExtraNamespaceLoc[0], diag::err_nested_namespaces_with_double_colon)
          << SourceRange(ExtraNamespaceLoc.front(), ExtraIdentLoc.back());
    }
    Diag(Tok, Ident ? diag::err_expected_lbrace :
         diag::err_expected_ident_lbrace);
    return 0;
  }

  if (getCurScope()->isClassScope() || getCurScope()->isTemplateParamScope() || 
      getCurScope()->isInObjcMethodScope() || getCurScope()->getBlockParent() || 
      getCurScope()->getFnParent()) {
    if (!ExtraIdent.empty()) {
      Diag(ExtraNamespaceLoc[0], diag::err_nested_namespaces_with_double_colon)
          << SourceRange(ExtraNamespaceLoc.front(), ExtraIdentLoc.back());
    }
    Diag(T.getOpenLocation(), diag::err_namespace_nonnamespace_scope);
    SkipUntil(tok::r_brace, false);
    return 0;
  }

  if (!ExtraIdent.empty()) {
    TentativeParsingAction TPA(*this);
    SkipUntil(tok::r_brace, /*StopAtSemi*/false, /*DontConsume*/true);
    Token rBraceToken = Tok;
    TPA.Revert();

    if (!rBraceToken.is(tok::r_brace)) {
      Diag(ExtraNamespaceLoc[0], diag::err_nested_namespaces_with_double_colon)
          << SourceRange(ExtraNamespaceLoc.front(), ExtraIdentLoc.back());
    } else {
      std::string NamespaceFix;
      for (std::vector<IdentifierInfo*>::iterator I = ExtraIdent.begin(),
           E = ExtraIdent.end(); I != E; ++I) {
        NamespaceFix += " { namespace ";
        NamespaceFix += (*I)->getName();
      }

      std::string RBraces;
      for (unsigned i = 0, e = ExtraIdent.size(); i != e; ++i)
        RBraces +=  "} ";

      Diag(ExtraNamespaceLoc[0], diag::err_nested_namespaces_with_double_colon)
          << FixItHint::CreateReplacement(SourceRange(ExtraNamespaceLoc.front(),
                                                      ExtraIdentLoc.back()),
                                          NamespaceFix)
          << FixItHint::CreateInsertion(rBraceToken.getLocation(), RBraces);
    }
  }

  // If we're still good, complain about inline namespaces in non-C++0x now.
  if (InlineLoc.isValid())
    Diag(InlineLoc, getLangOpts().CPlusPlus0x ?
         diag::warn_cxx98_compat_inline_namespace : diag::ext_inline_namespace);

  // Enter a scope for the namespace.
  ParseScope NamespaceScope(this, Scope::DeclScope);

  Decl *NamespcDecl =
    Actions.ActOnStartNamespaceDef(getCurScope(), InlineLoc, NamespaceLoc,
                                   IdentLoc, Ident, T.getOpenLocation(), 
                                   attrs.getList());

  PrettyDeclStackTraceEntry CrashInfo(Actions, NamespcDecl, NamespaceLoc,
                                      "parsing namespace");

  // Parse the contents of the namespace.  This includes parsing recovery on 
  // any improperly nested namespaces.
  ParseInnerNamespace(ExtraIdentLoc, ExtraIdent, ExtraNamespaceLoc, 0,
                      InlineLoc, attrs, T);

  // Leave the namespace scope.
  NamespaceScope.Exit();

  DeclEnd = T.getCloseLocation();
  Actions.ActOnFinishNamespaceDef(NamespcDecl, DeclEnd);

  return NamespcDecl;
}

/// ParseInnerNamespace - Parse the contents of a namespace.
void Parser::ParseInnerNamespace(std::vector<SourceLocation>& IdentLoc,
                                 std::vector<IdentifierInfo*>& Ident,
                                 std::vector<SourceLocation>& NamespaceLoc,
                                 unsigned int index, SourceLocation& InlineLoc,
                                 ParsedAttributes& attrs,
                                 BalancedDelimiterTracker &Tracker) {
  if (index == Ident.size()) {
    while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
      ParsedAttributesWithRange attrs(AttrFactory);
      MaybeParseCXX0XAttributes(attrs);
      MaybeParseMicrosoftAttributes(attrs);
      ParseExternalDeclaration(attrs);
    }

    // The caller is what called check -- we are simply calling
    // the close for it.
    Tracker.consumeClose();

    return;
  }

  // Parse improperly nested namespaces.
  ParseScope NamespaceScope(this, Scope::DeclScope);
  Decl *NamespcDecl =
    Actions.ActOnStartNamespaceDef(getCurScope(), SourceLocation(),
                                   NamespaceLoc[index], IdentLoc[index],
                                   Ident[index], Tracker.getOpenLocation(), 
                                   attrs.getList());

  ParseInnerNamespace(IdentLoc, Ident, NamespaceLoc, ++index, InlineLoc,
                      attrs, Tracker);

  NamespaceScope.Exit();

  Actions.ActOnFinishNamespaceDef(NamespcDecl, Tracker.getCloseLocation());
}

/// ParseNamespaceAlias - Parse the part after the '=' in a namespace
/// alias definition.
///
Decl *Parser::ParseNamespaceAlias(SourceLocation NamespaceLoc,
                                  SourceLocation AliasLoc,
                                  IdentifierInfo *Alias,
                                  SourceLocation &DeclEnd) {
  assert(Tok.is(tok::equal) && "Not equal token");

  ConsumeToken(); // eat the '='.

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteNamespaceAliasDecl(getCurScope());
    cutOffParsing();
    return 0;
  }

  CXXScopeSpec SS;
  // Parse (optional) nested-name-specifier.
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);

  if (SS.isInvalid() || Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_namespace_name);
    // Skip to end of the definition and eat the ';'.
    SkipUntil(tok::semi);
    return 0;
  }

  // Parse identifier.
  IdentifierInfo *Ident = Tok.getIdentifierInfo();
  SourceLocation IdentLoc = ConsumeToken();

  // Eat the ';'.
  DeclEnd = Tok.getLocation();
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after_namespace_name,
                   "", tok::semi);

  return Actions.ActOnNamespaceAliasDef(getCurScope(), NamespaceLoc, AliasLoc, Alias,
                                        SS, IdentLoc, Ident);
}

/// ParseLinkage - We know that the current token is a string_literal
/// and just before that, that extern was seen.
///
///       linkage-specification: [C++ 7.5p2: dcl.link]
///         'extern' string-literal '{' declaration-seq[opt] '}'
///         'extern' string-literal declaration
///
Decl *Parser::ParseLinkage(ParsingDeclSpec &DS, unsigned Context) {
  assert(Tok.is(tok::string_literal) && "Not a string literal!");
  SmallString<8> LangBuffer;
  bool Invalid = false;
  StringRef Lang = PP.getSpelling(Tok, LangBuffer, &Invalid);
  if (Invalid)
    return 0;

  // FIXME: This is incorrect: linkage-specifiers are parsed in translation
  // phase 7, so string-literal concatenation is supposed to occur.
  //   extern "" "C" "" "+" "+" { } is legal.
  if (Tok.hasUDSuffix())
    Diag(Tok, diag::err_invalid_string_udl);
  SourceLocation Loc = ConsumeStringToken();

  ParseScope LinkageScope(this, Scope::DeclScope);
  Decl *LinkageSpec
    = Actions.ActOnStartLinkageSpecification(getCurScope(),
                                             DS.getSourceRange().getBegin(),
                                             Loc, Lang,
                                      Tok.is(tok::l_brace) ? Tok.getLocation()
                                                           : SourceLocation());

  ParsedAttributesWithRange attrs(AttrFactory);
  MaybeParseCXX0XAttributes(attrs);
  MaybeParseMicrosoftAttributes(attrs);

  if (Tok.isNot(tok::l_brace)) {
    // Reset the source range in DS, as the leading "extern"
    // does not really belong to the inner declaration ...
    DS.SetRangeStart(SourceLocation());
    DS.SetRangeEnd(SourceLocation());
    // ... but anyway remember that such an "extern" was seen.
    DS.setExternInLinkageSpec(true);
    ParseExternalDeclaration(attrs, &DS);
    return Actions.ActOnFinishLinkageSpecification(getCurScope(), LinkageSpec,
                                                   SourceLocation());
  }

  DS.abort();

  ProhibitAttributes(attrs);

  BalancedDelimiterTracker T(*this, tok::l_brace);
  T.consumeOpen();
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    ParsedAttributesWithRange attrs(AttrFactory);
    MaybeParseCXX0XAttributes(attrs);
    MaybeParseMicrosoftAttributes(attrs);
    ParseExternalDeclaration(attrs);
  }

  T.consumeClose();
  return Actions.ActOnFinishLinkageSpecification(getCurScope(), LinkageSpec,
                                                 T.getCloseLocation());
}

/// ParseUsingDirectiveOrDeclaration - Parse C++ using using-declaration or
/// using-directive. Assumes that current token is 'using'.
Decl *Parser::ParseUsingDirectiveOrDeclaration(unsigned Context,
                                         const ParsedTemplateInfo &TemplateInfo,
                                               SourceLocation &DeclEnd,
                                             ParsedAttributesWithRange &attrs,
                                               Decl **OwnedType) {
  assert(Tok.is(tok::kw_using) && "Not using token");
  ObjCDeclContextSwitch ObjCDC(*this);
  
  // Eat 'using'.
  SourceLocation UsingLoc = ConsumeToken();

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteUsing(getCurScope());
    cutOffParsing();
    return 0;
  }

  // 'using namespace' means this is a using-directive.
  if (Tok.is(tok::kw_namespace)) {
    // Template parameters are always an error here.
    if (TemplateInfo.Kind) {
      SourceRange R = TemplateInfo.getSourceRange();
      Diag(UsingLoc, diag::err_templated_using_directive)
        << R << FixItHint::CreateRemoval(R);
    }

    return ParseUsingDirective(Context, UsingLoc, DeclEnd, attrs);
  }

  // Otherwise, it must be a using-declaration or an alias-declaration.

  // Using declarations can't have attributes.
  ProhibitAttributes(attrs);

  return ParseUsingDeclaration(Context, TemplateInfo, UsingLoc, DeclEnd,
                                    AS_none, OwnedType);
}

/// ParseUsingDirective - Parse C++ using-directive, assumes
/// that current token is 'namespace' and 'using' was already parsed.
///
///       using-directive: [C++ 7.3.p4: namespace.udir]
///        'using' 'namespace' ::[opt] nested-name-specifier[opt]
///                 namespace-name ;
/// [GNU] using-directive:
///        'using' 'namespace' ::[opt] nested-name-specifier[opt]
///                 namespace-name attributes[opt] ;
///
Decl *Parser::ParseUsingDirective(unsigned Context,
                                  SourceLocation UsingLoc,
                                  SourceLocation &DeclEnd,
                                  ParsedAttributes &attrs) {
  assert(Tok.is(tok::kw_namespace) && "Not 'namespace' token");

  // Eat 'namespace'.
  SourceLocation NamespcLoc = ConsumeToken();

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteUsingDirective(getCurScope());
    cutOffParsing();
    return 0;
  }

  CXXScopeSpec SS;
  // Parse (optional) nested-name-specifier.
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);

  IdentifierInfo *NamespcName = 0;
  SourceLocation IdentLoc = SourceLocation();

  // Parse namespace-name.
  if (SS.isInvalid() || Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_namespace_name);
    // If there was invalid namespace name, skip to end of decl, and eat ';'.
    SkipUntil(tok::semi);
    // FIXME: Are there cases, when we would like to call ActOnUsingDirective?
    return 0;
  }

  // Parse identifier.
  NamespcName = Tok.getIdentifierInfo();
  IdentLoc = ConsumeToken();

  // Parse (optional) attributes (most likely GNU strong-using extension).
  bool GNUAttr = false;
  if (Tok.is(tok::kw___attribute)) {
    GNUAttr = true;
    ParseGNUAttributes(attrs);
  }

  // Eat ';'.
  DeclEnd = Tok.getLocation();
  ExpectAndConsume(tok::semi,
                   GNUAttr ? diag::err_expected_semi_after_attribute_list
                           : diag::err_expected_semi_after_namespace_name, 
                   "", tok::semi);

  return Actions.ActOnUsingDirective(getCurScope(), UsingLoc, NamespcLoc, SS,
                                     IdentLoc, NamespcName, attrs.getList());
}

/// ParseUsingDeclaration - Parse C++ using-declaration or alias-declaration.
/// Assumes that 'using' was already seen.
///
///     using-declaration: [C++ 7.3.p3: namespace.udecl]
///       'using' 'typename'[opt] ::[opt] nested-name-specifier
///               unqualified-id
///       'using' :: unqualified-id
///
///     alias-declaration: C++0x [decl.typedef]p2
///       'using' identifier = type-id ;
///
Decl *Parser::ParseUsingDeclaration(unsigned Context,
                                    const ParsedTemplateInfo &TemplateInfo,
                                    SourceLocation UsingLoc,
                                    SourceLocation &DeclEnd,
                                    AccessSpecifier AS,
                                    Decl **OwnedType) {
  CXXScopeSpec SS;
  SourceLocation TypenameLoc;
  bool IsTypeName;

  // Ignore optional 'typename'.
  // FIXME: This is wrong; we should parse this as a typename-specifier.
  if (Tok.is(tok::kw_typename)) {
    TypenameLoc = Tok.getLocation();
    ConsumeToken();
    IsTypeName = true;
  }
  else
    IsTypeName = false;

  // Parse nested-name-specifier.
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);

  // Check nested-name specifier.
  if (SS.isInvalid()) {
    SkipUntil(tok::semi);
    return 0;
  }

  // Parse the unqualified-id. We allow parsing of both constructor and
  // destructor names and allow the action module to diagnose any semantic
  // errors.
  SourceLocation TemplateKWLoc;
  UnqualifiedId Name;
  if (ParseUnqualifiedId(SS,
                         /*EnteringContext=*/false,
                         /*AllowDestructorName=*/true,
                         /*AllowConstructorName=*/true,
                         ParsedType(),
                         TemplateKWLoc,
                         Name)) {
    SkipUntil(tok::semi);
    return 0;
  }

  ParsedAttributes attrs(AttrFactory);

  // Maybe this is an alias-declaration.
  bool IsAliasDecl = Tok.is(tok::equal);
  TypeResult TypeAlias;
  if (IsAliasDecl) {
    // TODO: Attribute support. C++0x attributes may appear before the equals.
    // Where can GNU attributes appear?
    ConsumeToken();

    Diag(Tok.getLocation(), getLangOpts().CPlusPlus0x ?
         diag::warn_cxx98_compat_alias_declaration :
         diag::ext_alias_declaration);

    // Type alias templates cannot be specialized.
    int SpecKind = -1;
    if (TemplateInfo.Kind == ParsedTemplateInfo::Template &&
        Name.getKind() == UnqualifiedId::IK_TemplateId)
      SpecKind = 0;
    if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitSpecialization)
      SpecKind = 1;
    if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation)
      SpecKind = 2;
    if (SpecKind != -1) {
      SourceRange Range;
      if (SpecKind == 0)
        Range = SourceRange(Name.TemplateId->LAngleLoc,
                            Name.TemplateId->RAngleLoc);
      else
        Range = TemplateInfo.getSourceRange();
      Diag(Range.getBegin(), diag::err_alias_declaration_specialization)
        << SpecKind << Range;
      SkipUntil(tok::semi);
      return 0;
    }

    // Name must be an identifier.
    if (Name.getKind() != UnqualifiedId::IK_Identifier) {
      Diag(Name.StartLocation, diag::err_alias_declaration_not_identifier);
      // No removal fixit: can't recover from this.
      SkipUntil(tok::semi);
      return 0;
    } else if (IsTypeName)
      Diag(TypenameLoc, diag::err_alias_declaration_not_identifier)
        << FixItHint::CreateRemoval(SourceRange(TypenameLoc,
                             SS.isNotEmpty() ? SS.getEndLoc() : TypenameLoc));
    else if (SS.isNotEmpty())
      Diag(SS.getBeginLoc(), diag::err_alias_declaration_not_identifier)
        << FixItHint::CreateRemoval(SS.getRange());

    TypeAlias = ParseTypeName(0, TemplateInfo.Kind ?
                              Declarator::AliasTemplateContext :
                              Declarator::AliasDeclContext, AS, OwnedType);
  } else
    // Parse (optional) attributes (most likely GNU strong-using extension).
    MaybeParseGNUAttributes(attrs);

  // Eat ';'.
  DeclEnd = Tok.getLocation();
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after,
                   !attrs.empty() ? "attributes list" :
                   IsAliasDecl ? "alias declaration" : "using declaration",
                   tok::semi);

  // Diagnose an attempt to declare a templated using-declaration.
  // In C++0x, alias-declarations can be templates:
  //   template <...> using id = type;
  if (TemplateInfo.Kind && !IsAliasDecl) {
    SourceRange R = TemplateInfo.getSourceRange();
    Diag(UsingLoc, diag::err_templated_using_declaration)
      << R << FixItHint::CreateRemoval(R);

    // Unfortunately, we have to bail out instead of recovering by
    // ignoring the parameters, just in case the nested name specifier
    // depends on the parameters.
    return 0;
  }

  // "typename" keyword is allowed for identifiers only,
  // because it may be a type definition.
  if (IsTypeName && Name.getKind() != UnqualifiedId::IK_Identifier) {
    Diag(Name.getSourceRange().getBegin(), diag::err_typename_identifiers_only)
      << FixItHint::CreateRemoval(SourceRange(TypenameLoc));
    // Proceed parsing, but reset the IsTypeName flag.
    IsTypeName = false;
  }

  if (IsAliasDecl) {
    TemplateParameterLists *TemplateParams = TemplateInfo.TemplateParams;
    MultiTemplateParamsArg TemplateParamsArg(Actions,
      TemplateParams ? TemplateParams->data() : 0,
      TemplateParams ? TemplateParams->size() : 0);
    return Actions.ActOnAliasDeclaration(getCurScope(), AS, TemplateParamsArg,
                                         UsingLoc, Name, TypeAlias);
  }

  return Actions.ActOnUsingDeclaration(getCurScope(), AS, true, UsingLoc, SS,
                                       Name, attrs.getList(),
                                       IsTypeName, TypenameLoc);
}

/// ParseStaticAssertDeclaration - Parse C++0x or C11 static_assert-declaration.
///
/// [C++0x] static_assert-declaration:
///           static_assert ( constant-expression  ,  string-literal  ) ;
///
/// [C11]   static_assert-declaration:
///           _Static_assert ( constant-expression  ,  string-literal  ) ;
///
Decl *Parser::ParseStaticAssertDeclaration(SourceLocation &DeclEnd){
  assert((Tok.is(tok::kw_static_assert) || Tok.is(tok::kw__Static_assert)) &&
         "Not a static_assert declaration");

  if (Tok.is(tok::kw__Static_assert) && !getLangOpts().C11)
    Diag(Tok, diag::ext_c11_static_assert);
  if (Tok.is(tok::kw_static_assert))
    Diag(Tok, diag::warn_cxx98_compat_static_assert);

  SourceLocation StaticAssertLoc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.consumeOpen()) {
    Diag(Tok, diag::err_expected_lparen);
    return 0;
  }

  ExprResult AssertExpr(ParseConstantExpression());
  if (AssertExpr.isInvalid()) {
    SkipUntil(tok::semi);
    return 0;
  }

  if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "", tok::semi))
    return 0;

  if (!isTokenStringLiteral()) {
    Diag(Tok, diag::err_expected_string_literal);
    SkipUntil(tok::semi);
    return 0;
  }

  ExprResult AssertMessage(ParseStringLiteralExpression());
  if (AssertMessage.isInvalid()) {
    SkipUntil(tok::semi);
    return 0;
  }

  T.consumeClose();

  DeclEnd = Tok.getLocation();
  ExpectAndConsumeSemi(diag::err_expected_semi_after_static_assert);

  return Actions.ActOnStaticAssertDeclaration(StaticAssertLoc,
                                              AssertExpr.take(),
                                              AssertMessage.take(),
                                              T.getCloseLocation());
}

/// ParseDecltypeSpecifier - Parse a C++0x decltype specifier.
///
/// 'decltype' ( expression )
///
SourceLocation Parser::ParseDecltypeSpecifier(DeclSpec &DS) {
  assert((Tok.is(tok::kw_decltype) || Tok.is(tok::annot_decltype))
           && "Not a decltype specifier");
  

  ExprResult Result;
  SourceLocation StartLoc = Tok.getLocation();
  SourceLocation EndLoc;

  if (Tok.is(tok::annot_decltype)) {
    Result = getExprAnnotation(Tok);
    EndLoc = Tok.getAnnotationEndLoc();
    ConsumeToken();
    if (Result.isInvalid()) {
      DS.SetTypeSpecError();
      return EndLoc;
    }
  } else {
    if (Tok.getIdentifierInfo()->isStr("decltype"))
      Diag(Tok, diag::warn_cxx98_compat_decltype);

    ConsumeToken();

    BalancedDelimiterTracker T(*this, tok::l_paren);
    if (T.expectAndConsume(diag::err_expected_lparen_after,
                           "decltype", tok::r_paren)) {
      DS.SetTypeSpecError();
      return T.getOpenLocation() == Tok.getLocation() ?
             StartLoc : T.getOpenLocation();
    }

    // Parse the expression

    // C++0x [dcl.type.simple]p4:
    //   The operand of the decltype specifier is an unevaluated operand.
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated,
                                                 0, /*IsDecltype=*/true);
    Result = ParseExpression();
    if (Result.isInvalid()) {
      SkipUntil(tok::r_paren);
      DS.SetTypeSpecError();
      return StartLoc;
    }

    // Match the ')'
    T.consumeClose();
    if (T.getCloseLocation().isInvalid()) {
      DS.SetTypeSpecError();
      // FIXME: this should return the location of the last token
      //        that was consumed (by "consumeClose()")
      return T.getCloseLocation();
    }

    Result = Actions.ActOnDecltypeExpression(Result.take());
    if (Result.isInvalid()) {
      DS.SetTypeSpecError();
      return T.getCloseLocation();
    }

    EndLoc = T.getCloseLocation();
  }

  const char *PrevSpec = 0;
  unsigned DiagID;
  // Check for duplicate type specifiers (e.g. "int decltype(a)").
  if (DS.SetTypeSpecType(DeclSpec::TST_decltype, StartLoc, PrevSpec,
                         DiagID, Result.release())) {
    Diag(StartLoc, DiagID) << PrevSpec;
    DS.SetTypeSpecError();
  }
  return EndLoc;
}

void Parser::AnnotateExistingDecltypeSpecifier(const DeclSpec& DS, 
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc) {
  // make sure we have a token we can turn into an annotation token
  if (PP.isBacktrackEnabled())
    PP.RevertCachedTokens(1);
  else
    PP.EnterToken(Tok);

  Tok.setKind(tok::annot_decltype);
  setExprAnnotation(Tok, DS.getTypeSpecType() == TST_decltype ? 
                         DS.getRepAsExpr() : ExprResult());
  Tok.setAnnotationEndLoc(EndLoc);
  Tok.setLocation(StartLoc);
  PP.AnnotateCachedTokens(Tok);
}

void Parser::ParseUnderlyingTypeSpecifier(DeclSpec &DS) {
  assert(Tok.is(tok::kw___underlying_type) &&
         "Not an underlying type specifier");

  SourceLocation StartLoc = ConsumeToken();
  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                       "__underlying_type", tok::r_paren)) {
    return;
  }

  TypeResult Result = ParseTypeName();
  if (Result.isInvalid()) {
    SkipUntil(tok::r_paren);
    return;
  }

  // Match the ')'
  T.consumeClose();
  if (T.getCloseLocation().isInvalid())
    return;

  const char *PrevSpec = 0;
  unsigned DiagID;
  if (DS.SetTypeSpecType(DeclSpec::TST_underlyingType, StartLoc, PrevSpec,
                         DiagID, Result.release()))
    Diag(StartLoc, DiagID) << PrevSpec;
}

/// ParseBaseTypeSpecifier - Parse a C++ base-type-specifier which is either a
/// class name or decltype-specifier. Note that we only check that the result 
/// names a type; semantic analysis will need to verify that the type names a 
/// class. The result is either a type or null, depending on whether a type 
/// name was found.
///
///       base-type-specifier: [C++ 10.1]
///         class-or-decltype
///       class-or-decltype: [C++ 10.1]
///         nested-name-specifier[opt] class-name
///         decltype-specifier
///       class-name: [C++ 9.1]
///         identifier
///         simple-template-id
///
Parser::TypeResult Parser::ParseBaseTypeSpecifier(SourceLocation &BaseLoc,
                                                  SourceLocation &EndLocation) {
  // Ignore attempts to use typename
  if (Tok.is(tok::kw_typename)) {
    Diag(Tok, diag::err_expected_class_name_not_template)
      << FixItHint::CreateRemoval(Tok.getLocation());
    ConsumeToken();
  }

  // Parse optional nested-name-specifier
  CXXScopeSpec SS;
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);

  BaseLoc = Tok.getLocation();

  // Parse decltype-specifier
  // tok == kw_decltype is just error recovery, it can only happen when SS 
  // isn't empty
  if (Tok.is(tok::kw_decltype) || Tok.is(tok::annot_decltype)) {
    if (SS.isNotEmpty())
      Diag(SS.getBeginLoc(), diag::err_unexpected_scope_on_base_decltype)
        << FixItHint::CreateRemoval(SS.getRange());
    // Fake up a Declarator to use with ActOnTypeName.
    DeclSpec DS(AttrFactory);

    EndLocation = ParseDecltypeSpecifier(DS);

    Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
    return Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
  }

  // Check whether we have a template-id that names a type.
  if (Tok.is(tok::annot_template_id)) {
    TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);
    if (TemplateId->Kind == TNK_Type_template ||
        TemplateId->Kind == TNK_Dependent_template_name) {
      AnnotateTemplateIdTokenAsType();

      assert(Tok.is(tok::annot_typename) && "template-id -> type failed");
      ParsedType Type = getTypeAnnotation(Tok);
      EndLocation = Tok.getAnnotationEndLoc();
      ConsumeToken();

      if (Type)
        return Type;
      return true;
    }

    // Fall through to produce an error below.
  }

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_class_name);
    return true;
  }

  IdentifierInfo *Id = Tok.getIdentifierInfo();
  SourceLocation IdLoc = ConsumeToken();

  if (Tok.is(tok::less)) {
    // It looks the user intended to write a template-id here, but the
    // template-name was wrong. Try to fix that.
    TemplateNameKind TNK = TNK_Type_template;
    TemplateTy Template;
    if (!Actions.DiagnoseUnknownTemplateName(*Id, IdLoc, getCurScope(),
                                             &SS, Template, TNK)) {
      Diag(IdLoc, diag::err_unknown_template_name)
        << Id;
    }

    if (!Template)
      return true;

    // Form the template name
    UnqualifiedId TemplateName;
    TemplateName.setIdentifier(Id, IdLoc);

    // Parse the full template-id, then turn it into a type.
    if (AnnotateTemplateIdToken(Template, TNK, SS, SourceLocation(),
                                TemplateName, true))
      return true;
    if (TNK == TNK_Dependent_template_name)
      AnnotateTemplateIdTokenAsType();

    // If we didn't end up with a typename token, there's nothing more we
    // can do.
    if (Tok.isNot(tok::annot_typename))
      return true;

    // Retrieve the type from the annotation token, consume that token, and
    // return.
    EndLocation = Tok.getAnnotationEndLoc();
    ParsedType Type = getTypeAnnotation(Tok);
    ConsumeToken();
    return Type;
  }

  // We have an identifier; check whether it is actually a type.
  ParsedType Type = Actions.getTypeName(*Id, IdLoc, getCurScope(), &SS, true,
                                        false, ParsedType(),
                                        /*IsCtorOrDtorName=*/false,
                                        /*NonTrivialTypeSourceInfo=*/true);
  if (!Type) {
    Diag(IdLoc, diag::err_expected_class_name);
    return true;
  }

  // Consume the identifier.
  EndLocation = IdLoc;

  // Fake up a Declarator to use with ActOnTypeName.
  DeclSpec DS(AttrFactory);
  DS.SetRangeStart(IdLoc);
  DS.SetRangeEnd(EndLocation);
  DS.getTypeSpecScope() = SS;

  const char *PrevSpec = 0;
  unsigned DiagID;
  DS.SetTypeSpecType(TST_typename, IdLoc, PrevSpec, DiagID, Type);

  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  return Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
}

/// ParseClassSpecifier - Parse a C++ class-specifier [C++ class] or
/// elaborated-type-specifier [C++ dcl.type.elab]; we can't tell which
/// until we reach the start of a definition or see a token that
/// cannot start a definition.
///
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
void Parser::ParseClassSpecifier(tok::TokenKind TagTokKind,
                                 SourceLocation StartLoc, DeclSpec &DS,
                                 const ParsedTemplateInfo &TemplateInfo,
                                 AccessSpecifier AS, 
                                 bool EnteringContext, DeclSpecContext DSC) {
  DeclSpec::TST TagType;
  if (TagTokKind == tok::kw_struct)
    TagType = DeclSpec::TST_struct;
  else if (TagTokKind == tok::kw_class)
    TagType = DeclSpec::TST_class;
  else {
    assert(TagTokKind == tok::kw_union && "Not a class specifier");
    TagType = DeclSpec::TST_union;
  }

  if (Tok.is(tok::code_completion)) {
    // Code completion for a struct, class, or union name.
    Actions.CodeCompleteTag(getCurScope(), TagType);
    return cutOffParsing();
  }

  // C++03 [temp.explicit] 14.7.2/8:
  //   The usual access checking rules do not apply to names used to specify
  //   explicit instantiations.
  //
  // As an extension we do not perform access checking on the names used to
  // specify explicit specializations either. This is important to allow
  // specializing traits classes for private types.
  Sema::SuppressAccessChecksRAII SuppressAccess(Actions,
    TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation ||
    TemplateInfo.Kind == ParsedTemplateInfo::ExplicitSpecialization);

  ParsedAttributes attrs(AttrFactory);
  // If attributes exist after tag, parse them.
  if (Tok.is(tok::kw___attribute))
    ParseGNUAttributes(attrs);

  // If declspecs exist after tag, parse them.
  while (Tok.is(tok::kw___declspec))
    ParseMicrosoftDeclSpec(attrs);

  // If C++0x attributes exist here, parse them.
  // FIXME: Are we consistent with the ordering of parsing of different
  // styles of attributes?
  MaybeParseCXX0XAttributes(attrs);

  if (TagType == DeclSpec::TST_struct &&
      !Tok.is(tok::identifier) &&
      Tok.getIdentifierInfo() &&
      (Tok.is(tok::kw___is_arithmetic) ||
       Tok.is(tok::kw___is_convertible) ||
       Tok.is(tok::kw___is_empty) ||
       Tok.is(tok::kw___is_floating_point) ||
       Tok.is(tok::kw___is_function) ||
       Tok.is(tok::kw___is_fundamental) ||
       Tok.is(tok::kw___is_integral) ||
       Tok.is(tok::kw___is_member_function_pointer) ||
       Tok.is(tok::kw___is_member_pointer) ||
       Tok.is(tok::kw___is_pod) ||
       Tok.is(tok::kw___is_pointer) ||
       Tok.is(tok::kw___is_same) ||
       Tok.is(tok::kw___is_scalar) ||
       Tok.is(tok::kw___is_signed) ||
       Tok.is(tok::kw___is_unsigned) ||
       Tok.is(tok::kw___is_void))) {
    // GNU libstdc++ 4.2 and libc++ use certain intrinsic names as the
    // name of struct templates, but some are keywords in GCC >= 4.3
    // and Clang. Therefore, when we see the token sequence "struct
    // X", make X into a normal identifier rather than a keyword, to
    // allow libstdc++ 4.2 and libc++ to work properly.
    Tok.getIdentifierInfo()->RevertTokenIDToIdentifier();
    Tok.setKind(tok::identifier);
  }

  // Parse the (optional) nested-name-specifier.
  CXXScopeSpec &SS = DS.getTypeSpecScope();
  if (getLangOpts().CPlusPlus) {
    // "FOO : BAR" is not a potential typo for "FOO::BAR".
    ColonProtectionRAIIObject X(*this);

    if (ParseOptionalCXXScopeSpecifier(SS, ParsedType(), EnteringContext))
      DS.SetTypeSpecError();
    if (SS.isSet())
      if (Tok.isNot(tok::identifier) && Tok.isNot(tok::annot_template_id))
        Diag(Tok, diag::err_expected_ident);
  }

  TemplateParameterLists *TemplateParams = TemplateInfo.TemplateParams;

  // Parse the (optional) class name or simple-template-id.
  IdentifierInfo *Name = 0;
  SourceLocation NameLoc;
  TemplateIdAnnotation *TemplateId = 0;
  if (Tok.is(tok::identifier)) {
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();

    if (Tok.is(tok::less) && getLangOpts().CPlusPlus) {
      // The name was supposed to refer to a template, but didn't.
      // Eat the template argument list and try to continue parsing this as
      // a class (or template thereof).
      TemplateArgList TemplateArgs;
      SourceLocation LAngleLoc, RAngleLoc;
      if (ParseTemplateIdAfterTemplateName(TemplateTy(), NameLoc, SS,
                                           true, LAngleLoc,
                                           TemplateArgs, RAngleLoc)) {
        // We couldn't parse the template argument list at all, so don't
        // try to give any location information for the list.
        LAngleLoc = RAngleLoc = SourceLocation();
      }

      Diag(NameLoc, diag::err_explicit_spec_non_template)
        << (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation)
        << (TagType == DeclSpec::TST_class? 0
            : TagType == DeclSpec::TST_struct? 1
            : 2)
        << Name
        << SourceRange(LAngleLoc, RAngleLoc);

      // Strip off the last template parameter list if it was empty, since
      // we've removed its template argument list.
      if (TemplateParams && TemplateInfo.LastParameterListWasEmpty) {
        if (TemplateParams && TemplateParams->size() > 1) {
          TemplateParams->pop_back();
        } else {
          TemplateParams = 0;
          const_cast<ParsedTemplateInfo&>(TemplateInfo).Kind
            = ParsedTemplateInfo::NonTemplate;
        }
      } else if (TemplateInfo.Kind
                                == ParsedTemplateInfo::ExplicitInstantiation) {
        // Pretend this is just a forward declaration.
        TemplateParams = 0;
        const_cast<ParsedTemplateInfo&>(TemplateInfo).Kind
          = ParsedTemplateInfo::NonTemplate;
        const_cast<ParsedTemplateInfo&>(TemplateInfo).TemplateLoc
          = SourceLocation();
        const_cast<ParsedTemplateInfo&>(TemplateInfo).ExternLoc
          = SourceLocation();
      }
    }
  } else if (Tok.is(tok::annot_template_id)) {
    TemplateId = takeTemplateIdAnnotation(Tok);
    NameLoc = ConsumeToken();

    if (TemplateId->Kind != TNK_Type_template &&
        TemplateId->Kind != TNK_Dependent_template_name) {
      // The template-name in the simple-template-id refers to
      // something other than a class template. Give an appropriate
      // error message and skip to the ';'.
      SourceRange Range(NameLoc);
      if (SS.isNotEmpty())
        Range.setBegin(SS.getBeginLoc());

      Diag(TemplateId->LAngleLoc, diag::err_template_spec_syntax_non_template)
        << Name << static_cast<int>(TemplateId->Kind) << Range;

      DS.SetTypeSpecError();
      SkipUntil(tok::semi, false, true);
      return;
    }
  }

  // As soon as we're finished parsing the class's template-id, turn access
  // checking back on.
  SuppressAccess.done();

  // There are four options here.
  //  - If we are in a trailing return type, this is always just a reference,
  //    and we must not try to parse a definition. For instance,
  //      [] () -> struct S { };
  //    does not define a type.
  //  - If we have 'struct foo {...', 'struct foo :...',
  //    'struct foo final :' or 'struct foo final {', then this is a definition.
  //  - If we have 'struct foo;', then this is either a forward declaration
  //    or a friend declaration, which have to be treated differently.
  //  - Otherwise we have something like 'struct foo xyz', a reference.
  // However, in type-specifier-seq's, things look like declarations but are
  // just references, e.g.
  //   new struct s;
  // or
  //   &T::operator struct s;
  // For these, DSC is DSC_type_specifier.
  Sema::TagUseKind TUK;
  if (DSC == DSC_trailing)
    TUK = Sema::TUK_Reference;
  else if (Tok.is(tok::l_brace) ||
           (getLangOpts().CPlusPlus && Tok.is(tok::colon)) ||
           (isCXX0XFinalKeyword() &&
            (NextToken().is(tok::l_brace) || NextToken().is(tok::colon)))) {
    if (DS.isFriendSpecified()) {
      // C++ [class.friend]p2:
      //   A class shall not be defined in a friend declaration.
      Diag(Tok.getLocation(), diag::err_friend_decl_defines_type)
        << SourceRange(DS.getFriendSpecLoc());

      // Skip everything up to the semicolon, so that this looks like a proper
      // friend class (or template thereof) declaration.
      SkipUntil(tok::semi, true, true);
      TUK = Sema::TUK_Friend;
    } else {
      // Okay, this is a class definition.
      TUK = Sema::TUK_Definition;
    }
  } else if (Tok.is(tok::semi) && DSC != DSC_type_specifier)
    TUK = DS.isFriendSpecified() ? Sema::TUK_Friend : Sema::TUK_Declaration;
  else
    TUK = Sema::TUK_Reference;

  if (!Name && !TemplateId && (DS.getTypeSpecType() == DeclSpec::TST_error ||
                               TUK != Sema::TUK_Definition)) {
    if (DS.getTypeSpecType() != DeclSpec::TST_error) {
      // We have a declaration or reference to an anonymous class.
      Diag(StartLoc, diag::err_anon_type_definition)
        << DeclSpec::getSpecifierName(TagType);
    }

    SkipUntil(tok::comma, true);
    return;
  }

  // Create the tag portion of the class or class template.
  DeclResult TagOrTempResult = true; // invalid
  TypeResult TypeResult = true; // invalid

  bool Owned = false;
  if (TemplateId) {
    // Explicit specialization, class template partial specialization,
    // or explicit instantiation.
    ASTTemplateArgsPtr TemplateArgsPtr(Actions,
                                       TemplateId->getTemplateArgs(),
                                       TemplateId->NumArgs);
    if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation &&
        TUK == Sema::TUK_Declaration) {
      // This is an explicit instantiation of a class template.
      TagOrTempResult
        = Actions.ActOnExplicitInstantiation(getCurScope(),
                                             TemplateInfo.ExternLoc,
                                             TemplateInfo.TemplateLoc,
                                             TagType,
                                             StartLoc,
                                             SS,
                                             TemplateId->Template,
                                             TemplateId->TemplateNameLoc,
                                             TemplateId->LAngleLoc,
                                             TemplateArgsPtr,
                                             TemplateId->RAngleLoc,
                                             attrs.getList());

    // Friend template-ids are treated as references unless
    // they have template headers, in which case they're ill-formed
    // (FIXME: "template <class T> friend class A<T>::B<int>;").
    // We diagnose this error in ActOnClassTemplateSpecialization.
    } else if (TUK == Sema::TUK_Reference ||
               (TUK == Sema::TUK_Friend &&
                TemplateInfo.Kind == ParsedTemplateInfo::NonTemplate)) {
      TypeResult = Actions.ActOnTagTemplateIdType(TUK, TagType, StartLoc,
                                                  TemplateId->SS,
                                                  TemplateId->TemplateKWLoc,
                                                  TemplateId->Template,
                                                  TemplateId->TemplateNameLoc,
                                                  TemplateId->LAngleLoc,
                                                  TemplateArgsPtr,
                                                  TemplateId->RAngleLoc);
    } else {
      // This is an explicit specialization or a class template
      // partial specialization.
      TemplateParameterLists FakedParamLists;

      if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation) {
        // This looks like an explicit instantiation, because we have
        // something like
        //
        //   template class Foo<X>
        //
        // but it actually has a definition. Most likely, this was
        // meant to be an explicit specialization, but the user forgot
        // the '<>' after 'template'.
        assert(TUK == Sema::TUK_Definition && "Expected a definition here");

        SourceLocation LAngleLoc
          = PP.getLocForEndOfToken(TemplateInfo.TemplateLoc);
        Diag(TemplateId->TemplateNameLoc,
             diag::err_explicit_instantiation_with_definition)
          << SourceRange(TemplateInfo.TemplateLoc)
          << FixItHint::CreateInsertion(LAngleLoc, "<>");

        // Create a fake template parameter list that contains only
        // "template<>", so that we treat this construct as a class
        // template specialization.
        FakedParamLists.push_back(
          Actions.ActOnTemplateParameterList(0, SourceLocation(),
                                             TemplateInfo.TemplateLoc,
                                             LAngleLoc,
                                             0, 0,
                                             LAngleLoc));
        TemplateParams = &FakedParamLists;
      }

      // Build the class template specialization.
      TagOrTempResult
        = Actions.ActOnClassTemplateSpecialization(getCurScope(), TagType, TUK,
                       StartLoc, DS.getModulePrivateSpecLoc(), SS,
                       TemplateId->Template,
                       TemplateId->TemplateNameLoc,
                       TemplateId->LAngleLoc,
                       TemplateArgsPtr,
                       TemplateId->RAngleLoc,
                       attrs.getList(),
                       MultiTemplateParamsArg(Actions,
                                    TemplateParams? &(*TemplateParams)[0] : 0,
                                 TemplateParams? TemplateParams->size() : 0));
    }
  } else if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation &&
             TUK == Sema::TUK_Declaration) {
    // Explicit instantiation of a member of a class template
    // specialization, e.g.,
    //
    //   template struct Outer<int>::Inner;
    //
    TagOrTempResult
      = Actions.ActOnExplicitInstantiation(getCurScope(),
                                           TemplateInfo.ExternLoc,
                                           TemplateInfo.TemplateLoc,
                                           TagType, StartLoc, SS, Name,
                                           NameLoc, attrs.getList());
  } else if (TUK == Sema::TUK_Friend &&
             TemplateInfo.Kind != ParsedTemplateInfo::NonTemplate) {
    TagOrTempResult =
      Actions.ActOnTemplatedFriendTag(getCurScope(), DS.getFriendSpecLoc(),
                                      TagType, StartLoc, SS,
                                      Name, NameLoc, attrs.getList(),
                                      MultiTemplateParamsArg(Actions,
                                    TemplateParams? &(*TemplateParams)[0] : 0,
                                 TemplateParams? TemplateParams->size() : 0));
  } else {
    if (TemplateInfo.Kind == ParsedTemplateInfo::ExplicitInstantiation &&
        TUK == Sema::TUK_Definition) {
      // FIXME: Diagnose this particular error.
    }

    bool IsDependent = false;

    // Don't pass down template parameter lists if this is just a tag
    // reference.  For example, we don't need the template parameters here:
    //   template <class T> class A *makeA(T t);
    MultiTemplateParamsArg TParams;
    if (TUK != Sema::TUK_Reference && TemplateParams)
      TParams =
        MultiTemplateParamsArg(&(*TemplateParams)[0], TemplateParams->size());

    // Declaration or definition of a class type
    TagOrTempResult = Actions.ActOnTag(getCurScope(), TagType, TUK, StartLoc,
                                       SS, Name, NameLoc, attrs.getList(), AS,
                                       DS.getModulePrivateSpecLoc(),
                                       TParams, Owned, IsDependent,
                                       SourceLocation(), false,
                                       clang::TypeResult());

    // If ActOnTag said the type was dependent, try again with the
    // less common call.
    if (IsDependent) {
      assert(TUK == Sema::TUK_Reference || TUK == Sema::TUK_Friend);
      TypeResult = Actions.ActOnDependentTag(getCurScope(), TagType, TUK,
                                             SS, Name, StartLoc, NameLoc);
    }
  }

  // If there is a body, parse it and inform the actions module.
  if (TUK == Sema::TUK_Definition) {
    assert(Tok.is(tok::l_brace) ||
           (getLangOpts().CPlusPlus && Tok.is(tok::colon)) ||
           isCXX0XFinalKeyword());
    if (getLangOpts().CPlusPlus)
      ParseCXXMemberSpecification(StartLoc, TagType, TagOrTempResult.get());
    else
      ParseStructUnionBody(StartLoc, TagType, TagOrTempResult.get());
  }

  const char *PrevSpec = 0;
  unsigned DiagID;
  bool Result;
  if (!TypeResult.isInvalid()) {
    Result = DS.SetTypeSpecType(DeclSpec::TST_typename, StartLoc,
                                NameLoc.isValid() ? NameLoc : StartLoc,
                                PrevSpec, DiagID, TypeResult.get());
  } else if (!TagOrTempResult.isInvalid()) {
    Result = DS.SetTypeSpecType(TagType, StartLoc,
                                NameLoc.isValid() ? NameLoc : StartLoc,
                                PrevSpec, DiagID, TagOrTempResult.get(), Owned);
  } else {
    DS.SetTypeSpecError();
    return;
  }

  if (Result)
    Diag(StartLoc, DiagID) << PrevSpec;

  // At this point, we've successfully parsed a class-specifier in 'definition'
  // form (e.g. "struct foo { int x; }".  While we could just return here, we're
  // going to look at what comes after it to improve error recovery.  If an
  // impossible token occurs next, we assume that the programmer forgot a ; at
  // the end of the declaration and recover that way.
  //
  // This switch enumerates the valid "follow" set for definition.
  if (TUK == Sema::TUK_Definition) {
    bool ExpectedSemi = true;
    switch (Tok.getKind()) {
    default: break;
    case tok::semi:               // struct foo {...} ;
    case tok::star:               // struct foo {...} *         P;
    case tok::amp:                // struct foo {...} &         R = ...
    case tok::identifier:         // struct foo {...} V         ;
    case tok::r_paren:            //(struct foo {...} )         {4}
    case tok::annot_cxxscope:     // struct foo {...} a::       b;
    case tok::annot_typename:     // struct foo {...} a         ::b;
    case tok::annot_template_id:  // struct foo {...} a<int>    ::b;
    case tok::l_paren:            // struct foo {...} (         x);
    case tok::comma:              // __builtin_offsetof(struct foo{...} ,
      ExpectedSemi = false;
      break;
    // Type qualifiers
    case tok::kw_const:           // struct foo {...} const     x;
    case tok::kw_volatile:        // struct foo {...} volatile  x;
    case tok::kw_restrict:        // struct foo {...} restrict  x;
    case tok::kw_inline:          // struct foo {...} inline    foo() {};
    // Storage-class specifiers
    case tok::kw_static:          // struct foo {...} static    x;
    case tok::kw_extern:          // struct foo {...} extern    x;
    case tok::kw_typedef:         // struct foo {...} typedef   x;
    case tok::kw_register:        // struct foo {...} register  x;
    case tok::kw_auto:            // struct foo {...} auto      x;
    case tok::kw_mutable:         // struct foo {...} mutable   x;
    case tok::kw_constexpr:       // struct foo {...} constexpr x;
      // As shown above, type qualifiers and storage class specifiers absolutely
      // can occur after class specifiers according to the grammar.  However,
      // almost no one actually writes code like this.  If we see one of these,
      // it is much more likely that someone missed a semi colon and the
      // type/storage class specifier we're seeing is part of the *next*
      // intended declaration, as in:
      //
      //   struct foo { ... }
      //   typedef int X;
      //
      // We'd really like to emit a missing semicolon error instead of emitting
      // an error on the 'int' saying that you can't have two type specifiers in
      // the same declaration of X.  Because of this, we look ahead past this
      // token to see if it's a type specifier.  If so, we know the code is
      // otherwise invalid, so we can produce the expected semi error.
      if (!isKnownToBeTypeSpecifier(NextToken()))
        ExpectedSemi = false;
      break;

    case tok::r_brace:  // struct bar { struct foo {...} }
      // Missing ';' at end of struct is accepted as an extension in C mode.
      if (!getLangOpts().CPlusPlus)
        ExpectedSemi = false;
      break;
    }

    // C++ [temp]p3 In a template-declaration which defines a class, no
    // declarator is permitted.
    if (TemplateInfo.Kind)
      ExpectedSemi = true;

    if (ExpectedSemi) {
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after_tagdecl,
                       TagType == DeclSpec::TST_class ? "class"
                       : TagType == DeclSpec::TST_struct? "struct" : "union");
      // Push this token back into the preprocessor and change our current token
      // to ';' so that the rest of the code recovers as though there were an
      // ';' after the definition.
      PP.EnterToken(Tok);
      Tok.setKind(tok::semi);
    }
  }
}

/// ParseBaseClause - Parse the base-clause of a C++ class [C++ class.derived].
///
///       base-clause : [C++ class.derived]
///         ':' base-specifier-list
///       base-specifier-list:
///         base-specifier '...'[opt]
///         base-specifier-list ',' base-specifier '...'[opt]
void Parser::ParseBaseClause(Decl *ClassDecl) {
  assert(Tok.is(tok::colon) && "Not a base clause");
  ConsumeToken();

  // Build up an array of parsed base specifiers.
  SmallVector<CXXBaseSpecifier *, 8> BaseInfo;

  while (true) {
    // Parse a base-specifier.
    BaseResult Result = ParseBaseSpecifier(ClassDecl);
    if (Result.isInvalid()) {
      // Skip the rest of this base specifier, up until the comma or
      // opening brace.
      SkipUntil(tok::comma, tok::l_brace, true, true);
    } else {
      // Add this to our array of base specifiers.
      BaseInfo.push_back(Result.get());
    }

    // If the next token is a comma, consume it and keep reading
    // base-specifiers.
    if (Tok.isNot(tok::comma)) break;

    // Consume the comma.
    ConsumeToken();
  }

  // Attach the base specifiers
  Actions.ActOnBaseSpecifiers(ClassDecl, BaseInfo.data(), BaseInfo.size());
}

/// ParseBaseSpecifier - Parse a C++ base-specifier. A base-specifier is
/// one entry in the base class list of a class specifier, for example:
///    class foo : public bar, virtual private baz {
/// 'public bar' and 'virtual private baz' are each base-specifiers.
///
///       base-specifier: [C++ class.derived]
///         ::[opt] nested-name-specifier[opt] class-name
///         'virtual' access-specifier[opt] ::[opt] nested-name-specifier[opt]
///                        base-type-specifier
///         access-specifier 'virtual'[opt] ::[opt] nested-name-specifier[opt]
///                        base-type-specifier
Parser::BaseResult Parser::ParseBaseSpecifier(Decl *ClassDecl) {
  bool IsVirtual = false;
  SourceLocation StartLoc = Tok.getLocation();

  // Parse the 'virtual' keyword.
  if (Tok.is(tok::kw_virtual))  {
    ConsumeToken();
    IsVirtual = true;
  }

  // Parse an (optional) access specifier.
  AccessSpecifier Access = getAccessSpecifierIfPresent();
  if (Access != AS_none)
    ConsumeToken();

  // Parse the 'virtual' keyword (again!), in case it came after the
  // access specifier.
  if (Tok.is(tok::kw_virtual))  {
    SourceLocation VirtualLoc = ConsumeToken();
    if (IsVirtual) {
      // Complain about duplicate 'virtual'
      Diag(VirtualLoc, diag::err_dup_virtual)
        << FixItHint::CreateRemoval(VirtualLoc);
    }

    IsVirtual = true;
  }

  // Parse the class-name.
  SourceLocation EndLocation;
  SourceLocation BaseLoc;
  TypeResult BaseType = ParseBaseTypeSpecifier(BaseLoc, EndLocation);
  if (BaseType.isInvalid())
    return true;

  // Parse the optional ellipsis (for a pack expansion). The ellipsis is 
  // actually part of the base-specifier-list grammar productions, but we
  // parse it here for convenience.
  SourceLocation EllipsisLoc;
  if (Tok.is(tok::ellipsis))
    EllipsisLoc = ConsumeToken();
  
  // Find the complete source range for the base-specifier.
  SourceRange Range(StartLoc, EndLocation);

  // Notify semantic analysis that we have parsed a complete
  // base-specifier.
  return Actions.ActOnBaseSpecifier(ClassDecl, Range, IsVirtual, Access,
                                    BaseType.get(), BaseLoc, EllipsisLoc);
}

/// getAccessSpecifierIfPresent - Determine whether the next token is
/// a C++ access-specifier.
///
///       access-specifier: [C++ class.derived]
///         'private'
///         'protected'
///         'public'
AccessSpecifier Parser::getAccessSpecifierIfPresent() const {
  switch (Tok.getKind()) {
  default: return AS_none;
  case tok::kw_private: return AS_private;
  case tok::kw_protected: return AS_protected;
  case tok::kw_public: return AS_public;
  }
}

/// \brief If the given declarator has any parts for which parsing has to be
/// delayed, e.g., default arguments, create a late-parsed method declaration
/// record to handle the parsing at the end of the class definition.
void Parser::HandleMemberFunctionDeclDelays(Declarator& DeclaratorInfo,
                                            Decl *ThisDecl) {
  // We just declared a member function. If this member function
  // has any default arguments, we'll need to parse them later.
  LateParsedMethodDeclaration *LateMethod = 0;
  DeclaratorChunk::FunctionTypeInfo &FTI
    = DeclaratorInfo.getFunctionTypeInfo();

  for (unsigned ParamIdx = 0; ParamIdx < FTI.NumArgs; ++ParamIdx) {
    if (LateMethod || FTI.ArgInfo[ParamIdx].DefaultArgTokens) {
      if (!LateMethod) {
        // Push this method onto the stack of late-parsed method
        // declarations.
        LateMethod = new LateParsedMethodDeclaration(this, ThisDecl);
        getCurrentClass().LateParsedDeclarations.push_back(LateMethod);
        LateMethod->TemplateScope = getCurScope()->isTemplateParamScope();

        // Add all of the parameters prior to this one (they don't
        // have default arguments).
        LateMethod->DefaultArgs.reserve(FTI.NumArgs);
        for (unsigned I = 0; I < ParamIdx; ++I)
          LateMethod->DefaultArgs.push_back(
                             LateParsedDefaultArgument(FTI.ArgInfo[I].Param));
      }

      // Add this parameter to the list of parameters (it may or may
      // not have a default argument).
      LateMethod->DefaultArgs.push_back(
        LateParsedDefaultArgument(FTI.ArgInfo[ParamIdx].Param,
                                  FTI.ArgInfo[ParamIdx].DefaultArgTokens));
    }
  }
}

/// isCXX0XVirtSpecifier - Determine whether the given token is a C++0x
/// virt-specifier.
///
///       virt-specifier:
///         override
///         final
VirtSpecifiers::Specifier Parser::isCXX0XVirtSpecifier(const Token &Tok) const {
  if (!getLangOpts().CPlusPlus)
    return VirtSpecifiers::VS_None;

  if (Tok.is(tok::identifier)) {
    IdentifierInfo *II = Tok.getIdentifierInfo();

    // Initialize the contextual keywords.
    if (!Ident_final) {
      Ident_final = &PP.getIdentifierTable().get("final");
      Ident_override = &PP.getIdentifierTable().get("override");
    }

    if (II == Ident_override)
      return VirtSpecifiers::VS_Override;

    if (II == Ident_final)
      return VirtSpecifiers::VS_Final;
  }

  return VirtSpecifiers::VS_None;
}

/// ParseOptionalCXX0XVirtSpecifierSeq - Parse a virt-specifier-seq.
///
///       virt-specifier-seq:
///         virt-specifier
///         virt-specifier-seq virt-specifier
void Parser::ParseOptionalCXX0XVirtSpecifierSeq(VirtSpecifiers &VS) {
  while (true) {
    VirtSpecifiers::Specifier Specifier = isCXX0XVirtSpecifier();
    if (Specifier == VirtSpecifiers::VS_None)
      return;

    // C++ [class.mem]p8:
    //   A virt-specifier-seq shall contain at most one of each virt-specifier.
    const char *PrevSpec = 0;
    if (VS.SetSpecifier(Specifier, Tok.getLocation(), PrevSpec))
      Diag(Tok.getLocation(), diag::err_duplicate_virt_specifier)
        << PrevSpec
        << FixItHint::CreateRemoval(Tok.getLocation());

    Diag(Tok.getLocation(), getLangOpts().CPlusPlus0x ?
         diag::warn_cxx98_compat_override_control_keyword :
         diag::ext_override_control_keyword)
      << VirtSpecifiers::getSpecifierName(Specifier);
    ConsumeToken();
  }
}

/// isCXX0XFinalKeyword - Determine whether the next token is a C++0x
/// contextual 'final' keyword.
bool Parser::isCXX0XFinalKeyword() const {
  if (!getLangOpts().CPlusPlus)
    return false;

  if (!Tok.is(tok::identifier))
    return false;

  // Initialize the contextual keywords.
  if (!Ident_final) {
    Ident_final = &PP.getIdentifierTable().get("final");
    Ident_override = &PP.getIdentifierTable().get("override");
  }
  
  return Tok.getIdentifierInfo() == Ident_final;
}

/// ParseCXXClassMemberDeclaration - Parse a C++ class member declaration.
///
///       member-declaration:
///         decl-specifier-seq[opt] member-declarator-list[opt] ';'
///         function-definition ';'[opt]
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
/// 
///       pure-specifier:
///         '= 0'
///
///       constant-initializer:
///         '=' constant-expression
///
void Parser::ParseCXXClassMemberDeclaration(AccessSpecifier AS,
                                            AttributeList *AccessAttrs,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       ParsingDeclRAIIObject *TemplateDiags) {
  if (Tok.is(tok::at)) {
    if (getLangOpts().ObjC1 && NextToken().isObjCAtKeyword(tok::objc_defs))
      Diag(Tok, diag::err_at_defs_cxx);
    else
      Diag(Tok, diag::err_at_in_class);
    
    ConsumeToken();
    SkipUntil(tok::r_brace);
    return;
  }
  
  // Access declarations.
  if (!TemplateInfo.Kind &&
      (Tok.is(tok::identifier) || Tok.is(tok::coloncolon)) &&
      !TryAnnotateCXXScopeToken() &&
      Tok.is(tok::annot_cxxscope)) {
    bool isAccessDecl = false;
    if (NextToken().is(tok::identifier))
      isAccessDecl = GetLookAheadToken(2).is(tok::semi);
    else
      isAccessDecl = NextToken().is(tok::kw_operator);

    if (isAccessDecl) {
      // Collect the scope specifier token we annotated earlier.
      CXXScopeSpec SS;
      ParseOptionalCXXScopeSpecifier(SS, ParsedType(), 
                                     /*EnteringContext=*/false);

      // Try to parse an unqualified-id.
      SourceLocation TemplateKWLoc;
      UnqualifiedId Name;
      if (ParseUnqualifiedId(SS, false, true, true, ParsedType(),
                             TemplateKWLoc, Name)) {
        SkipUntil(tok::semi);
        return;
      }

      // TODO: recover from mistakenly-qualified operator declarations.
      if (ExpectAndConsume(tok::semi,
                           diag::err_expected_semi_after,
                           "access declaration",
                           tok::semi))
        return;

      Actions.ActOnUsingDeclaration(getCurScope(), AS,
                                    false, SourceLocation(),
                                    SS, Name,
                                    /* AttrList */ 0,
                                    /* IsTypeName */ false,
                                    SourceLocation());
      return;
    }
  }

  // static_assert-declaration
  if (Tok.is(tok::kw_static_assert) || Tok.is(tok::kw__Static_assert)) {
    // FIXME: Check for templates
    SourceLocation DeclEnd;
    ParseStaticAssertDeclaration(DeclEnd);
    return;
  }

  if (Tok.is(tok::kw_template)) {
    assert(!TemplateInfo.TemplateParams &&
           "Nested template improperly parsed?");
    SourceLocation DeclEnd;
    ParseDeclarationStartingWithTemplate(Declarator::MemberContext, DeclEnd,
                                         AS, AccessAttrs);
    return;
  }

  // Handle:  member-declaration ::= '__extension__' member-declaration
  if (Tok.is(tok::kw___extension__)) {
    // __extension__ silences extension warnings in the subexpression.
    ExtensionRAIIObject O(Diags);  // Use RAII to do this.
    ConsumeToken();
    return ParseCXXClassMemberDeclaration(AS, AccessAttrs,
                                          TemplateInfo, TemplateDiags);
  }

  // Don't parse FOO:BAR as if it were a typo for FOO::BAR, in this context it
  // is a bitfield.
  ColonProtectionRAIIObject X(*this);

  ParsedAttributesWithRange attrs(AttrFactory);
  // Optional C++0x attribute-specifier
  MaybeParseCXX0XAttributes(attrs);
  MaybeParseMicrosoftAttributes(attrs);

  if (Tok.is(tok::kw_using)) {
    ProhibitAttributes(attrs);

    // Eat 'using'.
    SourceLocation UsingLoc = ConsumeToken();

    if (Tok.is(tok::kw_namespace)) {
      Diag(UsingLoc, diag::err_using_namespace_in_class);
      SkipUntil(tok::semi, true, true);
    } else {
      SourceLocation DeclEnd;
      // Otherwise, it must be a using-declaration or an alias-declaration.
      ParseUsingDeclaration(Declarator::MemberContext, TemplateInfo,
                            UsingLoc, DeclEnd, AS);
    }
    return;
  }

  // Hold late-parsed attributes so we can attach a Decl to them later.
  LateParsedAttrList CommonLateParsedAttrs;

  // decl-specifier-seq:
  // Parse the common declaration-specifiers piece.
  ParsingDeclSpec DS(*this, TemplateDiags);
  DS.takeAttributesFrom(attrs);
  ParseDeclarationSpecifiers(DS, TemplateInfo, AS, DSC_class,
                             &CommonLateParsedAttrs);

  MultiTemplateParamsArg TemplateParams(Actions,
      TemplateInfo.TemplateParams? TemplateInfo.TemplateParams->data() : 0,
      TemplateInfo.TemplateParams? TemplateInfo.TemplateParams->size() : 0);

  if (Tok.is(tok::semi)) {
    ConsumeToken();
    Decl *TheDecl =
      Actions.ParsedFreeStandingDeclSpec(getCurScope(), AS, DS, TemplateParams);
    DS.complete(TheDecl);
    return;
  }

  ParsingDeclarator DeclaratorInfo(*this, DS, Declarator::MemberContext);
  VirtSpecifiers VS;

  // Hold late-parsed attributes so we can attach a Decl to them later.
  LateParsedAttrList LateParsedAttrs;

  SourceLocation EqualLoc;
  bool HasInitializer = false;
  ExprResult Init;
  if (Tok.isNot(tok::colon)) {
    // Don't parse FOO:BAR as if it were a typo for FOO::BAR.
    ColonProtectionRAIIObject X(*this);

    // Parse the first declarator.
    ParseDeclarator(DeclaratorInfo);
    // Error parsing the declarator?
    if (!DeclaratorInfo.hasName()) {
      // If so, skip until the semi-colon or a }.
      SkipUntil(tok::r_brace, true, true);
      if (Tok.is(tok::semi))
        ConsumeToken();
      return;
    }

    ParseOptionalCXX0XVirtSpecifierSeq(VS);

    // If attributes exist after the declarator, but before an '{', parse them.
    MaybeParseGNUAttributes(DeclaratorInfo, &LateParsedAttrs);

    // MSVC permits pure specifier on inline functions declared at class scope.
    // Hence check for =0 before checking for function definition.
    if (getLangOpts().MicrosoftExt && Tok.is(tok::equal) &&
        DeclaratorInfo.isFunctionDeclarator() && 
        NextToken().is(tok::numeric_constant)) {
      EqualLoc = ConsumeToken();
      Init = ParseInitializer();
      if (Init.isInvalid())
        SkipUntil(tok::comma, true, true);
      else
        HasInitializer = true;
    }

    FunctionDefinitionKind DefinitionKind = FDK_Declaration;
    // function-definition:
    //
    // In C++11, a non-function declarator followed by an open brace is a
    // braced-init-list for an in-class member initialization, not an
    // erroneous function definition.
    if (Tok.is(tok::l_brace) && !getLangOpts().CPlusPlus0x) {
      DefinitionKind = FDK_Definition;
    } else if (DeclaratorInfo.isFunctionDeclarator()) {
      if (Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try)) {
        DefinitionKind = FDK_Definition;
      } else if (Tok.is(tok::equal)) {
        const Token &KW = NextToken();
        if (KW.is(tok::kw_default))
          DefinitionKind = FDK_Defaulted;
        else if (KW.is(tok::kw_delete))
          DefinitionKind = FDK_Deleted;
      }
    }

    if (DefinitionKind) {
      if (!DeclaratorInfo.isFunctionDeclarator()) {
        Diag(DeclaratorInfo.getIdentifierLoc(), diag::err_func_def_no_params);
        ConsumeBrace();
        SkipUntil(tok::r_brace, /*StopAtSemi*/false);
        
        // Consume the optional ';'
        if (Tok.is(tok::semi))
          ConsumeToken();
        return;
      }

      if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef) {
        Diag(DeclaratorInfo.getIdentifierLoc(),
             diag::err_function_declared_typedef);
        // This recovery skips the entire function body. It would be nice
        // to simply call ParseCXXInlineMethodDef() below, however Sema
        // assumes the declarator represents a function, not a typedef.
        ConsumeBrace();
        SkipUntil(tok::r_brace, /*StopAtSemi*/false);

        // Consume the optional ';'
        if (Tok.is(tok::semi))
          ConsumeToken();
        return;
      }

      Decl *FunDecl =
        ParseCXXInlineMethodDef(AS, AccessAttrs, DeclaratorInfo, TemplateInfo,
                                VS, DefinitionKind, Init);

      for (unsigned i = 0, ni = CommonLateParsedAttrs.size(); i < ni; ++i) {
        CommonLateParsedAttrs[i]->addDecl(FunDecl);
      }
      for (unsigned i = 0, ni = LateParsedAttrs.size(); i < ni; ++i) {
        LateParsedAttrs[i]->addDecl(FunDecl);
      }
      LateParsedAttrs.clear();

      // Consume the ';' - it's optional unless we have a delete or default
      if (Tok.is(tok::semi)) {
        ConsumeToken();
      }

      return;
    }
  }

  // member-declarator-list:
  //   member-declarator
  //   member-declarator-list ',' member-declarator

  SmallVector<Decl *, 8> DeclsInGroup;
  ExprResult BitfieldSize;
  bool ExpectSemi = true;

  while (1) {
    // member-declarator:
    //   declarator pure-specifier[opt]
    //   declarator brace-or-equal-initializer[opt]
    //   identifier[opt] ':' constant-expression
    if (Tok.is(tok::colon)) {
      ConsumeToken();
      BitfieldSize = ParseConstantExpression();
      if (BitfieldSize.isInvalid())
        SkipUntil(tok::comma, true, true);
    }

    // If a simple-asm-expr is present, parse it.
    if (Tok.is(tok::kw_asm)) {
      SourceLocation Loc;
      ExprResult AsmLabel(ParseSimpleAsm(&Loc));
      if (AsmLabel.isInvalid())
        SkipUntil(tok::comma, true, true);
 
      DeclaratorInfo.setAsmLabel(AsmLabel.release());
      DeclaratorInfo.SetRangeEnd(Loc);
    }

    // If attributes exist after the declarator, parse them.
    MaybeParseGNUAttributes(DeclaratorInfo, &LateParsedAttrs);

    // FIXME: When g++ adds support for this, we'll need to check whether it
    // goes before or after the GNU attributes and __asm__.
    ParseOptionalCXX0XVirtSpecifierSeq(VS);

    bool HasDeferredInitializer = false;
    if ((Tok.is(tok::equal) || Tok.is(tok::l_brace)) && !HasInitializer) {
      if (BitfieldSize.get()) {
        Diag(Tok, diag::err_bitfield_member_init);
        SkipUntil(tok::comma, true, true);
      } else {
        HasInitializer = true;
        HasDeferredInitializer = !DeclaratorInfo.isDeclarationOfFunction() &&
          DeclaratorInfo.getDeclSpec().getStorageClassSpec()
            != DeclSpec::SCS_static &&
          DeclaratorInfo.getDeclSpec().getStorageClassSpec()
            != DeclSpec::SCS_typedef;
      }
    }

    // NOTE: If Sema is the Action module and declarator is an instance field,
    // this call will *not* return the created decl; It will return null.
    // See Sema::ActOnCXXMemberDeclarator for details.

    Decl *ThisDecl = 0;
    if (DS.isFriendSpecified()) {
      // TODO: handle initializers, bitfields, 'delete'
      ThisDecl = Actions.ActOnFriendFunctionDecl(getCurScope(), DeclaratorInfo,
                                                 move(TemplateParams));
    } else {
      ThisDecl = Actions.ActOnCXXMemberDeclarator(getCurScope(), AS,
                                                  DeclaratorInfo,
                                                  move(TemplateParams),
                                                  BitfieldSize.release(),
                                                  VS, HasDeferredInitializer);
      if (AccessAttrs)
        Actions.ProcessDeclAttributeList(getCurScope(), ThisDecl, AccessAttrs,
                                         false, true);
    }
    
    // Set the Decl for any late parsed attributes
    for (unsigned i = 0, ni = CommonLateParsedAttrs.size(); i < ni; ++i) {
      CommonLateParsedAttrs[i]->addDecl(ThisDecl);
    }
    for (unsigned i = 0, ni = LateParsedAttrs.size(); i < ni; ++i) {
      LateParsedAttrs[i]->addDecl(ThisDecl);
    }
    LateParsedAttrs.clear();

    // Handle the initializer.
    if (HasDeferredInitializer) {
      // The initializer was deferred; parse it and cache the tokens.
      Diag(Tok, getLangOpts().CPlusPlus0x ?
           diag::warn_cxx98_compat_nonstatic_member_init :
           diag::ext_nonstatic_member_init);

      if (DeclaratorInfo.isArrayOfUnknownBound()) {
        // C++0x [dcl.array]p3: An array bound may also be omitted when the
        // declarator is followed by an initializer. 
        //
        // A brace-or-equal-initializer for a member-declarator is not an
        // initializer in the grammar, so this is ill-formed.
        Diag(Tok, diag::err_incomplete_array_member_init);
        SkipUntil(tok::comma, true, true);
        if (ThisDecl)
          // Avoid later warnings about a class member of incomplete type.
          ThisDecl->setInvalidDecl();
      } else
        ParseCXXNonStaticMemberInitializer(ThisDecl);
    } else if (HasInitializer) {
      // Normal initializer.
      if (!Init.isUsable())
        Init = ParseCXXMemberInitializer(ThisDecl,
                 DeclaratorInfo.isDeclarationOfFunction(), EqualLoc);
      
      if (Init.isInvalid())
        SkipUntil(tok::comma, true, true);
      else if (ThisDecl)
        Actions.AddInitializerToDecl(ThisDecl, Init.get(), EqualLoc.isInvalid(),
                                   DS.getTypeSpecType() == DeclSpec::TST_auto);      
    } else if (ThisDecl && DS.getStorageClassSpec() == DeclSpec::SCS_static) {
      // No initializer.
      Actions.ActOnUninitializedDecl(ThisDecl, 
                                   DS.getTypeSpecType() == DeclSpec::TST_auto);
    }
    
    if (ThisDecl) {
      Actions.FinalizeDeclaration(ThisDecl);
      DeclsInGroup.push_back(ThisDecl);
    }
    
    if (ThisDecl && DeclaratorInfo.isFunctionDeclarator() &&
        DeclaratorInfo.getDeclSpec().getStorageClassSpec()
          != DeclSpec::SCS_typedef) {
      HandleMemberFunctionDeclDelays(DeclaratorInfo, ThisDecl);
    }

    DeclaratorInfo.complete(ThisDecl);

    // If we don't have a comma, it is either the end of the list (a ';')
    // or an error, bail out.
    if (Tok.isNot(tok::comma))
      break;

    // Consume the comma.
    SourceLocation CommaLoc = ConsumeToken();

    if (Tok.isAtStartOfLine() &&
        !MightBeDeclarator(Declarator::MemberContext)) {
      // This comma was followed by a line-break and something which can't be
      // the start of a declarator. The comma was probably a typo for a
      // semicolon.
      Diag(CommaLoc, diag::err_expected_semi_declaration)
        << FixItHint::CreateReplacement(CommaLoc, ";");
      ExpectSemi = false;
      break;
    }

    // Parse the next declarator.
    DeclaratorInfo.clear();
    VS.clear();
    BitfieldSize = true;
    Init = true;
    HasInitializer = false;
    DeclaratorInfo.setCommaLoc(CommaLoc);

    // Attributes are only allowed on the second declarator.
    MaybeParseGNUAttributes(DeclaratorInfo);

    if (Tok.isNot(tok::colon))
      ParseDeclarator(DeclaratorInfo);
  }

  if (ExpectSemi &&
      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list)) {
    // Skip to end of block or statement.
    SkipUntil(tok::r_brace, true, true);
    // If we stopped at a ';', eat it.
    if (Tok.is(tok::semi)) ConsumeToken();
    return;
  }

  Actions.FinalizeDeclaratorGroup(getCurScope(), DS, DeclsInGroup.data(),
                                  DeclsInGroup.size());
}

/// ParseCXXMemberInitializer - Parse the brace-or-equal-initializer or
/// pure-specifier. Also detect and reject any attempted defaulted/deleted
/// function definition. The location of the '=', if any, will be placed in
/// EqualLoc.
///
///   pure-specifier:
///     '= 0'
///
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
///
/// Prior to C++0x, the assignment-expression in an initializer-clause must
/// be a constant-expression.
ExprResult Parser::ParseCXXMemberInitializer(Decl *D, bool IsFunction,
                                             SourceLocation &EqualLoc) {
  assert((Tok.is(tok::equal) || Tok.is(tok::l_brace))
         && "Data member initializer not starting with '=' or '{'");

  EnterExpressionEvaluationContext Context(Actions, 
                                           Sema::PotentiallyEvaluated,
                                           D);
  if (Tok.is(tok::equal)) {
    EqualLoc = ConsumeToken();
    if (Tok.is(tok::kw_delete)) {
      // In principle, an initializer of '= delete p;' is legal, but it will
      // never type-check. It's better to diagnose it as an ill-formed expression
      // than as an ill-formed deleted non-function member.
      // An initializer of '= delete p, foo' will never be parsed, because
      // a top-level comma always ends the initializer expression.
      const Token &Next = NextToken();
      if (IsFunction || Next.is(tok::semi) || Next.is(tok::comma) ||
           Next.is(tok::eof)) {
        if (IsFunction)
          Diag(ConsumeToken(), diag::err_default_delete_in_multiple_declaration)
            << 1 /* delete */;
        else
          Diag(ConsumeToken(), diag::err_deleted_non_function);
        return ExprResult();
      }
    } else if (Tok.is(tok::kw_default)) {
      if (IsFunction)
        Diag(Tok, diag::err_default_delete_in_multiple_declaration)
          << 0 /* default */;
      else
        Diag(ConsumeToken(), diag::err_default_special_members);
      return ExprResult();
    }

  }
  return ParseInitializer();
}

/// ParseCXXMemberSpecification - Parse the class definition.
///
///       member-specification:
///         member-declaration member-specification[opt]
///         access-specifier ':' member-specification[opt]
///
void Parser::ParseCXXMemberSpecification(SourceLocation RecordLoc,
                                         unsigned TagType, Decl *TagDecl) {
  assert((TagType == DeclSpec::TST_struct ||
         TagType == DeclSpec::TST_union  ||
         TagType == DeclSpec::TST_class) && "Invalid TagType!");

  PrettyDeclStackTraceEntry CrashInfo(Actions, TagDecl, RecordLoc,
                                      "parsing struct/union/class body");

  // Determine whether this is a non-nested class. Note that local
  // classes are *not* considered to be nested classes.
  bool NonNestedClass = true;
  if (!ClassStack.empty()) {
    for (const Scope *S = getCurScope(); S; S = S->getParent()) {
      if (S->isClassScope()) {
        // We're inside a class scope, so this is a nested class.
        NonNestedClass = false;
        break;
      }

      if ((S->getFlags() & Scope::FnScope)) {
        // If we're in a function or function template declared in the
        // body of a class, then this is a local class rather than a
        // nested class.
        const Scope *Parent = S->getParent();
        if (Parent->isTemplateParamScope())
          Parent = Parent->getParent();
        if (Parent->isClassScope())
          break;
      }
    }
  }

  // Enter a scope for the class.
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope);

  // Note that we are parsing a new (potentially-nested) class definition.
  ParsingClassDefinition ParsingDef(*this, TagDecl, NonNestedClass);

  if (TagDecl)
    Actions.ActOnTagStartDefinition(getCurScope(), TagDecl);

  SourceLocation FinalLoc;

  // Parse the optional 'final' keyword.
  if (getLangOpts().CPlusPlus && Tok.is(tok::identifier)) {
    assert(isCXX0XFinalKeyword() && "not a class definition");
    FinalLoc = ConsumeToken();

    Diag(FinalLoc, getLangOpts().CPlusPlus0x ?
         diag::warn_cxx98_compat_override_control_keyword :
         diag::ext_override_control_keyword) << "final";
  }

  if (Tok.is(tok::colon)) {
    ParseBaseClause(TagDecl);

    if (!Tok.is(tok::l_brace)) {
      Diag(Tok, diag::err_expected_lbrace_after_base_specifiers);

      if (TagDecl)
        Actions.ActOnTagDefinitionError(getCurScope(), TagDecl);
      return;
    }
  }

  assert(Tok.is(tok::l_brace));
  BalancedDelimiterTracker T(*this, tok::l_brace);
  T.consumeOpen();

  if (TagDecl)
    Actions.ActOnStartCXXMemberDeclarations(getCurScope(), TagDecl, FinalLoc,
                                            T.getOpenLocation());

  // C++ 11p3: Members of a class defined with the keyword class are private
  // by default. Members of a class defined with the keywords struct or union
  // are public by default.
  AccessSpecifier CurAS;
  if (TagType == DeclSpec::TST_class)
    CurAS = AS_private;
  else
    CurAS = AS_public;
  ParsedAttributes AccessAttrs(AttrFactory);

  if (TagDecl) {
    // While we still have something to read, read the member-declarations.
    while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
      // Each iteration of this loop reads one member-declaration.

      if (getLangOpts().MicrosoftExt && (Tok.is(tok::kw___if_exists) ||
          Tok.is(tok::kw___if_not_exists))) {
        ParseMicrosoftIfExistsClassDeclaration((DeclSpec::TST)TagType, CurAS);
        continue;
      }

      // Check for extraneous top-level semicolon.
      if (Tok.is(tok::semi)) {
        Diag(Tok, diag::ext_extra_struct_semi)
          << DeclSpec::getSpecifierName((DeclSpec::TST)TagType)
          << FixItHint::CreateRemoval(Tok.getLocation());
        ConsumeToken();
        continue;
      }

      if (Tok.is(tok::annot_pragma_vis)) {
        HandlePragmaVisibility();
        continue;
      }

      if (Tok.is(tok::annot_pragma_pack)) {
        HandlePragmaPack();
        continue;
      }

      AccessSpecifier AS = getAccessSpecifierIfPresent();
      if (AS != AS_none) {
        // Current token is a C++ access specifier.
        CurAS = AS;
        SourceLocation ASLoc = Tok.getLocation();
        unsigned TokLength = Tok.getLength();
        ConsumeToken();
        AccessAttrs.clear();
        MaybeParseGNUAttributes(AccessAttrs);

        SourceLocation EndLoc;
        if (Tok.is(tok::colon)) {
          EndLoc = Tok.getLocation();
          ConsumeToken();
        } else if (Tok.is(tok::semi)) {
          EndLoc = Tok.getLocation();
          ConsumeToken();
          Diag(EndLoc, diag::err_expected_colon) 
            << FixItHint::CreateReplacement(EndLoc, ":");
        } else {
          EndLoc = ASLoc.getLocWithOffset(TokLength);
          Diag(EndLoc, diag::err_expected_colon) 
            << FixItHint::CreateInsertion(EndLoc, ":");
        }

        if (Actions.ActOnAccessSpecifier(AS, ASLoc, EndLoc,
                                         AccessAttrs.getList())) {
          // found another attribute than only annotations
          AccessAttrs.clear();
        }

        continue;
      }

      // FIXME: Make sure we don't have a template here.

      // Parse all the comma separated declarators.
      ParseCXXClassMemberDeclaration(CurAS, AccessAttrs.getList());
    }

    T.consumeClose();
  } else {
    SkipUntil(tok::r_brace, false, false);
  }

  // If attributes exist after class contents, parse them.
  ParsedAttributes attrs(AttrFactory);
  MaybeParseGNUAttributes(attrs);

  if (TagDecl)
    Actions.ActOnFinishCXXMemberSpecification(getCurScope(), RecordLoc, TagDecl,
                                              T.getOpenLocation(), 
                                              T.getCloseLocation(),
                                              attrs.getList());

  // C++11 [class.mem]p2:
  //   Within the class member-specification, the class is regarded as complete
  //   within function bodies, default arguments, and
  //   brace-or-equal-initializers for non-static data members (including such
  //   things in nested classes).
  if (TagDecl && NonNestedClass) {
    // We are not inside a nested class. This class and its nested classes
    // are complete and we can parse the delayed portions of method
    // declarations and the lexed inline method definitions, along with any
    // delayed attributes.
    SourceLocation SavedPrevTokLocation = PrevTokLocation;
    ParseLexedAttributes(getCurrentClass());
    ParseLexedMethodDeclarations(getCurrentClass());

    // We've finished with all pending member declarations.
    Actions.ActOnFinishCXXMemberDecls();

    ParseLexedMemberInitializers(getCurrentClass());
    ParseLexedMethodDefs(getCurrentClass());
    PrevTokLocation = SavedPrevTokLocation;
  }

  if (TagDecl)
    Actions.ActOnTagFinishDefinition(getCurScope(), TagDecl, 
                                     T.getCloseLocation());

  // Leave the class scope.
  ParsingDef.Pop();
  ClassScope.Exit();
}

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
/// [C++]  ctor-initializer:
///          ':' mem-initializer-list
///
/// [C++]  mem-initializer-list:
///          mem-initializer ...[opt]
///          mem-initializer ...[opt] , mem-initializer-list
void Parser::ParseConstructorInitializer(Decl *ConstructorDecl) {
  assert(Tok.is(tok::colon) && "Constructor initializer always starts with ':'");

  // Poison the SEH identifiers so they are flagged as illegal in constructor initializers
  PoisonSEHIdentifiersRAIIObject PoisonSEHIdentifiers(*this, true);
  SourceLocation ColonLoc = ConsumeToken();

  SmallVector<CXXCtorInitializer*, 4> MemInitializers;
  bool AnyErrors = false;

  do {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteConstructorInitializer(ConstructorDecl, 
                                                 MemInitializers.data(), 
                                                 MemInitializers.size());
      return cutOffParsing();
    } else {
      MemInitResult MemInit = ParseMemInitializer(ConstructorDecl);
      if (!MemInit.isInvalid())
        MemInitializers.push_back(MemInit.get());
      else
        AnyErrors = true;
    }
    
    if (Tok.is(tok::comma))
      ConsumeToken();
    else if (Tok.is(tok::l_brace))
      break;
    // If the next token looks like a base or member initializer, assume that
    // we're just missing a comma.
    else if (Tok.is(tok::identifier) || Tok.is(tok::coloncolon)) {
      SourceLocation Loc = PP.getLocForEndOfToken(PrevTokLocation);
      Diag(Loc, diag::err_ctor_init_missing_comma)
        << FixItHint::CreateInsertion(Loc, ", ");
    } else {
      // Skip over garbage, until we get to '{'.  Don't eat the '{'.
      Diag(Tok.getLocation(), diag::err_expected_lbrace_or_comma);
      SkipUntil(tok::l_brace, true, true);
      break;
    }
  } while (true);

  Actions.ActOnMemInitializers(ConstructorDecl, ColonLoc,
                               MemInitializers.data(), MemInitializers.size(),
                               AnyErrors);
}

/// ParseMemInitializer - Parse a C++ member initializer, which is
/// part of a constructor initializer that explicitly initializes one
/// member or base class (C++ [class.base.init]). See
/// ParseConstructorInitializer for an example.
///
/// [C++] mem-initializer:
///         mem-initializer-id '(' expression-list[opt] ')'
/// [C++0x] mem-initializer-id braced-init-list
///
/// [C++] mem-initializer-id:
///         '::'[opt] nested-name-specifier[opt] class-name
///         identifier
Parser::MemInitResult Parser::ParseMemInitializer(Decl *ConstructorDecl) {
  // parse '::'[opt] nested-name-specifier[opt]
  CXXScopeSpec SS;
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), /*EnteringContext=*/false);
  ParsedType TemplateTypeTy;
  if (Tok.is(tok::annot_template_id)) {
    TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);
    if (TemplateId->Kind == TNK_Type_template ||
        TemplateId->Kind == TNK_Dependent_template_name) {
      AnnotateTemplateIdTokenAsType();
      assert(Tok.is(tok::annot_typename) && "template-id -> type failed");
      TemplateTypeTy = getTypeAnnotation(Tok);
    }
  }
  // Uses of decltype will already have been converted to annot_decltype by
  // ParseOptionalCXXScopeSpecifier at this point.
  if (!TemplateTypeTy && Tok.isNot(tok::identifier)
      && Tok.isNot(tok::annot_decltype)) {
    Diag(Tok, diag::err_expected_member_or_base_name);
    return true;
  }

  IdentifierInfo *II = 0;
  DeclSpec DS(AttrFactory);
  SourceLocation IdLoc = Tok.getLocation();
  if (Tok.is(tok::annot_decltype)) {
    // Get the decltype expression, if there is one.
    ParseDecltypeSpecifier(DS);
  } else {
    if (Tok.is(tok::identifier))
      // Get the identifier. This may be a member name or a class name,
      // but we'll let the semantic analysis determine which it is.
      II = Tok.getIdentifierInfo();
    ConsumeToken();
  }


  // Parse the '('.
  if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)) {
    Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);

    ExprResult InitList = ParseBraceInitializer();
    if (InitList.isInvalid())
      return true;

    SourceLocation EllipsisLoc;
    if (Tok.is(tok::ellipsis))
      EllipsisLoc = ConsumeToken();

    return Actions.ActOnMemInitializer(ConstructorDecl, getCurScope(), SS, II,
                                       TemplateTypeTy, DS, IdLoc, 
                                       InitList.take(), EllipsisLoc);
  } else if(Tok.is(tok::l_paren)) {
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();

    // Parse the optional expression-list.
    ExprVector ArgExprs(Actions);
    CommaLocsTy CommaLocs;
    if (Tok.isNot(tok::r_paren) && ParseExpressionList(ArgExprs, CommaLocs)) {
      SkipUntil(tok::r_paren);
      return true;
    }

    T.consumeClose();

    SourceLocation EllipsisLoc;
    if (Tok.is(tok::ellipsis))
      EllipsisLoc = ConsumeToken();

    return Actions.ActOnMemInitializer(ConstructorDecl, getCurScope(), SS, II,
                                       TemplateTypeTy, DS, IdLoc,
                                       T.getOpenLocation(), ArgExprs.take(),
                                       ArgExprs.size(), T.getCloseLocation(),
                                       EllipsisLoc);
  }

  Diag(Tok, getLangOpts().CPlusPlus0x ? diag::err_expected_lparen_or_lbrace
                                  : diag::err_expected_lparen);
  return true;
}

/// \brief Parse a C++ exception-specification if present (C++0x [except.spec]).
///
///       exception-specification:
///         dynamic-exception-specification
///         noexcept-specification
///
///       noexcept-specification:
///         'noexcept'
///         'noexcept' '(' constant-expression ')'
ExceptionSpecificationType
Parser::tryParseExceptionSpecification(
                    SourceRange &SpecificationRange,
                    SmallVectorImpl<ParsedType> &DynamicExceptions,
                    SmallVectorImpl<SourceRange> &DynamicExceptionRanges,
                    ExprResult &NoexceptExpr) {
  ExceptionSpecificationType Result = EST_None;

  // See if there's a dynamic specification.
  if (Tok.is(tok::kw_throw)) {
    Result = ParseDynamicExceptionSpecification(SpecificationRange,
                                                DynamicExceptions,
                                                DynamicExceptionRanges);
    assert(DynamicExceptions.size() == DynamicExceptionRanges.size() &&
           "Produced different number of exception types and ranges.");
  }

  // If there's no noexcept specification, we're done.
  if (Tok.isNot(tok::kw_noexcept))
    return Result;

  Diag(Tok, diag::warn_cxx98_compat_noexcept_decl);

  // If we already had a dynamic specification, parse the noexcept for,
  // recovery, but emit a diagnostic and don't store the results.
  SourceRange NoexceptRange;
  ExceptionSpecificationType NoexceptType = EST_None;

  SourceLocation KeywordLoc = ConsumeToken();
  if (Tok.is(tok::l_paren)) {
    // There is an argument.
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();
    NoexceptType = EST_ComputedNoexcept;
    NoexceptExpr = ParseConstantExpression();
    // The argument must be contextually convertible to bool. We use
    // ActOnBooleanCondition for this purpose.
    if (!NoexceptExpr.isInvalid())
      NoexceptExpr = Actions.ActOnBooleanCondition(getCurScope(), KeywordLoc,
                                                   NoexceptExpr.get());
    T.consumeClose();
    NoexceptRange = SourceRange(KeywordLoc, T.getCloseLocation());
  } else {
    // There is no argument.
    NoexceptType = EST_BasicNoexcept;
    NoexceptRange = SourceRange(KeywordLoc, KeywordLoc);
  }

  if (Result == EST_None) {
    SpecificationRange = NoexceptRange;
    Result = NoexceptType;

    // If there's a dynamic specification after a noexcept specification,
    // parse that and ignore the results.
    if (Tok.is(tok::kw_throw)) {
      Diag(Tok.getLocation(), diag::err_dynamic_and_noexcept_specification);
      ParseDynamicExceptionSpecification(NoexceptRange, DynamicExceptions,
                                         DynamicExceptionRanges);
    }
  } else {
    Diag(Tok.getLocation(), diag::err_dynamic_and_noexcept_specification);
  }

  return Result;
}

/// ParseDynamicExceptionSpecification - Parse a C++
/// dynamic-exception-specification (C++ [except.spec]).
///
///       dynamic-exception-specification:
///         'throw' '(' type-id-list [opt] ')'
/// [MS]    'throw' '(' '...' ')'
///
///       type-id-list:
///         type-id ... [opt]
///         type-id-list ',' type-id ... [opt]
///
ExceptionSpecificationType Parser::ParseDynamicExceptionSpecification(
                                  SourceRange &SpecificationRange,
                                  SmallVectorImpl<ParsedType> &Exceptions,
                                  SmallVectorImpl<SourceRange> &Ranges) {
  assert(Tok.is(tok::kw_throw) && "expected throw");

  SpecificationRange.setBegin(ConsumeToken());
  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.consumeOpen()) {
    Diag(Tok, diag::err_expected_lparen_after) << "throw";
    SpecificationRange.setEnd(SpecificationRange.getBegin());
    return EST_DynamicNone;
  }

  // Parse throw(...), a Microsoft extension that means "this function
  // can throw anything".
  if (Tok.is(tok::ellipsis)) {
    SourceLocation EllipsisLoc = ConsumeToken();
    if (!getLangOpts().MicrosoftExt)
      Diag(EllipsisLoc, diag::ext_ellipsis_exception_spec);
    T.consumeClose();
    SpecificationRange.setEnd(T.getCloseLocation());
    return EST_MSAny;
  }

  // Parse the sequence of type-ids.
  SourceRange Range;
  while (Tok.isNot(tok::r_paren)) {
    TypeResult Res(ParseTypeName(&Range));

    if (Tok.is(tok::ellipsis)) {
      // C++0x [temp.variadic]p5:
      //   - In a dynamic-exception-specification (15.4); the pattern is a 
      //     type-id.
      SourceLocation Ellipsis = ConsumeToken();
      Range.setEnd(Ellipsis);
      if (!Res.isInvalid())
        Res = Actions.ActOnPackExpansion(Res.get(), Ellipsis);
    }

    if (!Res.isInvalid()) {
      Exceptions.push_back(Res.get());
      Ranges.push_back(Range);
    }
    
    if (Tok.is(tok::comma))
      ConsumeToken();
    else
      break;
  }

  T.consumeClose();
  SpecificationRange.setEnd(T.getCloseLocation());
  return Exceptions.empty() ? EST_DynamicNone : EST_Dynamic;
}

/// ParseTrailingReturnType - Parse a trailing return type on a new-style
/// function declaration.
TypeResult Parser::ParseTrailingReturnType(SourceRange &Range) {
  assert(Tok.is(tok::arrow) && "expected arrow");

  ConsumeToken();

  return ParseTypeName(&Range, Declarator::TrailingReturnContext);
}

/// \brief We have just started parsing the definition of a new class,
/// so push that class onto our stack of classes that is currently
/// being parsed.
Sema::ParsingClassState
Parser::PushParsingClass(Decl *ClassDecl, bool NonNestedClass) {
  assert((NonNestedClass || !ClassStack.empty()) &&
         "Nested class without outer class");
  ClassStack.push(new ParsingClass(ClassDecl, NonNestedClass));
  return Actions.PushParsingClass();
}

/// \brief Deallocate the given parsed class and all of its nested
/// classes.
void Parser::DeallocateParsedClasses(Parser::ParsingClass *Class) {
  for (unsigned I = 0, N = Class->LateParsedDeclarations.size(); I != N; ++I)
    delete Class->LateParsedDeclarations[I];
  delete Class;
}

/// \brief Pop the top class of the stack of classes that are
/// currently being parsed.
///
/// This routine should be called when we have finished parsing the
/// definition of a class, but have not yet popped the Scope
/// associated with the class's definition.
///
/// \returns true if the class we've popped is a top-level class,
/// false otherwise.
void Parser::PopParsingClass(Sema::ParsingClassState state) {
  assert(!ClassStack.empty() && "Mismatched push/pop for class parsing");

  Actions.PopParsingClass(state);

  ParsingClass *Victim = ClassStack.top();
  ClassStack.pop();
  if (Victim->TopLevelClass) {
    // Deallocate all of the nested classes of this class,
    // recursively: we don't need to keep any of this information.
    DeallocateParsedClasses(Victim);
    return;
  }
  assert(!ClassStack.empty() && "Missing top-level class?");

  if (Victim->LateParsedDeclarations.empty()) {
    // The victim is a nested class, but we will not need to perform
    // any processing after the definition of this class since it has
    // no members whose handling was delayed. Therefore, we can just
    // remove this nested class.
    DeallocateParsedClasses(Victim);
    return;
  }

  // This nested class has some members that will need to be processed
  // after the top-level class is completely defined. Therefore, add
  // it to the list of nested classes within its parent.
  assert(getCurScope()->isClassScope() && "Nested class outside of class scope?");
  ClassStack.top()->LateParsedDeclarations.push_back(new LateParsedClass(this, Victim));
  Victim->TemplateScope = getCurScope()->getParent()->isTemplateParamScope();
}

/// \brief Try to parse an 'identifier' which appears within an attribute-token.
///
/// \return the parsed identifier on success, and 0 if the next token is not an
/// attribute-token.
///
/// C++11 [dcl.attr.grammar]p3:
///   If a keyword or an alternative token that satisfies the syntactic
///   requirements of an identifier is contained in an attribute-token,
///   it is considered an identifier.
IdentifierInfo *Parser::TryParseCXX11AttributeIdentifier(SourceLocation &Loc) {
  switch (Tok.getKind()) {
  default:
    // Identifiers and keywords have identifier info attached.
    if (IdentifierInfo *II = Tok.getIdentifierInfo()) {
      Loc = ConsumeToken();
      return II;
    }
    return 0;

  case tok::ampamp:       // 'and'
  case tok::pipe:         // 'bitor'
  case tok::pipepipe:     // 'or'
  case tok::caret:        // 'xor'
  case tok::tilde:        // 'compl'
  case tok::amp:          // 'bitand'
  case tok::ampequal:     // 'and_eq'
  case tok::pipeequal:    // 'or_eq'
  case tok::caretequal:   // 'xor_eq'
  case tok::exclaim:      // 'not'
  case tok::exclaimequal: // 'not_eq'
    // Alternative tokens do not have identifier info, but their spelling
    // starts with an alphabetical character.
    llvm::SmallString<8> SpellingBuf;
    StringRef Spelling = PP.getSpelling(Tok.getLocation(), SpellingBuf);
    if (std::isalpha(Spelling[0])) {
      Loc = ConsumeToken();
      return &PP.getIdentifierTable().get(Spelling.data());
    }
    return 0;
  }
}

/// ParseCXX11AttributeSpecifier - Parse a C++11 attribute-specifier. Currently
/// only parses standard attributes.
///
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
///
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
void Parser::ParseCXX11AttributeSpecifier(ParsedAttributes &attrs,
                                          SourceLocation *endLoc) {
  if (Tok.is(tok::kw_alignas)) {
    Diag(Tok.getLocation(), diag::warn_cxx98_compat_alignas);
    ParseAlignmentSpecifier(attrs, endLoc);
    return;
  }

  assert(Tok.is(tok::l_square) && NextToken().is(tok::l_square)
      && "Not a C++11 attribute list");

  Diag(Tok.getLocation(), diag::warn_cxx98_compat_attribute);

  ConsumeBracket();
  ConsumeBracket();

  while (Tok.isNot(tok::r_square)) {
    // attribute not present
    if (Tok.is(tok::comma)) {
      ConsumeToken();
      continue;
    }

    SourceLocation ScopeLoc, AttrLoc;
    IdentifierInfo *ScopeName = 0, *AttrName = 0;

    AttrName = TryParseCXX11AttributeIdentifier(AttrLoc);
    if (!AttrName)
      // Break out to the "expected ']'" diagnostic.
      break;

    // scoped attribute
    if (Tok.is(tok::coloncolon)) {
      ConsumeToken();

      ScopeName = AttrName;
      ScopeLoc = AttrLoc;

      AttrName = TryParseCXX11AttributeIdentifier(AttrLoc);
      if (!AttrName) {
        Diag(Tok.getLocation(), diag::err_expected_ident);
        SkipUntil(tok::r_square, tok::comma, true, true);
        continue;
      }
    }

    bool AttrParsed = false;
    // No scoped names are supported; ideally we could put all non-standard
    // attributes into namespaces.
    if (!ScopeName) {
      switch (AttributeList::getKind(AttrName)) {
      // No arguments
      case AttributeList::AT_carries_dependency:
      case AttributeList::AT_noreturn: {
        if (Tok.is(tok::l_paren)) {
          Diag(Tok.getLocation(), diag::err_cxx11_attribute_forbids_arguments)
            << AttrName->getName();
          break;
        }

        attrs.addNew(AttrName, AttrLoc, 0, AttrLoc, 0,
                     SourceLocation(), 0, 0, false, true);
        AttrParsed = true;
        break;
      }

      // Silence warnings
      default: break;
      }
    }

    // Skip the entire parameter clause, if any
    if (!AttrParsed && Tok.is(tok::l_paren)) {
      ConsumeParen();
      // SkipUntil maintains the balancedness of tokens.
      SkipUntil(tok::r_paren, false);
    }

    if (Tok.is(tok::ellipsis)) {
      if (AttrParsed)
        Diag(Tok, diag::err_cxx11_attribute_forbids_ellipsis)
          << AttrName->getName();
      ConsumeToken();
    }
  }

  if (ExpectAndConsume(tok::r_square, diag::err_expected_rsquare))
    SkipUntil(tok::r_square, false);
  if (endLoc)
    *endLoc = Tok.getLocation();
  if (ExpectAndConsume(tok::r_square, diag::err_expected_rsquare))
    SkipUntil(tok::r_square, false);
}

/// ParseCXX11Attributes - Parse a C++0x attribute-specifier-seq.
///
/// attribute-specifier-seq:
///       attribute-specifier-seq[opt] attribute-specifier
void Parser::ParseCXX11Attributes(ParsedAttributesWithRange &attrs,
                                  SourceLocation *endLoc) {
  SourceLocation StartLoc = Tok.getLocation(), Loc;
  if (!endLoc)
    endLoc = &Loc;

  do {
    ParseCXX11AttributeSpecifier(attrs, endLoc);
  } while (isCXX11AttributeSpecifier());

  attrs.Range = SourceRange(StartLoc, *endLoc);
}

/// ParseMicrosoftAttributes - Parse a Microsoft attribute [Attr]
///
/// [MS] ms-attribute:
///             '[' token-seq ']'
///
/// [MS] ms-attribute-seq:
///             ms-attribute[opt]
///             ms-attribute ms-attribute-seq
void Parser::ParseMicrosoftAttributes(ParsedAttributes &attrs,
                                      SourceLocation *endLoc) {
  assert(Tok.is(tok::l_square) && "Not a Microsoft attribute list");

  while (Tok.is(tok::l_square)) {
    // FIXME: If this is actually a C++11 attribute, parse it as one.
    ConsumeBracket();
    SkipUntil(tok::r_square, true, true);
    if (endLoc) *endLoc = Tok.getLocation();
    ExpectAndConsume(tok::r_square, diag::err_expected_rsquare);
  }
}

void Parser::ParseMicrosoftIfExistsClassDeclaration(DeclSpec::TST TagType,
                                                    AccessSpecifier& CurAS) {
  IfExistsCondition Result;
  if (ParseMicrosoftIfExistsCondition(Result))
    return;
  
  BalancedDelimiterTracker Braces(*this, tok::l_brace);
  if (Braces.consumeOpen()) {
    Diag(Tok, diag::err_expected_lbrace);
    return;
  }

  switch (Result.Behavior) {
  case IEB_Parse:
    // Parse the declarations below.
    break;
        
  case IEB_Dependent:
    Diag(Result.KeywordLoc, diag::warn_microsoft_dependent_exists)
      << Result.IsIfExists;
    // Fall through to skip.
      
  case IEB_Skip:
    Braces.skipToEnd();
    return;
  }

  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    // __if_exists, __if_not_exists can nest.
    if ((Tok.is(tok::kw___if_exists) || Tok.is(tok::kw___if_not_exists))) {
      ParseMicrosoftIfExistsClassDeclaration((DeclSpec::TST)TagType, CurAS);
      continue;
    }

    // Check for extraneous top-level semicolon.
    if (Tok.is(tok::semi)) {
      Diag(Tok, diag::ext_extra_struct_semi)
        << DeclSpec::getSpecifierName((DeclSpec::TST)TagType)
        << FixItHint::CreateRemoval(Tok.getLocation());
      ConsumeToken();
      continue;
    }

    AccessSpecifier AS = getAccessSpecifierIfPresent();
    if (AS != AS_none) {
      // Current token is a C++ access specifier.
      CurAS = AS;
      SourceLocation ASLoc = Tok.getLocation();
      ConsumeToken();
      if (Tok.is(tok::colon))
        Actions.ActOnAccessSpecifier(AS, ASLoc, Tok.getLocation());
      else
        Diag(Tok, diag::err_expected_colon);
      ConsumeToken();
      continue;
    }

    // Parse all the comma separated declarators.
    ParseCXXClassMemberDeclaration(CurAS, 0);
  }
  
  Braces.consumeClose();
}
