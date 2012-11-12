//===--- ParseObjC.cpp - Objective C Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Objective-C portions of the Parser interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;


/// ParseObjCAtDirectives - Handle parts of the external-declaration production:
///       external-declaration: [C99 6.9]
/// [OBJC]  objc-class-definition
/// [OBJC]  objc-class-declaration
/// [OBJC]  objc-alias-declaration
/// [OBJC]  objc-protocol-definition
/// [OBJC]  objc-method-definition
/// [OBJC]  '@' 'end'
Parser::DeclGroupPtrTy Parser::ParseObjCAtDirectives() {
  SourceLocation AtLoc = ConsumeToken(); // the "@"

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCAtDirective(getCurScope());
    cutOffParsing();
    return DeclGroupPtrTy();
  }
    
  Decl *SingleDecl = 0;
  switch (Tok.getObjCKeywordID()) {
  case tok::objc_class:
    return ParseObjCAtClassDeclaration(AtLoc);
  case tok::objc_interface: {
    ParsedAttributes attrs(AttrFactory);
    SingleDecl = ParseObjCAtInterfaceDeclaration(AtLoc, attrs);
    break;
  }
  case tok::objc_protocol: {
    ParsedAttributes attrs(AttrFactory);
    return ParseObjCAtProtocolDeclaration(AtLoc, attrs);
  }
  case tok::objc_implementation:
    return ParseObjCAtImplementationDeclaration(AtLoc);
  case tok::objc_end:
    return ParseObjCAtEndDeclaration(AtLoc);
  case tok::objc_compatibility_alias:
    SingleDecl = ParseObjCAtAliasDeclaration(AtLoc);
    break;
  case tok::objc_synthesize:
    SingleDecl = ParseObjCPropertySynthesize(AtLoc);
    break;
  case tok::objc_dynamic:
    SingleDecl = ParseObjCPropertyDynamic(AtLoc);
    break;
  case tok::objc___experimental_modules_import:
    if (getLangOpts().Modules)
      return ParseModuleImport(AtLoc);
      
    // Fall through
      
  default:
    Diag(AtLoc, diag::err_unexpected_at);
    SkipUntil(tok::semi);
    SingleDecl = 0;
    break;
  }
  return Actions.ConvertDeclToDeclGroup(SingleDecl);
}

///
/// objc-class-declaration:
///    '@' 'class' identifier-list ';'
///
Parser::DeclGroupPtrTy
Parser::ParseObjCAtClassDeclaration(SourceLocation atLoc) {
  ConsumeToken(); // the identifier "class"
  SmallVector<IdentifierInfo *, 8> ClassNames;
  SmallVector<SourceLocation, 8> ClassLocs;


  while (1) {
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return Actions.ConvertDeclToDeclGroup(0);
    }
    ClassNames.push_back(Tok.getIdentifierInfo());
    ClassLocs.push_back(Tok.getLocation());
    ConsumeToken();

    if (Tok.isNot(tok::comma))
      break;

    ConsumeToken();
  }

  // Consume the ';'.
  if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@class"))
    return Actions.ConvertDeclToDeclGroup(0);

  return Actions.ActOnForwardClassDeclaration(atLoc, ClassNames.data(),
                                              ClassLocs.data(),
                                              ClassNames.size());
}

void Parser::CheckNestedObjCContexts(SourceLocation AtLoc)
{
  Sema::ObjCContainerKind ock = Actions.getObjCContainerKind();
  if (ock == Sema::OCK_None)
    return;

  Decl *Decl = Actions.getObjCDeclContext();
  if (CurParsedObjCImpl) {
    CurParsedObjCImpl->finish(AtLoc);
  } else {
    Actions.ActOnAtEnd(getCurScope(), AtLoc);
  }
  Diag(AtLoc, diag::err_objc_missing_end)
      << FixItHint::CreateInsertion(AtLoc, "@end\n");
  if (Decl)
    Diag(Decl->getLocStart(), diag::note_objc_container_start)
        << (int) ock;
}

///
///   objc-interface:
///     objc-class-interface-attributes[opt] objc-class-interface
///     objc-category-interface
///
///   objc-class-interface:
///     '@' 'interface' identifier objc-superclass[opt]
///       objc-protocol-refs[opt]
///       objc-class-instance-variables[opt]
///       objc-interface-decl-list
///     @end
///
///   objc-category-interface:
///     '@' 'interface' identifier '(' identifier[opt] ')'
///       objc-protocol-refs[opt]
///       objc-interface-decl-list
///     @end
///
///   objc-superclass:
///     ':' identifier
///
///   objc-class-interface-attributes:
///     __attribute__((visibility("default")))
///     __attribute__((visibility("hidden")))
///     __attribute__((deprecated))
///     __attribute__((unavailable))
///     __attribute__((objc_exception)) - used by NSException on 64-bit
///     __attribute__((objc_root_class))
///
Decl *Parser::ParseObjCAtInterfaceDeclaration(SourceLocation AtLoc,
                                              ParsedAttributes &attrs) {
  assert(Tok.isObjCAtKeyword(tok::objc_interface) &&
         "ParseObjCAtInterfaceDeclaration(): Expected @interface");
  CheckNestedObjCContexts(AtLoc);
  ConsumeToken(); // the "interface" identifier

  // Code completion after '@interface'.
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCInterfaceDecl(getCurScope());
    cutOffParsing();
    return 0;
  }

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing class or category name.
    return 0;
  }

  // We have a class or category name - consume it.
  IdentifierInfo *nameId = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken();
  if (Tok.is(tok::l_paren) && 
      !isKnownToBeTypeSpecifier(GetLookAheadToken(1))) { // we have a category.
    
    BalancedDelimiterTracker T(*this, tok::l_paren);
    T.consumeOpen();

    SourceLocation categoryLoc;
    IdentifierInfo *categoryId = 0;
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCInterfaceCategory(getCurScope(), nameId, nameLoc);
      cutOffParsing();
      return 0;
    }
    
    // For ObjC2, the category name is optional (not an error).
    if (Tok.is(tok::identifier)) {
      categoryId = Tok.getIdentifierInfo();
      categoryLoc = ConsumeToken();
    }
    else if (!getLangOpts().ObjC2) {
      Diag(Tok, diag::err_expected_ident); // missing category name.
      return 0;
    }
   
    T.consumeClose();
    if (T.getCloseLocation().isInvalid())
      return 0;
    
    if (!attrs.empty()) { // categories don't support attributes.
      Diag(nameLoc, diag::err_objc_no_attributes_on_category);
      attrs.clear();
    }
    
    // Next, we need to check for any protocol references.
    SourceLocation LAngleLoc, EndProtoLoc;
    SmallVector<Decl *, 8> ProtocolRefs;
    SmallVector<SourceLocation, 8> ProtocolLocs;
    if (Tok.is(tok::less) &&
        ParseObjCProtocolReferences(ProtocolRefs, ProtocolLocs, true,
                                    LAngleLoc, EndProtoLoc))
      return 0;

    Decl *CategoryType =
    Actions.ActOnStartCategoryInterface(AtLoc,
                                        nameId, nameLoc,
                                        categoryId, categoryLoc,
                                        ProtocolRefs.data(),
                                        ProtocolRefs.size(),
                                        ProtocolLocs.data(),
                                        EndProtoLoc);
    
    if (Tok.is(tok::l_brace))
      ParseObjCClassInstanceVariables(CategoryType, tok::objc_private, AtLoc);
      
    ParseObjCInterfaceDeclList(tok::objc_not_keyword, CategoryType);
    return CategoryType;
  }
  // Parse a class interface.
  IdentifierInfo *superClassId = 0;
  SourceLocation superClassLoc;

  if (Tok.is(tok::colon)) { // a super class is specified.
    ConsumeToken();

    // Code completion of superclass names.
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCSuperclass(getCurScope(), nameId, nameLoc);
      cutOffParsing();
      return 0;
    }

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident); // missing super class name.
      return 0;
    }
    superClassId = Tok.getIdentifierInfo();
    superClassLoc = ConsumeToken();
  }
  // Next, we need to check for any protocol references.
  SmallVector<Decl *, 8> ProtocolRefs;
  SmallVector<SourceLocation, 8> ProtocolLocs;
  SourceLocation LAngleLoc, EndProtoLoc;
  if (Tok.is(tok::less) &&
      ParseObjCProtocolReferences(ProtocolRefs, ProtocolLocs, true,
                                  LAngleLoc, EndProtoLoc))
    return 0;

  Decl *ClsType =
    Actions.ActOnStartClassInterface(AtLoc, nameId, nameLoc,
                                     superClassId, superClassLoc,
                                     ProtocolRefs.data(), ProtocolRefs.size(),
                                     ProtocolLocs.data(),
                                     EndProtoLoc, attrs.getList());

  if (Tok.is(tok::l_brace))
    ParseObjCClassInstanceVariables(ClsType, tok::objc_protected, AtLoc);

  ParseObjCInterfaceDeclList(tok::objc_interface, ClsType);
  return ClsType;
}

/// The Objective-C property callback.  This should be defined where
/// it's used, but instead it's been lifted to here to support VS2005.
struct Parser::ObjCPropertyCallback : FieldCallback {
private:
  virtual void anchor();
public:
  Parser &P;
  SmallVectorImpl<Decl *> &Props;
  ObjCDeclSpec &OCDS;
  SourceLocation AtLoc;
  SourceLocation LParenLoc;
  tok::ObjCKeywordKind MethodImplKind;
        
  ObjCPropertyCallback(Parser &P, 
                       SmallVectorImpl<Decl *> &Props,
                       ObjCDeclSpec &OCDS, SourceLocation AtLoc,
                       SourceLocation LParenLoc,
                       tok::ObjCKeywordKind MethodImplKind) :
    P(P), Props(Props), OCDS(OCDS), AtLoc(AtLoc), LParenLoc(LParenLoc),
    MethodImplKind(MethodImplKind) {
  }

  void invoke(ParsingFieldDeclarator &FD) {
    if (FD.D.getIdentifier() == 0) {
      P.Diag(AtLoc, diag::err_objc_property_requires_field_name)
        << FD.D.getSourceRange();
      return;
    }
    if (FD.BitfieldSize) {
      P.Diag(AtLoc, diag::err_objc_property_bitfield)
        << FD.D.getSourceRange();
      return;
    }

    // Install the property declarator into interfaceDecl.
    IdentifierInfo *SelName =
      OCDS.getGetterName() ? OCDS.getGetterName() : FD.D.getIdentifier();

    Selector GetterSel =
      P.PP.getSelectorTable().getNullarySelector(SelName);
    IdentifierInfo *SetterName = OCDS.getSetterName();
    Selector SetterSel;
    if (SetterName)
      SetterSel = P.PP.getSelectorTable().getSelector(1, &SetterName);
    else
      SetterSel = SelectorTable::constructSetterName(P.PP.getIdentifierTable(),
                                                     P.PP.getSelectorTable(),
                                                     FD.D.getIdentifier());
    bool isOverridingProperty = false;
    Decl *Property =
      P.Actions.ActOnProperty(P.getCurScope(), AtLoc, LParenLoc,
                              FD, OCDS,
                              GetterSel, SetterSel, 
                              &isOverridingProperty,
                              MethodImplKind);
    if (!isOverridingProperty)
      Props.push_back(Property);

    FD.complete(Property);
  }
};

void Parser::ObjCPropertyCallback::anchor() {
}

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
///
void Parser::ParseObjCInterfaceDeclList(tok::ObjCKeywordKind contextKey, 
                                        Decl *CDecl) {
  SmallVector<Decl *, 32> allMethods;
  SmallVector<Decl *, 16> allProperties;
  SmallVector<DeclGroupPtrTy, 8> allTUVariables;
  tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword;

  SourceRange AtEnd;
    
  while (1) {
    // If this is a method prototype, parse it.
    if (Tok.is(tok::minus) || Tok.is(tok::plus)) {
      if (Decl *methodPrototype =
          ParseObjCMethodPrototype(MethodImplKind, false))
        allMethods.push_back(methodPrototype);
      // Consume the ';' here, since ParseObjCMethodPrototype() is re-used for
      // method definitions.
      if (ExpectAndConsumeSemi(diag::err_expected_semi_after_method_proto)) {
        // We didn't find a semi and we error'ed out. Skip until a ';' or '@'.
        SkipUntil(tok::at, /*StopAtSemi=*/true, /*DontConsume=*/true);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
      continue;
    }
    if (Tok.is(tok::l_paren)) {
      Diag(Tok, diag::err_expected_minus_or_plus);
      ParseObjCMethodDecl(Tok.getLocation(), 
                          tok::minus, 
                          MethodImplKind, false);
      continue;
    }
    // Ignore excess semicolons.
    if (Tok.is(tok::semi)) {
      ConsumeToken();
      continue;
    }

    // If we got to the end of the file, exit the loop.
    if (Tok.is(tok::eof))
      break;

    // Code completion within an Objective-C interface.
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteOrdinaryName(getCurScope(), 
                            CurParsedObjCImpl? Sema::PCC_ObjCImplementation
                                             : Sema::PCC_ObjCInterface);
      return cutOffParsing();
    }
    
    // If we don't have an @ directive, parse it as a function definition.
    if (Tok.isNot(tok::at)) {
      // The code below does not consume '}'s because it is afraid of eating the
      // end of a namespace.  Because of the way this code is structured, an
      // erroneous r_brace would cause an infinite loop if not handled here.
      if (Tok.is(tok::r_brace))
        break;
      ParsedAttributesWithRange attrs(AttrFactory);
      allTUVariables.push_back(ParseDeclarationOrFunctionDefinition(attrs));
      continue;
    }

    // Otherwise, we have an @ directive, eat the @.
    SourceLocation AtLoc = ConsumeToken(); // the "@"
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCAtDirective(getCurScope());
      return cutOffParsing();
    }

    tok::ObjCKeywordKind DirectiveKind = Tok.getObjCKeywordID();

    if (DirectiveKind == tok::objc_end) { // @end -> terminate list
      AtEnd.setBegin(AtLoc);
      AtEnd.setEnd(Tok.getLocation());
      break;
    } else if (DirectiveKind == tok::objc_not_keyword) {
      Diag(Tok, diag::err_objc_unknown_at);
      SkipUntil(tok::semi);
      continue;
    }

    // Eat the identifier.
    ConsumeToken();

    switch (DirectiveKind) {
    default:
      // FIXME: If someone forgets an @end on a protocol, this loop will
      // continue to eat up tons of stuff and spew lots of nonsense errors.  It
      // would probably be better to bail out if we saw an @class or @interface
      // or something like that.
      Diag(AtLoc, diag::err_objc_illegal_interface_qual);
      // Skip until we see an '@' or '}' or ';'.
      SkipUntil(tok::r_brace, tok::at);
      break;
        
    case tok::objc_implementation:
    case tok::objc_interface:
      Diag(AtLoc, diag::err_objc_missing_end)
          << FixItHint::CreateInsertion(AtLoc, "@end\n");
      Diag(CDecl->getLocStart(), diag::note_objc_container_start)
          << (int) Actions.getObjCContainerKind();
      ConsumeToken();
      break;
        
    case tok::objc_required:
    case tok::objc_optional:
      // This is only valid on protocols.
      // FIXME: Should this check for ObjC2 being enabled?
      if (contextKey != tok::objc_protocol)
        Diag(AtLoc, diag::err_objc_directive_only_in_protocol);
      else
        MethodImplKind = DirectiveKind;
      break;

    case tok::objc_property:
      if (!getLangOpts().ObjC2)
        Diag(AtLoc, diag::err_objc_properties_require_objc2);

      ObjCDeclSpec OCDS;
      SourceLocation LParenLoc;
      // Parse property attribute list, if any.
      if (Tok.is(tok::l_paren)) {
        LParenLoc = Tok.getLocation();
        ParseObjCPropertyAttribute(OCDS);
      }

      ObjCPropertyCallback Callback(*this, allProperties,
                                    OCDS, AtLoc, LParenLoc, MethodImplKind);

      // Parse all the comma separated declarators.
      ParsingDeclSpec DS(*this);
      ParseStructDeclaration(DS, Callback);

      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list);
      break;
    }
  }

  // We break out of the big loop in two cases: when we see @end or when we see
  // EOF.  In the former case, eat the @end.  In the later case, emit an error.
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCAtDirective(getCurScope());
    return cutOffParsing();
  } else if (Tok.isObjCAtKeyword(tok::objc_end)) {
    ConsumeToken(); // the "end" identifier
  } else {
    Diag(Tok, diag::err_objc_missing_end)
        << FixItHint::CreateInsertion(Tok.getLocation(), "\n@end\n");
    Diag(CDecl->getLocStart(), diag::note_objc_container_start)
        << (int) Actions.getObjCContainerKind();
    AtEnd.setBegin(Tok.getLocation());
    AtEnd.setEnd(Tok.getLocation());
  }

  // Insert collected methods declarations into the @interface object.
  // This passes in an invalid SourceLocation for AtEndLoc when EOF is hit.
  Actions.ActOnAtEnd(getCurScope(), AtEnd,
                     allMethods.data(), allMethods.size(),
                     allProperties.data(), allProperties.size(),
                     allTUVariables.data(), allTUVariables.size());
}

///   Parse property attribute declarations.
///
///   property-attr-decl: '(' property-attrlist ')'
///   property-attrlist:
///     property-attribute
///     property-attrlist ',' property-attribute
///   property-attribute:
///     getter '=' identifier
///     setter '=' identifier ':'
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
///
void Parser::ParseObjCPropertyAttribute(ObjCDeclSpec &DS) {
  assert(Tok.getKind() == tok::l_paren);
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  while (1) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCPropertyFlags(getCurScope(), DS);
      return cutOffParsing();
    }
    const IdentifierInfo *II = Tok.getIdentifierInfo();

    // If this is not an identifier at all, bail out early.
    if (II == 0) {
      T.consumeClose();
      return;
    }

    SourceLocation AttrName = ConsumeToken(); // consume last attribute name

    if (II->isStr("readonly"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_readonly);
    else if (II->isStr("assign"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_assign);
    else if (II->isStr("unsafe_unretained"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_unsafe_unretained);
    else if (II->isStr("readwrite"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_readwrite);
    else if (II->isStr("retain"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_retain);
    else if (II->isStr("strong"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_strong);
    else if (II->isStr("copy"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_copy);
    else if (II->isStr("nonatomic"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_nonatomic);
    else if (II->isStr("atomic"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_atomic);
    else if (II->isStr("weak"))
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_weak);
    else if (II->isStr("getter") || II->isStr("setter")) {
      bool IsSetter = II->getNameStart()[0] == 's';

      // getter/setter require extra treatment.
      unsigned DiagID = IsSetter ? diag::err_objc_expected_equal_for_setter :
        diag::err_objc_expected_equal_for_getter;

      if (ExpectAndConsume(tok::equal, DiagID, "", tok::r_paren))
        return;

      if (Tok.is(tok::code_completion)) {
        if (IsSetter)
          Actions.CodeCompleteObjCPropertySetter(getCurScope());
        else
          Actions.CodeCompleteObjCPropertyGetter(getCurScope());
        return cutOffParsing();
      }

      
      SourceLocation SelLoc;
      IdentifierInfo *SelIdent = ParseObjCSelectorPiece(SelLoc);

      if (!SelIdent) {
        Diag(Tok, diag::err_objc_expected_selector_for_getter_setter)
          << IsSetter;
        SkipUntil(tok::r_paren);
        return;
      }

      if (IsSetter) {
        DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_setter);
        DS.setSetterName(SelIdent);

        if (ExpectAndConsume(tok::colon, 
                             diag::err_expected_colon_after_setter_name, "",
                             tok::r_paren))
          return;
      } else {
        DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_getter);
        DS.setGetterName(SelIdent);
      }
    } else {
      Diag(AttrName, diag::err_objc_expected_property_attr) << II;
      SkipUntil(tok::r_paren);
      return;
    }

    if (Tok.isNot(tok::comma))
      break;

    ConsumeToken();
  }

  T.consumeClose();
}

///   objc-method-proto:
///     objc-instance-method objc-method-decl objc-method-attributes[opt]
///     objc-class-method objc-method-decl objc-method-attributes[opt]
///
///   objc-instance-method: '-'
///   objc-class-method: '+'
///
///   objc-method-attributes:         [OBJC2]
///     __attribute__((deprecated))
///
Decl *Parser::ParseObjCMethodPrototype(tok::ObjCKeywordKind MethodImplKind,
                                       bool MethodDefinition) {
  assert((Tok.is(tok::minus) || Tok.is(tok::plus)) && "expected +/-");

  tok::TokenKind methodType = Tok.getKind();
  SourceLocation mLoc = ConsumeToken();
  Decl *MDecl = ParseObjCMethodDecl(mLoc, methodType, MethodImplKind,
                                    MethodDefinition);
  // Since this rule is used for both method declarations and definitions,
  // the caller is (optionally) responsible for consuming the ';'.
  return MDecl;
}

///   objc-selector:
///     identifier
///     one of
///       enum struct union if else while do for switch case default
///       break continue return goto asm sizeof typeof __alignof
///       unsigned long const short volatile signed restrict _Complex
///       in out inout bycopy byref oneway int char float double void _Bool
///
IdentifierInfo *Parser::ParseObjCSelectorPiece(SourceLocation &SelectorLoc) {

  switch (Tok.getKind()) {
  default:
    return 0;
  case tok::ampamp:
  case tok::ampequal:
  case tok::amp:
  case tok::pipe:
  case tok::tilde:
  case tok::exclaim:
  case tok::exclaimequal:
  case tok::pipepipe:
  case tok::pipeequal:
  case tok::caret:
  case tok::caretequal: {
    std::string ThisTok(PP.getSpelling(Tok));
    if (isalpha(ThisTok[0])) {
      IdentifierInfo *II = &PP.getIdentifierTable().get(ThisTok.data());
      Tok.setKind(tok::identifier);
      SelectorLoc = ConsumeToken();
      return II;
    }
    return 0; 
  }
      
  case tok::identifier:
  case tok::kw_asm:
  case tok::kw_auto:
  case tok::kw_bool:
  case tok::kw_break:
  case tok::kw_case:
  case tok::kw_catch:
  case tok::kw_char:
  case tok::kw_class:
  case tok::kw_const:
  case tok::kw_const_cast:
  case tok::kw_continue:
  case tok::kw_default:
  case tok::kw_delete:
  case tok::kw_do:
  case tok::kw_double:
  case tok::kw_dynamic_cast:
  case tok::kw_else:
  case tok::kw_enum:
  case tok::kw_explicit:
  case tok::kw_export:
  case tok::kw_extern:
  case tok::kw_false:
  case tok::kw_float:
  case tok::kw_for:
  case tok::kw_friend:
  case tok::kw_goto:
  case tok::kw_if:
  case tok::kw_inline:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw_mutable:
  case tok::kw_namespace:
  case tok::kw_new:
  case tok::kw_operator:
  case tok::kw_private:
  case tok::kw_protected:
  case tok::kw_public:
  case tok::kw_register:
  case tok::kw_reinterpret_cast:
  case tok::kw_restrict:
  case tok::kw_return:
  case tok::kw_short:
  case tok::kw_signed:
  case tok::kw_sizeof:
  case tok::kw_static:
  case tok::kw_static_cast:
  case tok::kw_struct:
  case tok::kw_switch:
  case tok::kw_template:
  case tok::kw_this:
  case tok::kw_throw:
  case tok::kw_true:
  case tok::kw_try:
  case tok::kw_typedef:
  case tok::kw_typeid:
  case tok::kw_typename:
  case tok::kw_typeof:
  case tok::kw_union:
  case tok::kw_unsigned:
  case tok::kw_using:
  case tok::kw_virtual:
  case tok::kw_void:
  case tok::kw_volatile:
  case tok::kw_wchar_t:
  case tok::kw_while:
  case tok::kw__Bool:
  case tok::kw__Complex:
  case tok::kw___alignof:
    IdentifierInfo *II = Tok.getIdentifierInfo();
    SelectorLoc = ConsumeToken();
    return II;
  }
}

///  objc-for-collection-in: 'in'
///
bool Parser::isTokIdentifier_in() const {
  // FIXME: May have to do additional look-ahead to only allow for
  // valid tokens following an 'in'; such as an identifier, unary operators,
  // '[' etc.
  return (getLangOpts().ObjC2 && Tok.is(tok::identifier) &&
          Tok.getIdentifierInfo() == ObjCTypeQuals[objc_in]);
}

/// ParseObjCTypeQualifierList - This routine parses the objective-c's type
/// qualifier list and builds their bitmask representation in the input
/// argument.
///
///   objc-type-qualifiers:
///     objc-type-qualifier
///     objc-type-qualifiers objc-type-qualifier
///
void Parser::ParseObjCTypeQualifierList(ObjCDeclSpec &DS,
                                        Declarator::TheContext Context) {
  assert(Context == Declarator::ObjCParameterContext ||
         Context == Declarator::ObjCResultContext);

  while (1) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCPassingType(getCurScope(), DS, 
                          Context == Declarator::ObjCParameterContext);
      return cutOffParsing();
    }
    
    if (Tok.isNot(tok::identifier))
      return;

    const IdentifierInfo *II = Tok.getIdentifierInfo();
    for (unsigned i = 0; i != objc_NumQuals; ++i) {
      if (II != ObjCTypeQuals[i])
        continue;

      ObjCDeclSpec::ObjCDeclQualifier Qual;
      switch (i) {
      default: llvm_unreachable("Unknown decl qualifier");
      case objc_in:     Qual = ObjCDeclSpec::DQ_In; break;
      case objc_out:    Qual = ObjCDeclSpec::DQ_Out; break;
      case objc_inout:  Qual = ObjCDeclSpec::DQ_Inout; break;
      case objc_oneway: Qual = ObjCDeclSpec::DQ_Oneway; break;
      case objc_bycopy: Qual = ObjCDeclSpec::DQ_Bycopy; break;
      case objc_byref:  Qual = ObjCDeclSpec::DQ_Byref; break;
      }
      DS.setObjCDeclQualifier(Qual);
      ConsumeToken();
      II = 0;
      break;
    }

    // If this wasn't a recognized qualifier, bail out.
    if (II) return;
  }
}

/// Take all the decl attributes out of the given list and add
/// them to the given attribute set.
static void takeDeclAttributes(ParsedAttributes &attrs,
                               AttributeList *list) {
  while (list) {
    AttributeList *cur = list;
    list = cur->getNext();

    if (!cur->isUsedAsTypeAttr()) {
      // Clear out the next pointer.  We're really completely
      // destroying the internal invariants of the declarator here,
      // but it doesn't matter because we're done with it.
      cur->setNext(0);
      attrs.add(cur);
    }
  }
}

/// takeDeclAttributes - Take all the decl attributes from the given
/// declarator and add them to the given list.
static void takeDeclAttributes(ParsedAttributes &attrs,
                               Declarator &D) {
  // First, take ownership of all attributes.
  attrs.getPool().takeAllFrom(D.getAttributePool());
  attrs.getPool().takeAllFrom(D.getDeclSpec().getAttributePool());

  // Now actually move the attributes over.
  takeDeclAttributes(attrs, D.getDeclSpec().getAttributes().getList());
  takeDeclAttributes(attrs, D.getAttributes());
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i)
    takeDeclAttributes(attrs,
                  const_cast<AttributeList*>(D.getTypeObject(i).getAttrs()));
}

///   objc-type-name:
///     '(' objc-type-qualifiers[opt] type-name ')'
///     '(' objc-type-qualifiers[opt] ')'
///
ParsedType Parser::ParseObjCTypeName(ObjCDeclSpec &DS, 
                                     Declarator::TheContext context,
                                     ParsedAttributes *paramAttrs) {
  assert(context == Declarator::ObjCParameterContext ||
         context == Declarator::ObjCResultContext);
  assert((paramAttrs != 0) == (context == Declarator::ObjCParameterContext));

  assert(Tok.is(tok::l_paren) && "expected (");

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  SourceLocation TypeStartLoc = Tok.getLocation();
  ObjCDeclContextSwitch ObjCDC(*this);

  // Parse type qualifiers, in, inout, etc.
  ParseObjCTypeQualifierList(DS, context);

  ParsedType Ty;
  if (isTypeSpecifierQualifier()) {
    // Parse an abstract declarator.
    DeclSpec declSpec(AttrFactory);
    declSpec.setObjCQualifiers(&DS);
    ParseSpecifierQualifierList(declSpec);
    declSpec.SetRangeEnd(Tok.getLocation());
    Declarator declarator(declSpec, context);
    ParseDeclarator(declarator);

    // If that's not invalid, extract a type.
    if (!declarator.isInvalidType()) {
      TypeResult type = Actions.ActOnTypeName(getCurScope(), declarator);
      if (!type.isInvalid())
        Ty = type.get();

      // If we're parsing a parameter, steal all the decl attributes
      // and add them to the decl spec.
      if (context == Declarator::ObjCParameterContext)
        takeDeclAttributes(*paramAttrs, declarator);
    }
  } else if (context == Declarator::ObjCResultContext &&
             Tok.is(tok::identifier)) {
    if (!Ident_instancetype)
      Ident_instancetype = PP.getIdentifierInfo("instancetype");
    
    if (Tok.getIdentifierInfo() == Ident_instancetype) {
      Ty = Actions.ActOnObjCInstanceType(Tok.getLocation());
      ConsumeToken();
    }
  }

  if (Tok.is(tok::r_paren))
    T.consumeClose();
  else if (Tok.getLocation() == TypeStartLoc) {
    // If we didn't eat any tokens, then this isn't a type.
    Diag(Tok, diag::err_expected_type);
    SkipUntil(tok::r_paren);
  } else {
    // Otherwise, we found *something*, but didn't get a ')' in the right
    // place.  Emit an error then return what we have as the type.
    T.consumeClose();
  }
  return Ty;
}

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
///
Decl *Parser::ParseObjCMethodDecl(SourceLocation mLoc,
                                  tok::TokenKind mType,
                                  tok::ObjCKeywordKind MethodImplKind,
                                  bool MethodDefinition) {
  ParsingDeclRAIIObject PD(*this, ParsingDeclRAIIObject::NoParent);

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCMethodDecl(getCurScope(), mType == tok::minus, 
                                       /*ReturnType=*/ ParsedType());
    cutOffParsing();
    return 0;
  }

  // Parse the return type if present.
  ParsedType ReturnType;
  ObjCDeclSpec DSRet;
  if (Tok.is(tok::l_paren))
    ReturnType = ParseObjCTypeName(DSRet, Declarator::ObjCResultContext, 0);

  // If attributes exist before the method, parse them.
  ParsedAttributes methodAttrs(AttrFactory);
  if (getLangOpts().ObjC2)
    MaybeParseGNUAttributes(methodAttrs);

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCMethodDecl(getCurScope(), mType == tok::minus, 
                                       ReturnType);
    cutOffParsing();
    return 0;
  }

  // Now parse the selector.
  SourceLocation selLoc;
  IdentifierInfo *SelIdent = ParseObjCSelectorPiece(selLoc);

  // An unnamed colon is valid.
  if (!SelIdent && Tok.isNot(tok::colon)) { // missing selector name.
    Diag(Tok, diag::err_expected_selector_for_method)
      << SourceRange(mLoc, Tok.getLocation());
    // Skip until we get a ; or @.
    SkipUntil(tok::at, true /*StopAtSemi*/, true /*don't consume*/);
    return 0;
  }

  SmallVector<DeclaratorChunk::ParamInfo, 8> CParamInfo;
  if (Tok.isNot(tok::colon)) {
    // If attributes exist after the method, parse them.
    if (getLangOpts().ObjC2)
      MaybeParseGNUAttributes(methodAttrs);

    Selector Sel = PP.getSelectorTable().getNullarySelector(SelIdent);
    Decl *Result
         = Actions.ActOnMethodDeclaration(getCurScope(), mLoc, Tok.getLocation(),
                                          mType, DSRet, ReturnType, 
                                          selLoc, Sel, 0, 
                                          CParamInfo.data(), CParamInfo.size(),
                                          methodAttrs.getList(), MethodImplKind,
                                          false, MethodDefinition);
    PD.complete(Result);
    return Result;
  }

  SmallVector<IdentifierInfo *, 12> KeyIdents;
  SmallVector<SourceLocation, 12> KeyLocs;
  SmallVector<Sema::ObjCArgInfo, 12> ArgInfos;
  ParseScope PrototypeScope(this,
                            Scope::FunctionPrototypeScope|Scope::DeclScope);

  AttributePool allParamAttrs(AttrFactory);
  while (1) {
    ParsedAttributes paramAttrs(AttrFactory);
    Sema::ObjCArgInfo ArgInfo;

    // Each iteration parses a single keyword argument.
    if (Tok.isNot(tok::colon)) {
      Diag(Tok, diag::err_expected_colon);
      break;
    }
    ConsumeToken(); // Eat the ':'.

    ArgInfo.Type = ParsedType();
    if (Tok.is(tok::l_paren)) // Parse the argument type if present.
      ArgInfo.Type = ParseObjCTypeName(ArgInfo.DeclSpec,
                                       Declarator::ObjCParameterContext,
                                       &paramAttrs);

    // If attributes exist before the argument name, parse them.
    // Regardless, collect all the attributes we've parsed so far.
    ArgInfo.ArgAttrs = 0;
    if (getLangOpts().ObjC2) {
      MaybeParseGNUAttributes(paramAttrs);
      ArgInfo.ArgAttrs = paramAttrs.getList();
    }

    // Code completion for the next piece of the selector.
    if (Tok.is(tok::code_completion)) {
      KeyIdents.push_back(SelIdent);
      Actions.CodeCompleteObjCMethodDeclSelector(getCurScope(), 
                                                 mType == tok::minus,
                                                 /*AtParameterName=*/true,
                                                 ReturnType,
                                                 KeyIdents.data(), 
                                                 KeyIdents.size());
      cutOffParsing();
      return 0;
    }
    
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident); // missing argument name.
      break;
    }

    ArgInfo.Name = Tok.getIdentifierInfo();
    ArgInfo.NameLoc = Tok.getLocation();
    ConsumeToken(); // Eat the identifier.

    ArgInfos.push_back(ArgInfo);
    KeyIdents.push_back(SelIdent);
    KeyLocs.push_back(selLoc);

    // Make sure the attributes persist.
    allParamAttrs.takeAllFrom(paramAttrs.getPool());

    // Code completion for the next piece of the selector.
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCMethodDeclSelector(getCurScope(), 
                                                 mType == tok::minus,
                                                 /*AtParameterName=*/false,
                                                 ReturnType,
                                                 KeyIdents.data(), 
                                                 KeyIdents.size());
      cutOffParsing();
      return 0;
    }
    
    // Check for another keyword selector.
    SelIdent = ParseObjCSelectorPiece(selLoc);
    if (!SelIdent && Tok.isNot(tok::colon))
      break;
    if (!SelIdent) {
      SourceLocation ColonLoc = Tok.getLocation();
      if (PP.getLocForEndOfToken(ArgInfo.NameLoc) == ColonLoc) {
        Diag(ArgInfo.NameLoc, diag::warn_missing_selector_name) << ArgInfo.Name;
        Diag(ArgInfo.NameLoc, diag::note_missing_selector_name) << ArgInfo.Name;
        Diag(ColonLoc, diag::note_force_empty_selector_name) << ArgInfo.Name;
      }
    }
    // We have a selector or a colon, continue parsing.
  }

  bool isVariadic = false;
  bool cStyleParamWarned = false;
  // Parse the (optional) parameter list.
  while (Tok.is(tok::comma)) {
    ConsumeToken();
    if (Tok.is(tok::ellipsis)) {
      isVariadic = true;
      ConsumeToken();
      break;
    }
    if (!cStyleParamWarned) {
      Diag(Tok, diag::warn_cstyle_param);
      cStyleParamWarned = true;
    }
    DeclSpec DS(AttrFactory);
    ParseDeclarationSpecifiers(DS);
    // Parse the declarator.
    Declarator ParmDecl(DS, Declarator::PrototypeContext);
    ParseDeclarator(ParmDecl);
    IdentifierInfo *ParmII = ParmDecl.getIdentifier();
    Decl *Param = Actions.ActOnParamDeclarator(getCurScope(), ParmDecl);
    CParamInfo.push_back(DeclaratorChunk::ParamInfo(ParmII,
                                                    ParmDecl.getIdentifierLoc(), 
                                                    Param,
                                                   0));
  }

  // FIXME: Add support for optional parameter list...
  // If attributes exist after the method, parse them.
  if (getLangOpts().ObjC2)
    MaybeParseGNUAttributes(methodAttrs);
  
  if (KeyIdents.size() == 0)
    return 0;
  
  Selector Sel = PP.getSelectorTable().getSelector(KeyIdents.size(),
                                                   &KeyIdents[0]);
  Decl *Result
       = Actions.ActOnMethodDeclaration(getCurScope(), mLoc, Tok.getLocation(),
                                        mType, DSRet, ReturnType, 
                                        KeyLocs, Sel, &ArgInfos[0], 
                                        CParamInfo.data(), CParamInfo.size(),
                                        methodAttrs.getList(),
                                        MethodImplKind, isVariadic, MethodDefinition);
  
  PD.complete(Result);
  return Result;
}

///   objc-protocol-refs:
///     '<' identifier-list '>'
///
bool Parser::
ParseObjCProtocolReferences(SmallVectorImpl<Decl *> &Protocols,
                            SmallVectorImpl<SourceLocation> &ProtocolLocs,
                            bool WarnOnDeclarations,
                            SourceLocation &LAngleLoc, SourceLocation &EndLoc) {
  assert(Tok.is(tok::less) && "expected <");

  LAngleLoc = ConsumeToken(); // the "<"

  SmallVector<IdentifierLocPair, 8> ProtocolIdents;

  while (1) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCProtocolReferences(ProtocolIdents.data(), 
                                                 ProtocolIdents.size());
      cutOffParsing();
      return true;
    }

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::greater);
      return true;
    }
    ProtocolIdents.push_back(std::make_pair(Tok.getIdentifierInfo(),
                                       Tok.getLocation()));
    ProtocolLocs.push_back(Tok.getLocation());
    ConsumeToken();

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken();
  }

  // Consume the '>'.
  if (Tok.isNot(tok::greater)) {
    Diag(Tok, diag::err_expected_greater);
    return true;
  }

  EndLoc = ConsumeToken();

  // Convert the list of protocols identifiers into a list of protocol decls.
  Actions.FindProtocolDeclaration(WarnOnDeclarations,
                                  &ProtocolIdents[0], ProtocolIdents.size(),
                                  Protocols);
  return false;
}

/// \brief Parse the Objective-C protocol qualifiers that follow a typename
/// in a decl-specifier-seq, starting at the '<'.
bool Parser::ParseObjCProtocolQualifiers(DeclSpec &DS) {
  assert(Tok.is(tok::less) && "Protocol qualifiers start with '<'");
  assert(getLangOpts().ObjC1 && "Protocol qualifiers only exist in Objective-C");
  SourceLocation LAngleLoc, EndProtoLoc;
  SmallVector<Decl *, 8> ProtocolDecl;
  SmallVector<SourceLocation, 8> ProtocolLocs;
  bool Result = ParseObjCProtocolReferences(ProtocolDecl, ProtocolLocs, false,
                                            LAngleLoc, EndProtoLoc);
  DS.setProtocolQualifiers(ProtocolDecl.data(), ProtocolDecl.size(),
                           ProtocolLocs.data(), LAngleLoc);
  if (EndProtoLoc.isValid())
    DS.SetRangeEnd(EndProtoLoc);
  return Result;
}


///   objc-class-instance-variables:
///     '{' objc-instance-variable-decl-list[opt] '}'
///
///   objc-instance-variable-decl-list:
///     objc-visibility-spec
///     objc-instance-variable-decl ';'
///     ';'
///     objc-instance-variable-decl-list objc-visibility-spec
///     objc-instance-variable-decl-list objc-instance-variable-decl ';'
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
///
void Parser::ParseObjCClassInstanceVariables(Decl *interfaceDecl,
                                             tok::ObjCKeywordKind visibility,
                                             SourceLocation atLoc) {
  assert(Tok.is(tok::l_brace) && "expected {");
  SmallVector<Decl *, 32> AllIvarDecls;
    
  ParseScope ClassScope(this, Scope::DeclScope|Scope::ClassScope);
  ObjCDeclContextSwitch ObjCDC(*this);

  BalancedDelimiterTracker T(*this, tok::l_brace);
  T.consumeOpen();

  // While we still have something to read, read the instance variables.
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    // Each iteration of this loop reads one objc-instance-variable-decl.

    // Check for extraneous top-level semicolon.
    if (Tok.is(tok::semi)) {
      ConsumeExtraSemi(InstanceVariableList);
      continue;
    }

    // Set the default visibility to private.
    if (Tok.is(tok::at)) { // parse objc-visibility-spec
      ConsumeToken(); // eat the @ sign
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCAtVisibility(getCurScope());
        return cutOffParsing();
      }
      
      switch (Tok.getObjCKeywordID()) {
      case tok::objc_private:
      case tok::objc_public:
      case tok::objc_protected:
      case tok::objc_package:
        visibility = Tok.getObjCKeywordID();
        ConsumeToken();
        continue;
      default:
        Diag(Tok, diag::err_objc_illegal_visibility_spec);
        continue;
      }
    }

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteOrdinaryName(getCurScope(), 
                                       Sema::PCC_ObjCInstanceVariableList);
      return cutOffParsing();
    }
    
    struct ObjCIvarCallback : FieldCallback {
      Parser &P;
      Decl *IDecl;
      tok::ObjCKeywordKind visibility;
      SmallVectorImpl<Decl *> &AllIvarDecls;

      ObjCIvarCallback(Parser &P, Decl *IDecl, tok::ObjCKeywordKind V,
                       SmallVectorImpl<Decl *> &AllIvarDecls) :
        P(P), IDecl(IDecl), visibility(V), AllIvarDecls(AllIvarDecls) {
      }

      void invoke(ParsingFieldDeclarator &FD) {
        P.Actions.ActOnObjCContainerStartDefinition(IDecl);
        // Install the declarator into the interface decl.
        Decl *Field
          = P.Actions.ActOnIvar(P.getCurScope(),
                                FD.D.getDeclSpec().getSourceRange().getBegin(),
                                FD.D, FD.BitfieldSize, visibility);
        P.Actions.ActOnObjCContainerFinishDefinition();
        if (Field)
          AllIvarDecls.push_back(Field);
        FD.complete(Field);
      }
    } Callback(*this, interfaceDecl, visibility, AllIvarDecls);
    
    // Parse all the comma separated declarators.
    ParsingDeclSpec DS(*this);
    ParseStructDeclaration(DS, Callback);

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_semi_decl_list);
      // Skip to end of block or statement
      SkipUntil(tok::r_brace, true, true);
    }
  }
  T.consumeClose();

  Actions.ActOnObjCContainerStartDefinition(interfaceDecl);
  Actions.ActOnLastBitfield(T.getCloseLocation(), AllIvarDecls);
  Actions.ActOnObjCContainerFinishDefinition();
  // Call ActOnFields() even if we don't have any decls. This is useful
  // for code rewriting tools that need to be aware of the empty list.
  Actions.ActOnFields(getCurScope(), atLoc, interfaceDecl,
                      AllIvarDecls,
                      T.getOpenLocation(), T.getCloseLocation(), 0);
  return;
}

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
///
///   "\@protocol identifier ;" should be resolved as "\@protocol
///   identifier-list ;": objc-interface-decl-list may not start with a
///   semicolon in the first alternative if objc-protocol-refs are omitted.
Parser::DeclGroupPtrTy 
Parser::ParseObjCAtProtocolDeclaration(SourceLocation AtLoc,
                                       ParsedAttributes &attrs) {
  assert(Tok.isObjCAtKeyword(tok::objc_protocol) &&
         "ParseObjCAtProtocolDeclaration(): Expected @protocol");
  ConsumeToken(); // the "protocol" identifier

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCProtocolDecl(getCurScope());
    cutOffParsing();
    return DeclGroupPtrTy();
  }

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing protocol name.
    return DeclGroupPtrTy();
  }
  // Save the protocol name, then consume it.
  IdentifierInfo *protocolName = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken();

  if (Tok.is(tok::semi)) { // forward declaration of one protocol.
    IdentifierLocPair ProtoInfo(protocolName, nameLoc);
    ConsumeToken();
    return Actions.ActOnForwardProtocolDeclaration(AtLoc, &ProtoInfo, 1,
                                                   attrs.getList());
  }

  CheckNestedObjCContexts(AtLoc);

  if (Tok.is(tok::comma)) { // list of forward declarations.
    SmallVector<IdentifierLocPair, 8> ProtocolRefs;
    ProtocolRefs.push_back(std::make_pair(protocolName, nameLoc));

    // Parse the list of forward declarations.
    while (1) {
      ConsumeToken(); // the ','
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::semi);
        return DeclGroupPtrTy();
      }
      ProtocolRefs.push_back(IdentifierLocPair(Tok.getIdentifierInfo(),
                                               Tok.getLocation()));
      ConsumeToken(); // the identifier

      if (Tok.isNot(tok::comma))
        break;
    }
    // Consume the ';'.
    if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@protocol"))
      return DeclGroupPtrTy();

    return Actions.ActOnForwardProtocolDeclaration(AtLoc,
                                                   &ProtocolRefs[0],
                                                   ProtocolRefs.size(),
                                                   attrs.getList());
  }

  // Last, and definitely not least, parse a protocol declaration.
  SourceLocation LAngleLoc, EndProtoLoc;

  SmallVector<Decl *, 8> ProtocolRefs;
  SmallVector<SourceLocation, 8> ProtocolLocs;
  if (Tok.is(tok::less) &&
      ParseObjCProtocolReferences(ProtocolRefs, ProtocolLocs, false,
                                  LAngleLoc, EndProtoLoc))
    return DeclGroupPtrTy();

  Decl *ProtoType =
    Actions.ActOnStartProtocolInterface(AtLoc, protocolName, nameLoc,
                                        ProtocolRefs.data(),
                                        ProtocolRefs.size(),
                                        ProtocolLocs.data(),
                                        EndProtoLoc, attrs.getList());

  ParseObjCInterfaceDeclList(tok::objc_protocol, ProtoType);
  return Actions.ConvertDeclToDeclGroup(ProtoType);
}

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
Parser::DeclGroupPtrTy
Parser::ParseObjCAtImplementationDeclaration(SourceLocation AtLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_implementation) &&
         "ParseObjCAtImplementationDeclaration(): Expected @implementation");
  CheckNestedObjCContexts(AtLoc);
  ConsumeToken(); // the "implementation" identifier

  // Code completion after '@implementation'.
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCImplementationDecl(getCurScope());
    cutOffParsing();
    return DeclGroupPtrTy();
  }

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing class or category name.
    return DeclGroupPtrTy();
  }
  // We have a class or category name - consume it.
  IdentifierInfo *nameId = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken(); // consume class or category name
  Decl *ObjCImpDecl = 0;

  if (Tok.is(tok::l_paren)) {
    // we have a category implementation.
    ConsumeParen();
    SourceLocation categoryLoc, rparenLoc;
    IdentifierInfo *categoryId = 0;

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCImplementationCategory(getCurScope(), nameId, nameLoc);
      cutOffParsing();
      return DeclGroupPtrTy();
    }
    
    if (Tok.is(tok::identifier)) {
      categoryId = Tok.getIdentifierInfo();
      categoryLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_ident); // missing category name.
      return DeclGroupPtrTy();
    }
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren, false); // don't stop at ';'
      return DeclGroupPtrTy();
    }
    rparenLoc = ConsumeParen();
    ObjCImpDecl = Actions.ActOnStartCategoryImplementation(
                                    AtLoc, nameId, nameLoc, categoryId,
                                    categoryLoc);

  } else {
    // We have a class implementation
    SourceLocation superClassLoc;
    IdentifierInfo *superClassId = 0;
    if (Tok.is(tok::colon)) {
      // We have a super class
      ConsumeToken();
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident); // missing super class name.
        return DeclGroupPtrTy();
      }
      superClassId = Tok.getIdentifierInfo();
      superClassLoc = ConsumeToken(); // Consume super class name
    }
    ObjCImpDecl = Actions.ActOnStartClassImplementation(
                                    AtLoc, nameId, nameLoc,
                                    superClassId, superClassLoc);
  
    if (Tok.is(tok::l_brace)) // we have ivars
      ParseObjCClassInstanceVariables(ObjCImpDecl, tok::objc_private, AtLoc);
  }
  assert(ObjCImpDecl);

  SmallVector<Decl *, 8> DeclsInGroup;

  {
    ObjCImplParsingDataRAII ObjCImplParsing(*this, ObjCImpDecl);
    while (!ObjCImplParsing.isFinished() && Tok.isNot(tok::eof)) {
      ParsedAttributesWithRange attrs(AttrFactory);
      MaybeParseCXX0XAttributes(attrs);
      MaybeParseMicrosoftAttributes(attrs);
      if (DeclGroupPtrTy DGP = ParseExternalDeclaration(attrs)) {
        DeclGroupRef DG = DGP.get();
        DeclsInGroup.append(DG.begin(), DG.end());
      }
    }
  }

  return Actions.ActOnFinishObjCImplementation(ObjCImpDecl, DeclsInGroup);
}

Parser::DeclGroupPtrTy
Parser::ParseObjCAtEndDeclaration(SourceRange atEnd) {
  assert(Tok.isObjCAtKeyword(tok::objc_end) &&
         "ParseObjCAtEndDeclaration(): Expected @end");
  ConsumeToken(); // the "end" identifier
  if (CurParsedObjCImpl)
    CurParsedObjCImpl->finish(atEnd);
  else
    // missing @implementation
    Diag(atEnd.getBegin(), diag::err_expected_objc_container);
  return DeclGroupPtrTy();
}

Parser::ObjCImplParsingDataRAII::~ObjCImplParsingDataRAII() {
  if (!Finished) {
    finish(P.Tok.getLocation());
    if (P.Tok.is(tok::eof)) {
      P.Diag(P.Tok, diag::err_objc_missing_end)
          << FixItHint::CreateInsertion(P.Tok.getLocation(), "\n@end\n");
      P.Diag(Dcl->getLocStart(), diag::note_objc_container_start)
          << Sema::OCK_Implementation;
    }
  }
  P.CurParsedObjCImpl = 0;
  assert(LateParsedObjCMethods.empty());
}

void Parser::ObjCImplParsingDataRAII::finish(SourceRange AtEnd) {
  assert(!Finished);
  P.Actions.DefaultSynthesizeProperties(P.getCurScope(), Dcl);
  for (size_t i = 0; i < LateParsedObjCMethods.size(); ++i)
    P.ParseLexedObjCMethodDefs(*LateParsedObjCMethods[i], 
                               true/*Methods*/);

  P.Actions.ActOnAtEnd(P.getCurScope(), AtEnd);

  if (HasCFunction)
    for (size_t i = 0; i < LateParsedObjCMethods.size(); ++i)
      P.ParseLexedObjCMethodDefs(*LateParsedObjCMethods[i], 
                                 false/*c-functions*/);
  
  /// \brief Clear and free the cached objc methods.
  for (LateParsedObjCMethodContainer::iterator
         I = LateParsedObjCMethods.begin(),
         E = LateParsedObjCMethods.end(); I != E; ++I)
    delete *I;
  LateParsedObjCMethods.clear();

  Finished = true;
}

///   compatibility-alias-decl:
///     @compatibility_alias alias-name  class-name ';'
///
Decl *Parser::ParseObjCAtAliasDeclaration(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_compatibility_alias) &&
         "ParseObjCAtAliasDeclaration(): Expected @compatibility_alias");
  ConsumeToken(); // consume compatibility_alias
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  IdentifierInfo *aliasId = Tok.getIdentifierInfo();
  SourceLocation aliasLoc = ConsumeToken(); // consume alias-name
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  IdentifierInfo *classId = Tok.getIdentifierInfo();
  SourceLocation classLoc = ConsumeToken(); // consume class-name;
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after, 
                   "@compatibility_alias");
  return Actions.ActOnCompatibilityAlias(atLoc, aliasId, aliasLoc,
                                         classId, classLoc);
}

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
///
Decl *Parser::ParseObjCPropertySynthesize(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_synthesize) &&
         "ParseObjCPropertyDynamic(): Expected '@synthesize'");
  ConsumeToken(); // consume synthesize

  while (true) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCPropertyDefinition(getCurScope());
      cutOffParsing();
      return 0;
    }
    
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_synthesized_property_name);
      SkipUntil(tok::semi);
      return 0;
    }
    
    IdentifierInfo *propertyIvar = 0;
    IdentifierInfo *propertyId = Tok.getIdentifierInfo();
    SourceLocation propertyLoc = ConsumeToken(); // consume property name
    SourceLocation propertyIvarLoc;
    if (Tok.is(tok::equal)) {
      // property '=' ivar-name
      ConsumeToken(); // consume '='
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCPropertySynthesizeIvar(getCurScope(), propertyId);
        cutOffParsing();
        return 0;
      }
      
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }
      propertyIvar = Tok.getIdentifierInfo();
      propertyIvarLoc = ConsumeToken(); // consume ivar-name
    }
    Actions.ActOnPropertyImplDecl(getCurScope(), atLoc, propertyLoc, true,
                                  propertyId, propertyIvar, propertyIvarLoc);
    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // consume ','
  }
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@synthesize");
  return 0;
}

///   property-dynamic:
///     @dynamic  property-list
///
///   property-list:
///     identifier
///     property-list ',' identifier
///
Decl *Parser::ParseObjCPropertyDynamic(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_dynamic) &&
         "ParseObjCPropertyDynamic(): Expected '@dynamic'");
  ConsumeToken(); // consume dynamic
  while (true) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteObjCPropertyDefinition(getCurScope());
      cutOffParsing();
      return 0;
    }
    
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return 0;
    }
    
    IdentifierInfo *propertyId = Tok.getIdentifierInfo();
    SourceLocation propertyLoc = ConsumeToken(); // consume property name
    Actions.ActOnPropertyImplDecl(getCurScope(), atLoc, propertyLoc, false,
                                  propertyId, 0, SourceLocation());

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // consume ','
  }
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@dynamic");
  return 0;
}

///  objc-throw-statement:
///    throw expression[opt];
///
StmtResult Parser::ParseObjCThrowStmt(SourceLocation atLoc) {
  ExprResult Res;
  ConsumeToken(); // consume throw
  if (Tok.isNot(tok::semi)) {
    Res = ParseExpression();
    if (Res.isInvalid()) {
      SkipUntil(tok::semi);
      return StmtError();
    }
  }
  // consume ';'
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@throw");
  return Actions.ActOnObjCAtThrowStmt(atLoc, Res.take(), getCurScope());
}

/// objc-synchronized-statement:
///   @synchronized '(' expression ')' compound-statement
///
StmtResult
Parser::ParseObjCSynchronizedStmt(SourceLocation atLoc) {
  ConsumeToken(); // consume synchronized
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "@synchronized";
    return StmtError();
  }

  // The operand is surrounded with parentheses.
  ConsumeParen();  // '('
  ExprResult operand(ParseExpression());

  if (Tok.is(tok::r_paren)) {
    ConsumeParen();  // ')'
  } else {
    if (!operand.isInvalid())
      Diag(Tok, diag::err_expected_rparen);

    // Skip forward until we see a left brace, but don't consume it.
    SkipUntil(tok::l_brace, true, true);
  }

  // Require a compound statement.
  if (Tok.isNot(tok::l_brace)) {
    if (!operand.isInvalid())
      Diag(Tok, diag::err_expected_lbrace);
    return StmtError();
  }

  // Check the @synchronized operand now.
  if (!operand.isInvalid())
    operand = Actions.ActOnObjCAtSynchronizedOperand(atLoc, operand.take());

  // Parse the compound statement within a new scope.
  ParseScope bodyScope(this, Scope::DeclScope);
  StmtResult body(ParseCompoundStatementBody());
  bodyScope.Exit();

  // If there was a semantic or parse error earlier with the
  // operand, fail now.
  if (operand.isInvalid())
    return StmtError();

  if (body.isInvalid())
    body = Actions.ActOnNullStmt(Tok.getLocation());

  return Actions.ActOnObjCAtSynchronizedStmt(atLoc, operand.get(), body.get());
}

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
///
StmtResult Parser::ParseObjCTryStmt(SourceLocation atLoc) {
  bool catch_or_finally_seen = false;

  ConsumeToken(); // consume try
  if (Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_lbrace);
    return StmtError();
  }
  StmtVector CatchStmts;
  StmtResult FinallyStmt;
  ParseScope TryScope(this, Scope::DeclScope);
  StmtResult TryBody(ParseCompoundStatementBody());
  TryScope.Exit();
  if (TryBody.isInvalid())
    TryBody = Actions.ActOnNullStmt(Tok.getLocation());

  while (Tok.is(tok::at)) {
    // At this point, we need to lookahead to determine if this @ is the start
    // of an @catch or @finally.  We don't want to consume the @ token if this
    // is an @try or @encode or something else.
    Token AfterAt = GetLookAheadToken(1);
    if (!AfterAt.isObjCAtKeyword(tok::objc_catch) &&
        !AfterAt.isObjCAtKeyword(tok::objc_finally))
      break;

    SourceLocation AtCatchFinallyLoc = ConsumeToken();
    if (Tok.isObjCAtKeyword(tok::objc_catch)) {
      Decl *FirstPart = 0;
      ConsumeToken(); // consume catch
      if (Tok.is(tok::l_paren)) {
        ConsumeParen();
        ParseScope CatchScope(this, Scope::DeclScope|Scope::AtCatchScope);
        if (Tok.isNot(tok::ellipsis)) {
          DeclSpec DS(AttrFactory);
          ParseDeclarationSpecifiers(DS);
          Declarator ParmDecl(DS, Declarator::ObjCCatchContext);
          ParseDeclarator(ParmDecl);

          // Inform the actions module about the declarator, so it
          // gets added to the current scope.
          FirstPart = Actions.ActOnObjCExceptionDecl(getCurScope(), ParmDecl);
        } else
          ConsumeToken(); // consume '...'

        SourceLocation RParenLoc;

        if (Tok.is(tok::r_paren))
          RParenLoc = ConsumeParen();
        else // Skip over garbage, until we get to ')'.  Eat the ')'.
          SkipUntil(tok::r_paren, true, false);

        StmtResult CatchBody(true);
        if (Tok.is(tok::l_brace))
          CatchBody = ParseCompoundStatementBody();
        else
          Diag(Tok, diag::err_expected_lbrace);
        if (CatchBody.isInvalid())
          CatchBody = Actions.ActOnNullStmt(Tok.getLocation());
        
        StmtResult Catch = Actions.ActOnObjCAtCatchStmt(AtCatchFinallyLoc,
                                                              RParenLoc, 
                                                              FirstPart, 
                                                              CatchBody.take());
        if (!Catch.isInvalid())
          CatchStmts.push_back(Catch.release());
        
      } else {
        Diag(AtCatchFinallyLoc, diag::err_expected_lparen_after)
          << "@catch clause";
        return StmtError();
      }
      catch_or_finally_seen = true;
    } else {
      assert(Tok.isObjCAtKeyword(tok::objc_finally) && "Lookahead confused?");
      ConsumeToken(); // consume finally
      ParseScope FinallyScope(this, Scope::DeclScope);

      StmtResult FinallyBody(true);
      if (Tok.is(tok::l_brace))
        FinallyBody = ParseCompoundStatementBody();
      else
        Diag(Tok, diag::err_expected_lbrace);
      if (FinallyBody.isInvalid())
        FinallyBody = Actions.ActOnNullStmt(Tok.getLocation());
      FinallyStmt = Actions.ActOnObjCAtFinallyStmt(AtCatchFinallyLoc,
                                                   FinallyBody.take());
      catch_or_finally_seen = true;
      break;
    }
  }
  if (!catch_or_finally_seen) {
    Diag(atLoc, diag::err_missing_catch_finally);
    return StmtError();
  }
  
  return Actions.ActOnObjCAtTryStmt(atLoc, TryBody.take(), 
                                    CatchStmts,
                                    FinallyStmt.take());
}

/// objc-autoreleasepool-statement:
///   @autoreleasepool compound-statement
///
StmtResult
Parser::ParseObjCAutoreleasePoolStmt(SourceLocation atLoc) {
  ConsumeToken(); // consume autoreleasepool
  if (Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_lbrace);
    return StmtError();
  }
  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope BodyScope(this, Scope::DeclScope);

  StmtResult AutoreleasePoolBody(ParseCompoundStatementBody());

  BodyScope.Exit();
  if (AutoreleasePoolBody.isInvalid())
    AutoreleasePoolBody = Actions.ActOnNullStmt(Tok.getLocation());
  return Actions.ActOnObjCAutoreleasePoolStmt(atLoc, 
                                                AutoreleasePoolBody.take());
}

/// StashAwayMethodOrFunctionBodyTokens -  Consume the tokens and store them 
/// for later parsing.
void Parser::StashAwayMethodOrFunctionBodyTokens(Decl *MDecl) {
  LexedMethod* LM = new LexedMethod(this, MDecl);
  CurParsedObjCImpl->LateParsedObjCMethods.push_back(LM);
  CachedTokens &Toks = LM->Toks;
  // Begin by storing the '{' or 'try' or ':' token.
  Toks.push_back(Tok);
  if (Tok.is(tok::kw_try)) {
    ConsumeToken();
    if (Tok.is(tok::colon)) {
      Toks.push_back(Tok);
      ConsumeToken();
      while (Tok.isNot(tok::l_brace)) {
        ConsumeAndStoreUntil(tok::l_paren, Toks, /*StopAtSemi=*/false);
        ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/false);
      }
    }
    Toks.push_back(Tok); // also store '{'
  }
  else if (Tok.is(tok::colon)) {
    ConsumeToken();
    while (Tok.isNot(tok::l_brace)) {
      ConsumeAndStoreUntil(tok::l_paren, Toks, /*StopAtSemi=*/false);
      ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/false);
    }
    Toks.push_back(Tok); // also store '{'
  }
  ConsumeBrace();
  // Consume everything up to (and including) the matching right brace.
  ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
  while (Tok.is(tok::kw_catch)) {
    ConsumeAndStoreUntil(tok::l_brace, Toks, /*StopAtSemi=*/false);
    ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
  }
}

///   objc-method-def: objc-method-proto ';'[opt] '{' body '}'
///
Decl *Parser::ParseObjCMethodDefinition() {
  Decl *MDecl = ParseObjCMethodPrototype();

  PrettyDeclStackTraceEntry CrashInfo(Actions, MDecl, Tok.getLocation(),
                                      "parsing Objective-C method");

  // parse optional ';'
  if (Tok.is(tok::semi)) {
    if (CurParsedObjCImpl) {
      Diag(Tok, diag::warn_semicolon_before_method_body)
        << FixItHint::CreateRemoval(Tok.getLocation());
    }
    ConsumeToken();
  }

  // We should have an opening brace now.
  if (Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_method_body);

    // Skip over garbage, until we get to '{'.  Don't eat the '{'.
    SkipUntil(tok::l_brace, true, true);

    // If we didn't find the '{', bail out.
    if (Tok.isNot(tok::l_brace))
      return 0;
  }

  if (!MDecl) {
    ConsumeBrace();
    SkipUntil(tok::r_brace, /*StopAtSemi=*/false);
    return 0;
  }

  // Allow the rest of sema to find private method decl implementations.
  Actions.AddAnyMethodToGlobalPool(MDecl);
  assert (CurParsedObjCImpl 
          && "ParseObjCMethodDefinition - Method out of @implementation");
  // Consume the tokens and store them for later parsing.
  StashAwayMethodOrFunctionBodyTokens(MDecl);
  return MDecl;
}

StmtResult Parser::ParseObjCAtStatement(SourceLocation AtLoc) {
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCAtStatement(getCurScope());
    cutOffParsing();
    return StmtError();
  }
  
  if (Tok.isObjCAtKeyword(tok::objc_try))
    return ParseObjCTryStmt(AtLoc);
  
  if (Tok.isObjCAtKeyword(tok::objc_throw))
    return ParseObjCThrowStmt(AtLoc);
  
  if (Tok.isObjCAtKeyword(tok::objc_synchronized))
    return ParseObjCSynchronizedStmt(AtLoc);

  if (Tok.isObjCAtKeyword(tok::objc_autoreleasepool))
    return ParseObjCAutoreleasePoolStmt(AtLoc);
  
  ExprResult Res(ParseExpressionWithLeadingAt(AtLoc));
  if (Res.isInvalid()) {
    // If the expression is invalid, skip ahead to the next semicolon. Not
    // doing this opens us up to the possibility of infinite loops if
    // ParseExpression does not consume any tokens.
    SkipUntil(tok::semi);
    return StmtError();
  }
  
  // Otherwise, eat the semicolon.
  ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
  return Actions.ActOnExprStmt(Actions.MakeFullExpr(Res.take()));
}

ExprResult Parser::ParseObjCAtExpression(SourceLocation AtLoc) {
  switch (Tok.getKind()) {
  case tok::code_completion:
    Actions.CodeCompleteObjCAtExpression(getCurScope());
    cutOffParsing();
    return ExprError();

  case tok::minus:
  case tok::plus: {
    tok::TokenKind Kind = Tok.getKind();
    SourceLocation OpLoc = ConsumeToken();

    if (!Tok.is(tok::numeric_constant)) {
      const char *Symbol = 0;
      switch (Kind) {
      case tok::minus: Symbol = "-"; break;
      case tok::plus: Symbol = "+"; break;
      default: llvm_unreachable("missing unary operator case");
      }
      Diag(Tok, diag::err_nsnumber_nonliteral_unary)
        << Symbol;
      return ExprError();
    }

    ExprResult Lit(Actions.ActOnNumericConstant(Tok));
    if (Lit.isInvalid()) {
      return Lit;
    }
    ConsumeToken(); // Consume the literal token.

    Lit = Actions.ActOnUnaryOp(getCurScope(), OpLoc, Kind, Lit.take());
    if (Lit.isInvalid())
      return Lit;

    return ParsePostfixExpressionSuffix(
             Actions.BuildObjCNumericLiteral(AtLoc, Lit.take()));
  }

  case tok::string_literal:    // primary-expression: string-literal
  case tok::wide_string_literal:
    return ParsePostfixExpressionSuffix(ParseObjCStringLiteral(AtLoc));

  case tok::char_constant:
    return ParsePostfixExpressionSuffix(ParseObjCCharacterLiteral(AtLoc));
      
  case tok::numeric_constant:
    return ParsePostfixExpressionSuffix(ParseObjCNumericLiteral(AtLoc));

  case tok::kw_true:  // Objective-C++, etc.
  case tok::kw___objc_yes: // c/c++/objc/objc++ __objc_yes
    return ParsePostfixExpressionSuffix(ParseObjCBooleanLiteral(AtLoc, true));
  case tok::kw_false: // Objective-C++, etc.
  case tok::kw___objc_no: // c/c++/objc/objc++ __objc_no
    return ParsePostfixExpressionSuffix(ParseObjCBooleanLiteral(AtLoc, false));
    
  case tok::l_square:
    // Objective-C array literal
    return ParsePostfixExpressionSuffix(ParseObjCArrayLiteral(AtLoc));
          
  case tok::l_brace:
    // Objective-C dictionary literal
    return ParsePostfixExpressionSuffix(ParseObjCDictionaryLiteral(AtLoc));
          
  case tok::l_paren:
    // Objective-C boxed expression
    return ParsePostfixExpressionSuffix(ParseObjCBoxedExpr(AtLoc));
          
  default:
    if (Tok.getIdentifierInfo() == 0)
      return ExprError(Diag(AtLoc, diag::err_unexpected_at));

    switch (Tok.getIdentifierInfo()->getObjCKeywordID()) {
    case tok::objc_encode:
      return ParsePostfixExpressionSuffix(ParseObjCEncodeExpression(AtLoc));
    case tok::objc_protocol:
      return ParsePostfixExpressionSuffix(ParseObjCProtocolExpression(AtLoc));
    case tok::objc_selector:
      return ParsePostfixExpressionSuffix(ParseObjCSelectorExpression(AtLoc));
      default: {
        const char *str = 0;
        if (GetLookAheadToken(1).is(tok::l_brace)) {
          char ch = Tok.getIdentifierInfo()->getNameStart()[0];
          str =  
            ch == 't' ? "try" 
                      : (ch == 'f' ? "finally" 
                                   : (ch == 'a' ? "autoreleasepool" : 0));
        }
        if (str) {
          SourceLocation kwLoc = Tok.getLocation();
          return ExprError(Diag(AtLoc, diag::err_unexpected_at) << 
                             FixItHint::CreateReplacement(kwLoc, str));
        }
        else
          return ExprError(Diag(AtLoc, diag::err_unexpected_at));
      }
    }
  }
}

/// \brief Parse the receiver of an Objective-C++ message send.
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
///   objc-receiver: [C++]
///     'super' [not parsed here]
///     expression
///     simple-type-specifier
///     typename-specifier
bool Parser::ParseObjCXXMessageReceiver(bool &IsExpr, void *&TypeOrExpr) {
  InMessageExpressionRAIIObject InMessage(*this, true);

  if (Tok.is(tok::identifier) || Tok.is(tok::coloncolon) || 
      Tok.is(tok::kw_typename) || Tok.is(tok::annot_cxxscope))
    TryAnnotateTypeOrScopeToken();

  if (!Actions.isSimpleTypeSpecifier(Tok.getKind())) {
    //   objc-receiver:
    //     expression
    ExprResult Receiver = ParseExpression();
    if (Receiver.isInvalid())
      return true;

    IsExpr = true;
    TypeOrExpr = Receiver.take();
    return false;
  }

  // objc-receiver:
  //   typename-specifier
  //   simple-type-specifier
  //   expression (that starts with one of the above)
  DeclSpec DS(AttrFactory);
  ParseCXXSimpleTypeSpecifier(DS);
  
  if (Tok.is(tok::l_paren)) {
    // If we see an opening parentheses at this point, we are
    // actually parsing an expression that starts with a
    // function-style cast, e.g.,
    //
    //   postfix-expression:
    //     simple-type-specifier ( expression-list [opt] )
    //     typename-specifier ( expression-list [opt] )
    //
    // Parse the remainder of this case, then the (optional)
    // postfix-expression suffix, followed by the (optional)
    // right-hand side of the binary expression. We have an
    // instance method.
    ExprResult Receiver = ParseCXXTypeConstructExpression(DS);
    if (!Receiver.isInvalid())
      Receiver = ParsePostfixExpressionSuffix(Receiver.take());
    if (!Receiver.isInvalid())
      Receiver = ParseRHSOfBinaryExpression(Receiver.take(), prec::Comma);
    if (Receiver.isInvalid())
      return true;

    IsExpr = true;
    TypeOrExpr = Receiver.take();
    return false;
  }
  
  // We have a class message. Turn the simple-type-specifier or
  // typename-specifier we parsed into a type and parse the
  // remainder of the class message.
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  TypeResult Type = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
  if (Type.isInvalid())
    return true;

  IsExpr = false;
  TypeOrExpr = Type.get().getAsOpaquePtr();
  return false;
}

/// \brief Determine whether the parser is currently referring to a an
/// Objective-C message send, using a simplified heuristic to avoid overhead.
///
/// This routine will only return true for a subset of valid message-send
/// expressions.
bool Parser::isSimpleObjCMessageExpression() {
  assert(Tok.is(tok::l_square) && getLangOpts().ObjC1 &&
         "Incorrect start for isSimpleObjCMessageExpression");
  return GetLookAheadToken(1).is(tok::identifier) &&
         GetLookAheadToken(2).is(tok::identifier);
}

bool Parser::isStartOfObjCClassMessageMissingOpenBracket() {
  if (!getLangOpts().ObjC1 || !NextToken().is(tok::identifier) || 
      InMessageExpression)
    return false;
  
  
  ParsedType Type;

  if (Tok.is(tok::annot_typename)) 
    Type = getTypeAnnotation(Tok);
  else if (Tok.is(tok::identifier))
    Type = Actions.getTypeName(*Tok.getIdentifierInfo(), Tok.getLocation(), 
                               getCurScope());
  else
    return false;
  
  if (!Type.get().isNull() && Type.get()->isObjCObjectOrInterfaceType()) {
    const Token &AfterNext = GetLookAheadToken(2);
    if (AfterNext.is(tok::colon) || AfterNext.is(tok::r_square)) {
      if (Tok.is(tok::identifier))
        TryAnnotateTypeOrScopeToken();
      
      return Tok.is(tok::annot_typename);
    }
  }

  return false;
}

///   objc-message-expr:
///     '[' objc-receiver objc-message-args ']'
///
///   objc-receiver: [C]
///     'super'
///     expression
///     class-name
///     type-name
///
ExprResult Parser::ParseObjCMessageExpression() {
  assert(Tok.is(tok::l_square) && "'[' expected");
  SourceLocation LBracLoc = ConsumeBracket(); // consume '['

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCMessageReceiver(getCurScope());
    cutOffParsing();
    return ExprError();
  }
  
  InMessageExpressionRAIIObject InMessage(*this, true);
  
  if (getLangOpts().CPlusPlus) {
    // We completely separate the C and C++ cases because C++ requires
    // more complicated (read: slower) parsing. 
    
    // Handle send to super.  
    // FIXME: This doesn't benefit from the same typo-correction we
    // get in Objective-C.
    if (Tok.is(tok::identifier) && Tok.getIdentifierInfo() == Ident_super &&
        NextToken().isNot(tok::period) && getCurScope()->isInObjcMethodScope())
      return ParseObjCMessageExpressionBody(LBracLoc, ConsumeToken(),
                                            ParsedType(), 0);

    // Parse the receiver, which is either a type or an expression.
    bool IsExpr;
    void *TypeOrExpr = NULL;
    if (ParseObjCXXMessageReceiver(IsExpr, TypeOrExpr)) {
      SkipUntil(tok::r_square);
      return ExprError();
    }

    if (IsExpr)
      return ParseObjCMessageExpressionBody(LBracLoc, SourceLocation(),
                                            ParsedType(),
                                            static_cast<Expr*>(TypeOrExpr));

    return ParseObjCMessageExpressionBody(LBracLoc, SourceLocation(), 
                              ParsedType::getFromOpaquePtr(TypeOrExpr),
                                          0);
  }
  
  if (Tok.is(tok::identifier)) {
    IdentifierInfo *Name = Tok.getIdentifierInfo();
    SourceLocation NameLoc = Tok.getLocation();
    ParsedType ReceiverType;
    switch (Actions.getObjCMessageKind(getCurScope(), Name, NameLoc,
                                       Name == Ident_super,
                                       NextToken().is(tok::period),
                                       ReceiverType)) {
    case Sema::ObjCSuperMessage:
      return ParseObjCMessageExpressionBody(LBracLoc, ConsumeToken(),
                                            ParsedType(), 0);

    case Sema::ObjCClassMessage:
      if (!ReceiverType) {
        SkipUntil(tok::r_square);
        return ExprError();
      }

      ConsumeToken(); // the type name

      return ParseObjCMessageExpressionBody(LBracLoc, SourceLocation(), 
                                            ReceiverType, 0);
        
    case Sema::ObjCInstanceMessage:
      // Fall through to parse an expression.
      break;
    }
  }
  
  // Otherwise, an arbitrary expression can be the receiver of a send.
  ExprResult Res(ParseExpression());
  if (Res.isInvalid()) {
    SkipUntil(tok::r_square);
    return Res;
  }

  return ParseObjCMessageExpressionBody(LBracLoc, SourceLocation(),
                                        ParsedType(), Res.take());
}

/// \brief Parse the remainder of an Objective-C message following the
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
///
ExprResult
Parser::ParseObjCMessageExpressionBody(SourceLocation LBracLoc,
                                       SourceLocation SuperLoc,
                                       ParsedType ReceiverType,
                                       ExprArg ReceiverExpr) {
  InMessageExpressionRAIIObject InMessage(*this, true);

  if (Tok.is(tok::code_completion)) {
    if (SuperLoc.isValid())
      Actions.CodeCompleteObjCSuperMessage(getCurScope(), SuperLoc, 0, 0,
                                           false);
    else if (ReceiverType)
      Actions.CodeCompleteObjCClassMessage(getCurScope(), ReceiverType, 0, 0,
                                           false);
    else
      Actions.CodeCompleteObjCInstanceMessage(getCurScope(), ReceiverExpr,
                                              0, 0, false);
    cutOffParsing();
    return ExprError();
  }
  
  // Parse objc-selector
  SourceLocation Loc;
  IdentifierInfo *selIdent = ParseObjCSelectorPiece(Loc);

  SmallVector<IdentifierInfo *, 12> KeyIdents;
  SmallVector<SourceLocation, 12> KeyLocs;
  ExprVector KeyExprs;

  if (Tok.is(tok::colon)) {
    while (1) {
      // Each iteration parses a single keyword argument.
      KeyIdents.push_back(selIdent);
      KeyLocs.push_back(Loc);

      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon);
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return ExprError();
      }

      ConsumeToken(); // Eat the ':'.
      ///  Parse the expression after ':'
      
      if (Tok.is(tok::code_completion)) {
        if (SuperLoc.isValid())
          Actions.CodeCompleteObjCSuperMessage(getCurScope(), SuperLoc, 
                                               KeyIdents.data(), 
                                               KeyIdents.size(),
                                               /*AtArgumentEpression=*/true);
        else if (ReceiverType)
          Actions.CodeCompleteObjCClassMessage(getCurScope(), ReceiverType,
                                               KeyIdents.data(), 
                                               KeyIdents.size(),
                                               /*AtArgumentEpression=*/true);
        else
          Actions.CodeCompleteObjCInstanceMessage(getCurScope(), ReceiverExpr,
                                                  KeyIdents.data(), 
                                                  KeyIdents.size(),
                                                  /*AtArgumentEpression=*/true);

        cutOffParsing();
        return ExprError();
      }
      
      ExprResult Res(ParseAssignmentExpression());
      if (Res.isInvalid()) {
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return Res;
      }

      // We have a valid expression.
      KeyExprs.push_back(Res.release());

      // Code completion after each argument.
      if (Tok.is(tok::code_completion)) {
        if (SuperLoc.isValid())
          Actions.CodeCompleteObjCSuperMessage(getCurScope(), SuperLoc, 
                                               KeyIdents.data(), 
                                               KeyIdents.size(),
                                               /*AtArgumentEpression=*/false);
        else if (ReceiverType)
          Actions.CodeCompleteObjCClassMessage(getCurScope(), ReceiverType,
                                               KeyIdents.data(), 
                                               KeyIdents.size(),
                                               /*AtArgumentEpression=*/false);
        else
          Actions.CodeCompleteObjCInstanceMessage(getCurScope(), ReceiverExpr,
                                                  KeyIdents.data(), 
                                                  KeyIdents.size(),
                                                /*AtArgumentEpression=*/false);
        cutOffParsing();
        return ExprError();
      }
            
      // Check for another keyword selector.
      selIdent = ParseObjCSelectorPiece(Loc);
      if (!selIdent && Tok.isNot(tok::colon))
        break;
      // We have a selector or a colon, continue parsing.
    }
    // Parse the, optional, argument list, comma separated.
    while (Tok.is(tok::comma)) {
      SourceLocation commaLoc = ConsumeToken(); // Eat the ','.
      ///  Parse the expression after ','
      ExprResult Res(ParseAssignmentExpression());
      if (Res.isInvalid()) {
        if (Tok.is(tok::colon)) {
          Diag(commaLoc, diag::note_extra_comma_message_arg) <<
            FixItHint::CreateRemoval(commaLoc);
        }
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return Res;
      }

      // We have a valid expression.
      KeyExprs.push_back(Res.release());
    }
  } else if (!selIdent) {
    Diag(Tok, diag::err_expected_ident); // missing selector name.

    // We must manually skip to a ']', otherwise the expression skipper will
    // stop at the ']' when it skips to the ';'.  We want it to skip beyond
    // the enclosing expression.
    SkipUntil(tok::r_square);
    return ExprError();
  }
    
  if (Tok.isNot(tok::r_square)) {
    if (Tok.is(tok::identifier))
      Diag(Tok, diag::err_expected_colon);
    else
      Diag(Tok, diag::err_expected_rsquare);
    // We must manually skip to a ']', otherwise the expression skipper will
    // stop at the ']' when it skips to the ';'.  We want it to skip beyond
    // the enclosing expression.
    SkipUntil(tok::r_square);
    return ExprError();
  }

  SourceLocation RBracLoc = ConsumeBracket(); // consume ']'

  unsigned nKeys = KeyIdents.size();
  if (nKeys == 0) {
    KeyIdents.push_back(selIdent);
    KeyLocs.push_back(Loc);
  }
  Selector Sel = PP.getSelectorTable().getSelector(nKeys, &KeyIdents[0]);

  if (SuperLoc.isValid())
    return Actions.ActOnSuperMessage(getCurScope(), SuperLoc, Sel,
                                     LBracLoc, KeyLocs, RBracLoc, KeyExprs);
  else if (ReceiverType)
    return Actions.ActOnClassMessage(getCurScope(), ReceiverType, Sel,
                                     LBracLoc, KeyLocs, RBracLoc, KeyExprs);
  return Actions.ActOnInstanceMessage(getCurScope(), ReceiverExpr, Sel,
                                      LBracLoc, KeyLocs, RBracLoc, KeyExprs);
}

ExprResult Parser::ParseObjCStringLiteral(SourceLocation AtLoc) {
  ExprResult Res(ParseStringLiteralExpression());
  if (Res.isInvalid()) return Res;

  // @"foo" @"bar" is a valid concatenated string.  Eat any subsequent string
  // expressions.  At this point, we know that the only valid thing that starts
  // with '@' is an @"".
  SmallVector<SourceLocation, 4> AtLocs;
  ExprVector AtStrings;
  AtLocs.push_back(AtLoc);
  AtStrings.push_back(Res.release());

  while (Tok.is(tok::at)) {
    AtLocs.push_back(ConsumeToken()); // eat the @.

    // Invalid unless there is a string literal.
    if (!isTokenStringLiteral())
      return ExprError(Diag(Tok, diag::err_objc_concat_string));

    ExprResult Lit(ParseStringLiteralExpression());
    if (Lit.isInvalid())
      return Lit;

    AtStrings.push_back(Lit.release());
  }

  return Owned(Actions.ParseObjCStringLiteral(&AtLocs[0], AtStrings.data(),
                                              AtStrings.size()));
}

/// ParseObjCBooleanLiteral -
/// objc-scalar-literal : '@' boolean-keyword
///                        ;
/// boolean-keyword: 'true' | 'false' | '__objc_yes' | '__objc_no'
///                        ;
ExprResult Parser::ParseObjCBooleanLiteral(SourceLocation AtLoc, 
                                           bool ArgValue) {
  SourceLocation EndLoc = ConsumeToken();             // consume the keyword.
  return Actions.ActOnObjCBoolLiteral(AtLoc, EndLoc, ArgValue);
}

/// ParseObjCCharacterLiteral -
/// objc-scalar-literal : '@' character-literal
///                        ;
ExprResult Parser::ParseObjCCharacterLiteral(SourceLocation AtLoc) {
  ExprResult Lit(Actions.ActOnCharacterConstant(Tok));
  if (Lit.isInvalid()) {
    return Lit;
  }
  ConsumeToken(); // Consume the literal token.
  return Owned(Actions.BuildObjCNumericLiteral(AtLoc, Lit.take()));
}

/// ParseObjCNumericLiteral -
/// objc-scalar-literal : '@' scalar-literal
///                        ;
/// scalar-literal : | numeric-constant			/* any numeric constant. */
///                    ;
ExprResult Parser::ParseObjCNumericLiteral(SourceLocation AtLoc) {
  ExprResult Lit(Actions.ActOnNumericConstant(Tok));
  if (Lit.isInvalid()) {
    return Lit;
  }
  ConsumeToken(); // Consume the literal token.
  return Owned(Actions.BuildObjCNumericLiteral(AtLoc, Lit.take()));
}

/// ParseObjCBoxedExpr -
/// objc-box-expression:
///       @( assignment-expression )
ExprResult
Parser::ParseObjCBoxedExpr(SourceLocation AtLoc) {
  if (Tok.isNot(tok::l_paren))
    return ExprError(Diag(Tok, diag::err_expected_lparen_after) << "@");

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();
  ExprResult ValueExpr(ParseAssignmentExpression());
  if (T.consumeClose())
    return ExprError();

  if (ValueExpr.isInvalid())
    return ExprError();

  // Wrap the sub-expression in a parenthesized expression, to distinguish
  // a boxed expression from a literal.
  SourceLocation LPLoc = T.getOpenLocation(), RPLoc = T.getCloseLocation();
  ValueExpr = Actions.ActOnParenExpr(LPLoc, RPLoc, ValueExpr.take());
  return Owned(Actions.BuildObjCBoxedExpr(SourceRange(AtLoc, RPLoc),
                                          ValueExpr.take()));
}

ExprResult Parser::ParseObjCArrayLiteral(SourceLocation AtLoc) {
  ExprVector ElementExprs;                   // array elements.
  ConsumeBracket(); // consume the l_square.

  while (Tok.isNot(tok::r_square)) {
    // Parse list of array element expressions (all must be id types).
    ExprResult Res(ParseAssignmentExpression());
    if (Res.isInvalid()) {
      // We must manually skip to a ']', otherwise the expression skipper will
      // stop at the ']' when it skips to the ';'.  We want it to skip beyond
      // the enclosing expression.
      SkipUntil(tok::r_square);
      return Res;
    }    
    
    // Parse the ellipsis that indicates a pack expansion.
    if (Tok.is(tok::ellipsis))
      Res = Actions.ActOnPackExpansion(Res.get(), ConsumeToken());    
    if (Res.isInvalid())
      return true;

    ElementExprs.push_back(Res.release());

    if (Tok.is(tok::comma))
      ConsumeToken(); // Eat the ','.
    else if (Tok.isNot(tok::r_square))
     return ExprError(Diag(Tok, diag::err_expected_rsquare_or_comma));
  }
  SourceLocation EndLoc = ConsumeBracket(); // location of ']'
  MultiExprArg Args(ElementExprs);
  return Owned(Actions.BuildObjCArrayLiteral(SourceRange(AtLoc, EndLoc), Args));
}

ExprResult Parser::ParseObjCDictionaryLiteral(SourceLocation AtLoc) {
  SmallVector<ObjCDictionaryElement, 4> Elements; // dictionary elements.
  ConsumeBrace(); // consume the l_square.
  while (Tok.isNot(tok::r_brace)) {
    // Parse the comma separated key : value expressions.
    ExprResult KeyExpr;
    {
      ColonProtectionRAIIObject X(*this);
      KeyExpr = ParseAssignmentExpression();
      if (KeyExpr.isInvalid()) {
        // We must manually skip to a '}', otherwise the expression skipper will
        // stop at the '}' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_brace);
        return KeyExpr;
      }
    }

    if (Tok.is(tok::colon)) {
      ConsumeToken();
    } else {
      return ExprError(Diag(Tok, diag::err_expected_colon));
    }
    
    ExprResult ValueExpr(ParseAssignmentExpression());
    if (ValueExpr.isInvalid()) {
      // We must manually skip to a '}', otherwise the expression skipper will
      // stop at the '}' when it skips to the ';'.  We want it to skip beyond
      // the enclosing expression.
      SkipUntil(tok::r_brace);
      return ValueExpr;
    }
    
    // Parse the ellipsis that designates this as a pack expansion.
    SourceLocation EllipsisLoc;
    if (Tok.is(tok::ellipsis) && getLangOpts().CPlusPlus)
      EllipsisLoc = ConsumeToken();
    
    // We have a valid expression. Collect it in a vector so we can
    // build the argument list.
    ObjCDictionaryElement Element = { 
      KeyExpr.get(), ValueExpr.get(), EllipsisLoc, llvm::Optional<unsigned>()
    };
    Elements.push_back(Element);
    
    if (Tok.is(tok::comma))
      ConsumeToken(); // Eat the ','.
    else if (Tok.isNot(tok::r_brace))
      return ExprError(Diag(Tok, diag::err_expected_rbrace_or_comma));
  }
  SourceLocation EndLoc = ConsumeBrace();
  
  // Create the ObjCDictionaryLiteral.
  return Owned(Actions.BuildObjCDictionaryLiteral(SourceRange(AtLoc, EndLoc),
                                                  Elements.data(),
                                                  Elements.size()));
}

///    objc-encode-expression:
///      \@encode ( type-name )
ExprResult
Parser::ParseObjCEncodeExpression(SourceLocation AtLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_encode) && "Not an @encode expression!");

  SourceLocation EncLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren))
    return ExprError(Diag(Tok, diag::err_expected_lparen_after) << "@encode");

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  TypeResult Ty = ParseTypeName();

  T.consumeClose();

  if (Ty.isInvalid())
    return ExprError();

  return Owned(Actions.ParseObjCEncodeExpression(AtLoc, EncLoc,
                                                 T.getOpenLocation(), Ty.get(),
                                                 T.getCloseLocation()));
}

///     objc-protocol-expression
///       \@protocol ( protocol-name )
ExprResult
Parser::ParseObjCProtocolExpression(SourceLocation AtLoc) {
  SourceLocation ProtoLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren))
    return ExprError(Diag(Tok, diag::err_expected_lparen_after) << "@protocol");

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  if (Tok.isNot(tok::identifier))
    return ExprError(Diag(Tok, diag::err_expected_ident));

  IdentifierInfo *protocolId = Tok.getIdentifierInfo();
  SourceLocation ProtoIdLoc = ConsumeToken();

  T.consumeClose();

  return Owned(Actions.ParseObjCProtocolExpression(protocolId, AtLoc, ProtoLoc,
                                                   T.getOpenLocation(),
                                                   ProtoIdLoc,
                                                   T.getCloseLocation()));
}

///     objc-selector-expression
///       @selector '(' objc-keyword-selector ')'
ExprResult Parser::ParseObjCSelectorExpression(SourceLocation AtLoc) {
  SourceLocation SelectorLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren))
    return ExprError(Diag(Tok, diag::err_expected_lparen_after) << "@selector");

  SmallVector<IdentifierInfo *, 12> KeyIdents;
  SourceLocation sLoc;
  
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteObjCSelector(getCurScope(), KeyIdents.data(),
                                     KeyIdents.size());
    cutOffParsing();
    return ExprError();
  }
  
  IdentifierInfo *SelIdent = ParseObjCSelectorPiece(sLoc);
  if (!SelIdent &&  // missing selector name.
      Tok.isNot(tok::colon) && Tok.isNot(tok::coloncolon))
    return ExprError(Diag(Tok, diag::err_expected_ident));

  KeyIdents.push_back(SelIdent);
  unsigned nColons = 0;
  if (Tok.isNot(tok::r_paren)) {
    while (1) {
      if (Tok.is(tok::coloncolon)) { // Handle :: in C++.
        ++nColons;
        KeyIdents.push_back(0);
      } else if (Tok.isNot(tok::colon))
        return ExprError(Diag(Tok, diag::err_expected_colon));

      ++nColons;
      ConsumeToken(); // Eat the ':' or '::'.
      if (Tok.is(tok::r_paren))
        break;
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCSelector(getCurScope(), KeyIdents.data(),
                                         KeyIdents.size());
        cutOffParsing();
        return ExprError();
      }

      // Check for another keyword selector.
      SourceLocation Loc;
      SelIdent = ParseObjCSelectorPiece(Loc);
      KeyIdents.push_back(SelIdent);
      if (!SelIdent && Tok.isNot(tok::colon) && Tok.isNot(tok::coloncolon))
        break;
    }
  }
  T.consumeClose();
  Selector Sel = PP.getSelectorTable().getSelector(nColons, &KeyIdents[0]);
  return Owned(Actions.ParseObjCSelectorExpression(Sel, AtLoc, SelectorLoc,
                                                   T.getOpenLocation(),
                                                   T.getCloseLocation()));
 }

void Parser::ParseLexedObjCMethodDefs(LexedMethod &LM, bool parseMethod) {
  // MCDecl might be null due to error in method or c-function  prototype, etc.
  Decl *MCDecl = LM.D;
  bool skip = MCDecl && 
              ((parseMethod && !Actions.isObjCMethodDecl(MCDecl)) ||
              (!parseMethod && Actions.isObjCMethodDecl(MCDecl)));
  if (skip)
    return;
  
  // Save the current token position.
  SourceLocation OrigLoc = Tok.getLocation();

  assert(!LM.Toks.empty() && "ParseLexedObjCMethodDef - Empty body!");
  // Append the current token at the end of the new token stream so that it
  // doesn't get lost.
  LM.Toks.push_back(Tok);
  PP.EnterTokenStream(LM.Toks.data(), LM.Toks.size(), true, false);
  
  // Consume the previously pushed token.
  ConsumeAnyToken();
    
  assert((Tok.is(tok::l_brace) || Tok.is(tok::kw_try) ||
          Tok.is(tok::colon)) && 
          "Inline objective-c method not starting with '{' or 'try' or ':'");
  // Enter a scope for the method or c-fucntion body.
  ParseScope BodyScope(this,
                       parseMethod
                       ? Scope::ObjCMethodScope|Scope::FnScope|Scope::DeclScope
                       : Scope::FnScope|Scope::DeclScope);
    
  // Tell the actions module that we have entered a method or c-function definition 
  // with the specified Declarator for the method/function.
  if (parseMethod)
    Actions.ActOnStartOfObjCMethodDef(getCurScope(), MCDecl);
  else
    Actions.ActOnStartOfFunctionDef(getCurScope(), MCDecl);
  if (Tok.is(tok::kw_try))
    MCDecl = ParseFunctionTryBlock(MCDecl, BodyScope);
  else {
    if (Tok.is(tok::colon))
      ParseConstructorInitializer(MCDecl);
    MCDecl = ParseFunctionStatementBody(MCDecl, BodyScope);
  }
  
  if (Tok.getLocation() != OrigLoc) {
    // Due to parsing error, we either went over the cached tokens or
    // there are still cached tokens left. If it's the latter case skip the
    // leftover tokens.
    // Since this is an uncommon situation that should be avoided, use the
    // expensive isBeforeInTranslationUnit call.
    if (PP.getSourceManager().isBeforeInTranslationUnit(Tok.getLocation(),
                                                     OrigLoc))
      while (Tok.getLocation() != OrigLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();
  }
  
  return;
}
