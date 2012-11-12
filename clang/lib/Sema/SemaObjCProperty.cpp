//===--- SemaObjCProperty.cpp - Semantic Analysis for ObjC @property ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Objective C @property and
//  @synthesize declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// Grammar actions.
//===----------------------------------------------------------------------===//

/// getImpliedARCOwnership - Given a set of property attributes and a
/// type, infer an expected lifetime.  The type's ownership qualification
/// is not considered.
///
/// Returns OCL_None if the attributes as stated do not imply an ownership.
/// Never returns OCL_Autoreleasing.
static Qualifiers::ObjCLifetime getImpliedARCOwnership(
                               ObjCPropertyDecl::PropertyAttributeKind attrs,
                                                QualType type) {
  // retain, strong, copy, weak, and unsafe_unretained are only legal
  // on properties of retainable pointer type.
  if (attrs & (ObjCPropertyDecl::OBJC_PR_retain |
               ObjCPropertyDecl::OBJC_PR_strong |
               ObjCPropertyDecl::OBJC_PR_copy)) {
    return Qualifiers::OCL_Strong;
  } else if (attrs & ObjCPropertyDecl::OBJC_PR_weak) {
    return Qualifiers::OCL_Weak;
  } else if (attrs & ObjCPropertyDecl::OBJC_PR_unsafe_unretained) {
    return Qualifiers::OCL_ExplicitNone;
  }

  // assign can appear on other types, so we have to check the
  // property type.
  if (attrs & ObjCPropertyDecl::OBJC_PR_assign &&
      type->isObjCRetainableType()) {
    return Qualifiers::OCL_ExplicitNone;
  }

  return Qualifiers::OCL_None;
}

/// Check the internal consistency of a property declaration.
static void checkARCPropertyDecl(Sema &S, ObjCPropertyDecl *property) {
  if (property->isInvalidDecl()) return;

  ObjCPropertyDecl::PropertyAttributeKind propertyKind
    = property->getPropertyAttributes();
  Qualifiers::ObjCLifetime propertyLifetime
    = property->getType().getObjCLifetime();

  // Nothing to do if we don't have a lifetime.
  if (propertyLifetime == Qualifiers::OCL_None) return;

  Qualifiers::ObjCLifetime expectedLifetime
    = getImpliedARCOwnership(propertyKind, property->getType());
  if (!expectedLifetime) {
    // We have a lifetime qualifier but no dominating property
    // attribute.  That's okay, but restore reasonable invariants by
    // setting the property attribute according to the lifetime
    // qualifier.
    ObjCPropertyDecl::PropertyAttributeKind attr;
    if (propertyLifetime == Qualifiers::OCL_Strong) {
      attr = ObjCPropertyDecl::OBJC_PR_strong;
    } else if (propertyLifetime == Qualifiers::OCL_Weak) {
      attr = ObjCPropertyDecl::OBJC_PR_weak;
    } else {
      assert(propertyLifetime == Qualifiers::OCL_ExplicitNone);
      attr = ObjCPropertyDecl::OBJC_PR_unsafe_unretained;
    }
    property->setPropertyAttributes(attr);
    return;
  }

  if (propertyLifetime == expectedLifetime) return;

  property->setInvalidDecl();
  S.Diag(property->getLocation(),
         diag::err_arc_inconsistent_property_ownership)
    << property->getDeclName()
    << expectedLifetime
    << propertyLifetime;
}

static unsigned deduceWeakPropertyFromType(Sema &S, QualType T) {
  if ((S.getLangOpts().getGC() != LangOptions::NonGC && 
       T.isObjCGCWeak()) ||
      (S.getLangOpts().ObjCAutoRefCount &&
       T.getObjCLifetime() == Qualifiers::OCL_Weak))
    return ObjCDeclSpec::DQ_PR_weak;
  return 0;
}

Decl *Sema::ActOnProperty(Scope *S, SourceLocation AtLoc,
                          SourceLocation LParenLoc,
                          FieldDeclarator &FD,
                          ObjCDeclSpec &ODS,
                          Selector GetterSel,
                          Selector SetterSel,
                          bool *isOverridingProperty,
                          tok::ObjCKeywordKind MethodImplKind,
                          DeclContext *lexicalDC) {
  unsigned Attributes = ODS.getPropertyAttributes();
  TypeSourceInfo *TSI = GetTypeForDeclarator(FD.D, S);
  QualType T = TSI->getType();
  Attributes |= deduceWeakPropertyFromType(*this, T);
  
  bool isReadWrite = ((Attributes & ObjCDeclSpec::DQ_PR_readwrite) ||
                      // default is readwrite!
                      !(Attributes & ObjCDeclSpec::DQ_PR_readonly));
  // property is defaulted to 'assign' if it is readwrite and is
  // not retain or copy
  bool isAssign = ((Attributes & ObjCDeclSpec::DQ_PR_assign) ||
                   (isReadWrite &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_retain) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_strong) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_copy) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) &&
                    !(Attributes & ObjCDeclSpec::DQ_PR_weak)));

  // Proceed with constructing the ObjCPropertDecls.
  ObjCContainerDecl *ClassDecl = cast<ObjCContainerDecl>(CurContext);
  if (ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl))
    if (CDecl->IsClassExtension()) {
      Decl *Res = HandlePropertyInClassExtension(S, AtLoc, LParenLoc,
                                           FD, GetterSel, SetterSel,
                                           isAssign, isReadWrite,
                                           Attributes,
                                           ODS.getPropertyAttributes(),
                                           isOverridingProperty, TSI,
                                           MethodImplKind);
      if (Res) {
        CheckObjCPropertyAttributes(Res, AtLoc, Attributes, false);
        if (getLangOpts().ObjCAutoRefCount)
          checkARCPropertyDecl(*this, cast<ObjCPropertyDecl>(Res));
      }
      ActOnDocumentableDecl(Res);
      return Res;
    }
  
  ObjCPropertyDecl *Res = CreatePropertyDecl(S, ClassDecl, AtLoc, LParenLoc, FD,
                                             GetterSel, SetterSel,
                                             isAssign, isReadWrite,
                                             Attributes,
                                             ODS.getPropertyAttributes(),
                                             TSI, MethodImplKind);
  if (lexicalDC)
    Res->setLexicalDeclContext(lexicalDC);

  // Validate the attributes on the @property.
  CheckObjCPropertyAttributes(Res, AtLoc, Attributes, 
                              (isa<ObjCInterfaceDecl>(ClassDecl) ||
                               isa<ObjCProtocolDecl>(ClassDecl)));

  if (getLangOpts().ObjCAutoRefCount)
    checkARCPropertyDecl(*this, Res);

  ActOnDocumentableDecl(Res);
  return Res;
}

static ObjCPropertyDecl::PropertyAttributeKind
makePropertyAttributesAsWritten(unsigned Attributes) {
  unsigned attributesAsWritten = 0;
  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_readonly;
  if (Attributes & ObjCDeclSpec::DQ_PR_readwrite)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_readwrite;
  if (Attributes & ObjCDeclSpec::DQ_PR_getter)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_getter;
  if (Attributes & ObjCDeclSpec::DQ_PR_setter)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_setter;
  if (Attributes & ObjCDeclSpec::DQ_PR_assign)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_assign;
  if (Attributes & ObjCDeclSpec::DQ_PR_retain)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_retain;
  if (Attributes & ObjCDeclSpec::DQ_PR_strong)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_strong;
  if (Attributes & ObjCDeclSpec::DQ_PR_weak)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_weak;
  if (Attributes & ObjCDeclSpec::DQ_PR_copy)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_copy;
  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_unsafe_unretained;
  if (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_nonatomic;
  if (Attributes & ObjCDeclSpec::DQ_PR_atomic)
    attributesAsWritten |= ObjCPropertyDecl::OBJC_PR_atomic;
  
  return (ObjCPropertyDecl::PropertyAttributeKind)attributesAsWritten;
}

static bool LocPropertyAttribute( ASTContext &Context, const char *attrName, 
                                 SourceLocation LParenLoc, SourceLocation &Loc) {
  if (LParenLoc.isMacroID())
    return false;
  
  SourceManager &SM = Context.getSourceManager();
  std::pair<FileID, unsigned> locInfo = SM.getDecomposedLoc(LParenLoc);
  // Try to load the file buffer.
  bool invalidTemp = false;
  StringRef file = SM.getBufferData(locInfo.first, &invalidTemp);
  if (invalidTemp)
    return false;
  const char *tokenBegin = file.data() + locInfo.second;
  
  // Lex from the start of the given location.
  Lexer lexer(SM.getLocForStartOfFile(locInfo.first),
              Context.getLangOpts(),
              file.begin(), tokenBegin, file.end());
  Token Tok;
  do {
    lexer.LexFromRawLexer(Tok);
    if (Tok.is(tok::raw_identifier) &&
        StringRef(Tok.getRawIdentifierData(), Tok.getLength()) == attrName) {
      Loc = Tok.getLocation();
      return true;
    }
  } while (Tok.isNot(tok::r_paren));
  return false;
  
}

static unsigned getOwnershipRule(unsigned attr) {
  return attr & (ObjCPropertyDecl::OBJC_PR_assign |
                 ObjCPropertyDecl::OBJC_PR_retain |
                 ObjCPropertyDecl::OBJC_PR_copy   |
                 ObjCPropertyDecl::OBJC_PR_weak   |
                 ObjCPropertyDecl::OBJC_PR_strong |
                 ObjCPropertyDecl::OBJC_PR_unsafe_unretained);
}

Decl *
Sema::HandlePropertyInClassExtension(Scope *S,
                                     SourceLocation AtLoc,
                                     SourceLocation LParenLoc,
                                     FieldDeclarator &FD,
                                     Selector GetterSel, Selector SetterSel,
                                     const bool isAssign,
                                     const bool isReadWrite,
                                     const unsigned Attributes,
                                     const unsigned AttributesAsWritten,
                                     bool *isOverridingProperty,
                                     TypeSourceInfo *T,
                                     tok::ObjCKeywordKind MethodImplKind) {
  ObjCCategoryDecl *CDecl = cast<ObjCCategoryDecl>(CurContext);
  // Diagnose if this property is already in continuation class.
  DeclContext *DC = CurContext;
  IdentifierInfo *PropertyId = FD.D.getIdentifier();
  ObjCInterfaceDecl *CCPrimary = CDecl->getClassInterface();
  
  if (CCPrimary)
    // Check for duplicate declaration of this property in current and
    // other class extensions.
    for (const ObjCCategoryDecl *ClsExtDecl = 
         CCPrimary->getFirstClassExtension();
         ClsExtDecl; ClsExtDecl = ClsExtDecl->getNextClassExtension()) {
      if (ObjCPropertyDecl *prevDecl =
          ObjCPropertyDecl::findPropertyDecl(ClsExtDecl, PropertyId)) {
        Diag(AtLoc, diag::err_duplicate_property);
        Diag(prevDecl->getLocation(), diag::note_property_declare);
        return 0;
      }
    }
  
  // Create a new ObjCPropertyDecl with the DeclContext being
  // the class extension.
  // FIXME. We should really be using CreatePropertyDecl for this.
  ObjCPropertyDecl *PDecl =
    ObjCPropertyDecl::Create(Context, DC, FD.D.getIdentifierLoc(),
                             PropertyId, AtLoc, LParenLoc, T);
  PDecl->setPropertyAttributesAsWritten(
                          makePropertyAttributesAsWritten(AttributesAsWritten));
  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);
  if (Attributes & ObjCDeclSpec::DQ_PR_readwrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);
  // Set setter/getter selector name. Needed later.
  PDecl->setGetterName(GetterSel);
  PDecl->setSetterName(SetterSel);
  ProcessDeclAttributes(S, PDecl, FD.D);
  DC->addDecl(PDecl);

  // We need to look in the @interface to see if the @property was
  // already declared.
  if (!CCPrimary) {
    Diag(CDecl->getLocation(), diag::err_continuation_class);
    *isOverridingProperty = true;
    return 0;
  }

  // Find the property in continuation class's primary class only.
  ObjCPropertyDecl *PIDecl =
    CCPrimary->FindPropertyVisibleInPrimaryClass(PropertyId);

  if (!PIDecl) {
    // No matching property found in the primary class. Just fall thru
    // and add property to continuation class's primary class.
    ObjCPropertyDecl *PrimaryPDecl =
      CreatePropertyDecl(S, CCPrimary, AtLoc, LParenLoc,
                         FD, GetterSel, SetterSel, isAssign, isReadWrite,
                         Attributes,AttributesAsWritten, T, MethodImplKind, DC);

    // A case of continuation class adding a new property in the class. This
    // is not what it was meant for. However, gcc supports it and so should we.
    // Make sure setter/getters are declared here.
    ProcessPropertyDecl(PrimaryPDecl, CCPrimary, /* redeclaredProperty = */ 0,
                        /* lexicalDC = */ CDecl);
    PDecl->setGetterMethodDecl(PrimaryPDecl->getGetterMethodDecl());
    PDecl->setSetterMethodDecl(PrimaryPDecl->getSetterMethodDecl());
    if (ASTMutationListener *L = Context.getASTMutationListener())
      L->AddedObjCPropertyInClassExtension(PrimaryPDecl, /*OrigProp=*/0, CDecl);
    return PrimaryPDecl;
  }
  if (!Context.hasSameType(PIDecl->getType(), PDecl->getType())) {
    bool IncompatibleObjC = false;
    QualType ConvertedType;
    // Relax the strict type matching for property type in continuation class.
    // Allow property object type of continuation class to be different as long
    // as it narrows the object type in its primary class property. Note that
    // this conversion is safe only because the wider type is for a 'readonly'
    // property in primary class and 'narrowed' type for a 'readwrite' property
    // in continuation class.
    if (!isa<ObjCObjectPointerType>(PIDecl->getType()) ||
        !isa<ObjCObjectPointerType>(PDecl->getType()) ||
        (!isObjCPointerConversion(PDecl->getType(), PIDecl->getType(), 
                                  ConvertedType, IncompatibleObjC))
        || IncompatibleObjC) {
      Diag(AtLoc, 
          diag::err_type_mismatch_continuation_class) << PDecl->getType();
      Diag(PIDecl->getLocation(), diag::note_property_declare);
      return 0;
    }
  }
    
  // The property 'PIDecl's readonly attribute will be over-ridden
  // with continuation class's readwrite property attribute!
  unsigned PIkind = PIDecl->getPropertyAttributesAsWritten();
  if (isReadWrite && (PIkind & ObjCPropertyDecl::OBJC_PR_readonly)) {
    PIkind |= deduceWeakPropertyFromType(*this, PIDecl->getType());
    unsigned ClassExtensionMemoryModel = getOwnershipRule(Attributes);
    unsigned PrimaryClassMemoryModel = getOwnershipRule(PIkind);
    if (PrimaryClassMemoryModel && ClassExtensionMemoryModel &&
        (PrimaryClassMemoryModel != ClassExtensionMemoryModel)) {
      Diag(AtLoc, diag::warn_property_attr_mismatch);
      Diag(PIDecl->getLocation(), diag::note_property_declare);
    }
    DeclContext *DC = cast<DeclContext>(CCPrimary);
    if (!ObjCPropertyDecl::findPropertyDecl(DC,
                                 PIDecl->getDeclName().getAsIdentifierInfo())) {
      // Protocol is not in the primary class. Must build one for it.
      ObjCDeclSpec ProtocolPropertyODS;
      // FIXME. Assuming that ObjCDeclSpec::ObjCPropertyAttributeKind
      // and ObjCPropertyDecl::PropertyAttributeKind have identical
      // values.  Should consolidate both into one enum type.
      ProtocolPropertyODS.
      setPropertyAttributes((ObjCDeclSpec::ObjCPropertyAttributeKind)
                            PIkind);
      // Must re-establish the context from class extension to primary
      // class context.
      ContextRAII SavedContext(*this, CCPrimary);
      
      Decl *ProtocolPtrTy =
        ActOnProperty(S, AtLoc, LParenLoc, FD, ProtocolPropertyODS,
                      PIDecl->getGetterName(),
                      PIDecl->getSetterName(),
                      isOverridingProperty,
                      MethodImplKind,
                      /* lexicalDC = */ CDecl);
      PIDecl = cast<ObjCPropertyDecl>(ProtocolPtrTy);
    }
    PIDecl->makeitReadWriteAttribute();
    if (Attributes & ObjCDeclSpec::DQ_PR_retain)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);
    if (Attributes & ObjCDeclSpec::DQ_PR_strong)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);
    if (Attributes & ObjCDeclSpec::DQ_PR_copy)
      PIDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);
    PIDecl->setSetterName(SetterSel);
  } else {
    // Tailor the diagnostics for the common case where a readwrite
    // property is declared both in the @interface and the continuation.
    // This is a common error where the user often intended the original
    // declaration to be readonly.
    unsigned diag =
      (Attributes & ObjCDeclSpec::DQ_PR_readwrite) &&
      (PIkind & ObjCPropertyDecl::OBJC_PR_readwrite)
      ? diag::err_use_continuation_class_redeclaration_readwrite
      : diag::err_use_continuation_class;
    Diag(AtLoc, diag)
      << CCPrimary->getDeclName();
    Diag(PIDecl->getLocation(), diag::note_property_declare);
    return 0;
  }
  *isOverridingProperty = true;
  // Make sure setter decl is synthesized, and added to primary class's list.
  ProcessPropertyDecl(PIDecl, CCPrimary, PDecl, CDecl);
  PDecl->setGetterMethodDecl(PIDecl->getGetterMethodDecl());
  PDecl->setSetterMethodDecl(PIDecl->getSetterMethodDecl());
  if (ASTMutationListener *L = Context.getASTMutationListener())
    L->AddedObjCPropertyInClassExtension(PDecl, PIDecl, CDecl);
  return PDecl;
}

ObjCPropertyDecl *Sema::CreatePropertyDecl(Scope *S,
                                           ObjCContainerDecl *CDecl,
                                           SourceLocation AtLoc,
                                           SourceLocation LParenLoc,
                                           FieldDeclarator &FD,
                                           Selector GetterSel,
                                           Selector SetterSel,
                                           const bool isAssign,
                                           const bool isReadWrite,
                                           const unsigned Attributes,
                                           const unsigned AttributesAsWritten,
                                           TypeSourceInfo *TInfo,
                                           tok::ObjCKeywordKind MethodImplKind,
                                           DeclContext *lexicalDC){
  IdentifierInfo *PropertyId = FD.D.getIdentifier();
  QualType T = TInfo->getType();

  // Issue a warning if property is 'assign' as default and its object, which is
  // gc'able conforms to NSCopying protocol
  if (getLangOpts().getGC() != LangOptions::NonGC &&
      isAssign && !(Attributes & ObjCDeclSpec::DQ_PR_assign))
    if (const ObjCObjectPointerType *ObjPtrTy =
          T->getAs<ObjCObjectPointerType>()) {
      ObjCInterfaceDecl *IDecl = ObjPtrTy->getObjectType()->getInterface();
      if (IDecl)
        if (ObjCProtocolDecl* PNSCopying =
            LookupProtocol(&Context.Idents.get("NSCopying"), AtLoc))
          if (IDecl->ClassImplementsProtocol(PNSCopying, true))
            Diag(AtLoc, diag::warn_implements_nscopying) << PropertyId;
    }
  if (T->isObjCObjectType())
    Diag(FD.D.getIdentifierLoc(), diag::err_statically_allocated_object);

  DeclContext *DC = cast<DeclContext>(CDecl);
  ObjCPropertyDecl *PDecl = ObjCPropertyDecl::Create(Context, DC,
                                                     FD.D.getIdentifierLoc(),
                                                     PropertyId, AtLoc, LParenLoc, TInfo);

  if (ObjCPropertyDecl *prevDecl =
        ObjCPropertyDecl::findPropertyDecl(DC, PropertyId)) {
    Diag(PDecl->getLocation(), diag::err_duplicate_property);
    Diag(prevDecl->getLocation(), diag::note_property_declare);
    PDecl->setInvalidDecl();
  }
  else {
    DC->addDecl(PDecl);
    if (lexicalDC)
      PDecl->setLexicalDeclContext(lexicalDC);
  }

  if (T->isArrayType() || T->isFunctionType()) {
    Diag(AtLoc, diag::err_property_type) << T;
    PDecl->setInvalidDecl();
  }

  ProcessDeclAttributes(S, PDecl, FD.D);

  // Regardless of setter/getter attribute, we save the default getter/setter
  // selector names in anticipation of declaration of setter/getter methods.
  PDecl->setGetterName(GetterSel);
  PDecl->setSetterName(SetterSel);
  PDecl->setPropertyAttributesAsWritten(
                          makePropertyAttributesAsWritten(AttributesAsWritten));

  if (Attributes & ObjCDeclSpec::DQ_PR_readonly)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readonly);

  if (Attributes & ObjCDeclSpec::DQ_PR_getter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_getter);

  if (Attributes & ObjCDeclSpec::DQ_PR_setter)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_setter);

  if (isReadWrite)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_readwrite);

  if (Attributes & ObjCDeclSpec::DQ_PR_retain)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_retain);

  if (Attributes & ObjCDeclSpec::DQ_PR_strong)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);

  if (Attributes & ObjCDeclSpec::DQ_PR_weak)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_weak);

  if (Attributes & ObjCDeclSpec::DQ_PR_copy)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_copy);

  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_unsafe_unretained);

  if (isAssign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);

  // In the semantic attributes, one of nonatomic or atomic is always set.
  if (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_nonatomic);
  else
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_atomic);

  // 'unsafe_unretained' is alias for 'assign'.
  if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_assign);
  if (isAssign)
    PDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_unsafe_unretained);

  if (MethodImplKind == tok::objc_required)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Required);
  else if (MethodImplKind == tok::objc_optional)
    PDecl->setPropertyImplementation(ObjCPropertyDecl::Optional);

  return PDecl;
}

static void checkARCPropertyImpl(Sema &S, SourceLocation propertyImplLoc,
                                 ObjCPropertyDecl *property,
                                 ObjCIvarDecl *ivar) {
  if (property->isInvalidDecl() || ivar->isInvalidDecl()) return;

  QualType ivarType = ivar->getType();
  Qualifiers::ObjCLifetime ivarLifetime = ivarType.getObjCLifetime();

  // The lifetime implied by the property's attributes.
  Qualifiers::ObjCLifetime propertyLifetime =
    getImpliedARCOwnership(property->getPropertyAttributes(),
                           property->getType());

  // We're fine if they match.
  if (propertyLifetime == ivarLifetime) return;

  // These aren't valid lifetimes for object ivars;  don't diagnose twice.
  if (ivarLifetime == Qualifiers::OCL_None ||
      ivarLifetime == Qualifiers::OCL_Autoreleasing)
    return;

  // If the ivar is private, and it's implicitly __unsafe_unretained
  // becaues of its type, then pretend it was actually implicitly
  // __strong.  This is only sound because we're processing the
  // property implementation before parsing any method bodies.
  if (ivarLifetime == Qualifiers::OCL_ExplicitNone &&
      propertyLifetime == Qualifiers::OCL_Strong &&
      ivar->getAccessControl() == ObjCIvarDecl::Private) {
    SplitQualType split = ivarType.split();
    if (split.Quals.hasObjCLifetime()) {
      assert(ivarType->isObjCARCImplicitlyUnretainedType());
      split.Quals.setObjCLifetime(Qualifiers::OCL_Strong);
      ivarType = S.Context.getQualifiedType(split);
      ivar->setType(ivarType);
      return;
    }
  }

  switch (propertyLifetime) {
  case Qualifiers::OCL_Strong:
    S.Diag(propertyImplLoc, diag::err_arc_strong_property_ownership)
      << property->getDeclName()
      << ivar->getDeclName()
      << ivarLifetime;
    break;

  case Qualifiers::OCL_Weak:
    S.Diag(propertyImplLoc, diag::error_weak_property)
      << property->getDeclName()
      << ivar->getDeclName();
    break;

  case Qualifiers::OCL_ExplicitNone:
    S.Diag(propertyImplLoc, diag::err_arc_assign_property_ownership)
      << property->getDeclName()
      << ivar->getDeclName()
      << ((property->getPropertyAttributesAsWritten() 
           & ObjCPropertyDecl::OBJC_PR_assign) != 0);
    break;

  case Qualifiers::OCL_Autoreleasing:
    llvm_unreachable("properties cannot be autoreleasing");

  case Qualifiers::OCL_None:
    // Any other property should be ignored.
    return;
  }

  S.Diag(property->getLocation(), diag::note_property_declare);
}

/// setImpliedPropertyAttributeForReadOnlyProperty -
/// This routine evaludates life-time attributes for a 'readonly'
/// property with no known lifetime of its own, using backing
/// 'ivar's attribute, if any. If no backing 'ivar', property's
/// life-time is assumed 'strong'.
static void setImpliedPropertyAttributeForReadOnlyProperty(
              ObjCPropertyDecl *property, ObjCIvarDecl *ivar) {
  Qualifiers::ObjCLifetime propertyLifetime = 
    getImpliedARCOwnership(property->getPropertyAttributes(),
                           property->getType());
  if (propertyLifetime != Qualifiers::OCL_None)
    return;
  
  if (!ivar) {
    // if no backing ivar, make property 'strong'.
    property->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);
    return;
  }
  // property assumes owenership of backing ivar.
  QualType ivarType = ivar->getType();
  Qualifiers::ObjCLifetime ivarLifetime = ivarType.getObjCLifetime();
  if (ivarLifetime == Qualifiers::OCL_Strong)
    property->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);
  else if (ivarLifetime == Qualifiers::OCL_Weak)
    property->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_weak);
  return;
}

/// DiagnoseClassAndClassExtPropertyMismatch - diagnose inconsistant property
/// attribute declared in primary class and attributes overridden in any of its
/// class extensions.
static void
DiagnoseClassAndClassExtPropertyMismatch(Sema &S, ObjCInterfaceDecl *ClassDecl, 
                                         ObjCPropertyDecl *property) {
  unsigned Attributes = property->getPropertyAttributesAsWritten();
  bool warn = (Attributes & ObjCDeclSpec::DQ_PR_readonly);
  for (const ObjCCategoryDecl *CDecl = ClassDecl->getFirstClassExtension();
       CDecl; CDecl = CDecl->getNextClassExtension()) {
    ObjCPropertyDecl *ClassExtProperty = 0;
    for (ObjCContainerDecl::prop_iterator P = CDecl->prop_begin(),
         E = CDecl->prop_end(); P != E; ++P) {
      if ((*P)->getIdentifier() == property->getIdentifier()) {
        ClassExtProperty = *P;
        break;
      }
    }
    if (ClassExtProperty) {
      warn = false;
      unsigned classExtPropertyAttr = 
        ClassExtProperty->getPropertyAttributesAsWritten();
      // We are issuing the warning that we postponed because class extensions
      // can override readonly->readwrite and 'setter' attributes originally
      // placed on class's property declaration now make sense in the overridden
      // property.
      if (Attributes & ObjCDeclSpec::DQ_PR_readonly) {
        if (!classExtPropertyAttr ||
            (classExtPropertyAttr & 
              (ObjCDeclSpec::DQ_PR_readwrite|
               ObjCDeclSpec::DQ_PR_assign |
               ObjCDeclSpec::DQ_PR_unsafe_unretained |
               ObjCDeclSpec::DQ_PR_copy |
               ObjCDeclSpec::DQ_PR_retain |
               ObjCDeclSpec::DQ_PR_strong)))
          continue;
        warn = true;
        break;
      }
    }
  }
  if (warn) {
    unsigned setterAttrs = (ObjCDeclSpec::DQ_PR_assign |
                            ObjCDeclSpec::DQ_PR_unsafe_unretained |
                            ObjCDeclSpec::DQ_PR_copy |
                            ObjCDeclSpec::DQ_PR_retain |
                            ObjCDeclSpec::DQ_PR_strong);
    if (Attributes & setterAttrs) {
      const char * which =     
      (Attributes & ObjCDeclSpec::DQ_PR_assign) ?
      "assign" :
      (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) ?
      "unsafe_unretained" :
      (Attributes & ObjCDeclSpec::DQ_PR_copy) ?
      "copy" : 
      (Attributes & ObjCDeclSpec::DQ_PR_retain) ?
      "retain" : "strong";
      
      S.Diag(property->getLocation(), 
             diag::warn_objc_property_attr_mutually_exclusive)
      << "readonly" << which;
    }
  }
  
  
}

/// ActOnPropertyImplDecl - This routine performs semantic checks and
/// builds the AST node for a property implementation declaration; declared
/// as \@synthesize or \@dynamic.
///
Decl *Sema::ActOnPropertyImplDecl(Scope *S,
                                  SourceLocation AtLoc,
                                  SourceLocation PropertyLoc,
                                  bool Synthesize,
                                  IdentifierInfo *PropertyId,
                                  IdentifierInfo *PropertyIvar,
                                  SourceLocation PropertyIvarLoc) {
  ObjCContainerDecl *ClassImpDecl =
    dyn_cast<ObjCContainerDecl>(CurContext);
  // Make sure we have a context for the property implementation declaration.
  if (!ClassImpDecl) {
    Diag(AtLoc, diag::error_missing_property_context);
    return 0;
  }
  if (PropertyIvarLoc.isInvalid())
    PropertyIvarLoc = PropertyLoc;
  SourceLocation PropertyDiagLoc = PropertyLoc;
  if (PropertyDiagLoc.isInvalid())
    PropertyDiagLoc = ClassImpDecl->getLocStart();
  ObjCPropertyDecl *property = 0;
  ObjCInterfaceDecl* IDecl = 0;
  // Find the class or category class where this property must have
  // a declaration.
  ObjCImplementationDecl *IC = 0;
  ObjCCategoryImplDecl* CatImplClass = 0;
  if ((IC = dyn_cast<ObjCImplementationDecl>(ClassImpDecl))) {
    IDecl = IC->getClassInterface();
    // We always synthesize an interface for an implementation
    // without an interface decl. So, IDecl is always non-zero.
    assert(IDecl &&
           "ActOnPropertyImplDecl - @implementation without @interface");

    // Look for this property declaration in the @implementation's @interface
    property = IDecl->FindPropertyDeclaration(PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_property_decl) << IDecl->getDeclName();
      return 0;
    }
    unsigned PIkind = property->getPropertyAttributesAsWritten();
    if ((PIkind & (ObjCPropertyDecl::OBJC_PR_atomic |
                   ObjCPropertyDecl::OBJC_PR_nonatomic) ) == 0) {
      if (AtLoc.isValid())
        Diag(AtLoc, diag::warn_implicit_atomic_property);
      else
        Diag(IC->getLocation(), diag::warn_auto_implicit_atomic_property);
      Diag(property->getLocation(), diag::note_property_declare);
    }
    
    if (const ObjCCategoryDecl *CD =
        dyn_cast<ObjCCategoryDecl>(property->getDeclContext())) {
      if (!CD->IsClassExtension()) {
        Diag(PropertyLoc, diag::error_category_property) << CD->getDeclName();
        Diag(property->getLocation(), diag::note_property_declare);
        return 0;
      }
    }
    
    if (Synthesize&&
        (PIkind & ObjCPropertyDecl::OBJC_PR_readonly) &&
        property->hasAttr<IBOutletAttr>() &&
        !AtLoc.isValid()) {
      Diag(IC->getLocation(), diag::warn_auto_readonly_iboutlet_property);
      Diag(property->getLocation(), diag::note_property_declare);
      SourceLocation readonlyLoc;
      if (LocPropertyAttribute(Context, "readonly", 
                               property->getLParenLoc(), readonlyLoc)) {
        SourceLocation endLoc = 
          readonlyLoc.getLocWithOffset(strlen("readonly")-1);
        SourceRange ReadonlySourceRange(readonlyLoc, endLoc);
        Diag(property->getLocation(), 
             diag::note_auto_readonly_iboutlet_fixup_suggest) <<
        FixItHint::CreateReplacement(ReadonlySourceRange, "readwrite");
      }
    }
    
    DiagnoseClassAndClassExtPropertyMismatch(*this, IDecl, property);
        
  } else if ((CatImplClass = dyn_cast<ObjCCategoryImplDecl>(ClassImpDecl))) {
    if (Synthesize) {
      Diag(AtLoc, diag::error_synthesize_category_decl);
      return 0;
    }
    IDecl = CatImplClass->getClassInterface();
    if (!IDecl) {
      Diag(AtLoc, diag::error_missing_property_interface);
      return 0;
    }
    ObjCCategoryDecl *Category =
    IDecl->FindCategoryDeclaration(CatImplClass->getIdentifier());

    // If category for this implementation not found, it is an error which
    // has already been reported eralier.
    if (!Category)
      return 0;
    // Look for this property declaration in @implementation's category
    property = Category->FindPropertyDeclaration(PropertyId);
    if (!property) {
      Diag(PropertyLoc, diag::error_bad_category_property_decl)
      << Category->getDeclName();
      return 0;
    }
  } else {
    Diag(AtLoc, diag::error_bad_property_context);
    return 0;
  }
  ObjCIvarDecl *Ivar = 0;
  bool CompleteTypeErr = false;
  bool compat = true;
  // Check that we have a valid, previously declared ivar for @synthesize
  if (Synthesize) {
    // @synthesize
    if (!PropertyIvar)
      PropertyIvar = PropertyId;
    // Check that this is a previously declared 'ivar' in 'IDecl' interface
    ObjCInterfaceDecl *ClassDeclared;
    Ivar = IDecl->lookupInstanceVariable(PropertyIvar, ClassDeclared);
    QualType PropType = property->getType();
    QualType PropertyIvarType = PropType.getNonReferenceType();

    if (RequireCompleteType(PropertyDiagLoc, PropertyIvarType,
                            diag::err_incomplete_synthesized_property,
                            property->getDeclName())) {
      Diag(property->getLocation(), diag::note_property_declare);
      CompleteTypeErr = true;
    }

    if (getLangOpts().ObjCAutoRefCount &&
        (property->getPropertyAttributesAsWritten() &
         ObjCPropertyDecl::OBJC_PR_readonly) &&
        PropertyIvarType->isObjCRetainableType()) {
      setImpliedPropertyAttributeForReadOnlyProperty(property, Ivar);    
    }
    
    ObjCPropertyDecl::PropertyAttributeKind kind 
      = property->getPropertyAttributes();

    // Add GC __weak to the ivar type if the property is weak.
    if ((kind & ObjCPropertyDecl::OBJC_PR_weak) && 
        getLangOpts().getGC() != LangOptions::NonGC) {
      assert(!getLangOpts().ObjCAutoRefCount);
      if (PropertyIvarType.isObjCGCStrong()) {
        Diag(PropertyDiagLoc, diag::err_gc_weak_property_strong_type);
        Diag(property->getLocation(), diag::note_property_declare);
      } else {
        PropertyIvarType =
          Context.getObjCGCQualType(PropertyIvarType, Qualifiers::Weak);
      }
    }
    if (AtLoc.isInvalid()) {
      // Check when default synthesizing a property that there is 
      // an ivar matching property name and issue warning; since this
      // is the most common case of not using an ivar used for backing
      // property in non-default synthesis case.
      ObjCInterfaceDecl *ClassDeclared=0;
      ObjCIvarDecl *originalIvar = 
      IDecl->lookupInstanceVariable(property->getIdentifier(), 
                                    ClassDeclared);
      if (originalIvar) {
        Diag(PropertyDiagLoc, 
             diag::warn_autosynthesis_property_ivar_match)
        << PropertyId << (Ivar == 0) << PropertyIvar 
        << originalIvar->getIdentifier();
        Diag(property->getLocation(), diag::note_property_declare);
        Diag(originalIvar->getLocation(), diag::note_ivar_decl);
      }
    }
    
    if (!Ivar) {
      // In ARC, give the ivar a lifetime qualifier based on the
      // property attributes.
      if (getLangOpts().ObjCAutoRefCount &&
          !PropertyIvarType.getObjCLifetime() &&
          PropertyIvarType->isObjCRetainableType()) {

        // It's an error if we have to do this and the user didn't
        // explicitly write an ownership attribute on the property.
        if (!property->hasWrittenStorageAttribute() &&
            !(kind & ObjCPropertyDecl::OBJC_PR_strong)) {
          Diag(PropertyDiagLoc,
               diag::err_arc_objc_property_default_assign_on_object);
          Diag(property->getLocation(), diag::note_property_declare);
        } else {
          Qualifiers::ObjCLifetime lifetime =
            getImpliedARCOwnership(kind, PropertyIvarType);
          assert(lifetime && "no lifetime for property?");
          if (lifetime == Qualifiers::OCL_Weak) {
            bool err = false;
            if (const ObjCObjectPointerType *ObjT =
                PropertyIvarType->getAs<ObjCObjectPointerType>()) {
              const ObjCInterfaceDecl *ObjI = ObjT->getInterfaceDecl();
              if (ObjI && ObjI->isArcWeakrefUnavailable()) {
                Diag(PropertyDiagLoc, diag::err_arc_weak_unavailable_property);
                Diag(property->getLocation(), diag::note_property_declare);
                err = true;
              }
            }
            if (!err && !getLangOpts().ObjCARCWeak) {
              Diag(PropertyDiagLoc, diag::err_arc_weak_no_runtime);
              Diag(property->getLocation(), diag::note_property_declare);
            }
          }
          
          Qualifiers qs;
          qs.addObjCLifetime(lifetime);
          PropertyIvarType = Context.getQualifiedType(PropertyIvarType, qs);   
        }
      }

      if (kind & ObjCPropertyDecl::OBJC_PR_weak &&
          !getLangOpts().ObjCAutoRefCount &&
          getLangOpts().getGC() == LangOptions::NonGC) {
        Diag(PropertyDiagLoc, diag::error_synthesize_weak_non_arc_or_gc);
        Diag(property->getLocation(), diag::note_property_declare);
      }

      Ivar = ObjCIvarDecl::Create(Context, ClassImpDecl,
                                  PropertyIvarLoc,PropertyIvarLoc, PropertyIvar,
                                  PropertyIvarType, /*Dinfo=*/0,
                                  ObjCIvarDecl::Private,
                                  (Expr *)0, true);
      if (CompleteTypeErr)
        Ivar->setInvalidDecl();
      ClassImpDecl->addDecl(Ivar);
      IDecl->makeDeclVisibleInContext(Ivar);

      if (getLangOpts().ObjCRuntime.isFragile())
        Diag(PropertyDiagLoc, diag::error_missing_property_ivar_decl)
            << PropertyId;
      // Note! I deliberately want it to fall thru so, we have a
      // a property implementation and to avoid future warnings.
    } else if (getLangOpts().ObjCRuntime.isNonFragile() &&
               !declaresSameEntity(ClassDeclared, IDecl)) {
      Diag(PropertyDiagLoc, diag::error_ivar_in_superclass_use)
      << property->getDeclName() << Ivar->getDeclName()
      << ClassDeclared->getDeclName();
      Diag(Ivar->getLocation(), diag::note_previous_access_declaration)
      << Ivar << Ivar->getName();
      // Note! I deliberately want it to fall thru so more errors are caught.
    }
    property->setPropertyIvarDecl(Ivar);

    QualType IvarType = Context.getCanonicalType(Ivar->getType());

    // Check that type of property and its ivar are type compatible.
    if (!Context.hasSameType(PropertyIvarType, IvarType)) {
      if (isa<ObjCObjectPointerType>(PropertyIvarType) 
          && isa<ObjCObjectPointerType>(IvarType))
        compat =
          Context.canAssignObjCInterfaces(
                                  PropertyIvarType->getAs<ObjCObjectPointerType>(),
                                  IvarType->getAs<ObjCObjectPointerType>());
      else {
        compat = (CheckAssignmentConstraints(PropertyIvarLoc, PropertyIvarType,
                                             IvarType)
                    == Compatible);
      }
      if (!compat) {
        Diag(PropertyDiagLoc, diag::error_property_ivar_type)
          << property->getDeclName() << PropType
          << Ivar->getDeclName() << IvarType;
        Diag(Ivar->getLocation(), diag::note_ivar_decl);
        // Note! I deliberately want it to fall thru so, we have a
        // a property implementation and to avoid future warnings.
      }
      else {
        // FIXME! Rules for properties are somewhat different that those
        // for assignments. Use a new routine to consolidate all cases;
        // specifically for property redeclarations as well as for ivars.
        QualType lhsType =Context.getCanonicalType(PropertyIvarType).getUnqualifiedType();
        QualType rhsType =Context.getCanonicalType(IvarType).getUnqualifiedType();
        if (lhsType != rhsType &&
            lhsType->isArithmeticType()) {
          Diag(PropertyDiagLoc, diag::error_property_ivar_type)
            << property->getDeclName() << PropType
            << Ivar->getDeclName() << IvarType;
          Diag(Ivar->getLocation(), diag::note_ivar_decl);
          // Fall thru - see previous comment
        }
      }
      // __weak is explicit. So it works on Canonical type.
      if ((PropType.isObjCGCWeak() && !IvarType.isObjCGCWeak() &&
           getLangOpts().getGC() != LangOptions::NonGC)) {
        Diag(PropertyDiagLoc, diag::error_weak_property)
        << property->getDeclName() << Ivar->getDeclName();
        Diag(Ivar->getLocation(), diag::note_ivar_decl);
        // Fall thru - see previous comment
      }
      // Fall thru - see previous comment
      if ((property->getType()->isObjCObjectPointerType() ||
           PropType.isObjCGCStrong()) && IvarType.isObjCGCWeak() &&
          getLangOpts().getGC() != LangOptions::NonGC) {
        Diag(PropertyDiagLoc, diag::error_strong_property)
        << property->getDeclName() << Ivar->getDeclName();
        // Fall thru - see previous comment
      }
    }
    if (getLangOpts().ObjCAutoRefCount)
      checkARCPropertyImpl(*this, PropertyLoc, property, Ivar);
  } else if (PropertyIvar)
    // @dynamic
    Diag(PropertyDiagLoc, diag::error_dynamic_property_ivar_decl);
    
  assert (property && "ActOnPropertyImplDecl - property declaration missing");
  ObjCPropertyImplDecl *PIDecl =
  ObjCPropertyImplDecl::Create(Context, CurContext, AtLoc, PropertyLoc,
                               property,
                               (Synthesize ?
                                ObjCPropertyImplDecl::Synthesize
                                : ObjCPropertyImplDecl::Dynamic),
                               Ivar, PropertyIvarLoc);

  if (CompleteTypeErr || !compat)
    PIDecl->setInvalidDecl();

  if (ObjCMethodDecl *getterMethod = property->getGetterMethodDecl()) {
    getterMethod->createImplicitParams(Context, IDecl);
    if (getLangOpts().CPlusPlus && Synthesize && !CompleteTypeErr &&
        Ivar->getType()->isRecordType()) {
      // For Objective-C++, need to synthesize the AST for the IVAR object to be
      // returned by the getter as it must conform to C++'s copy-return rules.
      // FIXME. Eventually we want to do this for Objective-C as well.
      SynthesizedFunctionScope Scope(*this, getterMethod);
      ImplicitParamDecl *SelfDecl = getterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl, false, SelfDecl->getType(),
                                  VK_RValue, PropertyDiagLoc);
      MarkDeclRefReferenced(SelfExpr);
      Expr *IvarRefExpr =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), PropertyDiagLoc,
                                      SelfExpr, true, true);
      ExprResult Res = 
        PerformCopyInitialization(InitializedEntity::InitializeResult(
                                    PropertyDiagLoc,
                                    getterMethod->getResultType(),
                                    /*NRVO=*/false),
                                  PropertyDiagLoc,
                                  Owned(IvarRefExpr));
      if (!Res.isInvalid()) {
        Expr *ResExpr = Res.takeAs<Expr>();
        if (ResExpr)
          ResExpr = MaybeCreateExprWithCleanups(ResExpr);
        PIDecl->setGetterCXXConstructor(ResExpr);
      }
    }
    if (property->hasAttr<NSReturnsNotRetainedAttr>() &&
        !getterMethod->hasAttr<NSReturnsNotRetainedAttr>()) {
      Diag(getterMethod->getLocation(), 
           diag::warn_property_getter_owning_mismatch);
      Diag(property->getLocation(), diag::note_property_declare);
    }
  }
  if (ObjCMethodDecl *setterMethod = property->getSetterMethodDecl()) {
    setterMethod->createImplicitParams(Context, IDecl);
    if (getLangOpts().CPlusPlus && Synthesize && !CompleteTypeErr &&
        Ivar->getType()->isRecordType()) {
      // FIXME. Eventually we want to do this for Objective-C as well.
      SynthesizedFunctionScope Scope(*this, setterMethod);
      ImplicitParamDecl *SelfDecl = setterMethod->getSelfDecl();
      DeclRefExpr *SelfExpr = 
        new (Context) DeclRefExpr(SelfDecl, false, SelfDecl->getType(),
                                  VK_RValue, PropertyDiagLoc);
      MarkDeclRefReferenced(SelfExpr);
      Expr *lhs =
        new (Context) ObjCIvarRefExpr(Ivar, Ivar->getType(), PropertyDiagLoc,
                                      SelfExpr, true, true);
      ObjCMethodDecl::param_iterator P = setterMethod->param_begin();
      ParmVarDecl *Param = (*P);
      QualType T = Param->getType().getNonReferenceType();
      DeclRefExpr *rhs = new (Context) DeclRefExpr(Param, false, T,
                                                   VK_LValue, PropertyDiagLoc);
      MarkDeclRefReferenced(rhs);
      ExprResult Res = BuildBinOp(S, PropertyDiagLoc, 
                                  BO_Assign, lhs, rhs);
      if (property->getPropertyAttributes() & 
          ObjCPropertyDecl::OBJC_PR_atomic) {
        Expr *callExpr = Res.takeAs<Expr>();
        if (const CXXOperatorCallExpr *CXXCE = 
              dyn_cast_or_null<CXXOperatorCallExpr>(callExpr))
          if (const FunctionDecl *FuncDecl = CXXCE->getDirectCallee())
            if (!FuncDecl->isTrivial())
              if (property->getType()->isReferenceType()) {
                Diag(PropertyDiagLoc, 
                     diag::err_atomic_property_nontrivial_assign_op)
                    << property->getType();
                Diag(FuncDecl->getLocStart(), 
                     diag::note_callee_decl) << FuncDecl;
              }
      }
      PIDecl->setSetterCXXAssignment(Res.takeAs<Expr>());
    }
  }
  
  if (IC) {
    if (Synthesize)
      if (ObjCPropertyImplDecl *PPIDecl =
          IC->FindPropertyImplIvarDecl(PropertyIvar)) {
        Diag(PropertyLoc, diag::error_duplicate_ivar_use)
        << PropertyId << PPIDecl->getPropertyDecl()->getIdentifier()
        << PropertyIvar;
        Diag(PPIDecl->getLocation(), diag::note_previous_use);
      }

    if (ObjCPropertyImplDecl *PPIDecl
        = IC->FindPropertyImplDecl(PropertyId)) {
      Diag(PropertyLoc, diag::error_property_implemented) << PropertyId;
      Diag(PPIDecl->getLocation(), diag::note_previous_declaration);
      return 0;
    }
    IC->addPropertyImplementation(PIDecl);
    if (getLangOpts().ObjCDefaultSynthProperties &&
        getLangOpts().ObjCRuntime.isNonFragile() &&
        !IDecl->isObjCRequiresPropertyDefs()) {
      // Diagnose if an ivar was lazily synthesdized due to a previous
      // use and if 1) property is @dynamic or 2) property is synthesized
      // but it requires an ivar of different name.
      ObjCInterfaceDecl *ClassDeclared=0;
      ObjCIvarDecl *Ivar = 0;
      if (!Synthesize)
        Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      else {
        if (PropertyIvar && PropertyIvar != PropertyId)
          Ivar = IDecl->lookupInstanceVariable(PropertyId, ClassDeclared);
      }
      // Issue diagnostics only if Ivar belongs to current class.
      if (Ivar && Ivar->getSynthesize() && 
          declaresSameEntity(IC->getClassInterface(), ClassDeclared)) {
        Diag(Ivar->getLocation(), diag::err_undeclared_var_use) 
        << PropertyId;
        Ivar->setInvalidDecl();
      }
    }
  } else {
    if (Synthesize)
      if (ObjCPropertyImplDecl *PPIDecl =
          CatImplClass->FindPropertyImplIvarDecl(PropertyIvar)) {
        Diag(PropertyDiagLoc, diag::error_duplicate_ivar_use)
        << PropertyId << PPIDecl->getPropertyDecl()->getIdentifier()
        << PropertyIvar;
        Diag(PPIDecl->getLocation(), diag::note_previous_use);
      }

    if (ObjCPropertyImplDecl *PPIDecl =
        CatImplClass->FindPropertyImplDecl(PropertyId)) {
      Diag(PropertyDiagLoc, diag::error_property_implemented) << PropertyId;
      Diag(PPIDecl->getLocation(), diag::note_previous_declaration);
      return 0;
    }
    CatImplClass->addPropertyImplementation(PIDecl);
  }

  return PIDecl;
}

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// DiagnosePropertyMismatch - Compares two properties for their
/// attributes and types and warns on a variety of inconsistencies.
///
void
Sema::DiagnosePropertyMismatch(ObjCPropertyDecl *Property,
                               ObjCPropertyDecl *SuperProperty,
                               const IdentifierInfo *inheritedName) {
  ObjCPropertyDecl::PropertyAttributeKind CAttr =
  Property->getPropertyAttributes();
  ObjCPropertyDecl::PropertyAttributeKind SAttr =
  SuperProperty->getPropertyAttributes();
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_readonly)
      && (SAttr & ObjCPropertyDecl::OBJC_PR_readwrite))
    Diag(Property->getLocation(), diag::warn_readonly_property)
      << Property->getDeclName() << inheritedName;
  if ((CAttr & ObjCPropertyDecl::OBJC_PR_copy)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_copy))
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "copy" << inheritedName;
  else if (!(SAttr & ObjCPropertyDecl::OBJC_PR_readonly)){
    unsigned CAttrRetain = 
      (CAttr & 
       (ObjCPropertyDecl::OBJC_PR_retain | ObjCPropertyDecl::OBJC_PR_strong));
    unsigned SAttrRetain = 
      (SAttr & 
       (ObjCPropertyDecl::OBJC_PR_retain | ObjCPropertyDecl::OBJC_PR_strong));
    bool CStrong = (CAttrRetain != 0);
    bool SStrong = (SAttrRetain != 0);
    if (CStrong != SStrong)
      Diag(Property->getLocation(), diag::warn_property_attribute)
        << Property->getDeclName() << "retain (or strong)" << inheritedName;
  }

  if ((CAttr & ObjCPropertyDecl::OBJC_PR_nonatomic)
      != (SAttr & ObjCPropertyDecl::OBJC_PR_nonatomic))
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "atomic" << inheritedName;
  if (Property->getSetterName() != SuperProperty->getSetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "setter" << inheritedName;
  if (Property->getGetterName() != SuperProperty->getGetterName())
    Diag(Property->getLocation(), diag::warn_property_attribute)
      << Property->getDeclName() << "getter" << inheritedName;

  QualType LHSType =
    Context.getCanonicalType(SuperProperty->getType());
  QualType RHSType =
    Context.getCanonicalType(Property->getType());

  if (!Context.propertyTypesAreCompatible(LHSType, RHSType)) {
    // Do cases not handled in above.
    // FIXME. For future support of covariant property types, revisit this.
    bool IncompatibleObjC = false;
    QualType ConvertedType;
    if (!isObjCPointerConversion(RHSType, LHSType, 
                                 ConvertedType, IncompatibleObjC) ||
        IncompatibleObjC) {
        Diag(Property->getLocation(), diag::warn_property_types_are_incompatible)
        << Property->getType() << SuperProperty->getType() << inheritedName;
      Diag(SuperProperty->getLocation(), diag::note_property_declare);
    }
  }
}

bool Sema::DiagnosePropertyAccessorMismatch(ObjCPropertyDecl *property,
                                            ObjCMethodDecl *GetterMethod,
                                            SourceLocation Loc) {
  if (!GetterMethod)
    return false;
  QualType GetterType = GetterMethod->getResultType().getNonReferenceType();
  QualType PropertyIvarType = property->getType().getNonReferenceType();
  bool compat = Context.hasSameType(PropertyIvarType, GetterType);
  if (!compat) {
    if (isa<ObjCObjectPointerType>(PropertyIvarType) && 
        isa<ObjCObjectPointerType>(GetterType))
      compat =
        Context.canAssignObjCInterfaces(
                                      GetterType->getAs<ObjCObjectPointerType>(),
                                      PropertyIvarType->getAs<ObjCObjectPointerType>());
    else if (CheckAssignmentConstraints(Loc, GetterType, PropertyIvarType) 
              != Compatible) {
          Diag(Loc, diag::error_property_accessor_type)
            << property->getDeclName() << PropertyIvarType
            << GetterMethod->getSelector() << GetterType;
          Diag(GetterMethod->getLocation(), diag::note_declared_at);
          return true;
    } else {
      compat = true;
      QualType lhsType =Context.getCanonicalType(PropertyIvarType).getUnqualifiedType();
      QualType rhsType =Context.getCanonicalType(GetterType).getUnqualifiedType();
      if (lhsType != rhsType && lhsType->isArithmeticType())
        compat = false;
    }
  }
  
  if (!compat) {
    Diag(Loc, diag::warn_accessor_property_type_mismatch)
    << property->getDeclName()
    << GetterMethod->getSelector();
    Diag(GetterMethod->getLocation(), diag::note_declared_at);
    return true;
  }

  return false;
}

/// ComparePropertiesInBaseAndSuper - This routine compares property
/// declarations in base and its super class, if any, and issues
/// diagnostics in a variety of inconsistent situations.
///
void Sema::ComparePropertiesInBaseAndSuper(ObjCInterfaceDecl *IDecl) {
  ObjCInterfaceDecl *SDecl = IDecl->getSuperClass();
  if (!SDecl)
    return;
  // FIXME: O(N^2)
  for (ObjCInterfaceDecl::prop_iterator S = SDecl->prop_begin(),
       E = SDecl->prop_end(); S != E; ++S) {
    ObjCPropertyDecl *SuperPDecl = *S;
    // Does property in super class has declaration in current class?
    for (ObjCInterfaceDecl::prop_iterator I = IDecl->prop_begin(),
         E = IDecl->prop_end(); I != E; ++I) {
      ObjCPropertyDecl *PDecl = *I;
      if (SuperPDecl->getIdentifier() == PDecl->getIdentifier())
          DiagnosePropertyMismatch(PDecl, SuperPDecl,
                                   SDecl->getIdentifier());
    }
  }
}

/// MatchOneProtocolPropertiesInClass - This routine goes thru the list
/// of properties declared in a protocol and compares their attribute against
/// the same property declared in the class or category.
void
Sema::MatchOneProtocolPropertiesInClass(Decl *CDecl,
                                          ObjCProtocolDecl *PDecl) {
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);
  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "MatchOneProtocolPropertiesInClass");
    if (!CatDecl->IsClassExtension())
      for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
           E = PDecl->prop_end(); P != E; ++P) {
        ObjCPropertyDecl *Pr = *P;
        ObjCCategoryDecl::prop_iterator CP, CE;
        // Is this property already in  category's list of properties?
        for (CP = CatDecl->prop_begin(), CE = CatDecl->prop_end(); CP!=CE; ++CP)
          if (CP->getIdentifier() == Pr->getIdentifier())
            break;
        if (CP != CE)
          // Property protocol already exist in class. Diagnose any mismatch.
          DiagnosePropertyMismatch(*CP, Pr, PDecl->getIdentifier());
      }
    return;
  }
  for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
       E = PDecl->prop_end(); P != E; ++P) {
    ObjCPropertyDecl *Pr = *P;
    ObjCInterfaceDecl::prop_iterator CP, CE;
    // Is this property already in  class's list of properties?
    for (CP = IDecl->prop_begin(), CE = IDecl->prop_end(); CP != CE; ++CP)
      if (CP->getIdentifier() == Pr->getIdentifier())
        break;
    if (CP != CE)
      // Property protocol already exist in class. Diagnose any mismatch.
      DiagnosePropertyMismatch(*CP, Pr, PDecl->getIdentifier());
    }
}

/// CompareProperties - This routine compares properties
/// declared in 'ClassOrProtocol' objects (which can be a class or an
/// inherited protocol with the list of properties for class/category 'CDecl'
///
void Sema::CompareProperties(Decl *CDecl, Decl *ClassOrProtocol) {
  Decl *ClassDecl = ClassOrProtocol;
  ObjCInterfaceDecl *IDecl = dyn_cast_or_null<ObjCInterfaceDecl>(CDecl);

  if (!IDecl) {
    // Category
    ObjCCategoryDecl *CatDecl = static_cast<ObjCCategoryDecl*>(CDecl);
    assert (CatDecl && "CompareProperties");
    if (ObjCCategoryDecl *MDecl = dyn_cast<ObjCCategoryDecl>(ClassDecl)) {
      for (ObjCCategoryDecl::protocol_iterator P = MDecl->protocol_begin(),
           E = MDecl->protocol_end(); P != E; ++P)
      // Match properties of category with those of protocol (*P)
      MatchOneProtocolPropertiesInClass(CatDecl, *P);

      // Go thru the list of protocols for this category and recursively match
      // their properties with those in the category.
      for (ObjCCategoryDecl::protocol_iterator P = CatDecl->protocol_begin(),
           E = CatDecl->protocol_end(); P != E; ++P)
        CompareProperties(CatDecl, *P);
    } else {
      ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
      for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
           E = MD->protocol_end(); P != E; ++P)
        MatchOneProtocolPropertiesInClass(CatDecl, *P);
    }
    return;
  }

  if (ObjCInterfaceDecl *MDecl = dyn_cast<ObjCInterfaceDecl>(ClassDecl)) {
    for (ObjCInterfaceDecl::all_protocol_iterator
          P = MDecl->all_referenced_protocol_begin(),
          E = MDecl->all_referenced_protocol_end(); P != E; ++P)
      // Match properties of class IDecl with those of protocol (*P).
      MatchOneProtocolPropertiesInClass(IDecl, *P);

    // Go thru the list of protocols for this class and recursively match
    // their properties with those declared in the class.
    for (ObjCInterfaceDecl::all_protocol_iterator
          P = IDecl->all_referenced_protocol_begin(),
          E = IDecl->all_referenced_protocol_end(); P != E; ++P)
      CompareProperties(IDecl, *P);
  } else {
    ObjCProtocolDecl *MD = cast<ObjCProtocolDecl>(ClassDecl);
    for (ObjCProtocolDecl::protocol_iterator P = MD->protocol_begin(),
         E = MD->protocol_end(); P != E; ++P)
      MatchOneProtocolPropertiesInClass(IDecl, *P);
  }
}

/// isPropertyReadonly - Return true if property is readonly, by searching
/// for the property in the class and in its categories and implementations
///
bool Sema::isPropertyReadonly(ObjCPropertyDecl *PDecl,
                              ObjCInterfaceDecl *IDecl) {
  // by far the most common case.
  if (!PDecl->isReadOnly())
    return false;
  // Even if property is ready only, if interface has a user defined setter,
  // it is not considered read only.
  if (IDecl->getInstanceMethod(PDecl->getSetterName()))
    return false;

  // Main class has the property as 'readonly'. Must search
  // through the category list to see if the property's
  // attribute has been over-ridden to 'readwrite'.
  for (ObjCCategoryDecl *Category = IDecl->getCategoryList();
       Category; Category = Category->getNextClassCategory()) {
    // Even if property is ready only, if a category has a user defined setter,
    // it is not considered read only.
    if (Category->getInstanceMethod(PDecl->getSetterName()))
      return false;
    ObjCPropertyDecl *P =
      Category->FindPropertyDeclaration(PDecl->getIdentifier());
    if (P && !P->isReadOnly())
      return false;
  }

  // Also, check for definition of a setter method in the implementation if
  // all else failed.
  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(CurContext)) {
    if (ObjCImplementationDecl *IMD =
        dyn_cast<ObjCImplementationDecl>(OMD->getDeclContext())) {
      if (IMD->getInstanceMethod(PDecl->getSetterName()))
        return false;
    } else if (ObjCCategoryImplDecl *CIMD =
               dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext())) {
      if (CIMD->getInstanceMethod(PDecl->getSetterName()))
        return false;
    }
  }
  // Lastly, look through the implementation (if one is in scope).
  if (ObjCImplementationDecl *ImpDecl = IDecl->getImplementation())
    if (ImpDecl->getInstanceMethod(PDecl->getSetterName()))
      return false;
  // If all fails, look at the super class.
  if (ObjCInterfaceDecl *SIDecl = IDecl->getSuperClass())
    return isPropertyReadonly(PDecl, SIDecl);
  return true;
}

/// CollectImmediateProperties - This routine collects all properties in
/// the class and its conforming protocols; but not those it its super class.
void Sema::CollectImmediateProperties(ObjCContainerDecl *CDecl,
            ObjCContainerDecl::PropertyMap &PropMap,
            ObjCContainerDecl::PropertyMap &SuperPropMap) {
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    for (ObjCContainerDecl::prop_iterator P = IDecl->prop_begin(),
         E = IDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = *P;
      PropMap[Prop->getIdentifier()] = Prop;
    }
    // scan through class's protocols.
    for (ObjCInterfaceDecl::all_protocol_iterator
         PI = IDecl->all_referenced_protocol_begin(),
         E = IDecl->all_referenced_protocol_end(); PI != E; ++PI)
        CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
  if (ObjCCategoryDecl *CATDecl = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    if (!CATDecl->IsClassExtension())
      for (ObjCContainerDecl::prop_iterator P = CATDecl->prop_begin(),
           E = CATDecl->prop_end(); P != E; ++P) {
        ObjCPropertyDecl *Prop = *P;
        PropMap[Prop->getIdentifier()] = Prop;
      }
    // scan through class's protocols.
    for (ObjCCategoryDecl::protocol_iterator PI = CATDecl->protocol_begin(),
         E = CATDecl->protocol_end(); PI != E; ++PI)
      CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
  else if (ObjCProtocolDecl *PDecl = dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::prop_iterator P = PDecl->prop_begin(),
         E = PDecl->prop_end(); P != E; ++P) {
      ObjCPropertyDecl *Prop = *P;
      ObjCPropertyDecl *PropertyFromSuper = SuperPropMap[Prop->getIdentifier()];
      // Exclude property for protocols which conform to class's super-class, 
      // as super-class has to implement the property.
      if (!PropertyFromSuper || 
          PropertyFromSuper->getIdentifier() != Prop->getIdentifier()) {
        ObjCPropertyDecl *&PropEntry = PropMap[Prop->getIdentifier()];
        if (!PropEntry)
          PropEntry = Prop;
      }
    }
    // scan through protocol's protocols.
    for (ObjCProtocolDecl::protocol_iterator PI = PDecl->protocol_begin(),
         E = PDecl->protocol_end(); PI != E; ++PI)
      CollectImmediateProperties((*PI), PropMap, SuperPropMap);
  }
}

/// CollectSuperClassPropertyImplementations - This routine collects list of
/// properties to be implemented in super class(s) and also coming from their
/// conforming protocols.
static void CollectSuperClassPropertyImplementations(ObjCInterfaceDecl *CDecl,
                                    ObjCInterfaceDecl::PropertyMap &PropMap) {
  if (ObjCInterfaceDecl *SDecl = CDecl->getSuperClass()) {
    while (SDecl) {
      SDecl->collectPropertiesToImplement(PropMap);
      SDecl = SDecl->getSuperClass();
    }
  }
}

/// \brief Default synthesizes all properties which must be synthesized
/// in class's \@implementation.
void Sema::DefaultSynthesizeProperties(Scope *S, ObjCImplDecl* IMPDecl,
                                       ObjCInterfaceDecl *IDecl) {
  
  ObjCInterfaceDecl::PropertyMap PropMap;
  IDecl->collectPropertiesToImplement(PropMap);
  if (PropMap.empty())
    return;
  ObjCInterfaceDecl::PropertyMap SuperPropMap;
  CollectSuperClassPropertyImplementations(IDecl, SuperPropMap);
  
  for (ObjCInterfaceDecl::PropertyMap::iterator
       P = PropMap.begin(), E = PropMap.end(); P != E; ++P) {
    ObjCPropertyDecl *Prop = P->second;
    // If property to be implemented in the super class, ignore.
    if (SuperPropMap[Prop->getIdentifier()])
      continue;
    // Is there a matching property synthesize/dynamic?
    if (Prop->isInvalidDecl() ||
        Prop->getPropertyImplementation() == ObjCPropertyDecl::Optional ||
        IMPDecl->FindPropertyImplIvarDecl(Prop->getIdentifier()))
      continue;
    // Property may have been synthesized by user.
    if (IMPDecl->FindPropertyImplDecl(Prop->getIdentifier()))
      continue;
    if (IMPDecl->getInstanceMethod(Prop->getGetterName())) {
      if (Prop->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_readonly)
        continue;
      if (IMPDecl->getInstanceMethod(Prop->getSetterName()))
        continue;
    }
    if (isa<ObjCProtocolDecl>(Prop->getDeclContext())) {
      // We won't auto-synthesize properties declared in protocols.
      Diag(IMPDecl->getLocation(), 
           diag::warn_auto_synthesizing_protocol_property);
      Diag(Prop->getLocation(), diag::note_property_declare);
      continue;
    }

    // We use invalid SourceLocations for the synthesized ivars since they
    // aren't really synthesized at a particular location; they just exist.
    // Saying that they are located at the @implementation isn't really going
    // to help users.
    ObjCPropertyImplDecl *PIDecl = dyn_cast_or_null<ObjCPropertyImplDecl>(
      ActOnPropertyImplDecl(S, SourceLocation(), SourceLocation(),
                            true,
                            /* property = */ Prop->getIdentifier(),
                            /* ivar = */ Prop->getDefaultSynthIvarName(Context),
                            Prop->getLocation()));
    if (PIDecl) {
      Diag(Prop->getLocation(), diag::warn_missing_explicit_synthesis);
      Diag(IMPDecl->getLocation(), diag::note_while_in_implementation);
    }
  }
}

void Sema::DefaultSynthesizeProperties(Scope *S, Decl *D) {
  if (!LangOpts.ObjCDefaultSynthProperties || LangOpts.ObjCRuntime.isFragile())
    return;
  ObjCImplementationDecl *IC=dyn_cast_or_null<ObjCImplementationDecl>(D);
  if (!IC)
    return;
  if (ObjCInterfaceDecl* IDecl = IC->getClassInterface())
    if (!IDecl->isObjCRequiresPropertyDefs())
      DefaultSynthesizeProperties(S, IC, IDecl);
}

void Sema::DiagnoseUnimplementedProperties(Scope *S, ObjCImplDecl* IMPDecl,
                                      ObjCContainerDecl *CDecl,
                                      const SelectorSet &InsMap) {
  ObjCContainerDecl::PropertyMap SuperPropMap;
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(CDecl))
    CollectSuperClassPropertyImplementations(IDecl, SuperPropMap);
  
  ObjCContainerDecl::PropertyMap PropMap;
  CollectImmediateProperties(CDecl, PropMap, SuperPropMap);
  if (PropMap.empty())
    return;

  llvm::DenseSet<ObjCPropertyDecl *> PropImplMap;
  for (ObjCImplDecl::propimpl_iterator
       I = IMPDecl->propimpl_begin(),
       EI = IMPDecl->propimpl_end(); I != EI; ++I)
    PropImplMap.insert(I->getPropertyDecl());

  for (ObjCContainerDecl::PropertyMap::iterator
       P = PropMap.begin(), E = PropMap.end(); P != E; ++P) {
    ObjCPropertyDecl *Prop = P->second;
    // Is there a matching propery synthesize/dynamic?
    if (Prop->isInvalidDecl() ||
        Prop->getPropertyImplementation() == ObjCPropertyDecl::Optional ||
        PropImplMap.count(Prop) || Prop->hasAttr<UnavailableAttr>())
      continue;
    if (!InsMap.count(Prop->getGetterName())) {
      Diag(IMPDecl->getLocation(),
           isa<ObjCCategoryDecl>(CDecl) ?
            diag::warn_setter_getter_impl_required_in_category :
            diag::warn_setter_getter_impl_required)
      << Prop->getDeclName() << Prop->getGetterName();
      Diag(Prop->getLocation(),
           diag::note_property_declare);
      if (LangOpts.ObjCDefaultSynthProperties && LangOpts.ObjCRuntime.isNonFragile())
        if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(CDecl))
          if (const ObjCInterfaceDecl *RID = ID->isObjCRequiresPropertyDefs())
            Diag(RID->getLocation(), diag::note_suppressed_class_declare);
            
    }

    if (!Prop->isReadOnly() && !InsMap.count(Prop->getSetterName())) {
      Diag(IMPDecl->getLocation(),
           isa<ObjCCategoryDecl>(CDecl) ?
           diag::warn_setter_getter_impl_required_in_category :
           diag::warn_setter_getter_impl_required)
      << Prop->getDeclName() << Prop->getSetterName();
      Diag(Prop->getLocation(),
           diag::note_property_declare);
      if (LangOpts.ObjCDefaultSynthProperties && LangOpts.ObjCRuntime.isNonFragile())
        if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(CDecl))
          if (const ObjCInterfaceDecl *RID = ID->isObjCRequiresPropertyDefs())
            Diag(RID->getLocation(), diag::note_suppressed_class_declare);
    }
  }
}

void
Sema::AtomicPropertySetterGetterRules (ObjCImplDecl* IMPDecl,
                                       ObjCContainerDecl* IDecl) {
  // Rules apply in non-GC mode only
  if (getLangOpts().getGC() != LangOptions::NonGC)
    return;
  for (ObjCContainerDecl::prop_iterator I = IDecl->prop_begin(),
       E = IDecl->prop_end();
       I != E; ++I) {
    ObjCPropertyDecl *Property = *I;
    ObjCMethodDecl *GetterMethod = 0;
    ObjCMethodDecl *SetterMethod = 0;
    bool LookedUpGetterSetter = false;

    unsigned Attributes = Property->getPropertyAttributes();
    unsigned AttributesAsWritten = Property->getPropertyAttributesAsWritten();

    if (!(AttributesAsWritten & ObjCPropertyDecl::OBJC_PR_atomic) &&
        !(AttributesAsWritten & ObjCPropertyDecl::OBJC_PR_nonatomic)) {
      GetterMethod = IMPDecl->getInstanceMethod(Property->getGetterName());
      SetterMethod = IMPDecl->getInstanceMethod(Property->getSetterName());
      LookedUpGetterSetter = true;
      if (GetterMethod) {
        Diag(GetterMethod->getLocation(),
             diag::warn_default_atomic_custom_getter_setter)
          << Property->getIdentifier() << 0;
        Diag(Property->getLocation(), diag::note_property_declare);
      }
      if (SetterMethod) {
        Diag(SetterMethod->getLocation(),
             diag::warn_default_atomic_custom_getter_setter)
          << Property->getIdentifier() << 1;
        Diag(Property->getLocation(), diag::note_property_declare);
      }
    }

    // We only care about readwrite atomic property.
    if ((Attributes & ObjCPropertyDecl::OBJC_PR_nonatomic) ||
        !(Attributes & ObjCPropertyDecl::OBJC_PR_readwrite))
      continue;
    if (const ObjCPropertyImplDecl *PIDecl
         = IMPDecl->FindPropertyImplDecl(Property->getIdentifier())) {
      if (PIDecl->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
        continue;
      if (!LookedUpGetterSetter) {
        GetterMethod = IMPDecl->getInstanceMethod(Property->getGetterName());
        SetterMethod = IMPDecl->getInstanceMethod(Property->getSetterName());
        LookedUpGetterSetter = true;
      }
      if ((GetterMethod && !SetterMethod) || (!GetterMethod && SetterMethod)) {
        SourceLocation MethodLoc =
          (GetterMethod ? GetterMethod->getLocation()
                        : SetterMethod->getLocation());
        Diag(MethodLoc, diag::warn_atomic_property_rule)
          << Property->getIdentifier() << (GetterMethod != 0)
          << (SetterMethod != 0);
        // fixit stuff.
        if (!AttributesAsWritten) {
          if (Property->getLParenLoc().isValid()) {
            // @property () ... case.
            SourceRange PropSourceRange(Property->getAtLoc(), 
                                        Property->getLParenLoc());
            Diag(Property->getLocation(), diag::note_atomic_property_fixup_suggest) <<
              FixItHint::CreateReplacement(PropSourceRange, "@property (nonatomic");
          }
          else {
            //@property id etc.
            SourceLocation endLoc = 
              Property->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
            endLoc = endLoc.getLocWithOffset(-1);
            SourceRange PropSourceRange(Property->getAtLoc(), endLoc);
            Diag(Property->getLocation(), diag::note_atomic_property_fixup_suggest) <<
              FixItHint::CreateReplacement(PropSourceRange, "@property (nonatomic) ");
          }
        }
        else if (!(AttributesAsWritten & ObjCPropertyDecl::OBJC_PR_atomic)) {
          // @property () ... case.
          SourceLocation endLoc = Property->getLParenLoc();
          SourceRange PropSourceRange(Property->getAtLoc(), endLoc);
          Diag(Property->getLocation(), diag::note_atomic_property_fixup_suggest) <<
           FixItHint::CreateReplacement(PropSourceRange, "@property (nonatomic, ");
        }
        else
          Diag(MethodLoc, diag::note_atomic_property_fixup_suggest);
        Diag(Property->getLocation(), diag::note_property_declare);
      }
    }
  }
}

void Sema::DiagnoseOwningPropertyGetterSynthesis(const ObjCImplementationDecl *D) {
  if (getLangOpts().getGC() == LangOptions::GCOnly)
    return;

  for (ObjCImplementationDecl::propimpl_iterator
         i = D->propimpl_begin(), e = D->propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;
    if (PID->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      continue;
    
    const ObjCPropertyDecl *PD = PID->getPropertyDecl();
    if (PD && !PD->hasAttr<NSReturnsNotRetainedAttr>() &&
        !D->getInstanceMethod(PD->getGetterName())) {
      ObjCMethodDecl *method = PD->getGetterMethodDecl();
      if (!method)
        continue;
      ObjCMethodFamily family = method->getMethodFamily();
      if (family == OMF_alloc || family == OMF_copy ||
          family == OMF_mutableCopy || family == OMF_new) {
        if (getLangOpts().ObjCAutoRefCount)
          Diag(PID->getLocation(), diag::err_ownin_getter_rule);
        else
          Diag(PID->getLocation(), diag::warn_owning_getter_rule);
        Diag(PD->getLocation(), diag::note_property_declare);
      }
    }
  }
}

/// AddPropertyAttrs - Propagates attributes from a property to the
/// implicitly-declared getter or setter for that property.
static void AddPropertyAttrs(Sema &S, ObjCMethodDecl *PropertyMethod,
                             ObjCPropertyDecl *Property) {
  // Should we just clone all attributes over?
  for (Decl::attr_iterator A = Property->attr_begin(), 
                        AEnd = Property->attr_end(); 
       A != AEnd; ++A) {
    if (isa<DeprecatedAttr>(*A) || 
        isa<UnavailableAttr>(*A) || 
        isa<AvailabilityAttr>(*A))
      PropertyMethod->addAttr((*A)->clone(S.Context));
  }
}

/// ProcessPropertyDecl - Make sure that any user-defined setter/getter methods
/// have the property type and issue diagnostics if they don't.
/// Also synthesize a getter/setter method if none exist (and update the
/// appropriate lookup tables. FIXME: Should reconsider if adding synthesized
/// methods is the "right" thing to do.
void Sema::ProcessPropertyDecl(ObjCPropertyDecl *property,
                               ObjCContainerDecl *CD,
                               ObjCPropertyDecl *redeclaredProperty,
                               ObjCContainerDecl *lexicalDC) {

  ObjCMethodDecl *GetterMethod, *SetterMethod;

  GetterMethod = CD->getInstanceMethod(property->getGetterName());
  SetterMethod = CD->getInstanceMethod(property->getSetterName());
  DiagnosePropertyAccessorMismatch(property, GetterMethod,
                                   property->getLocation());

  if (SetterMethod) {
    ObjCPropertyDecl::PropertyAttributeKind CAttr =
      property->getPropertyAttributes();
    if ((!(CAttr & ObjCPropertyDecl::OBJC_PR_readonly)) &&
        Context.getCanonicalType(SetterMethod->getResultType()) !=
          Context.VoidTy)
      Diag(SetterMethod->getLocation(), diag::err_setter_type_void);
    if (SetterMethod->param_size() != 1 ||
        !Context.hasSameUnqualifiedType(
          (*SetterMethod->param_begin())->getType().getNonReferenceType(), 
          property->getType().getNonReferenceType())) {
      Diag(property->getLocation(),
           diag::warn_accessor_property_type_mismatch)
        << property->getDeclName()
        << SetterMethod->getSelector();
      Diag(SetterMethod->getLocation(), diag::note_declared_at);
    }
  }

  // Synthesize getter/setter methods if none exist.
  // Find the default getter and if one not found, add one.
  // FIXME: The synthesized property we set here is misleading. We almost always
  // synthesize these methods unless the user explicitly provided prototypes
  // (which is odd, but allowed). Sema should be typechecking that the
  // declarations jive in that situation (which it is not currently).
  if (!GetterMethod) {
    // No instance method of same name as property getter name was found.
    // Declare a getter method and add it to the list of methods
    // for this class.
    SourceLocation Loc = redeclaredProperty ? 
      redeclaredProperty->getLocation() :
      property->getLocation();

    GetterMethod = ObjCMethodDecl::Create(Context, Loc, Loc,
                             property->getGetterName(),
                             property->getType(), 0, CD, /*isInstance=*/true,
                             /*isVariadic=*/false, /*isPropertyAccessor=*/true,
                             /*isImplicitlyDeclared=*/true, /*isDefined=*/false,
                             (property->getPropertyImplementation() ==
                              ObjCPropertyDecl::Optional) ?
                             ObjCMethodDecl::Optional :
                             ObjCMethodDecl::Required);
    CD->addDecl(GetterMethod);

    AddPropertyAttrs(*this, GetterMethod, property);

    // FIXME: Eventually this shouldn't be needed, as the lexical context
    // and the real context should be the same.
    if (lexicalDC)
      GetterMethod->setLexicalDeclContext(lexicalDC);
    if (property->hasAttr<NSReturnsNotRetainedAttr>())
      GetterMethod->addAttr(
        ::new (Context) NSReturnsNotRetainedAttr(Loc, Context));
  } else
    // A user declared getter will be synthesize when @synthesize of
    // the property with the same name is seen in the @implementation
    GetterMethod->setPropertyAccessor(true);
  property->setGetterMethodDecl(GetterMethod);

  // Skip setter if property is read-only.
  if (!property->isReadOnly()) {
    // Find the default setter and if one not found, add one.
    if (!SetterMethod) {
      // No instance method of same name as property setter name was found.
      // Declare a setter method and add it to the list of methods
      // for this class.
      SourceLocation Loc = redeclaredProperty ? 
        redeclaredProperty->getLocation() :
        property->getLocation();

      SetterMethod =
        ObjCMethodDecl::Create(Context, Loc, Loc,
                               property->getSetterName(), Context.VoidTy, 0,
                               CD, /*isInstance=*/true, /*isVariadic=*/false,
                               /*isPropertyAccessor=*/true,
                               /*isImplicitlyDeclared=*/true,
                               /*isDefined=*/false,
                               (property->getPropertyImplementation() ==
                                ObjCPropertyDecl::Optional) ?
                                ObjCMethodDecl::Optional :
                                ObjCMethodDecl::Required);

      // Invent the arguments for the setter. We don't bother making a
      // nice name for the argument.
      ParmVarDecl *Argument = ParmVarDecl::Create(Context, SetterMethod,
                                                  Loc, Loc,
                                                  property->getIdentifier(),
                                    property->getType().getUnqualifiedType(),
                                                  /*TInfo=*/0,
                                                  SC_None,
                                                  SC_None,
                                                  0);
      SetterMethod->setMethodParams(Context, Argument,
                                    ArrayRef<SourceLocation>());

      AddPropertyAttrs(*this, SetterMethod, property);

      CD->addDecl(SetterMethod);
      // FIXME: Eventually this shouldn't be needed, as the lexical context
      // and the real context should be the same.
      if (lexicalDC)
        SetterMethod->setLexicalDeclContext(lexicalDC);
    } else
      // A user declared setter will be synthesize when @synthesize of
      // the property with the same name is seen in the @implementation
      SetterMethod->setPropertyAccessor(true);
    property->setSetterMethodDecl(SetterMethod);
  }
  // Add any synthesized methods to the global pool. This allows us to
  // handle the following, which is supported by GCC (and part of the design).
  //
  // @interface Foo
  // @property double bar;
  // @end
  //
  // void thisIsUnfortunate() {
  //   id foo;
  //   double bar = [foo bar];
  // }
  //
  if (GetterMethod)
    AddInstanceMethodToGlobalPool(GetterMethod);
  if (SetterMethod)
    AddInstanceMethodToGlobalPool(SetterMethod);

  ObjCInterfaceDecl *CurrentClass = dyn_cast<ObjCInterfaceDecl>(CD);
  if (!CurrentClass) {
    if (ObjCCategoryDecl *Cat = dyn_cast<ObjCCategoryDecl>(CD))
      CurrentClass = Cat->getClassInterface();
    else if (ObjCImplDecl *Impl = dyn_cast<ObjCImplDecl>(CD))
      CurrentClass = Impl->getClassInterface();
  }
  if (GetterMethod)
    CheckObjCMethodOverrides(GetterMethod, CurrentClass, Sema::RTC_Unknown);
  if (SetterMethod)
    CheckObjCMethodOverrides(SetterMethod, CurrentClass, Sema::RTC_Unknown);
}

void Sema::CheckObjCPropertyAttributes(Decl *PDecl,
                                       SourceLocation Loc,
                                       unsigned &Attributes,
                                       bool propertyInPrimaryClass) {
  // FIXME: Improve the reported location.
  if (!PDecl || PDecl->isInvalidDecl())
    return;

  ObjCPropertyDecl *PropertyDecl = cast<ObjCPropertyDecl>(PDecl);
  QualType PropertyTy = PropertyDecl->getType(); 

  if (getLangOpts().ObjCAutoRefCount &&
      (Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      PropertyTy->isObjCRetainableType()) {
    // 'readonly' property with no obvious lifetime.
    // its life time will be determined by its backing ivar.
    unsigned rel = (ObjCDeclSpec::DQ_PR_unsafe_unretained |
                    ObjCDeclSpec::DQ_PR_copy |
                    ObjCDeclSpec::DQ_PR_retain |
                    ObjCDeclSpec::DQ_PR_strong |
                    ObjCDeclSpec::DQ_PR_weak |
                    ObjCDeclSpec::DQ_PR_assign);
    if ((Attributes & rel) == 0)
      return;
  }
  
  if (propertyInPrimaryClass) {
    // we postpone most property diagnosis until class's implementation
    // because, its readonly attribute may be overridden in its class 
    // extensions making other attributes, which make no sense, to make sense.
    if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
        (Attributes & ObjCDeclSpec::DQ_PR_readwrite))
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive) 
        << "readonly" << "readwrite";
  }
  // readonly and readwrite/assign/retain/copy conflict.
  else if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
           (Attributes & (ObjCDeclSpec::DQ_PR_readwrite |
                     ObjCDeclSpec::DQ_PR_assign |
                     ObjCDeclSpec::DQ_PR_unsafe_unretained |
                     ObjCDeclSpec::DQ_PR_copy |
                     ObjCDeclSpec::DQ_PR_retain |
                     ObjCDeclSpec::DQ_PR_strong))) {
    const char * which = (Attributes & ObjCDeclSpec::DQ_PR_readwrite) ?
                          "readwrite" :
                         (Attributes & ObjCDeclSpec::DQ_PR_assign) ?
                          "assign" :
                         (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) ?
                          "unsafe_unretained" :
                         (Attributes & ObjCDeclSpec::DQ_PR_copy) ?
                          "copy" : "retain";

    Diag(Loc, (Attributes & (ObjCDeclSpec::DQ_PR_readwrite)) ?
                 diag::err_objc_property_attr_mutually_exclusive :
                 diag::warn_objc_property_attr_mutually_exclusive)
      << "readonly" << which;
  }

  // Check for copy or retain on non-object types.
  if ((Attributes & (ObjCDeclSpec::DQ_PR_weak | ObjCDeclSpec::DQ_PR_copy |
                    ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong)) &&
      !PropertyTy->isObjCRetainableType() &&
      !PropertyDecl->getAttr<ObjCNSObjectAttr>()) {
    Diag(Loc, diag::err_objc_property_requires_object)
      << (Attributes & ObjCDeclSpec::DQ_PR_weak ? "weak" :
          Attributes & ObjCDeclSpec::DQ_PR_copy ? "copy" : "retain (or strong)");
    Attributes &= ~(ObjCDeclSpec::DQ_PR_weak   | ObjCDeclSpec::DQ_PR_copy |
                    ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong);
    PropertyDecl->setInvalidDecl();
  }

  // Check for more than one of { assign, copy, retain }.
  if (Attributes & ObjCDeclSpec::DQ_PR_assign) {
    if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "copy";
      Attributes &= ~ObjCDeclSpec::DQ_PR_copy;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (getLangOpts().ObjCAutoRefCount  &&
        (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "assign" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  } else if (Attributes & ObjCDeclSpec::DQ_PR_unsafe_unretained) {
    if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "copy";
      Attributes &= ~ObjCDeclSpec::DQ_PR_copy;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (getLangOpts().ObjCAutoRefCount  &&
        (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "unsafe_unretained" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  } else if (Attributes & ObjCDeclSpec::DQ_PR_copy) {
    if (Attributes & ObjCDeclSpec::DQ_PR_retain) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "retain";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_strong) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "strong";
      Attributes &= ~ObjCDeclSpec::DQ_PR_strong;
    }
    if (Attributes & ObjCDeclSpec::DQ_PR_weak) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "copy" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
    }
  }
  else if ((Attributes & ObjCDeclSpec::DQ_PR_retain) &&
           (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "retain" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_retain;
  }
  else if ((Attributes & ObjCDeclSpec::DQ_PR_strong) &&
           (Attributes & ObjCDeclSpec::DQ_PR_weak)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "strong" << "weak";
      Attributes &= ~ObjCDeclSpec::DQ_PR_weak;
  }

  if ((Attributes & ObjCDeclSpec::DQ_PR_atomic) &&
      (Attributes & ObjCDeclSpec::DQ_PR_nonatomic)) {
      Diag(Loc, diag::err_objc_property_attr_mutually_exclusive)
        << "atomic" << "nonatomic";
      Attributes &= ~ObjCDeclSpec::DQ_PR_atomic;
  }

  // Warn if user supplied no assignment attribute, property is
  // readwrite, and this is an object type.
  if (!(Attributes & (ObjCDeclSpec::DQ_PR_assign | ObjCDeclSpec::DQ_PR_copy |
                      ObjCDeclSpec::DQ_PR_unsafe_unretained |
                      ObjCDeclSpec::DQ_PR_retain | ObjCDeclSpec::DQ_PR_strong |
                      ObjCDeclSpec::DQ_PR_weak)) &&
      PropertyTy->isObjCObjectPointerType()) {
      if (getLangOpts().ObjCAutoRefCount)
        // With arc,  @property definitions should default to (strong) when 
        // not specified; including when property is 'readonly'.
        PropertyDecl->setPropertyAttributes(ObjCPropertyDecl::OBJC_PR_strong);
      else if (!(Attributes & ObjCDeclSpec::DQ_PR_readonly)) {
        bool isAnyClassTy = 
          (PropertyTy->isObjCClassType() || 
           PropertyTy->isObjCQualifiedClassType());
        // In non-gc, non-arc mode, 'Class' is treated as a 'void *' no need to
        // issue any warning.
        if (isAnyClassTy && getLangOpts().getGC() == LangOptions::NonGC)
          ;
        else if (propertyInPrimaryClass) {
          // Don't issue warning on property with no life time in class 
          // extension as it is inherited from property in primary class.
          // Skip this warning in gc-only mode.
          if (getLangOpts().getGC() != LangOptions::GCOnly)
            Diag(Loc, diag::warn_objc_property_no_assignment_attribute);

          // If non-gc code warn that this is likely inappropriate.
          if (getLangOpts().getGC() == LangOptions::NonGC)
            Diag(Loc, diag::warn_objc_property_default_assign_on_object);
        }
      }

    // FIXME: Implement warning dependent on NSCopying being
    // implemented. See also:
    // <rdar://5168496&4855821&5607453&5096644&4947311&5698469&4947014&5168496>
    // (please trim this list while you are at it).
  }

  if (!(Attributes & ObjCDeclSpec::DQ_PR_copy)
      &&!(Attributes & ObjCDeclSpec::DQ_PR_readonly)
      && getLangOpts().getGC() == LangOptions::GCOnly
      && PropertyTy->isBlockPointerType())
    Diag(Loc, diag::warn_objc_property_copy_missing_on_block);
  else if ((Attributes & ObjCDeclSpec::DQ_PR_retain) &&
           !(Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
           !(Attributes & ObjCDeclSpec::DQ_PR_strong) &&
           PropertyTy->isBlockPointerType())
      Diag(Loc, diag::warn_objc_property_retain_of_block);
  
  if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      (Attributes & ObjCDeclSpec::DQ_PR_setter))
    Diag(Loc, diag::warn_objc_readonly_property_has_setter);
      
}
