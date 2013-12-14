//===--- SemaTemplateInstantiateDecl.cpp - C++ Template Decl Instantiation ===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation for declarations.
//
//===----------------------------------------------------------------------===/
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Template.h"

using namespace clang;

static bool isDeclWithinFunction(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();
  if (DC->isFunctionOrMethod())
    return true;

  if (DC->isRecord())
    return cast<CXXRecordDecl>(DC)->isLocalClass();

  return false;
}

bool TemplateDeclInstantiator::SubstQualifier(const DeclaratorDecl *OldDecl,
                                              DeclaratorDecl *NewDecl) {
  if (!OldDecl->getQualifierLoc())
    return false;

  NestedNameSpecifierLoc NewQualifierLoc
    = SemaRef.SubstNestedNameSpecifierLoc(OldDecl->getQualifierLoc(),
                                          TemplateArgs);

  if (!NewQualifierLoc)
    return true;

  NewDecl->setQualifierInfo(NewQualifierLoc);
  return false;
}

bool TemplateDeclInstantiator::SubstQualifier(const TagDecl *OldDecl,
                                              TagDecl *NewDecl) {
  if (!OldDecl->getQualifierLoc())
    return false;

  NestedNameSpecifierLoc NewQualifierLoc
  = SemaRef.SubstNestedNameSpecifierLoc(OldDecl->getQualifierLoc(),
                                        TemplateArgs);

  if (!NewQualifierLoc)
    return true;

  NewDecl->setQualifierInfo(NewQualifierLoc);
  return false;
}

// Include attribute instantiation code.
#include "clang/Sema/AttrTemplateInstantiate.inc"

static void instantiateDependentAlignedAttr(
    Sema &S, const MultiLevelTemplateArgumentList &TemplateArgs,
    const AlignedAttr *Aligned, Decl *New, bool IsPackExpansion) {
  if (Aligned->isAlignmentExpr()) {
    // The alignment expression is a constant expression.
    EnterExpressionEvaluationContext Unevaluated(S, Sema::ConstantEvaluated);
    ExprResult Result = S.SubstExpr(Aligned->getAlignmentExpr(), TemplateArgs);
    if (!Result.isInvalid())
      S.AddAlignedAttr(Aligned->getLocation(), New, Result.takeAs<Expr>(),
                       Aligned->getSpellingListIndex(), IsPackExpansion);
  } else {
    TypeSourceInfo *Result = S.SubstType(Aligned->getAlignmentType(),
                                         TemplateArgs, Aligned->getLocation(),
                                         DeclarationName());
    if (Result)
      S.AddAlignedAttr(Aligned->getLocation(), New, Result,
                       Aligned->getSpellingListIndex(), IsPackExpansion);
  }
}

static void instantiateDependentAlignedAttr(
    Sema &S, const MultiLevelTemplateArgumentList &TemplateArgs,
    const AlignedAttr *Aligned, Decl *New) {
  if (!Aligned->isPackExpansion()) {
    instantiateDependentAlignedAttr(S, TemplateArgs, Aligned, New, false);
    return;
  }

  SmallVector<UnexpandedParameterPack, 2> Unexpanded;
  if (Aligned->isAlignmentExpr())
    S.collectUnexpandedParameterPacks(Aligned->getAlignmentExpr(),
                                      Unexpanded);
  else
    S.collectUnexpandedParameterPacks(Aligned->getAlignmentType()->getTypeLoc(),
                                      Unexpanded);
  assert(!Unexpanded.empty() && "Pack expansion without parameter packs?");

  // Determine whether we can expand this attribute pack yet.
  bool Expand = true, RetainExpansion = false;
  Optional<unsigned> NumExpansions;
  // FIXME: Use the actual location of the ellipsis.
  SourceLocation EllipsisLoc = Aligned->getLocation();
  if (S.CheckParameterPacksForExpansion(EllipsisLoc, Aligned->getRange(),
                                        Unexpanded, TemplateArgs, Expand,
                                        RetainExpansion, NumExpansions))
    return;

  if (!Expand) {
    Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(S, -1);
    instantiateDependentAlignedAttr(S, TemplateArgs, Aligned, New, true);
  } else {
    for (unsigned I = 0; I != *NumExpansions; ++I) {
      Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(S, I);
      instantiateDependentAlignedAttr(S, TemplateArgs, Aligned, New, false);
    }
  }
}

void Sema::InstantiateAttrs(const MultiLevelTemplateArgumentList &TemplateArgs,
                            const Decl *Tmpl, Decl *New,
                            LateInstantiatedAttrVec *LateAttrs,
                            LocalInstantiationScope *OuterMostScope) {
  for (AttrVec::const_iterator i = Tmpl->attr_begin(), e = Tmpl->attr_end();
       i != e; ++i) {
    const Attr *TmplAttr = *i;

    // FIXME: This should be generalized to more than just the AlignedAttr.
    const AlignedAttr *Aligned = dyn_cast<AlignedAttr>(TmplAttr);
    if (Aligned && Aligned->isAlignmentDependent()) {
      instantiateDependentAlignedAttr(*this, TemplateArgs, Aligned, New);
      continue;
    }

    assert(!TmplAttr->isPackExpansion());
    if (TmplAttr->isLateParsed() && LateAttrs) {
      // Late parsed attributes must be instantiated and attached after the
      // enclosing class has been instantiated.  See Sema::InstantiateClass.
      LocalInstantiationScope *Saved = 0;
      if (CurrentInstantiationScope)
        Saved = CurrentInstantiationScope->cloneScopes(OuterMostScope);
      LateAttrs->push_back(LateInstantiatedAttribute(TmplAttr, Saved, New));
    } else {
      // Allow 'this' within late-parsed attributes.
      NamedDecl *ND = dyn_cast<NamedDecl>(New);
      CXXRecordDecl *ThisContext =
          dyn_cast_or_null<CXXRecordDecl>(ND->getDeclContext());
      CXXThisScopeRAII ThisScope(*this, ThisContext, /*TypeQuals*/0,
                                 ND && ND->isCXXInstanceMember());

      Attr *NewAttr = sema::instantiateTemplateAttribute(TmplAttr, Context,
                                                         *this, TemplateArgs);
      if (NewAttr)
        New->addAttr(NewAttr);
    }
  }
}

Decl *
TemplateDeclInstantiator::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  llvm_unreachable("Translation units cannot be instantiated");
}

Decl *
TemplateDeclInstantiator::VisitLabelDecl(LabelDecl *D) {
  LabelDecl *Inst = LabelDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                      D->getIdentifier());
  Owner->addDecl(Inst);
  return Inst;
}

Decl *
TemplateDeclInstantiator::VisitNamespaceDecl(NamespaceDecl *D) {
  llvm_unreachable("Namespaces cannot be instantiated");
}

Decl *
TemplateDeclInstantiator::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  NamespaceAliasDecl *Inst
    = NamespaceAliasDecl::Create(SemaRef.Context, Owner,
                                 D->getNamespaceLoc(),
                                 D->getAliasLoc(),
                                 D->getIdentifier(),
                                 D->getQualifierLoc(),
                                 D->getTargetNameLoc(),
                                 D->getNamespace());
  Owner->addDecl(Inst);
  return Inst;
}

Decl *TemplateDeclInstantiator::InstantiateTypedefNameDecl(TypedefNameDecl *D,
                                                           bool IsTypeAlias) {
  bool Invalid = false;
  TypeSourceInfo *DI = D->getTypeSourceInfo();
  if (DI->getType()->isInstantiationDependentType() ||
      DI->getType()->isVariablyModifiedType()) {
    DI = SemaRef.SubstType(DI, TemplateArgs,
                           D->getLocation(), D->getDeclName());
    if (!DI) {
      Invalid = true;
      DI = SemaRef.Context.getTrivialTypeSourceInfo(SemaRef.Context.IntTy);
    }
  } else {
    SemaRef.MarkDeclarationsReferencedInType(D->getLocation(), DI->getType());
  }

  // HACK: g++ has a bug where it gets the value kind of ?: wrong.
  // libstdc++ relies upon this bug in its implementation of common_type.
  // If we happen to be processing that implementation, fake up the g++ ?:
  // semantics. See LWG issue 2141 for more information on the bug.
  const DecltypeType *DT = DI->getType()->getAs<DecltypeType>();
  CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D->getDeclContext());
  if (DT && RD && isa<ConditionalOperator>(DT->getUnderlyingExpr()) &&
      DT->isReferenceType() &&
      RD->getEnclosingNamespaceContext() == SemaRef.getStdNamespace() &&
      RD->getIdentifier() && RD->getIdentifier()->isStr("common_type") &&
      D->getIdentifier() && D->getIdentifier()->isStr("type") &&
      SemaRef.getSourceManager().isInSystemHeader(D->getLocStart()))
    // Fold it to the (non-reference) type which g++ would have produced.
    DI = SemaRef.Context.getTrivialTypeSourceInfo(
      DI->getType().getNonReferenceType());

  // Create the new typedef
  TypedefNameDecl *Typedef;
  if (IsTypeAlias)
    Typedef = TypeAliasDecl::Create(SemaRef.Context, Owner, D->getLocStart(),
                                    D->getLocation(), D->getIdentifier(), DI);
  else
    Typedef = TypedefDecl::Create(SemaRef.Context, Owner, D->getLocStart(),
                                  D->getLocation(), D->getIdentifier(), DI);
  if (Invalid)
    Typedef->setInvalidDecl();

  // If the old typedef was the name for linkage purposes of an anonymous
  // tag decl, re-establish that relationship for the new typedef.
  if (const TagType *oldTagType = D->getUnderlyingType()->getAs<TagType>()) {
    TagDecl *oldTag = oldTagType->getDecl();
    if (oldTag->getTypedefNameForAnonDecl() == D && !Invalid) {
      TagDecl *newTag = DI->getType()->castAs<TagType>()->getDecl();
      assert(!newTag->hasNameForLinkage());
      newTag->setTypedefNameForAnonDecl(Typedef);
    }
  }

  if (TypedefNameDecl *Prev = D->getPreviousDecl()) {
    NamedDecl *InstPrev = SemaRef.FindInstantiatedDecl(D->getLocation(), Prev,
                                                       TemplateArgs);
    if (!InstPrev)
      return 0;

    TypedefNameDecl *InstPrevTypedef = cast<TypedefNameDecl>(InstPrev);

    // If the typedef types are not identical, reject them.
    SemaRef.isIncompatibleTypedef(InstPrevTypedef, Typedef);

    Typedef->setPreviousDecl(InstPrevTypedef);
  }

  SemaRef.InstantiateAttrs(TemplateArgs, D, Typedef);

  Typedef->setAccess(D->getAccess());

  return Typedef;
}

Decl *TemplateDeclInstantiator::VisitTypedefDecl(TypedefDecl *D) {
  Decl *Typedef = InstantiateTypedefNameDecl(D, /*IsTypeAlias=*/false);
  Owner->addDecl(Typedef);
  return Typedef;
}

Decl *TemplateDeclInstantiator::VisitTypeAliasDecl(TypeAliasDecl *D) {
  Decl *Typedef = InstantiateTypedefNameDecl(D, /*IsTypeAlias=*/true);
  Owner->addDecl(Typedef);
  return Typedef;
}

Decl *
TemplateDeclInstantiator::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  // Create a local instantiation scope for this type alias template, which
  // will contain the instantiations of the template parameters.
  LocalInstantiationScope Scope(SemaRef);

  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return 0;

  TypeAliasDecl *Pattern = D->getTemplatedDecl();

  TypeAliasTemplateDecl *PrevAliasTemplate = 0;
  if (Pattern->getPreviousDecl()) {
    DeclContext::lookup_result Found = Owner->lookup(Pattern->getDeclName());
    if (!Found.empty()) {
      PrevAliasTemplate = dyn_cast<TypeAliasTemplateDecl>(Found.front());
    }
  }

  TypeAliasDecl *AliasInst = cast_or_null<TypeAliasDecl>(
    InstantiateTypedefNameDecl(Pattern, /*IsTypeAlias=*/true));
  if (!AliasInst)
    return 0;

  TypeAliasTemplateDecl *Inst
    = TypeAliasTemplateDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                    D->getDeclName(), InstParams, AliasInst);
  if (PrevAliasTemplate)
    Inst->setPreviousDecl(PrevAliasTemplate);

  Inst->setAccess(D->getAccess());

  if (!PrevAliasTemplate)
    Inst->setInstantiatedFromMemberTemplate(D);

  Owner->addDecl(Inst);

  return Inst;
}

Decl *TemplateDeclInstantiator::VisitVarDecl(VarDecl *D) {
  return VisitVarDecl(D, /*InstantiatingVarTemplate=*/false);
}

Decl *TemplateDeclInstantiator::VisitVarDecl(VarDecl *D,
                                             bool InstantiatingVarTemplate) {

  // If this is the variable for an anonymous struct or union,
  // instantiate the anonymous struct/union type first.
  if (const RecordType *RecordTy = D->getType()->getAs<RecordType>())
    if (RecordTy->getDecl()->isAnonymousStructOrUnion())
      if (!VisitCXXRecordDecl(cast<CXXRecordDecl>(RecordTy->getDecl())))
        return 0;

  // Do substitution on the type of the declaration
  TypeSourceInfo *DI = SemaRef.SubstType(D->getTypeSourceInfo(),
                                         TemplateArgs,
                                         D->getTypeSpecStartLoc(),
                                         D->getDeclName());
  if (!DI)
    return 0;

  if (DI->getType()->isFunctionType()) {
    SemaRef.Diag(D->getLocation(), diag::err_variable_instantiates_to_function)
      << D->isStaticDataMember() << DI->getType();
    return 0;
  }

  DeclContext *DC = Owner;
  if (D->isLocalExternDecl())
    SemaRef.adjustContextForLocalExternDecl(DC);

  // Build the instantiated declaration.
  VarDecl *Var = VarDecl::Create(SemaRef.Context, DC, D->getInnerLocStart(),
                                 D->getLocation(), D->getIdentifier(),
                                 DI->getType(), DI, D->getStorageClass());

  // In ARC, infer 'retaining' for variables of retainable type.
  if (SemaRef.getLangOpts().ObjCAutoRefCount && 
      SemaRef.inferObjCARCLifetime(Var))
    Var->setInvalidDecl();

  // Substitute the nested name specifier, if any.
  if (SubstQualifier(D, Var))
    return 0;

  SemaRef.BuildVariableInstantiation(Var, D, TemplateArgs, LateAttrs, Owner,
                                     StartingScope, InstantiatingVarTemplate);
  return Var;
}

Decl *TemplateDeclInstantiator::VisitAccessSpecDecl(AccessSpecDecl *D) {
  AccessSpecDecl* AD
    = AccessSpecDecl::Create(SemaRef.Context, D->getAccess(), Owner,
                             D->getAccessSpecifierLoc(), D->getColonLoc());
  Owner->addHiddenDecl(AD);
  return AD;
}

Decl *TemplateDeclInstantiator::VisitFieldDecl(FieldDecl *D) {
  bool Invalid = false;
  TypeSourceInfo *DI = D->getTypeSourceInfo();
  if (DI->getType()->isInstantiationDependentType() ||
      DI->getType()->isVariablyModifiedType())  {
    DI = SemaRef.SubstType(DI, TemplateArgs,
                           D->getLocation(), D->getDeclName());
    if (!DI) {
      DI = D->getTypeSourceInfo();
      Invalid = true;
    } else if (DI->getType()->isFunctionType()) {
      // C++ [temp.arg.type]p3:
      //   If a declaration acquires a function type through a type
      //   dependent on a template-parameter and this causes a
      //   declaration that does not use the syntactic form of a
      //   function declarator to have function type, the program is
      //   ill-formed.
      SemaRef.Diag(D->getLocation(), diag::err_field_instantiates_to_function)
        << DI->getType();
      Invalid = true;
    }
  } else {
    SemaRef.MarkDeclarationsReferencedInType(D->getLocation(), DI->getType());
  }

  Expr *BitWidth = D->getBitWidth();
  if (Invalid)
    BitWidth = 0;
  else if (BitWidth) {
    // The bit-width expression is a constant expression.
    EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                                 Sema::ConstantEvaluated);

    ExprResult InstantiatedBitWidth
      = SemaRef.SubstExpr(BitWidth, TemplateArgs);
    if (InstantiatedBitWidth.isInvalid()) {
      Invalid = true;
      BitWidth = 0;
    } else
      BitWidth = InstantiatedBitWidth.takeAs<Expr>();
  }

  FieldDecl *Field = SemaRef.CheckFieldDecl(D->getDeclName(),
                                            DI->getType(), DI,
                                            cast<RecordDecl>(Owner),
                                            D->getLocation(),
                                            D->isMutable(),
                                            BitWidth,
                                            D->getInClassInitStyle(),
                                            D->getInnerLocStart(),
                                            D->getAccess(),
                                            0);
  if (!Field) {
    cast<Decl>(Owner)->setInvalidDecl();
    return 0;
  }

  SemaRef.InstantiateAttrs(TemplateArgs, D, Field, LateAttrs, StartingScope);

  if (Field->hasAttrs())
    SemaRef.CheckAlignasUnderalignment(Field);

  if (Invalid)
    Field->setInvalidDecl();

  if (!Field->getDeclName()) {
    // Keep track of where this decl came from.
    SemaRef.Context.setInstantiatedFromUnnamedFieldDecl(Field, D);
  }
  if (CXXRecordDecl *Parent= dyn_cast<CXXRecordDecl>(Field->getDeclContext())) {
    if (Parent->isAnonymousStructOrUnion() &&
        Parent->getRedeclContext()->isFunctionOrMethod())
      SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Field);
  }

  Field->setImplicit(D->isImplicit());
  Field->setAccess(D->getAccess());
  Owner->addDecl(Field);

  return Field;
}

Decl *TemplateDeclInstantiator::VisitMSPropertyDecl(MSPropertyDecl *D) {
  bool Invalid = false;
  TypeSourceInfo *DI = D->getTypeSourceInfo();

  if (DI->getType()->isVariablyModifiedType()) {
    SemaRef.Diag(D->getLocation(), diag::err_property_is_variably_modified)
    << D->getName();
    Invalid = true;
  } else if (DI->getType()->isInstantiationDependentType())  {
    DI = SemaRef.SubstType(DI, TemplateArgs,
                           D->getLocation(), D->getDeclName());
    if (!DI) {
      DI = D->getTypeSourceInfo();
      Invalid = true;
    } else if (DI->getType()->isFunctionType()) {
      // C++ [temp.arg.type]p3:
      //   If a declaration acquires a function type through a type
      //   dependent on a template-parameter and this causes a
      //   declaration that does not use the syntactic form of a
      //   function declarator to have function type, the program is
      //   ill-formed.
      SemaRef.Diag(D->getLocation(), diag::err_field_instantiates_to_function)
      << DI->getType();
      Invalid = true;
    }
  } else {
    SemaRef.MarkDeclarationsReferencedInType(D->getLocation(), DI->getType());
  }

  MSPropertyDecl *Property = new (SemaRef.Context)
      MSPropertyDecl(Owner, D->getLocation(),
                     D->getDeclName(), DI->getType(), DI,
                     D->getLocStart(),
                     D->getGetterId(), D->getSetterId());

  SemaRef.InstantiateAttrs(TemplateArgs, D, Property, LateAttrs,
                           StartingScope);

  if (Invalid)
    Property->setInvalidDecl();

  Property->setAccess(D->getAccess());
  Owner->addDecl(Property);

  return Property;
}

Decl *TemplateDeclInstantiator::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  NamedDecl **NamedChain =
    new (SemaRef.Context)NamedDecl*[D->getChainingSize()];

  int i = 0;
  for (IndirectFieldDecl::chain_iterator PI =
       D->chain_begin(), PE = D->chain_end();
       PI != PE; ++PI) {
    NamedDecl *Next = SemaRef.FindInstantiatedDecl(D->getLocation(), *PI,
                                              TemplateArgs);
    if (!Next)
      return 0;

    NamedChain[i++] = Next;
  }

  QualType T = cast<FieldDecl>(NamedChain[i-1])->getType();
  IndirectFieldDecl* IndirectField
    = IndirectFieldDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                D->getIdentifier(), T,
                                NamedChain, D->getChainingSize());


  IndirectField->setImplicit(D->isImplicit());
  IndirectField->setAccess(D->getAccess());
  Owner->addDecl(IndirectField);
  return IndirectField;
}

Decl *TemplateDeclInstantiator::VisitFriendDecl(FriendDecl *D) {
  // Handle friend type expressions by simply substituting template
  // parameters into the pattern type and checking the result.
  if (TypeSourceInfo *Ty = D->getFriendType()) {
    TypeSourceInfo *InstTy;
    // If this is an unsupported friend, don't bother substituting template
    // arguments into it. The actual type referred to won't be used by any
    // parts of Clang, and may not be valid for instantiating. Just use the
    // same info for the instantiated friend.
    if (D->isUnsupportedFriend()) {
      InstTy = Ty;
    } else {
      InstTy = SemaRef.SubstType(Ty, TemplateArgs,
                                 D->getLocation(), DeclarationName());
    }
    if (!InstTy)
      return 0;

    FriendDecl *FD = SemaRef.CheckFriendTypeDecl(D->getLocStart(),
                                                 D->getFriendLoc(), InstTy);
    if (!FD)
      return 0;

    FD->setAccess(AS_public);
    FD->setUnsupportedFriend(D->isUnsupportedFriend());
    Owner->addDecl(FD);
    return FD;
  }

  NamedDecl *ND = D->getFriendDecl();
  assert(ND && "friend decl must be a decl or a type!");

  // All of the Visit implementations for the various potential friend
  // declarations have to be carefully written to work for friend
  // objects, with the most important detail being that the target
  // decl should almost certainly not be placed in Owner.
  Decl *NewND = Visit(ND);
  if (!NewND) return 0;

  FriendDecl *FD =
    FriendDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                       cast<NamedDecl>(NewND), D->getFriendLoc());
  FD->setAccess(AS_public);
  FD->setUnsupportedFriend(D->isUnsupportedFriend());
  Owner->addDecl(FD);
  return FD;
}

Decl *TemplateDeclInstantiator::VisitStaticAssertDecl(StaticAssertDecl *D) {
  Expr *AssertExpr = D->getAssertExpr();

  // The expression in a static assertion is a constant expression.
  EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                               Sema::ConstantEvaluated);

  ExprResult InstantiatedAssertExpr
    = SemaRef.SubstExpr(AssertExpr, TemplateArgs);
  if (InstantiatedAssertExpr.isInvalid())
    return 0;

  return SemaRef.BuildStaticAssertDeclaration(D->getLocation(),
                                              InstantiatedAssertExpr.get(),
                                              D->getMessage(),
                                              D->getRParenLoc(),
                                              D->isFailed());
}

Decl *TemplateDeclInstantiator::VisitEnumDecl(EnumDecl *D) {
  EnumDecl *PrevDecl = 0;
  if (D->getPreviousDecl()) {
    NamedDecl *Prev = SemaRef.FindInstantiatedDecl(D->getLocation(),
                                                   D->getPreviousDecl(),
                                                   TemplateArgs);
    if (!Prev) return 0;
    PrevDecl = cast<EnumDecl>(Prev);
  }

  EnumDecl *Enum = EnumDecl::Create(SemaRef.Context, Owner, D->getLocStart(),
                                    D->getLocation(), D->getIdentifier(),
                                    PrevDecl, D->isScoped(),
                                    D->isScopedUsingClassTag(), D->isFixed());
  if (D->isFixed()) {
    if (TypeSourceInfo *TI = D->getIntegerTypeSourceInfo()) {
      // If we have type source information for the underlying type, it means it
      // has been explicitly set by the user. Perform substitution on it before
      // moving on.
      SourceLocation UnderlyingLoc = TI->getTypeLoc().getBeginLoc();
      TypeSourceInfo *NewTI = SemaRef.SubstType(TI, TemplateArgs, UnderlyingLoc,
                                                DeclarationName());
      if (!NewTI || SemaRef.CheckEnumUnderlyingType(NewTI))
        Enum->setIntegerType(SemaRef.Context.IntTy);
      else
        Enum->setIntegerTypeSourceInfo(NewTI);
    } else {
      assert(!D->getIntegerType()->isDependentType()
             && "Dependent type without type source info");
      Enum->setIntegerType(D->getIntegerType());
    }
  }

  SemaRef.InstantiateAttrs(TemplateArgs, D, Enum);

  Enum->setInstantiationOfMemberEnum(D, TSK_ImplicitInstantiation);
  Enum->setAccess(D->getAccess());
  if (SubstQualifier(D, Enum)) return 0;
  Owner->addDecl(Enum);

  EnumDecl *Def = D->getDefinition();
  if (Def && Def != D) {
    // If this is an out-of-line definition of an enum member template, check
    // that the underlying types match in the instantiation of both
    // declarations.
    if (TypeSourceInfo *TI = Def->getIntegerTypeSourceInfo()) {
      SourceLocation UnderlyingLoc = TI->getTypeLoc().getBeginLoc();
      QualType DefnUnderlying =
        SemaRef.SubstType(TI->getType(), TemplateArgs,
                          UnderlyingLoc, DeclarationName());
      SemaRef.CheckEnumRedeclaration(Def->getLocation(), Def->isScoped(),
                                     DefnUnderlying, Enum);
    }
  }

  // C++11 [temp.inst]p1: The implicit instantiation of a class template
  // specialization causes the implicit instantiation of the declarations, but
  // not the definitions of scoped member enumerations.
  //
  // DR1484 clarifies that enumeration definitions inside of a template
  // declaration aren't considered entities that can be separately instantiated
  // from the rest of the entity they are declared inside of.
  if (isDeclWithinFunction(D) ? D == Def : Def && !Enum->isScoped()) {
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Enum);
    InstantiateEnumDefinition(Enum, Def);
  }

  return Enum;
}

void TemplateDeclInstantiator::InstantiateEnumDefinition(
    EnumDecl *Enum, EnumDecl *Pattern) {
  Enum->startDefinition();

  // Update the location to refer to the definition.
  Enum->setLocation(Pattern->getLocation());

  SmallVector<Decl*, 4> Enumerators;

  EnumConstantDecl *LastEnumConst = 0;
  for (EnumDecl::enumerator_iterator EC = Pattern->enumerator_begin(),
         ECEnd = Pattern->enumerator_end();
       EC != ECEnd; ++EC) {
    // The specified value for the enumerator.
    ExprResult Value = SemaRef.Owned((Expr *)0);
    if (Expr *UninstValue = EC->getInitExpr()) {
      // The enumerator's value expression is a constant expression.
      EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                                   Sema::ConstantEvaluated);

      Value = SemaRef.SubstExpr(UninstValue, TemplateArgs);
    }

    // Drop the initial value and continue.
    bool isInvalid = false;
    if (Value.isInvalid()) {
      Value = SemaRef.Owned((Expr *)0);
      isInvalid = true;
    }

    EnumConstantDecl *EnumConst
      = SemaRef.CheckEnumConstant(Enum, LastEnumConst,
                                  EC->getLocation(), EC->getIdentifier(),
                                  Value.get());

    if (isInvalid) {
      if (EnumConst)
        EnumConst->setInvalidDecl();
      Enum->setInvalidDecl();
    }

    if (EnumConst) {
      SemaRef.InstantiateAttrs(TemplateArgs, *EC, EnumConst);

      EnumConst->setAccess(Enum->getAccess());
      Enum->addDecl(EnumConst);
      Enumerators.push_back(EnumConst);
      LastEnumConst = EnumConst;

      if (Pattern->getDeclContext()->isFunctionOrMethod() &&
          !Enum->isScoped()) {
        // If the enumeration is within a function or method, record the enum
        // constant as a local.
        SemaRef.CurrentInstantiationScope->InstantiatedLocal(*EC, EnumConst);
      }
    }
  }

  // FIXME: Fixup LBraceLoc
  SemaRef.ActOnEnumBody(Enum->getLocation(), SourceLocation(),
                        Enum->getRBraceLoc(), Enum,
                        Enumerators,
                        0, 0);
}

Decl *TemplateDeclInstantiator::VisitEnumConstantDecl(EnumConstantDecl *D) {
  llvm_unreachable("EnumConstantDecls can only occur within EnumDecls.");
}

Decl *TemplateDeclInstantiator::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  bool isFriend = (D->getFriendObjectKind() != Decl::FOK_None);

  // Create a local instantiation scope for this class template, which
  // will contain the instantiations of the template parameters.
  LocalInstantiationScope Scope(SemaRef);
  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return NULL;

  CXXRecordDecl *Pattern = D->getTemplatedDecl();

  // Instantiate the qualifier.  We have to do this first in case
  // we're a friend declaration, because if we are then we need to put
  // the new declaration in the appropriate context.
  NestedNameSpecifierLoc QualifierLoc = Pattern->getQualifierLoc();
  if (QualifierLoc) {
    QualifierLoc = SemaRef.SubstNestedNameSpecifierLoc(QualifierLoc,
                                                       TemplateArgs);
    if (!QualifierLoc)
      return 0;
  }

  CXXRecordDecl *PrevDecl = 0;
  ClassTemplateDecl *PrevClassTemplate = 0;

  if (!isFriend && Pattern->getPreviousDecl()) {
    DeclContext::lookup_result Found = Owner->lookup(Pattern->getDeclName());
    if (!Found.empty()) {
      PrevClassTemplate = dyn_cast<ClassTemplateDecl>(Found.front());
      if (PrevClassTemplate)
        PrevDecl = PrevClassTemplate->getTemplatedDecl();
    }
  }

  // If this isn't a friend, then it's a member template, in which
  // case we just want to build the instantiation in the
  // specialization.  If it is a friend, we want to build it in
  // the appropriate context.
  DeclContext *DC = Owner;
  if (isFriend) {
    if (QualifierLoc) {
      CXXScopeSpec SS;
      SS.Adopt(QualifierLoc);
      DC = SemaRef.computeDeclContext(SS);
      if (!DC) return 0;
    } else {
      DC = SemaRef.FindInstantiatedContext(Pattern->getLocation(),
                                           Pattern->getDeclContext(),
                                           TemplateArgs);
    }

    // Look for a previous declaration of the template in the owning
    // context.
    LookupResult R(SemaRef, Pattern->getDeclName(), Pattern->getLocation(),
                   Sema::LookupOrdinaryName, Sema::ForRedeclaration);
    SemaRef.LookupQualifiedName(R, DC);

    if (R.isSingleResult()) {
      PrevClassTemplate = R.getAsSingle<ClassTemplateDecl>();
      if (PrevClassTemplate)
        PrevDecl = PrevClassTemplate->getTemplatedDecl();
    }

    if (!PrevClassTemplate && QualifierLoc) {
      SemaRef.Diag(Pattern->getLocation(), diag::err_not_tag_in_scope)
        << D->getTemplatedDecl()->getTagKind() << Pattern->getDeclName() << DC
        << QualifierLoc.getSourceRange();
      return 0;
    }

    bool AdoptedPreviousTemplateParams = false;
    if (PrevClassTemplate) {
      bool Complain = true;

      // HACK: libstdc++ 4.2.1 contains an ill-formed friend class
      // template for struct std::tr1::__detail::_Map_base, where the
      // template parameters of the friend declaration don't match the
      // template parameters of the original declaration. In this one
      // case, we don't complain about the ill-formed friend
      // declaration.
      if (isFriend && Pattern->getIdentifier() &&
          Pattern->getIdentifier()->isStr("_Map_base") &&
          DC->isNamespace() &&
          cast<NamespaceDecl>(DC)->getIdentifier() &&
          cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__detail")) {
        DeclContext *DCParent = DC->getParent();
        if (DCParent->isNamespace() &&
            cast<NamespaceDecl>(DCParent)->getIdentifier() &&
            cast<NamespaceDecl>(DCParent)->getIdentifier()->isStr("tr1")) {
          DeclContext *DCParent2 = DCParent->getParent();
          if (DCParent2->isNamespace() &&
              cast<NamespaceDecl>(DCParent2)->getIdentifier() &&
              cast<NamespaceDecl>(DCParent2)->getIdentifier()->isStr("std") &&
              DCParent2->getParent()->isTranslationUnit())
            Complain = false;
        }
      }

      TemplateParameterList *PrevParams
        = PrevClassTemplate->getTemplateParameters();

      // Make sure the parameter lists match.
      if (!SemaRef.TemplateParameterListsAreEqual(InstParams, PrevParams,
                                                  Complain,
                                                  Sema::TPL_TemplateMatch)) {
        if (Complain)
          return 0;

        AdoptedPreviousTemplateParams = true;
        InstParams = PrevParams;
      }

      // Do some additional validation, then merge default arguments
      // from the existing declarations.
      if (!AdoptedPreviousTemplateParams &&
          SemaRef.CheckTemplateParameterList(InstParams, PrevParams,
                                             Sema::TPC_ClassTemplate))
        return 0;
    }
  }

  CXXRecordDecl *RecordInst
    = CXXRecordDecl::Create(SemaRef.Context, Pattern->getTagKind(), DC,
                            Pattern->getLocStart(), Pattern->getLocation(),
                            Pattern->getIdentifier(), PrevDecl,
                            /*DelayTypeCreation=*/true);

  if (QualifierLoc)
    RecordInst->setQualifierInfo(QualifierLoc);

  ClassTemplateDecl *Inst
    = ClassTemplateDecl::Create(SemaRef.Context, DC, D->getLocation(),
                                D->getIdentifier(), InstParams, RecordInst,
                                PrevClassTemplate);
  RecordInst->setDescribedClassTemplate(Inst);

  if (isFriend) {
    if (PrevClassTemplate)
      Inst->setAccess(PrevClassTemplate->getAccess());
    else
      Inst->setAccess(D->getAccess());

    Inst->setObjectOfFriendDecl();
    // TODO: do we want to track the instantiation progeny of this
    // friend target decl?
  } else {
    Inst->setAccess(D->getAccess());
    if (!PrevClassTemplate)
      Inst->setInstantiatedFromMemberTemplate(D);
  }

  // Trigger creation of the type for the instantiation.
  SemaRef.Context.getInjectedClassNameType(RecordInst,
                                    Inst->getInjectedClassNameSpecialization());

  // Finish handling of friends.
  if (isFriend) {
    DC->makeDeclVisibleInContext(Inst);
    Inst->setLexicalDeclContext(Owner);
    RecordInst->setLexicalDeclContext(Owner);
    return Inst;
  }

  if (D->isOutOfLine()) {
    Inst->setLexicalDeclContext(D->getLexicalDeclContext());
    RecordInst->setLexicalDeclContext(D->getLexicalDeclContext());
  }

  Owner->addDecl(Inst);

  if (!PrevClassTemplate) {
    // Queue up any out-of-line partial specializations of this member
    // class template; the client will force their instantiation once
    // the enclosing class has been instantiated.
    SmallVector<ClassTemplatePartialSpecializationDecl *, 4> PartialSpecs;
    D->getPartialSpecializations(PartialSpecs);
    for (unsigned I = 0, N = PartialSpecs.size(); I != N; ++I)
      if (PartialSpecs[I]->getFirstDecl()->isOutOfLine())
        OutOfLinePartialSpecs.push_back(std::make_pair(Inst, PartialSpecs[I]));
  }

  return Inst;
}

Decl *
TemplateDeclInstantiator::VisitClassTemplatePartialSpecializationDecl(
                                   ClassTemplatePartialSpecializationDecl *D) {
  ClassTemplateDecl *ClassTemplate = D->getSpecializedTemplate();

  // Lookup the already-instantiated declaration in the instantiation
  // of the class template and return that.
  DeclContext::lookup_result Found
    = Owner->lookup(ClassTemplate->getDeclName());
  if (Found.empty())
    return 0;

  ClassTemplateDecl *InstClassTemplate
    = dyn_cast<ClassTemplateDecl>(Found.front());
  if (!InstClassTemplate)
    return 0;

  if (ClassTemplatePartialSpecializationDecl *Result
        = InstClassTemplate->findPartialSpecInstantiatedFromMember(D))
    return Result;

  return InstantiateClassTemplatePartialSpecialization(InstClassTemplate, D);
}

Decl *TemplateDeclInstantiator::VisitVarTemplateDecl(VarTemplateDecl *D) {
  assert(D->getTemplatedDecl()->isStaticDataMember() &&
         "Only static data member templates are allowed.");

  // Create a local instantiation scope for this variable template, which
  // will contain the instantiations of the template parameters.
  LocalInstantiationScope Scope(SemaRef);
  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return NULL;

  VarDecl *Pattern = D->getTemplatedDecl();
  VarTemplateDecl *PrevVarTemplate = 0;

  if (Pattern->getPreviousDecl()) {
    DeclContext::lookup_result Found = Owner->lookup(Pattern->getDeclName());
    if (!Found.empty())
      PrevVarTemplate = dyn_cast<VarTemplateDecl>(Found.front());
  }

  VarDecl *VarInst =
      cast_or_null<VarDecl>(VisitVarDecl(Pattern,
                                         /*InstantiatingVarTemplate=*/true));

  DeclContext *DC = Owner;

  VarTemplateDecl *Inst = VarTemplateDecl::Create(
      SemaRef.Context, DC, D->getLocation(), D->getIdentifier(), InstParams,
      VarInst, PrevVarTemplate);
  VarInst->setDescribedVarTemplate(Inst);

  Inst->setAccess(D->getAccess());
  if (!PrevVarTemplate)
    Inst->setInstantiatedFromMemberTemplate(D);

  if (D->isOutOfLine()) {
    Inst->setLexicalDeclContext(D->getLexicalDeclContext());
    VarInst->setLexicalDeclContext(D->getLexicalDeclContext());
  }

  Owner->addDecl(Inst);

  if (!PrevVarTemplate) {
    // Queue up any out-of-line partial specializations of this member
    // variable template; the client will force their instantiation once
    // the enclosing class has been instantiated.
    SmallVector<VarTemplatePartialSpecializationDecl *, 4> PartialSpecs;
    D->getPartialSpecializations(PartialSpecs);
    for (unsigned I = 0, N = PartialSpecs.size(); I != N; ++I)
      if (PartialSpecs[I]->getFirstDecl()->isOutOfLine())
        OutOfLineVarPartialSpecs.push_back(
            std::make_pair(Inst, PartialSpecs[I]));
  }

  return Inst;
}

Decl *TemplateDeclInstantiator::VisitVarTemplatePartialSpecializationDecl(
    VarTemplatePartialSpecializationDecl *D) {
  assert(D->isStaticDataMember() &&
         "Only static data member templates are allowed.");

  VarTemplateDecl *VarTemplate = D->getSpecializedTemplate();

  // Lookup the already-instantiated declaration and return that.
  DeclContext::lookup_result Found = Owner->lookup(VarTemplate->getDeclName());
  assert(!Found.empty() && "Instantiation found nothing?");

  VarTemplateDecl *InstVarTemplate = dyn_cast<VarTemplateDecl>(Found.front());
  assert(InstVarTemplate && "Instantiation did not find a variable template?");

  if (VarTemplatePartialSpecializationDecl *Result =
          InstVarTemplate->findPartialSpecInstantiatedFromMember(D))
    return Result;

  return InstantiateVarTemplatePartialSpecialization(InstVarTemplate, D);
}

Decl *
TemplateDeclInstantiator::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  // Create a local instantiation scope for this function template, which
  // will contain the instantiations of the template parameters and then get
  // merged with the local instantiation scope for the function template
  // itself.
  LocalInstantiationScope Scope(SemaRef);

  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return NULL;

  FunctionDecl *Instantiated = 0;
  if (CXXMethodDecl *DMethod = dyn_cast<CXXMethodDecl>(D->getTemplatedDecl()))
    Instantiated = cast_or_null<FunctionDecl>(VisitCXXMethodDecl(DMethod,
                                                                 InstParams));
  else
    Instantiated = cast_or_null<FunctionDecl>(VisitFunctionDecl(
                                                          D->getTemplatedDecl(),
                                                                InstParams));

  if (!Instantiated)
    return 0;

  // Link the instantiated function template declaration to the function
  // template from which it was instantiated.
  FunctionTemplateDecl *InstTemplate
    = Instantiated->getDescribedFunctionTemplate();
  InstTemplate->setAccess(D->getAccess());
  assert(InstTemplate &&
         "VisitFunctionDecl/CXXMethodDecl didn't create a template!");

  bool isFriend = (InstTemplate->getFriendObjectKind() != Decl::FOK_None);

  // Link the instantiation back to the pattern *unless* this is a
  // non-definition friend declaration.
  if (!InstTemplate->getInstantiatedFromMemberTemplate() &&
      !(isFriend && !D->getTemplatedDecl()->isThisDeclarationADefinition()))
    InstTemplate->setInstantiatedFromMemberTemplate(D);

  // Make declarations visible in the appropriate context.
  if (!isFriend) {
    Owner->addDecl(InstTemplate);
  } else if (InstTemplate->getDeclContext()->isRecord() &&
             !D->getPreviousDecl()) {
    SemaRef.CheckFriendAccess(InstTemplate);
  }

  return InstTemplate;
}

Decl *TemplateDeclInstantiator::VisitCXXRecordDecl(CXXRecordDecl *D) {
  CXXRecordDecl *PrevDecl = 0;
  if (D->isInjectedClassName())
    PrevDecl = cast<CXXRecordDecl>(Owner);
  else if (D->getPreviousDecl()) {
    NamedDecl *Prev = SemaRef.FindInstantiatedDecl(D->getLocation(),
                                                   D->getPreviousDecl(),
                                                   TemplateArgs);
    if (!Prev) return 0;
    PrevDecl = cast<CXXRecordDecl>(Prev);
  }

  CXXRecordDecl *Record
    = CXXRecordDecl::Create(SemaRef.Context, D->getTagKind(), Owner,
                            D->getLocStart(), D->getLocation(),
                            D->getIdentifier(), PrevDecl);

  // Substitute the nested name specifier, if any.
  if (SubstQualifier(D, Record))
    return 0;

  Record->setImplicit(D->isImplicit());
  // FIXME: Check against AS_none is an ugly hack to work around the issue that
  // the tag decls introduced by friend class declarations don't have an access
  // specifier. Remove once this area of the code gets sorted out.
  if (D->getAccess() != AS_none)
    Record->setAccess(D->getAccess());
  if (!D->isInjectedClassName())
    Record->setInstantiationOfMemberClass(D, TSK_ImplicitInstantiation);

  // If the original function was part of a friend declaration,
  // inherit its namespace state.
  if (D->getFriendObjectKind())
    Record->setObjectOfFriendDecl();

  // Make sure that anonymous structs and unions are recorded.
  if (D->isAnonymousStructOrUnion())
    Record->setAnonymousStructOrUnion(true);

  if (D->isLocalClass())
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Record);

  Owner->addDecl(Record);

  // DR1484 clarifies that the members of a local class are instantiated as part
  // of the instantiation of their enclosing entity.
  if (D->isCompleteDefinition() && D->isLocalClass()) {
    if (SemaRef.InstantiateClass(D->getLocation(), Record, D, TemplateArgs,
                                 TSK_ImplicitInstantiation,
                                 /*Complain=*/true)) {
      llvm_unreachable("InstantiateClass shouldn't fail here!");
    } else {
      SemaRef.InstantiateClassMembers(D->getLocation(), Record, TemplateArgs,
                                      TSK_ImplicitInstantiation);
    }
  }
  return Record;
}

/// \brief Adjust the given function type for an instantiation of the
/// given declaration, to cope with modifications to the function's type that
/// aren't reflected in the type-source information.
///
/// \param D The declaration we're instantiating.
/// \param TInfo The already-instantiated type.
static QualType adjustFunctionTypeForInstantiation(ASTContext &Context,
                                                   FunctionDecl *D,
                                                   TypeSourceInfo *TInfo) {
  const FunctionProtoType *OrigFunc
    = D->getType()->castAs<FunctionProtoType>();
  const FunctionProtoType *NewFunc
    = TInfo->getType()->castAs<FunctionProtoType>();
  if (OrigFunc->getExtInfo() == NewFunc->getExtInfo())
    return TInfo->getType();

  FunctionProtoType::ExtProtoInfo NewEPI = NewFunc->getExtProtoInfo();
  NewEPI.ExtInfo = OrigFunc->getExtInfo();
  return Context.getFunctionType(NewFunc->getResultType(),
                                 NewFunc->getArgTypes(), NewEPI);
}

/// Normal class members are of more specific types and therefore
/// don't make it here.  This function serves two purposes:
///   1) instantiating function templates
///   2) substituting friend declarations
/// FIXME: preserve function definitions in case #2
Decl *TemplateDeclInstantiator::VisitFunctionDecl(FunctionDecl *D,
                                       TemplateParameterList *TemplateParams) {
  // Check whether there is already a function template specialization for
  // this declaration.
  FunctionTemplateDecl *FunctionTemplate = D->getDescribedFunctionTemplate();
  if (FunctionTemplate && !TemplateParams) {
    ArrayRef<TemplateArgument> Innermost = TemplateArgs.getInnermost();

    void *InsertPos = 0;
    FunctionDecl *SpecFunc
      = FunctionTemplate->findSpecialization(Innermost.begin(), Innermost.size(),
                                             InsertPos);

    // If we already have a function template specialization, return it.
    if (SpecFunc)
      return SpecFunc;
  }

  bool isFriend;
  if (FunctionTemplate)
    isFriend = (FunctionTemplate->getFriendObjectKind() != Decl::FOK_None);
  else
    isFriend = (D->getFriendObjectKind() != Decl::FOK_None);

  bool MergeWithParentScope = (TemplateParams != 0) ||
    Owner->isFunctionOrMethod() ||
    !(isa<Decl>(Owner) &&
      cast<Decl>(Owner)->isDefinedOutsideFunctionOrMethod());
  LocalInstantiationScope Scope(SemaRef, MergeWithParentScope);

  SmallVector<ParmVarDecl *, 4> Params;
  TypeSourceInfo *TInfo = SubstFunctionType(D, Params);
  if (!TInfo)
    return 0;
  QualType T = adjustFunctionTypeForInstantiation(SemaRef.Context, D, TInfo);

  NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc();
  if (QualifierLoc) {
    QualifierLoc = SemaRef.SubstNestedNameSpecifierLoc(QualifierLoc,
                                                       TemplateArgs);
    if (!QualifierLoc)
      return 0;
  }

  // If we're instantiating a local function declaration, put the result
  // in the enclosing namespace; otherwise we need to find the instantiated
  // context.
  DeclContext *DC;
  if (D->isLocalExternDecl()) {
    DC = Owner;
    SemaRef.adjustContextForLocalExternDecl(DC);
  } else if (isFriend && QualifierLoc) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);
    DC = SemaRef.computeDeclContext(SS);
    if (!DC) return 0;
  } else {
    DC = SemaRef.FindInstantiatedContext(D->getLocation(), D->getDeclContext(),
                                         TemplateArgs);
  }

  FunctionDecl *Function =
      FunctionDecl::Create(SemaRef.Context, DC, D->getInnerLocStart(),
                           D->getNameInfo(), T, TInfo,
                           D->getCanonicalDecl()->getStorageClass(),
                           D->isInlineSpecified(), D->hasWrittenPrototype(),
                           D->isConstexpr());
  Function->setRangeEnd(D->getSourceRange().getEnd());

  if (D->isInlined())
    Function->setImplicitlyInline();

  if (QualifierLoc)
    Function->setQualifierInfo(QualifierLoc);

  if (D->isLocalExternDecl())
    Function->setLocalExternDecl();

  DeclContext *LexicalDC = Owner;
  if (!isFriend && D->isOutOfLine() && !D->isLocalExternDecl()) {
    assert(D->getDeclContext()->isFileContext());
    LexicalDC = D->getDeclContext();
  }

  Function->setLexicalDeclContext(LexicalDC);

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    if (Params[P])
      Params[P]->setOwningFunction(Function);
  Function->setParams(Params);

  SourceLocation InstantiateAtPOI;
  if (TemplateParams) {
    // Our resulting instantiation is actually a function template, since we
    // are substituting only the outer template parameters. For example, given
    //
    //   template<typename T>
    //   struct X {
    //     template<typename U> friend void f(T, U);
    //   };
    //
    //   X<int> x;
    //
    // We are instantiating the friend function template "f" within X<int>,
    // which means substituting int for T, but leaving "f" as a friend function
    // template.
    // Build the function template itself.
    FunctionTemplate = FunctionTemplateDecl::Create(SemaRef.Context, DC,
                                                    Function->getLocation(),
                                                    Function->getDeclName(),
                                                    TemplateParams, Function);
    Function->setDescribedFunctionTemplate(FunctionTemplate);

    FunctionTemplate->setLexicalDeclContext(LexicalDC);

    if (isFriend && D->isThisDeclarationADefinition()) {
      // TODO: should we remember this connection regardless of whether
      // the friend declaration provided a body?
      FunctionTemplate->setInstantiatedFromMemberTemplate(
                                           D->getDescribedFunctionTemplate());
    }
  } else if (FunctionTemplate) {
    // Record this function template specialization.
    ArrayRef<TemplateArgument> Innermost = TemplateArgs.getInnermost();
    Function->setFunctionTemplateSpecialization(FunctionTemplate,
                            TemplateArgumentList::CreateCopy(SemaRef.Context,
                                                             Innermost.begin(),
                                                             Innermost.size()),
                                                /*InsertPos=*/0);
  } else if (isFriend) {
    // Note, we need this connection even if the friend doesn't have a body.
    // Its body may exist but not have been attached yet due to deferred
    // parsing.
    // FIXME: It might be cleaner to set this when attaching the body to the
    // friend function declaration, however that would require finding all the
    // instantiations and modifying them.
    Function->setInstantiationOfMemberFunction(D, TSK_ImplicitInstantiation);
  }

  if (InitFunctionInstantiation(Function, D))
    Function->setInvalidDecl();

  bool isExplicitSpecialization = false;

  LookupResult Previous(
      SemaRef, Function->getDeclName(), SourceLocation(),
      D->isLocalExternDecl() ? Sema::LookupRedeclarationWithLinkage
                             : Sema::LookupOrdinaryName,
      Sema::ForRedeclaration);

  if (DependentFunctionTemplateSpecializationInfo *Info
        = D->getDependentSpecializationInfo()) {
    assert(isFriend && "non-friend has dependent specialization info?");

    // This needs to be set now for future sanity.
    Function->setObjectOfFriendDecl();

    // Instantiate the explicit template arguments.
    TemplateArgumentListInfo ExplicitArgs(Info->getLAngleLoc(),
                                          Info->getRAngleLoc());
    if (SemaRef.Subst(Info->getTemplateArgs(), Info->getNumTemplateArgs(),
                      ExplicitArgs, TemplateArgs))
      return 0;

    // Map the candidate templates to their instantiations.
    for (unsigned I = 0, E = Info->getNumTemplates(); I != E; ++I) {
      Decl *Temp = SemaRef.FindInstantiatedDecl(D->getLocation(),
                                                Info->getTemplate(I),
                                                TemplateArgs);
      if (!Temp) return 0;

      Previous.addDecl(cast<FunctionTemplateDecl>(Temp));
    }

    if (SemaRef.CheckFunctionTemplateSpecialization(Function,
                                                    &ExplicitArgs,
                                                    Previous))
      Function->setInvalidDecl();

    isExplicitSpecialization = true;

  } else if (TemplateParams || !FunctionTemplate) {
    // Look only into the namespace where the friend would be declared to
    // find a previous declaration. This is the innermost enclosing namespace,
    // as described in ActOnFriendFunctionDecl.
    SemaRef.LookupQualifiedName(Previous, DC);

    // In C++, the previous declaration we find might be a tag type
    // (class or enum). In this case, the new declaration will hide the
    // tag type. Note that this does does not apply if we're declaring a
    // typedef (C++ [dcl.typedef]p4).
    if (Previous.isSingleTagDecl())
      Previous.clear();
  }

  SemaRef.CheckFunctionDeclaration(/*Scope*/ 0, Function, Previous,
                                   isExplicitSpecialization);

  NamedDecl *PrincipalDecl = (TemplateParams
                              ? cast<NamedDecl>(FunctionTemplate)
                              : Function);

  // If the original function was part of a friend declaration,
  // inherit its namespace state and add it to the owner.
  if (isFriend) {
    PrincipalDecl->setObjectOfFriendDecl();
    DC->makeDeclVisibleInContext(PrincipalDecl);

    bool queuedInstantiation = false;

    // C++98 [temp.friend]p5: When a function is defined in a friend function
    //   declaration in a class template, the function is defined at each
    //   instantiation of the class template. The function is defined even if it
    //   is never used.
    // C++11 [temp.friend]p4: When a function is defined in a friend function
    //   declaration in a class template, the function is instantiated when the
    //   function is odr-used.
    //
    // If -Wc++98-compat is enabled, we go through the motions of checking for a
    // redefinition, but don't instantiate the function.
    if ((!SemaRef.getLangOpts().CPlusPlus11 ||
         SemaRef.Diags.getDiagnosticLevel(
             diag::warn_cxx98_compat_friend_redefinition,
             Function->getLocation())
           != DiagnosticsEngine::Ignored) &&
        D->isThisDeclarationADefinition()) {
      // Check for a function body.
      const FunctionDecl *Definition = 0;
      if (Function->isDefined(Definition) &&
          Definition->getTemplateSpecializationKind() == TSK_Undeclared) {
        SemaRef.Diag(Function->getLocation(),
                     SemaRef.getLangOpts().CPlusPlus11 ?
                       diag::warn_cxx98_compat_friend_redefinition :
                       diag::err_redefinition) << Function->getDeclName();
        SemaRef.Diag(Definition->getLocation(), diag::note_previous_definition);
        if (!SemaRef.getLangOpts().CPlusPlus11)
          Function->setInvalidDecl();
      }
      // Check for redefinitions due to other instantiations of this or
      // a similar friend function.
      else for (FunctionDecl::redecl_iterator R = Function->redecls_begin(),
                                           REnd = Function->redecls_end();
                R != REnd; ++R) {
        if (*R == Function)
          continue;
        switch (R->getFriendObjectKind()) {
        case Decl::FOK_None:
          if (!SemaRef.getLangOpts().CPlusPlus11 &&
              !queuedInstantiation && R->isUsed(false)) {
            if (MemberSpecializationInfo *MSInfo
                = Function->getMemberSpecializationInfo()) {
              if (MSInfo->getPointOfInstantiation().isInvalid()) {
                SourceLocation Loc = R->getLocation(); // FIXME
                MSInfo->setPointOfInstantiation(Loc);
                SemaRef.PendingLocalImplicitInstantiations.push_back(
                                                 std::make_pair(Function, Loc));
                queuedInstantiation = true;
              }
            }
          }
          break;
        default:
          if (const FunctionDecl *RPattern
              = R->getTemplateInstantiationPattern())
            if (RPattern->isDefined(RPattern)) {
              SemaRef.Diag(Function->getLocation(),
                           SemaRef.getLangOpts().CPlusPlus11 ?
                             diag::warn_cxx98_compat_friend_redefinition :
                             diag::err_redefinition)
                << Function->getDeclName();
              SemaRef.Diag(R->getLocation(), diag::note_previous_definition);
              if (!SemaRef.getLangOpts().CPlusPlus11)
                Function->setInvalidDecl();
              break;
            }
        }
      }
    }
  }

  if (Function->isLocalExternDecl() && !Function->getPreviousDecl())
    DC->makeDeclVisibleInContext(PrincipalDecl);

  if (Function->isOverloadedOperator() && !DC->isRecord() &&
      PrincipalDecl->isInIdentifierNamespace(Decl::IDNS_Ordinary))
    PrincipalDecl->setNonMemberOperator();

  assert(!D->isDefaulted() && "only methods should be defaulted");
  return Function;
}

Decl *
TemplateDeclInstantiator::VisitCXXMethodDecl(CXXMethodDecl *D,
                                      TemplateParameterList *TemplateParams,
                                      bool IsClassScopeSpecialization) {
  FunctionTemplateDecl *FunctionTemplate = D->getDescribedFunctionTemplate();
  if (FunctionTemplate && !TemplateParams) {
    // We are creating a function template specialization from a function
    // template. Check whether there is already a function template
    // specialization for this particular set of template arguments.
    ArrayRef<TemplateArgument> Innermost = TemplateArgs.getInnermost();

    void *InsertPos = 0;
    FunctionDecl *SpecFunc
      = FunctionTemplate->findSpecialization(Innermost.begin(), 
                                             Innermost.size(),
                                             InsertPos);

    // If we already have a function template specialization, return it.
    if (SpecFunc)
      return SpecFunc;
  }

  bool isFriend;
  if (FunctionTemplate)
    isFriend = (FunctionTemplate->getFriendObjectKind() != Decl::FOK_None);
  else
    isFriend = (D->getFriendObjectKind() != Decl::FOK_None);

  bool MergeWithParentScope = (TemplateParams != 0) ||
    !(isa<Decl>(Owner) &&
      cast<Decl>(Owner)->isDefinedOutsideFunctionOrMethod());
  LocalInstantiationScope Scope(SemaRef, MergeWithParentScope);

  // Instantiate enclosing template arguments for friends.
  SmallVector<TemplateParameterList *, 4> TempParamLists;
  unsigned NumTempParamLists = 0;
  if (isFriend && (NumTempParamLists = D->getNumTemplateParameterLists())) {
    TempParamLists.set_size(NumTempParamLists);
    for (unsigned I = 0; I != NumTempParamLists; ++I) {
      TemplateParameterList *TempParams = D->getTemplateParameterList(I);
      TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
      if (!InstParams)
        return NULL;
      TempParamLists[I] = InstParams;
    }
  }

  SmallVector<ParmVarDecl *, 4> Params;
  TypeSourceInfo *TInfo = SubstFunctionType(D, Params);
  if (!TInfo)
    return 0;
  QualType T = adjustFunctionTypeForInstantiation(SemaRef.Context, D, TInfo);

  NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc();
  if (QualifierLoc) {
    QualifierLoc = SemaRef.SubstNestedNameSpecifierLoc(QualifierLoc,
                                                 TemplateArgs);
    if (!QualifierLoc)
      return 0;
  }

  DeclContext *DC = Owner;
  if (isFriend) {
    if (QualifierLoc) {
      CXXScopeSpec SS;
      SS.Adopt(QualifierLoc);
      DC = SemaRef.computeDeclContext(SS);

      if (DC && SemaRef.RequireCompleteDeclContext(SS, DC))
        return 0;
    } else {
      DC = SemaRef.FindInstantiatedContext(D->getLocation(),
                                           D->getDeclContext(),
                                           TemplateArgs);
    }
    if (!DC) return 0;
  }

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(DC);
  CXXMethodDecl *Method = 0;

  SourceLocation StartLoc = D->getInnerLocStart();
  DeclarationNameInfo NameInfo
    = SemaRef.SubstDeclarationNameInfo(D->getNameInfo(), TemplateArgs);
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    Method = CXXConstructorDecl::Create(SemaRef.Context, Record,
                                        StartLoc, NameInfo, T, TInfo,
                                        Constructor->isExplicit(),
                                        Constructor->isInlineSpecified(),
                                        false, Constructor->isConstexpr());

    // Claim that the instantiation of a constructor or constructor template
    // inherits the same constructor that the template does.
    if (CXXConstructorDecl *Inh = const_cast<CXXConstructorDecl *>(
            Constructor->getInheritedConstructor())) {
      // If we're instantiating a specialization of a function template, our
      // "inherited constructor" will actually itself be a function template.
      // Instantiate a declaration of it, too.
      if (FunctionTemplate) {
        assert(!TemplateParams && Inh->getDescribedFunctionTemplate() &&
               !Inh->getParent()->isDependentContext() &&
               "inheriting constructor template in dependent context?");
        Sema::InstantiatingTemplate Inst(SemaRef, Constructor->getLocation(),
                                         Inh);
        if (Inst.isInvalid())
          return 0;
        Sema::ContextRAII SavedContext(SemaRef, Inh->getDeclContext());
        LocalInstantiationScope LocalScope(SemaRef);

        // Use the same template arguments that we deduced for the inheriting
        // constructor. There's no way they could be deduced differently.
        MultiLevelTemplateArgumentList InheritedArgs;
        InheritedArgs.addOuterTemplateArguments(TemplateArgs.getInnermost());
        Inh = cast_or_null<CXXConstructorDecl>(
            SemaRef.SubstDecl(Inh, Inh->getDeclContext(), InheritedArgs));
        if (!Inh)
          return 0;
      }
      cast<CXXConstructorDecl>(Method)->setInheritedConstructor(Inh);
    }
  } else if (CXXDestructorDecl *Destructor = dyn_cast<CXXDestructorDecl>(D)) {
    Method = CXXDestructorDecl::Create(SemaRef.Context, Record,
                                       StartLoc, NameInfo, T, TInfo,
                                       Destructor->isInlineSpecified(),
                                       false);
  } else if (CXXConversionDecl *Conversion = dyn_cast<CXXConversionDecl>(D)) {
    Method = CXXConversionDecl::Create(SemaRef.Context, Record,
                                       StartLoc, NameInfo, T, TInfo,
                                       Conversion->isInlineSpecified(),
                                       Conversion->isExplicit(),
                                       Conversion->isConstexpr(),
                                       Conversion->getLocEnd());
  } else {
    StorageClass SC = D->isStatic() ? SC_Static : SC_None;
    Method = CXXMethodDecl::Create(SemaRef.Context, Record,
                                   StartLoc, NameInfo, T, TInfo,
                                   SC, D->isInlineSpecified(),
                                   D->isConstexpr(), D->getLocEnd());
  }

  if (D->isInlined())
    Method->setImplicitlyInline();

  if (QualifierLoc)
    Method->setQualifierInfo(QualifierLoc);

  if (TemplateParams) {
    // Our resulting instantiation is actually a function template, since we
    // are substituting only the outer template parameters. For example, given
    //
    //   template<typename T>
    //   struct X {
    //     template<typename U> void f(T, U);
    //   };
    //
    //   X<int> x;
    //
    // We are instantiating the member template "f" within X<int>, which means
    // substituting int for T, but leaving "f" as a member function template.
    // Build the function template itself.
    FunctionTemplate = FunctionTemplateDecl::Create(SemaRef.Context, Record,
                                                    Method->getLocation(),
                                                    Method->getDeclName(),
                                                    TemplateParams, Method);
    if (isFriend) {
      FunctionTemplate->setLexicalDeclContext(Owner);
      FunctionTemplate->setObjectOfFriendDecl();
    } else if (D->isOutOfLine())
      FunctionTemplate->setLexicalDeclContext(D->getLexicalDeclContext());
    Method->setDescribedFunctionTemplate(FunctionTemplate);
  } else if (FunctionTemplate) {
    // Record this function template specialization.
    ArrayRef<TemplateArgument> Innermost = TemplateArgs.getInnermost();
    Method->setFunctionTemplateSpecialization(FunctionTemplate,
                         TemplateArgumentList::CreateCopy(SemaRef.Context,
                                                          Innermost.begin(),
                                                          Innermost.size()),
                                              /*InsertPos=*/0);
  } else if (!isFriend) {
    // Record that this is an instantiation of a member function.
    Method->setInstantiationOfMemberFunction(D, TSK_ImplicitInstantiation);
  }

  // If we are instantiating a member function defined
  // out-of-line, the instantiation will have the same lexical
  // context (which will be a namespace scope) as the template.
  if (isFriend) {
    if (NumTempParamLists)
      Method->setTemplateParameterListsInfo(SemaRef.Context,
                                            NumTempParamLists,
                                            TempParamLists.data());

    Method->setLexicalDeclContext(Owner);
    Method->setObjectOfFriendDecl();
  } else if (D->isOutOfLine())
    Method->setLexicalDeclContext(D->getLexicalDeclContext());

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Method);
  Method->setParams(Params);

  if (InitMethodInstantiation(Method, D))
    Method->setInvalidDecl();

  LookupResult Previous(SemaRef, NameInfo, Sema::LookupOrdinaryName,
                        Sema::ForRedeclaration);

  if (!FunctionTemplate || TemplateParams || isFriend) {
    SemaRef.LookupQualifiedName(Previous, Record);

    // In C++, the previous declaration we find might be a tag type
    // (class or enum). In this case, the new declaration will hide the
    // tag type. Note that this does does not apply if we're declaring a
    // typedef (C++ [dcl.typedef]p4).
    if (Previous.isSingleTagDecl())
      Previous.clear();
  }

  if (!IsClassScopeSpecialization)
    SemaRef.CheckFunctionDeclaration(0, Method, Previous, false);

  if (D->isPure())
    SemaRef.CheckPureMethod(Method, SourceRange());

  // Propagate access.  For a non-friend declaration, the access is
  // whatever we're propagating from.  For a friend, it should be the
  // previous declaration we just found.
  if (isFriend && Method->getPreviousDecl())
    Method->setAccess(Method->getPreviousDecl()->getAccess());
  else 
    Method->setAccess(D->getAccess());
  if (FunctionTemplate)
    FunctionTemplate->setAccess(Method->getAccess());

  SemaRef.CheckOverrideControl(Method);

  // If a function is defined as defaulted or deleted, mark it as such now.
  if (D->isExplicitlyDefaulted())
    SemaRef.SetDeclDefaulted(Method, Method->getLocation());
  if (D->isDeletedAsWritten())
    SemaRef.SetDeclDeleted(Method, Method->getLocation());

  // If there's a function template, let our caller handle it.
  if (FunctionTemplate) {
    // do nothing

  // Don't hide a (potentially) valid declaration with an invalid one.
  } else if (Method->isInvalidDecl() && !Previous.empty()) {
    // do nothing

  // Otherwise, check access to friends and make them visible.
  } else if (isFriend) {
    // We only need to re-check access for methods which we didn't
    // manage to match during parsing.
    if (!D->getPreviousDecl())
      SemaRef.CheckFriendAccess(Method);

    Record->makeDeclVisibleInContext(Method);

  // Otherwise, add the declaration.  We don't need to do this for
  // class-scope specializations because we'll have matched them with
  // the appropriate template.
  } else if (!IsClassScopeSpecialization) {
    Owner->addDecl(Method);
  }

  return Method;
}

Decl *TemplateDeclInstantiator::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *TemplateDeclInstantiator::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *TemplateDeclInstantiator::VisitCXXConversionDecl(CXXConversionDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *TemplateDeclInstantiator::VisitParmVarDecl(ParmVarDecl *D) {
  return SemaRef.SubstParmVarDecl(D, TemplateArgs, /*indexAdjustment*/ 0, None,
                                  /*ExpectParameterPack=*/ false);
}

Decl *TemplateDeclInstantiator::VisitTemplateTypeParmDecl(
                                                    TemplateTypeParmDecl *D) {
  // TODO: don't always clone when decls are refcounted.
  assert(D->getTypeForDecl()->isTemplateTypeParmType());

  TemplateTypeParmDecl *Inst =
    TemplateTypeParmDecl::Create(SemaRef.Context, Owner,
                                 D->getLocStart(), D->getLocation(),
                                 D->getDepth() - TemplateArgs.getNumLevels(),
                                 D->getIndex(), D->getIdentifier(),
                                 D->wasDeclaredWithTypename(),
                                 D->isParameterPack());
  Inst->setAccess(AS_public);

  if (D->hasDefaultArgument()) {
    TypeSourceInfo *InstantiatedDefaultArg =
        SemaRef.SubstType(D->getDefaultArgumentInfo(), TemplateArgs,
                          D->getDefaultArgumentLoc(), D->getDeclName());
    if (InstantiatedDefaultArg)
      Inst->setDefaultArgument(InstantiatedDefaultArg, false);
  }

  // Introduce this template parameter's instantiation into the instantiation
  // scope.
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Inst);

  return Inst;
}

Decl *TemplateDeclInstantiator::VisitNonTypeTemplateParmDecl(
                                                 NonTypeTemplateParmDecl *D) {
  // Substitute into the type of the non-type template parameter.
  TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
  SmallVector<TypeSourceInfo *, 4> ExpandedParameterPackTypesAsWritten;
  SmallVector<QualType, 4> ExpandedParameterPackTypes;
  bool IsExpandedParameterPack = false;
  TypeSourceInfo *DI;
  QualType T;
  bool Invalid = false;

  if (D->isExpandedParameterPack()) {
    // The non-type template parameter pack is an already-expanded pack
    // expansion of types. Substitute into each of the expanded types.
    ExpandedParameterPackTypes.reserve(D->getNumExpansionTypes());
    ExpandedParameterPackTypesAsWritten.reserve(D->getNumExpansionTypes());
    for (unsigned I = 0, N = D->getNumExpansionTypes(); I != N; ++I) {
      TypeSourceInfo *NewDI =SemaRef.SubstType(D->getExpansionTypeSourceInfo(I),
                                               TemplateArgs,
                                               D->getLocation(),
                                               D->getDeclName());
      if (!NewDI)
        return 0;

      ExpandedParameterPackTypesAsWritten.push_back(NewDI);
      QualType NewT =SemaRef.CheckNonTypeTemplateParameterType(NewDI->getType(),
                                                              D->getLocation());
      if (NewT.isNull())
        return 0;
      ExpandedParameterPackTypes.push_back(NewT);
    }

    IsExpandedParameterPack = true;
    DI = D->getTypeSourceInfo();
    T = DI->getType();
  } else if (D->isPackExpansion()) {
    // The non-type template parameter pack's type is a pack expansion of types.
    // Determine whether we need to expand this parameter pack into separate
    // types.
    PackExpansionTypeLoc Expansion = TL.castAs<PackExpansionTypeLoc>();
    TypeLoc Pattern = Expansion.getPatternLoc();
    SmallVector<UnexpandedParameterPack, 2> Unexpanded;
    SemaRef.collectUnexpandedParameterPacks(Pattern, Unexpanded);

    // Determine whether the set of unexpanded parameter packs can and should
    // be expanded.
    bool Expand = true;
    bool RetainExpansion = false;
    Optional<unsigned> OrigNumExpansions
      = Expansion.getTypePtr()->getNumExpansions();
    Optional<unsigned> NumExpansions = OrigNumExpansions;
    if (SemaRef.CheckParameterPacksForExpansion(Expansion.getEllipsisLoc(),
                                                Pattern.getSourceRange(),
                                                Unexpanded,
                                                TemplateArgs,
                                                Expand, RetainExpansion,
                                                NumExpansions))
      return 0;

    if (Expand) {
      for (unsigned I = 0; I != *NumExpansions; ++I) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, I);
        TypeSourceInfo *NewDI = SemaRef.SubstType(Pattern, TemplateArgs,
                                                  D->getLocation(),
                                                  D->getDeclName());
        if (!NewDI)
          return 0;

        ExpandedParameterPackTypesAsWritten.push_back(NewDI);
        QualType NewT = SemaRef.CheckNonTypeTemplateParameterType(
                                                              NewDI->getType(),
                                                              D->getLocation());
        if (NewT.isNull())
          return 0;
        ExpandedParameterPackTypes.push_back(NewT);
      }

      // Note that we have an expanded parameter pack. The "type" of this
      // expanded parameter pack is the original expansion type, but callers
      // will end up using the expanded parameter pack types for type-checking.
      IsExpandedParameterPack = true;
      DI = D->getTypeSourceInfo();
      T = DI->getType();
    } else {
      // We cannot fully expand the pack expansion now, so substitute into the
      // pattern and create a new pack expansion type.
      Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, -1);
      TypeSourceInfo *NewPattern = SemaRef.SubstType(Pattern, TemplateArgs,
                                                     D->getLocation(),
                                                     D->getDeclName());
      if (!NewPattern)
        return 0;

      DI = SemaRef.CheckPackExpansion(NewPattern, Expansion.getEllipsisLoc(),
                                      NumExpansions);
      if (!DI)
        return 0;

      T = DI->getType();
    }
  } else {
    // Simple case: substitution into a parameter that is not a parameter pack.
    DI = SemaRef.SubstType(D->getTypeSourceInfo(), TemplateArgs,
                           D->getLocation(), D->getDeclName());
    if (!DI)
      return 0;

    // Check that this type is acceptable for a non-type template parameter.
    T = SemaRef.CheckNonTypeTemplateParameterType(DI->getType(),
                                                  D->getLocation());
    if (T.isNull()) {
      T = SemaRef.Context.IntTy;
      Invalid = true;
    }
  }

  NonTypeTemplateParmDecl *Param;
  if (IsExpandedParameterPack)
    Param = NonTypeTemplateParmDecl::Create(SemaRef.Context, Owner,
                                            D->getInnerLocStart(),
                                            D->getLocation(),
                                    D->getDepth() - TemplateArgs.getNumLevels(),
                                            D->getPosition(),
                                            D->getIdentifier(), T,
                                            DI,
                                            ExpandedParameterPackTypes.data(),
                                            ExpandedParameterPackTypes.size(),
                                    ExpandedParameterPackTypesAsWritten.data());
  else
    Param = NonTypeTemplateParmDecl::Create(SemaRef.Context, Owner,
                                            D->getInnerLocStart(),
                                            D->getLocation(),
                                    D->getDepth() - TemplateArgs.getNumLevels(),
                                            D->getPosition(),
                                            D->getIdentifier(), T,
                                            D->isParameterPack(), DI);

  Param->setAccess(AS_public);
  if (Invalid)
    Param->setInvalidDecl();

  if (D->hasDefaultArgument()) {
    ExprResult Value = SemaRef.SubstExpr(D->getDefaultArgument(), TemplateArgs);
    if (!Value.isInvalid())
      Param->setDefaultArgument(Value.get(), false);
  }

  // Introduce this template parameter's instantiation into the instantiation
  // scope.
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Param);
  return Param;
}

static void collectUnexpandedParameterPacks(
    Sema &S,
    TemplateParameterList *Params,
    SmallVectorImpl<UnexpandedParameterPack> &Unexpanded) {
  for (TemplateParameterList::const_iterator I = Params->begin(),
                                             E = Params->end(); I != E; ++I) {
    if ((*I)->isTemplateParameterPack())
      continue;
    if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(*I))
      S.collectUnexpandedParameterPacks(NTTP->getTypeSourceInfo()->getTypeLoc(),
                                        Unexpanded);
    if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(*I))
      collectUnexpandedParameterPacks(S, TTP->getTemplateParameters(),
                                      Unexpanded);
  }
}

Decl *
TemplateDeclInstantiator::VisitTemplateTemplateParmDecl(
                                                  TemplateTemplateParmDecl *D) {
  // Instantiate the template parameter list of the template template parameter.
  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams;
  SmallVector<TemplateParameterList*, 8> ExpandedParams;

  bool IsExpandedParameterPack = false;

  if (D->isExpandedParameterPack()) {
    // The template template parameter pack is an already-expanded pack
    // expansion of template parameters. Substitute into each of the expanded
    // parameters.
    ExpandedParams.reserve(D->getNumExpansionTemplateParameters());
    for (unsigned I = 0, N = D->getNumExpansionTemplateParameters();
         I != N; ++I) {
      LocalInstantiationScope Scope(SemaRef);
      TemplateParameterList *Expansion =
        SubstTemplateParams(D->getExpansionTemplateParameters(I));
      if (!Expansion)
        return 0;
      ExpandedParams.push_back(Expansion);
    }

    IsExpandedParameterPack = true;
    InstParams = TempParams;
  } else if (D->isPackExpansion()) {
    // The template template parameter pack expands to a pack of template
    // template parameters. Determine whether we need to expand this parameter
    // pack into separate parameters.
    SmallVector<UnexpandedParameterPack, 2> Unexpanded;
    collectUnexpandedParameterPacks(SemaRef, D->getTemplateParameters(),
                                    Unexpanded);

    // Determine whether the set of unexpanded parameter packs can and should
    // be expanded.
    bool Expand = true;
    bool RetainExpansion = false;
    Optional<unsigned> NumExpansions;
    if (SemaRef.CheckParameterPacksForExpansion(D->getLocation(),
                                                TempParams->getSourceRange(),
                                                Unexpanded,
                                                TemplateArgs,
                                                Expand, RetainExpansion,
                                                NumExpansions))
      return 0;

    if (Expand) {
      for (unsigned I = 0; I != *NumExpansions; ++I) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, I);
        LocalInstantiationScope Scope(SemaRef);
        TemplateParameterList *Expansion = SubstTemplateParams(TempParams);
        if (!Expansion)
          return 0;
        ExpandedParams.push_back(Expansion);
      }

      // Note that we have an expanded parameter pack. The "type" of this
      // expanded parameter pack is the original expansion type, but callers
      // will end up using the expanded parameter pack types for type-checking.
      IsExpandedParameterPack = true;
      InstParams = TempParams;
    } else {
      // We cannot fully expand the pack expansion now, so just substitute
      // into the pattern.
      Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, -1);

      LocalInstantiationScope Scope(SemaRef);
      InstParams = SubstTemplateParams(TempParams);
      if (!InstParams)
        return 0;
    }
  } else {
    // Perform the actual substitution of template parameters within a new,
    // local instantiation scope.
    LocalInstantiationScope Scope(SemaRef);
    InstParams = SubstTemplateParams(TempParams);
    if (!InstParams)
      return 0;
  }

  // Build the template template parameter.
  TemplateTemplateParmDecl *Param;
  if (IsExpandedParameterPack)
    Param = TemplateTemplateParmDecl::Create(SemaRef.Context, Owner,
                                             D->getLocation(),
                                   D->getDepth() - TemplateArgs.getNumLevels(),
                                             D->getPosition(),
                                             D->getIdentifier(), InstParams,
                                             ExpandedParams);
  else
    Param = TemplateTemplateParmDecl::Create(SemaRef.Context, Owner,
                                             D->getLocation(),
                                   D->getDepth() - TemplateArgs.getNumLevels(),
                                             D->getPosition(),
                                             D->isParameterPack(),
                                             D->getIdentifier(), InstParams);
  if (D->hasDefaultArgument()) {
    NestedNameSpecifierLoc QualifierLoc =
        D->getDefaultArgument().getTemplateQualifierLoc();
    QualifierLoc =
        SemaRef.SubstNestedNameSpecifierLoc(QualifierLoc, TemplateArgs);
    TemplateName TName = SemaRef.SubstTemplateName(
        QualifierLoc, D->getDefaultArgument().getArgument().getAsTemplate(),
        D->getDefaultArgument().getTemplateNameLoc(), TemplateArgs);
    if (!TName.isNull())
      Param->setDefaultArgument(
          TemplateArgumentLoc(TemplateArgument(TName),
                              D->getDefaultArgument().getTemplateQualifierLoc(),
                              D->getDefaultArgument().getTemplateNameLoc()),
          false);
  }
  Param->setAccess(AS_public);

  // Introduce this template parameter's instantiation into the instantiation
  // scope.
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Param);

  return Param;
}

Decl *TemplateDeclInstantiator::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  // Using directives are never dependent (and never contain any types or
  // expressions), so they require no explicit instantiation work.

  UsingDirectiveDecl *Inst
    = UsingDirectiveDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                 D->getNamespaceKeyLocation(),
                                 D->getQualifierLoc(),
                                 D->getIdentLocation(),
                                 D->getNominatedNamespace(),
                                 D->getCommonAncestor());

  // Add the using directive to its declaration context
  // only if this is not a function or method.
  if (!Owner->isFunctionOrMethod())
    Owner->addDecl(Inst);

  return Inst;
}

Decl *TemplateDeclInstantiator::VisitUsingDecl(UsingDecl *D) {

  // The nested name specifier may be dependent, for example
  //     template <typename T> struct t {
  //       struct s1 { T f1(); };
  //       struct s2 : s1 { using s1::f1; };
  //     };
  //     template struct t<int>;
  // Here, in using s1::f1, s1 refers to t<T>::s1;
  // we need to substitute for t<int>::s1.
  NestedNameSpecifierLoc QualifierLoc
    = SemaRef.SubstNestedNameSpecifierLoc(D->getQualifierLoc(),
                                          TemplateArgs);
  if (!QualifierLoc)
    return 0;

  // The name info is non-dependent, so no transformation
  // is required.
  DeclarationNameInfo NameInfo = D->getNameInfo();

  // We only need to do redeclaration lookups if we're in a class
  // scope (in fact, it's not really even possible in non-class
  // scopes).
  bool CheckRedeclaration = Owner->isRecord();

  LookupResult Prev(SemaRef, NameInfo, Sema::LookupUsingDeclName,
                    Sema::ForRedeclaration);

  UsingDecl *NewUD = UsingDecl::Create(SemaRef.Context, Owner,
                                       D->getUsingLoc(),
                                       QualifierLoc,
                                       NameInfo,
                                       D->hasTypename());

  CXXScopeSpec SS;
  SS.Adopt(QualifierLoc);
  if (CheckRedeclaration) {
    Prev.setHideTags(false);
    SemaRef.LookupQualifiedName(Prev, Owner);

    // Check for invalid redeclarations.
    if (SemaRef.CheckUsingDeclRedeclaration(D->getUsingLoc(),
                                            D->hasTypename(), SS,
                                            D->getLocation(), Prev))
      NewUD->setInvalidDecl();

  }

  if (!NewUD->isInvalidDecl() &&
      SemaRef.CheckUsingDeclQualifier(D->getUsingLoc(), SS,
                                      D->getLocation()))
    NewUD->setInvalidDecl();

  SemaRef.Context.setInstantiatedFromUsingDecl(NewUD, D);
  NewUD->setAccess(D->getAccess());
  Owner->addDecl(NewUD);

  // Don't process the shadow decls for an invalid decl.
  if (NewUD->isInvalidDecl())
    return NewUD;

  if (NameInfo.getName().getNameKind() == DeclarationName::CXXConstructorName) {
    if (SemaRef.CheckInheritingConstructorUsingDecl(NewUD))
      NewUD->setInvalidDecl();
    return NewUD;
  }

  bool isFunctionScope = Owner->isFunctionOrMethod();

  // Process the shadow decls.
  for (UsingDecl::shadow_iterator I = D->shadow_begin(), E = D->shadow_end();
         I != E; ++I) {
    UsingShadowDecl *Shadow = *I;
    NamedDecl *InstTarget =
        cast_or_null<NamedDecl>(SemaRef.FindInstantiatedDecl(
            Shadow->getLocation(), Shadow->getTargetDecl(), TemplateArgs));
    if (!InstTarget)
      return 0;

    UsingShadowDecl *PrevDecl = 0;
    if (CheckRedeclaration) {
      if (SemaRef.CheckUsingShadowDecl(NewUD, InstTarget, Prev, PrevDecl))
        continue;
    } else if (UsingShadowDecl *OldPrev = Shadow->getPreviousDecl()) {
      PrevDecl = cast_or_null<UsingShadowDecl>(SemaRef.FindInstantiatedDecl(
          Shadow->getLocation(), OldPrev, TemplateArgs));
    }

    UsingShadowDecl *InstShadow =
        SemaRef.BuildUsingShadowDecl(/*Scope*/0, NewUD, InstTarget, PrevDecl);
    SemaRef.Context.setInstantiatedFromUsingShadowDecl(InstShadow, Shadow);

    if (isFunctionScope)
      SemaRef.CurrentInstantiationScope->InstantiatedLocal(Shadow, InstShadow);
  }

  return NewUD;
}

Decl *TemplateDeclInstantiator::VisitUsingShadowDecl(UsingShadowDecl *D) {
  // Ignore these;  we handle them in bulk when processing the UsingDecl.
  return 0;
}

Decl * TemplateDeclInstantiator
    ::VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) {
  NestedNameSpecifierLoc QualifierLoc
    = SemaRef.SubstNestedNameSpecifierLoc(D->getQualifierLoc(),
                                          TemplateArgs);
  if (!QualifierLoc)
    return 0;

  CXXScopeSpec SS;
  SS.Adopt(QualifierLoc);

  // Since NameInfo refers to a typename, it cannot be a C++ special name.
  // Hence, no transformation is required for it.
  DeclarationNameInfo NameInfo(D->getDeclName(), D->getLocation());
  NamedDecl *UD =
    SemaRef.BuildUsingDeclaration(/*Scope*/ 0, D->getAccess(),
                                  D->getUsingLoc(), SS, NameInfo, 0,
                                  /*instantiation*/ true,
                                  /*typename*/ true, D->getTypenameLoc());
  if (UD)
    SemaRef.Context.setInstantiatedFromUsingDecl(cast<UsingDecl>(UD), D);

  return UD;
}

Decl * TemplateDeclInstantiator
    ::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  NestedNameSpecifierLoc QualifierLoc
      = SemaRef.SubstNestedNameSpecifierLoc(D->getQualifierLoc(), TemplateArgs);
  if (!QualifierLoc)
    return 0;

  CXXScopeSpec SS;
  SS.Adopt(QualifierLoc);

  DeclarationNameInfo NameInfo
    = SemaRef.SubstDeclarationNameInfo(D->getNameInfo(), TemplateArgs);

  NamedDecl *UD =
    SemaRef.BuildUsingDeclaration(/*Scope*/ 0, D->getAccess(),
                                  D->getUsingLoc(), SS, NameInfo, 0,
                                  /*instantiation*/ true,
                                  /*typename*/ false, SourceLocation());
  if (UD)
    SemaRef.Context.setInstantiatedFromUsingDecl(cast<UsingDecl>(UD), D);

  return UD;
}


Decl *TemplateDeclInstantiator::VisitClassScopeFunctionSpecializationDecl(
                                     ClassScopeFunctionSpecializationDecl *Decl) {
  CXXMethodDecl *OldFD = Decl->getSpecialization();
  CXXMethodDecl *NewFD = cast<CXXMethodDecl>(VisitCXXMethodDecl(OldFD,
                                                                0, true));

  LookupResult Previous(SemaRef, NewFD->getNameInfo(), Sema::LookupOrdinaryName,
                        Sema::ForRedeclaration);

  TemplateArgumentListInfo TemplateArgs;
  TemplateArgumentListInfo* TemplateArgsPtr = 0;
  if (Decl->hasExplicitTemplateArgs()) {
    TemplateArgs = Decl->templateArgs();
    TemplateArgsPtr = &TemplateArgs;
  }

  SemaRef.LookupQualifiedName(Previous, SemaRef.CurContext);
  if (SemaRef.CheckFunctionTemplateSpecialization(NewFD, TemplateArgsPtr,
                                                  Previous)) {
    NewFD->setInvalidDecl();
    return NewFD;
  }

  // Associate the specialization with the pattern.
  FunctionDecl *Specialization = cast<FunctionDecl>(Previous.getFoundDecl());
  assert(Specialization && "Class scope Specialization is null");
  SemaRef.Context.setClassScopeSpecializationPattern(Specialization, OldFD);

  return NewFD;
}

Decl *TemplateDeclInstantiator::VisitOMPThreadPrivateDecl(
                                     OMPThreadPrivateDecl *D) {
  SmallVector<Expr *, 5> Vars;
  for (ArrayRef<Expr *>::iterator I = D->varlist_begin(),
                                  E = D->varlist_end();
       I != E; ++I) {
    Expr *Var = SemaRef.SubstExpr(*I, TemplateArgs).take();
    assert(isa<DeclRefExpr>(Var) && "threadprivate arg is not a DeclRefExpr");
    Vars.push_back(Var);
  }

  OMPThreadPrivateDecl *TD =
    SemaRef.CheckOMPThreadPrivateDecl(D->getLocation(), Vars);

  return TD;
}

Decl *TemplateDeclInstantiator::VisitFunctionDecl(FunctionDecl *D) {
  return VisitFunctionDecl(D, 0);
}

Decl *TemplateDeclInstantiator::VisitCXXMethodDecl(CXXMethodDecl *D) {
  return VisitCXXMethodDecl(D, 0);
}

Decl *TemplateDeclInstantiator::VisitRecordDecl(RecordDecl *D) {
  llvm_unreachable("There are only CXXRecordDecls in C++");
}

Decl *
TemplateDeclInstantiator::VisitClassTemplateSpecializationDecl(
    ClassTemplateSpecializationDecl *D) {
  // As a MS extension, we permit class-scope explicit specialization
  // of member class templates.
  ClassTemplateDecl *ClassTemplate = D->getSpecializedTemplate();
  assert(ClassTemplate->getDeclContext()->isRecord() &&
         D->getTemplateSpecializationKind() == TSK_ExplicitSpecialization &&
         "can only instantiate an explicit specialization "
         "for a member class template");

  // Lookup the already-instantiated declaration in the instantiation
  // of the class template. FIXME: Diagnose or assert if this fails?
  DeclContext::lookup_result Found
    = Owner->lookup(ClassTemplate->getDeclName());
  if (Found.empty())
    return 0;
  ClassTemplateDecl *InstClassTemplate
    = dyn_cast<ClassTemplateDecl>(Found.front());
  if (!InstClassTemplate)
    return 0;

  // Substitute into the template arguments of the class template explicit
  // specialization.
  TemplateSpecializationTypeLoc Loc = D->getTypeAsWritten()->getTypeLoc().
                                        castAs<TemplateSpecializationTypeLoc>();
  TemplateArgumentListInfo InstTemplateArgs(Loc.getLAngleLoc(),
                                            Loc.getRAngleLoc());
  SmallVector<TemplateArgumentLoc, 4> ArgLocs;
  for (unsigned I = 0; I != Loc.getNumArgs(); ++I)
    ArgLocs.push_back(Loc.getArgLoc(I));
  if (SemaRef.Subst(ArgLocs.data(), ArgLocs.size(),
                    InstTemplateArgs, TemplateArgs))
    return 0;

  // Check that the template argument list is well-formed for this
  // class template.
  SmallVector<TemplateArgument, 4> Converted;
  if (SemaRef.CheckTemplateArgumentList(InstClassTemplate,
                                        D->getLocation(),
                                        InstTemplateArgs,
                                        false,
                                        Converted))
    return 0;

  // Figure out where to insert this class template explicit specialization
  // in the member template's set of class template explicit specializations.
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *PrevDecl =
      InstClassTemplate->findSpecialization(Converted.data(), Converted.size(),
                                            InsertPos);

  // Check whether we've already seen a conflicting instantiation of this
  // declaration (for instance, if there was a prior implicit instantiation).
  bool Ignored;
  if (PrevDecl &&
      SemaRef.CheckSpecializationInstantiationRedecl(D->getLocation(),
                                                     D->getSpecializationKind(),
                                                     PrevDecl,
                                                     PrevDecl->getSpecializationKind(),
                                                     PrevDecl->getPointOfInstantiation(),
                                                     Ignored))
    return 0;

  // If PrevDecl was a definition and D is also a definition, diagnose.
  // This happens in cases like:
  //
  //   template<typename T, typename U>
  //   struct Outer {
  //     template<typename X> struct Inner;
  //     template<> struct Inner<T> {};
  //     template<> struct Inner<U> {};
  //   };
  //
  //   Outer<int, int> outer; // error: the explicit specializations of Inner
  //                          // have the same signature.
  if (PrevDecl && PrevDecl->getDefinition() &&
      D->isThisDeclarationADefinition()) {
    SemaRef.Diag(D->getLocation(), diag::err_redefinition) << PrevDecl;
    SemaRef.Diag(PrevDecl->getDefinition()->getLocation(),
                 diag::note_previous_definition);
    return 0;
  }

  // Create the class template partial specialization declaration.
  ClassTemplateSpecializationDecl *InstD
    = ClassTemplateSpecializationDecl::Create(SemaRef.Context,
                                              D->getTagKind(),
                                              Owner,
                                              D->getLocStart(),
                                              D->getLocation(),
                                              InstClassTemplate,
                                              Converted.data(),
                                              Converted.size(),
                                              PrevDecl);

  // Add this partial specialization to the set of class template partial
  // specializations.
  if (!PrevDecl)
    InstClassTemplate->AddSpecialization(InstD, InsertPos);

  // Substitute the nested name specifier, if any.
  if (SubstQualifier(D, InstD))
    return 0;

  // Build the canonical type that describes the converted template
  // arguments of the class template explicit specialization.
  QualType CanonType = SemaRef.Context.getTemplateSpecializationType(
      TemplateName(InstClassTemplate), Converted.data(), Converted.size(),
      SemaRef.Context.getRecordType(InstD));

  // Build the fully-sugared type for this class template
  // specialization as the user wrote in the specialization
  // itself. This means that we'll pretty-print the type retrieved
  // from the specialization's declaration the way that the user
  // actually wrote the specialization, rather than formatting the
  // name based on the "canonical" representation used to store the
  // template arguments in the specialization.
  TypeSourceInfo *WrittenTy = SemaRef.Context.getTemplateSpecializationTypeInfo(
      TemplateName(InstClassTemplate), D->getLocation(), InstTemplateArgs,
      CanonType);

  InstD->setAccess(D->getAccess());
  InstD->setInstantiationOfMemberClass(D, TSK_ImplicitInstantiation);
  InstD->setSpecializationKind(D->getSpecializationKind());
  InstD->setTypeAsWritten(WrittenTy);
  InstD->setExternLoc(D->getExternLoc());
  InstD->setTemplateKeywordLoc(D->getTemplateKeywordLoc());

  Owner->addDecl(InstD);

  // Instantiate the members of the class-scope explicit specialization eagerly.
  // We don't have support for lazy instantiation of an explicit specialization
  // yet, and MSVC eagerly instantiates in this case.
  if (D->isThisDeclarationADefinition() &&
      SemaRef.InstantiateClass(D->getLocation(), InstD, D, TemplateArgs,
                               TSK_ImplicitInstantiation,
                               /*Complain=*/true))
    return 0;

  return InstD;
}

Decl *TemplateDeclInstantiator::VisitVarTemplateSpecializationDecl(
    VarTemplateSpecializationDecl *D) {

  TemplateArgumentListInfo VarTemplateArgsInfo;
  VarTemplateDecl *VarTemplate = D->getSpecializedTemplate();
  assert(VarTemplate &&
         "A template specialization without specialized template?");

  // Substitute the current template arguments.
  const TemplateArgumentListInfo &TemplateArgsInfo = D->getTemplateArgsInfo();
  VarTemplateArgsInfo.setLAngleLoc(TemplateArgsInfo.getLAngleLoc());
  VarTemplateArgsInfo.setRAngleLoc(TemplateArgsInfo.getRAngleLoc());

  if (SemaRef.Subst(TemplateArgsInfo.getArgumentArray(),
                    TemplateArgsInfo.size(), VarTemplateArgsInfo, TemplateArgs))
    return 0;

  // Check that the template argument list is well-formed for this template.
  SmallVector<TemplateArgument, 4> Converted;
  bool ExpansionIntoFixedList = false;
  if (SemaRef.CheckTemplateArgumentList(
          VarTemplate, VarTemplate->getLocStart(),
          const_cast<TemplateArgumentListInfo &>(VarTemplateArgsInfo), false,
          Converted, &ExpansionIntoFixedList))
    return 0;

  // Find the variable template specialization declaration that
  // corresponds to these arguments.
  void *InsertPos = 0;
  if (VarTemplateSpecializationDecl *VarSpec = VarTemplate->findSpecialization(
          Converted.data(), Converted.size(), InsertPos))
    // If we already have a variable template specialization, return it.
    return VarSpec;

  return VisitVarTemplateSpecializationDecl(VarTemplate, D, InsertPos,
                                            VarTemplateArgsInfo, Converted);
}

Decl *TemplateDeclInstantiator::VisitVarTemplateSpecializationDecl(
    VarTemplateDecl *VarTemplate, VarDecl *D, void *InsertPos,
    const TemplateArgumentListInfo &TemplateArgsInfo,
    llvm::ArrayRef<TemplateArgument> Converted) {

  // If this is the variable for an anonymous struct or union,
  // instantiate the anonymous struct/union type first.
  if (const RecordType *RecordTy = D->getType()->getAs<RecordType>())
    if (RecordTy->getDecl()->isAnonymousStructOrUnion())
      if (!VisitCXXRecordDecl(cast<CXXRecordDecl>(RecordTy->getDecl())))
        return 0;

  // Do substitution on the type of the declaration
  TypeSourceInfo *DI =
      SemaRef.SubstType(D->getTypeSourceInfo(), TemplateArgs,
                        D->getTypeSpecStartLoc(), D->getDeclName());
  if (!DI)
    return 0;

  if (DI->getType()->isFunctionType()) {
    SemaRef.Diag(D->getLocation(), diag::err_variable_instantiates_to_function)
        << D->isStaticDataMember() << DI->getType();
    return 0;
  }

  // Build the instantiated declaration
  VarTemplateSpecializationDecl *Var = VarTemplateSpecializationDecl::Create(
      SemaRef.Context, Owner, D->getInnerLocStart(), D->getLocation(),
      VarTemplate, DI->getType(), DI, D->getStorageClass(), Converted.data(),
      Converted.size());
  Var->setTemplateArgsInfo(TemplateArgsInfo);
  if (InsertPos)
    VarTemplate->AddSpecialization(Var, InsertPos);

  // Substitute the nested name specifier, if any.
  if (SubstQualifier(D, Var))
    return 0;

  SemaRef.BuildVariableInstantiation(Var, D, TemplateArgs, LateAttrs,
                                     Owner, StartingScope);

  return Var;
}

Decl *TemplateDeclInstantiator::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D) {
  llvm_unreachable("@defs is not supported in Objective-C++");
}

Decl *TemplateDeclInstantiator::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  // FIXME: We need to be able to instantiate FriendTemplateDecls.
  unsigned DiagID = SemaRef.getDiagnostics().getCustomDiagID(
                                               DiagnosticsEngine::Error,
                                               "cannot instantiate %0 yet");
  SemaRef.Diag(D->getLocation(), DiagID)
    << D->getDeclKindName();

  return 0;
}

Decl *TemplateDeclInstantiator::VisitDecl(Decl *D) {
  llvm_unreachable("Unexpected decl");
}

Decl *Sema::SubstDecl(Decl *D, DeclContext *Owner,
                      const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateDeclInstantiator Instantiator(*this, Owner, TemplateArgs);
  if (D->isInvalidDecl())
    return 0;

  return Instantiator.Visit(D);
}

/// \brief Instantiates a nested template parameter list in the current
/// instantiation context.
///
/// \param L The parameter list to instantiate
///
/// \returns NULL if there was an error
TemplateParameterList *
TemplateDeclInstantiator::SubstTemplateParams(TemplateParameterList *L) {
  // Get errors for all the parameters before bailing out.
  bool Invalid = false;

  unsigned N = L->size();
  typedef SmallVector<NamedDecl *, 8> ParamVector;
  ParamVector Params;
  Params.reserve(N);
  for (TemplateParameterList::iterator PI = L->begin(), PE = L->end();
       PI != PE; ++PI) {
    NamedDecl *D = cast_or_null<NamedDecl>(Visit(*PI));
    Params.push_back(D);
    Invalid = Invalid || !D || D->isInvalidDecl();
  }

  // Clean up if we had an error.
  if (Invalid)
    return NULL;

  TemplateParameterList *InstL
    = TemplateParameterList::Create(SemaRef.Context, L->getTemplateLoc(),
                                    L->getLAngleLoc(), &Params.front(), N,
                                    L->getRAngleLoc());
  return InstL;
}

/// \brief Instantiate the declaration of a class template partial
/// specialization.
///
/// \param ClassTemplate the (instantiated) class template that is partially
// specialized by the instantiation of \p PartialSpec.
///
/// \param PartialSpec the (uninstantiated) class template partial
/// specialization that we are instantiating.
///
/// \returns The instantiated partial specialization, if successful; otherwise,
/// NULL to indicate an error.
ClassTemplatePartialSpecializationDecl *
TemplateDeclInstantiator::InstantiateClassTemplatePartialSpecialization(
                                            ClassTemplateDecl *ClassTemplate,
                          ClassTemplatePartialSpecializationDecl *PartialSpec) {
  // Create a local instantiation scope for this class template partial
  // specialization, which will contain the instantiations of the template
  // parameters.
  LocalInstantiationScope Scope(SemaRef);

  // Substitute into the template parameters of the class template partial
  // specialization.
  TemplateParameterList *TempParams = PartialSpec->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return 0;

  // Substitute into the template arguments of the class template partial
  // specialization.
  const ASTTemplateArgumentListInfo *TemplArgInfo
    = PartialSpec->getTemplateArgsAsWritten();
  TemplateArgumentListInfo InstTemplateArgs(TemplArgInfo->LAngleLoc,
                                            TemplArgInfo->RAngleLoc);
  if (SemaRef.Subst(TemplArgInfo->getTemplateArgs(),
                    TemplArgInfo->NumTemplateArgs,
                    InstTemplateArgs, TemplateArgs))
    return 0;

  // Check that the template argument list is well-formed for this
  // class template.
  SmallVector<TemplateArgument, 4> Converted;
  if (SemaRef.CheckTemplateArgumentList(ClassTemplate,
                                        PartialSpec->getLocation(),
                                        InstTemplateArgs,
                                        false,
                                        Converted))
    return 0;

  // Figure out where to insert this class template partial specialization
  // in the member template's set of class template partial specializations.
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *PrevDecl
    = ClassTemplate->findPartialSpecialization(Converted.data(),
                                               Converted.size(), InsertPos);

  // Build the canonical type that describes the converted template
  // arguments of the class template partial specialization.
  QualType CanonType
    = SemaRef.Context.getTemplateSpecializationType(TemplateName(ClassTemplate),
                                                    Converted.data(),
                                                    Converted.size());

  // Build the fully-sugared type for this class template
  // specialization as the user wrote in the specialization
  // itself. This means that we'll pretty-print the type retrieved
  // from the specialization's declaration the way that the user
  // actually wrote the specialization, rather than formatting the
  // name based on the "canonical" representation used to store the
  // template arguments in the specialization.
  TypeSourceInfo *WrittenTy
    = SemaRef.Context.getTemplateSpecializationTypeInfo(
                                                    TemplateName(ClassTemplate),
                                                    PartialSpec->getLocation(),
                                                    InstTemplateArgs,
                                                    CanonType);

  if (PrevDecl) {
    // We've already seen a partial specialization with the same template
    // parameters and template arguments. This can happen, for example, when
    // substituting the outer template arguments ends up causing two
    // class template partial specializations of a member class template
    // to have identical forms, e.g.,
    //
    //   template<typename T, typename U>
    //   struct Outer {
    //     template<typename X, typename Y> struct Inner;
    //     template<typename Y> struct Inner<T, Y>;
    //     template<typename Y> struct Inner<U, Y>;
    //   };
    //
    //   Outer<int, int> outer; // error: the partial specializations of Inner
    //                          // have the same signature.
    SemaRef.Diag(PartialSpec->getLocation(), diag::err_partial_spec_redeclared)
      << WrittenTy->getType();
    SemaRef.Diag(PrevDecl->getLocation(), diag::note_prev_partial_spec_here)
      << SemaRef.Context.getTypeDeclType(PrevDecl);
    return 0;
  }


  // Create the class template partial specialization declaration.
  ClassTemplatePartialSpecializationDecl *InstPartialSpec
    = ClassTemplatePartialSpecializationDecl::Create(SemaRef.Context,
                                                     PartialSpec->getTagKind(),
                                                     Owner,
                                                     PartialSpec->getLocStart(),
                                                     PartialSpec->getLocation(),
                                                     InstParams,
                                                     ClassTemplate,
                                                     Converted.data(),
                                                     Converted.size(),
                                                     InstTemplateArgs,
                                                     CanonType,
                                                     0);
  // Substitute the nested name specifier, if any.
  if (SubstQualifier(PartialSpec, InstPartialSpec))
    return 0;

  InstPartialSpec->setInstantiatedFromMember(PartialSpec);
  InstPartialSpec->setTypeAsWritten(WrittenTy);

  // Add this partial specialization to the set of class template partial
  // specializations.
  ClassTemplate->AddPartialSpecialization(InstPartialSpec, /*InsertPos=*/0);
  return InstPartialSpec;
}

/// \brief Instantiate the declaration of a variable template partial
/// specialization.
///
/// \param VarTemplate the (instantiated) variable template that is partially
/// specialized by the instantiation of \p PartialSpec.
///
/// \param PartialSpec the (uninstantiated) variable template partial
/// specialization that we are instantiating.
///
/// \returns The instantiated partial specialization, if successful; otherwise,
/// NULL to indicate an error.
VarTemplatePartialSpecializationDecl *
TemplateDeclInstantiator::InstantiateVarTemplatePartialSpecialization(
    VarTemplateDecl *VarTemplate,
    VarTemplatePartialSpecializationDecl *PartialSpec) {
  // Create a local instantiation scope for this variable template partial
  // specialization, which will contain the instantiations of the template
  // parameters.
  LocalInstantiationScope Scope(SemaRef);

  // Substitute into the template parameters of the variable template partial
  // specialization.
  TemplateParameterList *TempParams = PartialSpec->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams)
    return 0;

  // Substitute into the template arguments of the variable template partial
  // specialization.
  const ASTTemplateArgumentListInfo *TemplArgInfo
    = PartialSpec->getTemplateArgsAsWritten();
  TemplateArgumentListInfo InstTemplateArgs(TemplArgInfo->LAngleLoc,
                                            TemplArgInfo->RAngleLoc);
  if (SemaRef.Subst(TemplArgInfo->getTemplateArgs(),
                    TemplArgInfo->NumTemplateArgs,
                    InstTemplateArgs, TemplateArgs))
    return 0;

  // Check that the template argument list is well-formed for this
  // class template.
  SmallVector<TemplateArgument, 4> Converted;
  if (SemaRef.CheckTemplateArgumentList(VarTemplate, PartialSpec->getLocation(),
                                        InstTemplateArgs, false, Converted))
    return 0;

  // Figure out where to insert this variable template partial specialization
  // in the member template's set of variable template partial specializations.
  void *InsertPos = 0;
  VarTemplateSpecializationDecl *PrevDecl =
      VarTemplate->findPartialSpecialization(Converted.data(), Converted.size(),
                                             InsertPos);

  // Build the canonical type that describes the converted template
  // arguments of the variable template partial specialization.
  QualType CanonType = SemaRef.Context.getTemplateSpecializationType(
      TemplateName(VarTemplate), Converted.data(), Converted.size());

  // Build the fully-sugared type for this variable template
  // specialization as the user wrote in the specialization
  // itself. This means that we'll pretty-print the type retrieved
  // from the specialization's declaration the way that the user
  // actually wrote the specialization, rather than formatting the
  // name based on the "canonical" representation used to store the
  // template arguments in the specialization.
  TypeSourceInfo *WrittenTy = SemaRef.Context.getTemplateSpecializationTypeInfo(
      TemplateName(VarTemplate), PartialSpec->getLocation(), InstTemplateArgs,
      CanonType);

  if (PrevDecl) {
    // We've already seen a partial specialization with the same template
    // parameters and template arguments. This can happen, for example, when
    // substituting the outer template arguments ends up causing two
    // variable template partial specializations of a member variable template
    // to have identical forms, e.g.,
    //
    //   template<typename T, typename U>
    //   struct Outer {
    //     template<typename X, typename Y> pair<X,Y> p;
    //     template<typename Y> pair<T, Y> p;
    //     template<typename Y> pair<U, Y> p;
    //   };
    //
    //   Outer<int, int> outer; // error: the partial specializations of Inner
    //                          // have the same signature.
    SemaRef.Diag(PartialSpec->getLocation(),
                 diag::err_var_partial_spec_redeclared)
        << WrittenTy->getType();
    SemaRef.Diag(PrevDecl->getLocation(),
                 diag::note_var_prev_partial_spec_here);
    return 0;
  }

  // Do substitution on the type of the declaration
  TypeSourceInfo *DI = SemaRef.SubstType(
      PartialSpec->getTypeSourceInfo(), TemplateArgs,
      PartialSpec->getTypeSpecStartLoc(), PartialSpec->getDeclName());
  if (!DI)
    return 0;

  if (DI->getType()->isFunctionType()) {
    SemaRef.Diag(PartialSpec->getLocation(),
                 diag::err_variable_instantiates_to_function)
        << PartialSpec->isStaticDataMember() << DI->getType();
    return 0;
  }

  // Create the variable template partial specialization declaration.
  VarTemplatePartialSpecializationDecl *InstPartialSpec =
      VarTemplatePartialSpecializationDecl::Create(
          SemaRef.Context, Owner, PartialSpec->getInnerLocStart(),
          PartialSpec->getLocation(), InstParams, VarTemplate, DI->getType(),
          DI, PartialSpec->getStorageClass(), Converted.data(),
          Converted.size(), InstTemplateArgs);

  // Substitute the nested name specifier, if any.
  if (SubstQualifier(PartialSpec, InstPartialSpec))
    return 0;

  InstPartialSpec->setInstantiatedFromMember(PartialSpec);
  InstPartialSpec->setTypeAsWritten(WrittenTy);

  // Add this partial specialization to the set of variable template partial
  // specializations. The instantiation of the initializer is not necessary.
  VarTemplate->AddPartialSpecialization(InstPartialSpec, /*InsertPos=*/0);

  SemaRef.BuildVariableInstantiation(InstPartialSpec, PartialSpec, TemplateArgs,
                                     LateAttrs, Owner, StartingScope);

  return InstPartialSpec;
}

TypeSourceInfo*
TemplateDeclInstantiator::SubstFunctionType(FunctionDecl *D,
                              SmallVectorImpl<ParmVarDecl *> &Params) {
  TypeSourceInfo *OldTInfo = D->getTypeSourceInfo();
  assert(OldTInfo && "substituting function without type source info");
  assert(Params.empty() && "parameter vector is non-empty at start");
  
  CXXRecordDecl *ThisContext = 0;
  unsigned ThisTypeQuals = 0;
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    ThisContext = cast<CXXRecordDecl>(Owner);
    ThisTypeQuals = Method->getTypeQualifiers();
  }
  
  TypeSourceInfo *NewTInfo
    = SemaRef.SubstFunctionDeclType(OldTInfo, TemplateArgs,
                                    D->getTypeSpecStartLoc(),
                                    D->getDeclName(),
                                    ThisContext, ThisTypeQuals);
  if (!NewTInfo)
    return 0;

  TypeLoc OldTL = OldTInfo->getTypeLoc().IgnoreParens();
  if (FunctionProtoTypeLoc OldProtoLoc = OldTL.getAs<FunctionProtoTypeLoc>()) {
    if (NewTInfo != OldTInfo) {
      // Get parameters from the new type info.
      TypeLoc NewTL = NewTInfo->getTypeLoc().IgnoreParens();
      FunctionProtoTypeLoc NewProtoLoc = NewTL.castAs<FunctionProtoTypeLoc>();
      unsigned NewIdx = 0;
      for (unsigned OldIdx = 0, NumOldParams = OldProtoLoc.getNumArgs();
           OldIdx != NumOldParams; ++OldIdx) {
        ParmVarDecl *OldParam = OldProtoLoc.getArg(OldIdx);
        LocalInstantiationScope *Scope = SemaRef.CurrentInstantiationScope;

        Optional<unsigned> NumArgumentsInExpansion;
        if (OldParam->isParameterPack())
          NumArgumentsInExpansion =
              SemaRef.getNumArgumentsInExpansion(OldParam->getType(),
                                                 TemplateArgs);
        if (!NumArgumentsInExpansion) {
          // Simple case: normal parameter, or a parameter pack that's
          // instantiated to a (still-dependent) parameter pack.
          ParmVarDecl *NewParam = NewProtoLoc.getArg(NewIdx++);
          Params.push_back(NewParam);
          Scope->InstantiatedLocal(OldParam, NewParam);
        } else {
          // Parameter pack expansion: make the instantiation an argument pack.
          Scope->MakeInstantiatedLocalArgPack(OldParam);
          for (unsigned I = 0; I != *NumArgumentsInExpansion; ++I) {
            ParmVarDecl *NewParam = NewProtoLoc.getArg(NewIdx++);
            Params.push_back(NewParam);
            Scope->InstantiatedLocalPackArg(OldParam, NewParam);
          }
        }
      }
    } else {
      // The function type itself was not dependent and therefore no
      // substitution occurred. However, we still need to instantiate
      // the function parameters themselves.
      const FunctionProtoType *OldProto =
          cast<FunctionProtoType>(OldProtoLoc.getType());
      for (unsigned i = 0, i_end = OldProtoLoc.getNumArgs(); i != i_end; ++i) {
        ParmVarDecl *OldParam = OldProtoLoc.getArg(i);
        if (!OldParam) {
          Params.push_back(SemaRef.BuildParmVarDeclForTypedef(
              D, D->getLocation(), OldProto->getArgType(i)));
          continue;
        }

        ParmVarDecl *Parm =
            cast_or_null<ParmVarDecl>(VisitParmVarDecl(OldParam));
        if (!Parm)
          return 0;
        Params.push_back(Parm);
      }
    }
  } else {
    // If the type of this function, after ignoring parentheses, is not
    // *directly* a function type, then we're instantiating a function that
    // was declared via a typedef or with attributes, e.g.,
    //
    //   typedef int functype(int, int);
    //   functype func;
    //   int __cdecl meth(int, int);
    //
    // In this case, we'll just go instantiate the ParmVarDecls that we
    // synthesized in the method declaration.
    SmallVector<QualType, 4> ParamTypes;
    if (SemaRef.SubstParmTypes(D->getLocation(), D->param_begin(),
                               D->getNumParams(), TemplateArgs, ParamTypes,
                               &Params))
      return 0;
  }

  return NewTInfo;
}

/// Introduce the instantiated function parameters into the local
/// instantiation scope, and set the parameter names to those used
/// in the template.
static void addInstantiatedParametersToScope(Sema &S, FunctionDecl *Function,
                                             const FunctionDecl *PatternDecl,
                                             LocalInstantiationScope &Scope,
                           const MultiLevelTemplateArgumentList &TemplateArgs) {
  unsigned FParamIdx = 0;
  for (unsigned I = 0, N = PatternDecl->getNumParams(); I != N; ++I) {
    const ParmVarDecl *PatternParam = PatternDecl->getParamDecl(I);
    if (!PatternParam->isParameterPack()) {
      // Simple case: not a parameter pack.
      assert(FParamIdx < Function->getNumParams());
      ParmVarDecl *FunctionParam = Function->getParamDecl(FParamIdx);
      FunctionParam->setDeclName(PatternParam->getDeclName());
      Scope.InstantiatedLocal(PatternParam, FunctionParam);
      ++FParamIdx;
      continue;
    }

    // Expand the parameter pack.
    Scope.MakeInstantiatedLocalArgPack(PatternParam);
    Optional<unsigned> NumArgumentsInExpansion
      = S.getNumArgumentsInExpansion(PatternParam->getType(), TemplateArgs);
    assert(NumArgumentsInExpansion &&
           "should only be called when all template arguments are known");
    for (unsigned Arg = 0; Arg < *NumArgumentsInExpansion; ++Arg) {
      ParmVarDecl *FunctionParam = Function->getParamDecl(FParamIdx);
      FunctionParam->setDeclName(PatternParam->getDeclName());
      Scope.InstantiatedLocalPackArg(PatternParam, FunctionParam);
      ++FParamIdx;
    }
  }
}

static void InstantiateExceptionSpec(Sema &SemaRef, FunctionDecl *New,
                                     const FunctionProtoType *Proto,
                           const MultiLevelTemplateArgumentList &TemplateArgs) {
  assert(Proto->getExceptionSpecType() != EST_Uninstantiated);

  // C++11 [expr.prim.general]p3:
  //   If a declaration declares a member function or member function 
  //   template of a class X, the expression this is a prvalue of type 
  //   "pointer to cv-qualifier-seq X" between the optional cv-qualifer-seq
  //   and the end of the function-definition, member-declarator, or 
  //   declarator.    
  CXXRecordDecl *ThisContext = 0;
  unsigned ThisTypeQuals = 0;
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(New)) {
    ThisContext = Method->getParent();
    ThisTypeQuals = Method->getTypeQualifiers();
  }
  Sema::CXXThisScopeRAII ThisScope(SemaRef, ThisContext, ThisTypeQuals,
                                   SemaRef.getLangOpts().CPlusPlus11);

  // The function has an exception specification or a "noreturn"
  // attribute. Substitute into each of the exception types.
  SmallVector<QualType, 4> Exceptions;
  for (unsigned I = 0, N = Proto->getNumExceptions(); I != N; ++I) {
    // FIXME: Poor location information!
    if (const PackExpansionType *PackExpansion
          = Proto->getExceptionType(I)->getAs<PackExpansionType>()) {
      // We have a pack expansion. Instantiate it.
      SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      SemaRef.collectUnexpandedParameterPacks(PackExpansion->getPattern(),
                                              Unexpanded);
      assert(!Unexpanded.empty() &&
             "Pack expansion without parameter packs?");

      bool Expand = false;
      bool RetainExpansion = false;
      Optional<unsigned> NumExpansions = PackExpansion->getNumExpansions();
      if (SemaRef.CheckParameterPacksForExpansion(New->getLocation(),
                                                  SourceRange(),
                                                  Unexpanded,
                                                  TemplateArgs,
                                                  Expand,
                                                  RetainExpansion,
                                                  NumExpansions))
        break;

      if (!Expand) {
        // We can't expand this pack expansion into separate arguments yet;
        // just substitute into the pattern and create a new pack expansion
        // type.
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, -1);
        QualType T = SemaRef.SubstType(PackExpansion->getPattern(),
                                       TemplateArgs,
                                     New->getLocation(), New->getDeclName());
        if (T.isNull())
          break;

        T = SemaRef.Context.getPackExpansionType(T, NumExpansions);
        Exceptions.push_back(T);
        continue;
      }

      // Substitute into the pack expansion pattern for each template
      bool Invalid = false;
      for (unsigned ArgIdx = 0; ArgIdx != *NumExpansions; ++ArgIdx) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, ArgIdx);

        QualType T = SemaRef.SubstType(PackExpansion->getPattern(),
                                       TemplateArgs,
                                     New->getLocation(), New->getDeclName());
        if (T.isNull()) {
          Invalid = true;
          break;
        }

        Exceptions.push_back(T);
      }

      if (Invalid)
        break;

      continue;
    }

    QualType T
      = SemaRef.SubstType(Proto->getExceptionType(I), TemplateArgs,
                          New->getLocation(), New->getDeclName());
    if (T.isNull() ||
        SemaRef.CheckSpecifiedExceptionType(T, New->getLocation()))
      continue;

    Exceptions.push_back(T);
  }
  Expr *NoexceptExpr = 0;
  if (Expr *OldNoexceptExpr = Proto->getNoexceptExpr()) {
    EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                                 Sema::ConstantEvaluated);
    ExprResult E = SemaRef.SubstExpr(OldNoexceptExpr, TemplateArgs);
    if (E.isUsable())
      E = SemaRef.CheckBooleanCondition(E.get(), E.get()->getLocStart());

    if (E.isUsable()) {
      NoexceptExpr = E.take();
      if (!NoexceptExpr->isTypeDependent() &&
          !NoexceptExpr->isValueDependent())
        NoexceptExpr
          = SemaRef.VerifyIntegerConstantExpression(NoexceptExpr,
              0, diag::err_noexcept_needs_constant_expression,
              /*AllowFold*/ false).take();
    }
  }

  // Rebuild the function type
  const FunctionProtoType *NewProto
    = New->getType()->getAs<FunctionProtoType>();
  assert(NewProto && "Template instantiation without function prototype?");

  FunctionProtoType::ExtProtoInfo EPI = NewProto->getExtProtoInfo();
  EPI.ExceptionSpecType = Proto->getExceptionSpecType();
  EPI.NumExceptions = Exceptions.size();
  EPI.Exceptions = Exceptions.data();
  EPI.NoexceptExpr = NoexceptExpr;

  New->setType(SemaRef.Context.getFunctionType(NewProto->getResultType(),
                                               NewProto->getArgTypes(), EPI));
}

void Sema::InstantiateExceptionSpec(SourceLocation PointOfInstantiation,
                                    FunctionDecl *Decl) {
  const FunctionProtoType *Proto = Decl->getType()->castAs<FunctionProtoType>();
  if (Proto->getExceptionSpecType() != EST_Uninstantiated)
    return;

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Decl,
                             InstantiatingTemplate::ExceptionSpecification());
  if (Inst.isInvalid()) {
    // We hit the instantiation depth limit. Clear the exception specification
    // so that our callers don't have to cope with EST_Uninstantiated.
    FunctionProtoType::ExtProtoInfo EPI = Proto->getExtProtoInfo();
    EPI.ExceptionSpecType = EST_None;
    Decl->setType(Context.getFunctionType(Proto->getResultType(),
                                          Proto->getArgTypes(), EPI));
    return;
  }

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  Sema::ContextRAII savedContext(*this, Decl);
  LocalInstantiationScope Scope(*this);

  MultiLevelTemplateArgumentList TemplateArgs =
    getTemplateInstantiationArgs(Decl, 0, /*RelativeToPrimary*/true);

  FunctionDecl *Template = Proto->getExceptionSpecTemplate();
  addInstantiatedParametersToScope(*this, Decl, Template, Scope, TemplateArgs);

  ::InstantiateExceptionSpec(*this, Decl,
                             Template->getType()->castAs<FunctionProtoType>(),
                             TemplateArgs);
}

/// \brief Initializes the common fields of an instantiation function
/// declaration (New) from the corresponding fields of its template (Tmpl).
///
/// \returns true if there was an error
bool
TemplateDeclInstantiator::InitFunctionInstantiation(FunctionDecl *New,
                                                    FunctionDecl *Tmpl) {
  if (Tmpl->isDeleted())
    New->setDeletedAsWritten();

  // If we are performing substituting explicitly-specified template arguments
  // or deduced template arguments into a function template and we reach this
  // point, we are now past the point where SFINAE applies and have committed
  // to keeping the new function template specialization. We therefore
  // convert the active template instantiation for the function template
  // into a template instantiation for this specific function template
  // specialization, which is not a SFINAE context, so that we diagnose any
  // further errors in the declaration itself.
  typedef Sema::ActiveTemplateInstantiation ActiveInstType;
  ActiveInstType &ActiveInst = SemaRef.ActiveTemplateInstantiations.back();
  if (ActiveInst.Kind == ActiveInstType::ExplicitTemplateArgumentSubstitution ||
      ActiveInst.Kind == ActiveInstType::DeducedTemplateArgumentSubstitution) {
    if (FunctionTemplateDecl *FunTmpl
          = dyn_cast<FunctionTemplateDecl>(ActiveInst.Entity)) {
      assert(FunTmpl->getTemplatedDecl() == Tmpl &&
             "Deduction from the wrong function template?");
      (void) FunTmpl;
      ActiveInst.Kind = ActiveInstType::TemplateInstantiation;
      ActiveInst.Entity = New;
    }
  }

  const FunctionProtoType *Proto = Tmpl->getType()->getAs<FunctionProtoType>();
  assert(Proto && "Function template without prototype?");

  if (Proto->hasExceptionSpec() || Proto->getNoReturnAttr()) {
    FunctionProtoType::ExtProtoInfo EPI = Proto->getExtProtoInfo();

    // DR1330: In C++11, defer instantiation of a non-trivial
    // exception specification.
    if (SemaRef.getLangOpts().CPlusPlus11 &&
        EPI.ExceptionSpecType != EST_None &&
        EPI.ExceptionSpecType != EST_DynamicNone &&
        EPI.ExceptionSpecType != EST_BasicNoexcept) {
      FunctionDecl *ExceptionSpecTemplate = Tmpl;
      if (EPI.ExceptionSpecType == EST_Uninstantiated)
        ExceptionSpecTemplate = EPI.ExceptionSpecTemplate;
      ExceptionSpecificationType NewEST = EST_Uninstantiated;
      if (EPI.ExceptionSpecType == EST_Unevaluated)
        NewEST = EST_Unevaluated;

      // Mark the function has having an uninstantiated exception specification.
      const FunctionProtoType *NewProto
        = New->getType()->getAs<FunctionProtoType>();
      assert(NewProto && "Template instantiation without function prototype?");
      EPI = NewProto->getExtProtoInfo();
      EPI.ExceptionSpecType = NewEST;
      EPI.ExceptionSpecDecl = New;
      EPI.ExceptionSpecTemplate = ExceptionSpecTemplate;
      New->setType(SemaRef.Context.getFunctionType(
          NewProto->getResultType(), NewProto->getArgTypes(), EPI));
    } else {
      ::InstantiateExceptionSpec(SemaRef, New, Proto, TemplateArgs);
    }
  }

  // Get the definition. Leaves the variable unchanged if undefined.
  const FunctionDecl *Definition = Tmpl;
  Tmpl->isDefined(Definition);

  SemaRef.InstantiateAttrs(TemplateArgs, Definition, New,
                           LateAttrs, StartingScope);

  return false;
}

/// \brief Initializes common fields of an instantiated method
/// declaration (New) from the corresponding fields of its template
/// (Tmpl).
///
/// \returns true if there was an error
bool
TemplateDeclInstantiator::InitMethodInstantiation(CXXMethodDecl *New,
                                                  CXXMethodDecl *Tmpl) {
  if (InitFunctionInstantiation(New, Tmpl))
    return true;

  New->setAccess(Tmpl->getAccess());
  if (Tmpl->isVirtualAsWritten())
    New->setVirtualAsWritten(true);

  // FIXME: New needs a pointer to Tmpl
  return false;
}

/// \brief Instantiate the definition of the given function from its
/// template.
///
/// \param PointOfInstantiation the point at which the instantiation was
/// required. Note that this is not precisely a "point of instantiation"
/// for the function, but it's close.
///
/// \param Function the already-instantiated declaration of a
/// function template specialization or member function of a class template
/// specialization.
///
/// \param Recursive if true, recursively instantiates any functions that
/// are required by this instantiation.
///
/// \param DefinitionRequired if true, then we are performing an explicit
/// instantiation where the body of the function is required. Complain if
/// there is no such body.
void Sema::InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                         FunctionDecl *Function,
                                         bool Recursive,
                                         bool DefinitionRequired) {
  if (Function->isInvalidDecl() || Function->isDefined())
    return;

  // Never instantiate an explicit specialization except if it is a class scope
  // explicit specialization.
  if (Function->getTemplateSpecializationKind() == TSK_ExplicitSpecialization &&
      !Function->getClassScopeSpecializationPattern())
    return;

  // Find the function body that we'll be substituting.
  const FunctionDecl *PatternDecl = Function->getTemplateInstantiationPattern();
  assert(PatternDecl && "instantiating a non-template");

  Stmt *Pattern = PatternDecl->getBody(PatternDecl);
  assert(PatternDecl && "template definition is not a template");
  if (!Pattern) {
    // Try to find a defaulted definition
    PatternDecl->isDefined(PatternDecl);
  }
  assert(PatternDecl && "template definition is not a template");

  // Postpone late parsed template instantiations.
  if (PatternDecl->isLateTemplateParsed() &&
      !LateTemplateParser) {
    PendingInstantiations.push_back(
      std::make_pair(Function, PointOfInstantiation));
    return;
  }

  // Call the LateTemplateParser callback if there is a need to late parse
  // a templated function definition.
  if (!Pattern && PatternDecl->isLateTemplateParsed() &&
      LateTemplateParser) {
    // FIXME: Optimize to allow individual templates to be deserialized.
    if (PatternDecl->isFromASTFile())
      ExternalSource->ReadLateParsedTemplates(LateParsedTemplateMap);

    LateParsedTemplate *LPT = LateParsedTemplateMap.lookup(PatternDecl);
    assert(LPT && "missing LateParsedTemplate");
    LateTemplateParser(OpaqueParser, *LPT);
    Pattern = PatternDecl->getBody(PatternDecl);
  }

  if (!Pattern && !PatternDecl->isDefaulted()) {
    if (DefinitionRequired) {
      if (Function->getPrimaryTemplate())
        Diag(PointOfInstantiation,
             diag::err_explicit_instantiation_undefined_func_template)
          << Function->getPrimaryTemplate();
      else
        Diag(PointOfInstantiation,
             diag::err_explicit_instantiation_undefined_member)
          << 1 << Function->getDeclName() << Function->getDeclContext();

      if (PatternDecl)
        Diag(PatternDecl->getLocation(),
             diag::note_explicit_instantiation_here);
      Function->setInvalidDecl();
    } else if (Function->getTemplateSpecializationKind()
                 == TSK_ExplicitInstantiationDefinition) {
      PendingInstantiations.push_back(
        std::make_pair(Function, PointOfInstantiation));
    }

    return;
  }

  // C++1y [temp.explicit]p10:
  //   Except for inline functions, declarations with types deduced from their
  //   initializer or return value, and class template specializations, other
  //   explicit instantiation declarations have the effect of suppressing the
  //   implicit instantiation of the entity to which they refer.
  if (Function->getTemplateSpecializationKind()
        == TSK_ExplicitInstantiationDeclaration &&
      !PatternDecl->isInlined() &&
      !PatternDecl->getResultType()->getContainedAutoType())
    return;

  if (PatternDecl->isInlined())
    Function->setImplicitlyInline();

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Function);
  if (Inst.isInvalid())
    return;

  // Copy the inner loc start from the pattern.
  Function->setInnerLocStart(PatternDecl->getInnerLocStart());

  // If we're performing recursive template instantiation, create our own
  // queue of pending implicit instantiations that we will instantiate later,
  // while we're still within our own instantiation context.
  SmallVector<VTableUse, 16> SavedVTableUses;
  std::deque<PendingImplicitInstantiation> SavedPendingInstantiations;
  SavePendingLocalImplicitInstantiationsRAII
      SavedPendingLocalImplicitInstantiations(*this);
  if (Recursive) {
    VTableUses.swap(SavedVTableUses);
    PendingInstantiations.swap(SavedPendingInstantiations);
  }

  EnterExpressionEvaluationContext EvalContext(*this,
                                               Sema::PotentiallyEvaluated);

  // Introduce a new scope where local variable instantiations will be
  // recorded, unless we're actually a member function within a local
  // class, in which case we need to merge our results with the parent
  // scope (of the enclosing function).
  bool MergeWithParentScope = false;
  if (CXXRecordDecl *Rec = dyn_cast<CXXRecordDecl>(Function->getDeclContext()))
    MergeWithParentScope = Rec->isLocalClass();

  LocalInstantiationScope Scope(*this, MergeWithParentScope);

  if (PatternDecl->isDefaulted())
    SetDeclDefaulted(Function, PatternDecl->getLocation());
  else {
    ActOnStartOfFunctionDef(0, Function);

    // Enter the scope of this instantiation. We don't use
    // PushDeclContext because we don't have a scope.
    Sema::ContextRAII savedContext(*this, Function);

    MultiLevelTemplateArgumentList TemplateArgs =
      getTemplateInstantiationArgs(Function, 0, false, PatternDecl);

    addInstantiatedParametersToScope(*this, Function, PatternDecl, Scope,
                                     TemplateArgs);

    // If this is a constructor, instantiate the member initializers.
    if (const CXXConstructorDecl *Ctor =
          dyn_cast<CXXConstructorDecl>(PatternDecl)) {
      InstantiateMemInitializers(cast<CXXConstructorDecl>(Function), Ctor,
                                 TemplateArgs);
    }

    // Instantiate the function body.
    StmtResult Body = SubstStmt(Pattern, TemplateArgs);

    if (Body.isInvalid())
      Function->setInvalidDecl();

    ActOnFinishFunctionBody(Function, Body.get(),
                            /*IsInstantiation=*/true);

    PerformDependentDiagnostics(PatternDecl, TemplateArgs);

    savedContext.pop();
  }

  DeclGroupRef DG(Function);
  Consumer.HandleTopLevelDecl(DG);

  // This class may have local implicit instantiations that need to be
  // instantiation within this scope.
  PerformPendingInstantiations(/*LocalOnly=*/true);
  Scope.Exit();

  if (Recursive) {
    // Define any pending vtables.
    DefineUsedVTables();

    // Instantiate any pending implicit instantiations found during the
    // instantiation of this template.
    PerformPendingInstantiations();

    // Restore the set of pending vtables.
    assert(VTableUses.empty() &&
           "VTableUses should be empty before it is discarded.");
    VTableUses.swap(SavedVTableUses);

    // Restore the set of pending implicit instantiations.
    assert(PendingInstantiations.empty() &&
           "PendingInstantiations should be empty before it is discarded.");
    PendingInstantiations.swap(SavedPendingInstantiations);
  }
}

VarTemplateSpecializationDecl *Sema::BuildVarTemplateInstantiation(
    VarTemplateDecl *VarTemplate, VarDecl *FromVar,
    const TemplateArgumentList &TemplateArgList,
    const TemplateArgumentListInfo &TemplateArgsInfo,
    SmallVectorImpl<TemplateArgument> &Converted,
    SourceLocation PointOfInstantiation, void *InsertPos,
    LateInstantiatedAttrVec *LateAttrs,
    LocalInstantiationScope *StartingScope) {
  if (FromVar->isInvalidDecl())
    return 0;

  InstantiatingTemplate Inst(*this, PointOfInstantiation, FromVar);
  if (Inst.isInvalid())
    return 0;

  MultiLevelTemplateArgumentList TemplateArgLists;
  TemplateArgLists.addOuterTemplateArguments(&TemplateArgList);

  // Instantiate the first declaration of the variable template: for a partial
  // specialization of a static data member template, the first declaration may
  // or may not be the declaration in the class; if it's in the class, we want
  // to instantiate a member in the class (a declaration), and if it's outside,
  // we want to instantiate a definition.
  FromVar = FromVar->getFirstDecl();

  MultiLevelTemplateArgumentList MultiLevelList(TemplateArgList);
  TemplateDeclInstantiator Instantiator(*this, FromVar->getDeclContext(),
                                        MultiLevelList);

  // TODO: Set LateAttrs and StartingScope ...

  return cast_or_null<VarTemplateSpecializationDecl>(
      Instantiator.VisitVarTemplateSpecializationDecl(
          VarTemplate, FromVar, InsertPos, TemplateArgsInfo, Converted));
}

/// \brief Instantiates a variable template specialization by completing it
/// with appropriate type information and initializer.
VarTemplateSpecializationDecl *Sema::CompleteVarTemplateSpecializationDecl(
    VarTemplateSpecializationDecl *VarSpec, VarDecl *PatternDecl,
    const MultiLevelTemplateArgumentList &TemplateArgs) {

  // Do substitution on the type of the declaration
  TypeSourceInfo *DI =
      SubstType(PatternDecl->getTypeSourceInfo(), TemplateArgs,
                PatternDecl->getTypeSpecStartLoc(), PatternDecl->getDeclName());
  if (!DI)
    return 0;

  // Update the type of this variable template specialization.
  VarSpec->setType(DI->getType());

  // Instantiate the initializer.
  InstantiateVariableInitializer(VarSpec, PatternDecl, TemplateArgs);

  return VarSpec;
}

/// BuildVariableInstantiation - Used after a new variable has been created.
/// Sets basic variable data and decides whether to postpone the
/// variable instantiation.
void Sema::BuildVariableInstantiation(
    VarDecl *NewVar, VarDecl *OldVar,
    const MultiLevelTemplateArgumentList &TemplateArgs,
    LateInstantiatedAttrVec *LateAttrs, DeclContext *Owner,
    LocalInstantiationScope *StartingScope,
    bool InstantiatingVarTemplate) {

  // If we are instantiating a local extern declaration, the
  // instantiation belongs lexically to the containing function.
  // If we are instantiating a static data member defined
  // out-of-line, the instantiation will have the same lexical
  // context (which will be a namespace scope) as the template.
  if (OldVar->isLocalExternDecl()) {
    NewVar->setLocalExternDecl();
    NewVar->setLexicalDeclContext(Owner);
  } else if (OldVar->isOutOfLine())
    NewVar->setLexicalDeclContext(OldVar->getLexicalDeclContext());
  NewVar->setTSCSpec(OldVar->getTSCSpec());
  NewVar->setInitStyle(OldVar->getInitStyle());
  NewVar->setCXXForRangeDecl(OldVar->isCXXForRangeDecl());
  NewVar->setConstexpr(OldVar->isConstexpr());
  NewVar->setInitCapture(OldVar->isInitCapture());
  NewVar->setPreviousDeclInSameBlockScope(
      OldVar->isPreviousDeclInSameBlockScope());
  NewVar->setAccess(OldVar->getAccess());

  if (!OldVar->isStaticDataMember()) {
    if (OldVar->isUsed(false))
      NewVar->setIsUsed();
    NewVar->setReferenced(OldVar->isReferenced());
  }

  // See if the old variable had a type-specifier that defined an anonymous tag.
  // If it did, mark the new variable as being the declarator for the new
  // anonymous tag.
  if (const TagType *OldTagType = OldVar->getType()->getAs<TagType>()) {
    TagDecl *OldTag = OldTagType->getDecl();
    if (OldTag->getDeclaratorForAnonDecl() == OldVar) {
      TagDecl *NewTag = NewVar->getType()->castAs<TagType>()->getDecl();
      assert(!NewTag->hasNameForLinkage() &&
             !NewTag->hasDeclaratorForAnonDecl());
      NewTag->setDeclaratorForAnonDecl(NewVar);
    }
  }

  InstantiateAttrs(TemplateArgs, OldVar, NewVar, LateAttrs, StartingScope);

  if (NewVar->hasAttrs())
    CheckAlignasUnderalignment(NewVar);

  LookupResult Previous(
      *this, NewVar->getDeclName(), NewVar->getLocation(),
      NewVar->isLocalExternDecl() ? Sema::LookupRedeclarationWithLinkage
                                  : Sema::LookupOrdinaryName,
      Sema::ForRedeclaration);

  if (NewVar->isLocalExternDecl() && OldVar->getPreviousDecl()) {
    // We have a previous declaration. Use that one, so we merge with the
    // right type.
    if (NamedDecl *NewPrev = FindInstantiatedDecl(
            NewVar->getLocation(), OldVar->getPreviousDecl(), TemplateArgs))
      Previous.addDecl(NewPrev);
  } else if (!isa<VarTemplateSpecializationDecl>(NewVar) &&
             OldVar->hasLinkage())
    LookupQualifiedName(Previous, NewVar->getDeclContext(), false);
  CheckVariableDeclaration(NewVar, Previous);

  if (!InstantiatingVarTemplate) {
    NewVar->getLexicalDeclContext()->addHiddenDecl(NewVar);
    if (!NewVar->isLocalExternDecl() || !NewVar->getPreviousDecl())
      NewVar->getDeclContext()->makeDeclVisibleInContext(NewVar);
  }

  if (!OldVar->isOutOfLine()) {
    if (NewVar->getDeclContext()->isFunctionOrMethod())
      CurrentInstantiationScope->InstantiatedLocal(OldVar, NewVar);
  }

  // Link instantiations of static data members back to the template from
  // which they were instantiated.
  if (NewVar->isStaticDataMember() && !InstantiatingVarTemplate)
    NewVar->setInstantiationOfStaticDataMember(OldVar,
                                               TSK_ImplicitInstantiation);

  // Delay instantiation of the initializer for variable templates until a
  // definition of the variable is needed.
  if (!isa<VarTemplateSpecializationDecl>(NewVar) && !InstantiatingVarTemplate)
    InstantiateVariableInitializer(NewVar, OldVar, TemplateArgs);

  // Diagnose unused local variables with dependent types, where the diagnostic
  // will have been deferred.
  if (!NewVar->isInvalidDecl() &&
      NewVar->getDeclContext()->isFunctionOrMethod() && !NewVar->isUsed() &&
      OldVar->getType()->isDependentType())
    DiagnoseUnusedDecl(NewVar);
}

/// \brief Instantiate the initializer of a variable.
void Sema::InstantiateVariableInitializer(
    VarDecl *Var, VarDecl *OldVar,
    const MultiLevelTemplateArgumentList &TemplateArgs) {

  if (Var->getAnyInitializer())
    // We already have an initializer in the class.
    return;

  if (OldVar->getInit()) {
    if (Var->isStaticDataMember() && !OldVar->isOutOfLine())
      PushExpressionEvaluationContext(Sema::ConstantEvaluated, OldVar);
    else
      PushExpressionEvaluationContext(Sema::PotentiallyEvaluated, OldVar);

    // Instantiate the initializer.
    ExprResult Init =
        SubstInitializer(OldVar->getInit(), TemplateArgs,
                         OldVar->getInitStyle() == VarDecl::CallInit);
    if (!Init.isInvalid()) {
      bool TypeMayContainAuto = true;
      if (Init.get()) {
        bool DirectInit = OldVar->isDirectInit();
        AddInitializerToDecl(Var, Init.take(), DirectInit, TypeMayContainAuto);
      } else
        ActOnUninitializedDecl(Var, TypeMayContainAuto);
    } else {
      // FIXME: Not too happy about invalidating the declaration
      // because of a bogus initializer.
      Var->setInvalidDecl();
    }

    PopExpressionEvaluationContext();
  } else if ((!Var->isStaticDataMember() || Var->isOutOfLine()) &&
             !Var->isCXXForRangeDecl())
    ActOnUninitializedDecl(Var, false);
}

/// \brief Instantiate the definition of the given variable from its
/// template.
///
/// \param PointOfInstantiation the point at which the instantiation was
/// required. Note that this is not precisely a "point of instantiation"
/// for the function, but it's close.
///
/// \param Var the already-instantiated declaration of a static member
/// variable of a class template specialization.
///
/// \param Recursive if true, recursively instantiates any functions that
/// are required by this instantiation.
///
/// \param DefinitionRequired if true, then we are performing an explicit
/// instantiation where an out-of-line definition of the member variable
/// is required. Complain if there is no such definition.
void Sema::InstantiateStaticDataMemberDefinition(
                                          SourceLocation PointOfInstantiation,
                                                 VarDecl *Var,
                                                 bool Recursive,
                                                 bool DefinitionRequired) {
  InstantiateVariableDefinition(PointOfInstantiation, Var, Recursive,
                                DefinitionRequired);
}

void Sema::InstantiateVariableDefinition(SourceLocation PointOfInstantiation,
                                         VarDecl *Var, bool Recursive,
                                         bool DefinitionRequired) {
  if (Var->isInvalidDecl())
    return;

  VarTemplateSpecializationDecl *VarSpec =
      dyn_cast<VarTemplateSpecializationDecl>(Var);
  VarDecl *PatternDecl = 0, *Def = 0;
  MultiLevelTemplateArgumentList TemplateArgs =
      getTemplateInstantiationArgs(Var);

  if (VarSpec) {
    // If this is a variable template specialization, make sure that it is
    // non-dependent, then find its instantiation pattern.
    bool InstantiationDependent = false;
    assert(!TemplateSpecializationType::anyDependentTemplateArguments(
               VarSpec->getTemplateArgsInfo(), InstantiationDependent) &&
           "Only instantiate variable template specializations that are "
           "not type-dependent");
    (void)InstantiationDependent;

    // Find the variable initialization that we'll be substituting. If the
    // pattern was instantiated from a member template, look back further to
    // find the real pattern.
    assert(VarSpec->getSpecializedTemplate() &&
           "Specialization without specialized template?");
    llvm::PointerUnion<VarTemplateDecl *,
                       VarTemplatePartialSpecializationDecl *> PatternPtr =
        VarSpec->getSpecializedTemplateOrPartial();
    if (PatternPtr.is<VarTemplatePartialSpecializationDecl *>()) {
      VarTemplatePartialSpecializationDecl *Tmpl =
          PatternPtr.get<VarTemplatePartialSpecializationDecl *>();
      while (VarTemplatePartialSpecializationDecl *From =
                 Tmpl->getInstantiatedFromMember()) {
        if (Tmpl->isMemberSpecialization())
          break;

        Tmpl = From;
      }
      PatternDecl = Tmpl;
    } else {
      VarTemplateDecl *Tmpl = PatternPtr.get<VarTemplateDecl *>();
      while (VarTemplateDecl *From =
                 Tmpl->getInstantiatedFromMemberTemplate()) {
        if (Tmpl->isMemberSpecialization())
          break;

        Tmpl = From;
      }
      PatternDecl = Tmpl->getTemplatedDecl();
    }

    // If this is a static data member template, there might be an
    // uninstantiated initializer on the declaration. If so, instantiate
    // it now.
    if (PatternDecl->isStaticDataMember() &&
        (PatternDecl = PatternDecl->getFirstDecl())->hasInit() &&
        !Var->hasInit()) {
      // FIXME: Factor out the duplicated instantiation context setup/tear down
      // code here.
      InstantiatingTemplate Inst(*this, PointOfInstantiation, Var);
      if (Inst.isInvalid())
        return;

      // If we're performing recursive template instantiation, create our own
      // queue of pending implicit instantiations that we will instantiate
      // later, while we're still within our own instantiation context.
      SmallVector<VTableUse, 16> SavedVTableUses;
      std::deque<PendingImplicitInstantiation> SavedPendingInstantiations;
      if (Recursive) {
        VTableUses.swap(SavedVTableUses);
        PendingInstantiations.swap(SavedPendingInstantiations);
      }

      LocalInstantiationScope Local(*this);

      // Enter the scope of this instantiation. We don't use
      // PushDeclContext because we don't have a scope.
      ContextRAII PreviousContext(*this, Var->getDeclContext());
      InstantiateVariableInitializer(Var, PatternDecl, TemplateArgs);
      PreviousContext.pop();

      // FIXME: Need to inform the ASTConsumer that we instantiated the
      // initializer?

      // This variable may have local implicit instantiations that need to be
      // instantiated within this scope.
      PerformPendingInstantiations(/*LocalOnly=*/true);

      Local.Exit();

      if (Recursive) {
        // Define any newly required vtables.
        DefineUsedVTables();

        // Instantiate any pending implicit instantiations found during the
        // instantiation of this template.
        PerformPendingInstantiations();

        // Restore the set of pending vtables.
        assert(VTableUses.empty() &&
               "VTableUses should be empty before it is discarded.");
        VTableUses.swap(SavedVTableUses);

        // Restore the set of pending implicit instantiations.
        assert(PendingInstantiations.empty() &&
               "PendingInstantiations should be empty before it is discarded.");
        PendingInstantiations.swap(SavedPendingInstantiations);
      }
    }

    // Find actual definition
    Def = PatternDecl->getDefinition(getASTContext());
  } else {
    // If this is a static data member, find its out-of-line definition.
    assert(Var->isStaticDataMember() && "not a static data member?");
    PatternDecl = Var->getInstantiatedFromStaticDataMember();

    assert(PatternDecl && "data member was not instantiated from a template?");
    assert(PatternDecl->isStaticDataMember() && "not a static data member?");
    Def = PatternDecl->getOutOfLineDefinition();
  }

  // If we don't have a definition of the variable template, we won't perform
  // any instantiation. Rather, we rely on the user to instantiate this
  // definition (or provide a specialization for it) in another translation
  // unit.
  if (!Def) {
    if (DefinitionRequired) {
      if (VarSpec)
        Diag(PointOfInstantiation,
             diag::err_explicit_instantiation_undefined_var_template) << Var;
      else
        Diag(PointOfInstantiation,
             diag::err_explicit_instantiation_undefined_member)
            << 2 << Var->getDeclName() << Var->getDeclContext();
      Diag(PatternDecl->getLocation(),
           diag::note_explicit_instantiation_here);
      if (VarSpec)
        Var->setInvalidDecl();
    } else if (Var->getTemplateSpecializationKind()
                 == TSK_ExplicitInstantiationDefinition) {
      PendingInstantiations.push_back(
        std::make_pair(Var, PointOfInstantiation));
    }

    return;
  }

  TemplateSpecializationKind TSK = Var->getTemplateSpecializationKind();

  // Never instantiate an explicit specialization.
  if (TSK == TSK_ExplicitSpecialization)
    return;

  // C++11 [temp.explicit]p10:
  //   Except for inline functions, [...] explicit instantiation declarations
  //   have the effect of suppressing the implicit instantiation of the entity
  //   to which they refer.
  if (TSK == TSK_ExplicitInstantiationDeclaration)
    return;

  // Make sure to pass the instantiated variable to the consumer at the end.
  struct PassToConsumerRAII {
    ASTConsumer &Consumer;
    VarDecl *Var;

    PassToConsumerRAII(ASTConsumer &Consumer, VarDecl *Var)
      : Consumer(Consumer), Var(Var) { }

    ~PassToConsumerRAII() {
      Consumer.HandleCXXStaticMemberVarInstantiation(Var);
    }
  } PassToConsumerRAII(Consumer, Var);

  // If we already have a definition, we're done.
  if (VarDecl *Def = Var->getDefinition()) {
    // We may be explicitly instantiating something we've already implicitly
    // instantiated.
    Def->setTemplateSpecializationKind(Var->getTemplateSpecializationKind(),
                                       PointOfInstantiation);
    return;
  }

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Var);
  if (Inst.isInvalid())
    return;

  // If we're performing recursive template instantiation, create our own
  // queue of pending implicit instantiations that we will instantiate later,
  // while we're still within our own instantiation context.
  SmallVector<VTableUse, 16> SavedVTableUses;
  std::deque<PendingImplicitInstantiation> SavedPendingInstantiations;
  SavePendingLocalImplicitInstantiationsRAII
      SavedPendingLocalImplicitInstantiations(*this);
  if (Recursive) {
    VTableUses.swap(SavedVTableUses);
    PendingInstantiations.swap(SavedPendingInstantiations);
  }

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  ContextRAII PreviousContext(*this, Var->getDeclContext());
  LocalInstantiationScope Local(*this);

  VarDecl *OldVar = Var;
  if (!VarSpec)
    Var = cast_or_null<VarDecl>(SubstDecl(Def, Var->getDeclContext(),
                                          TemplateArgs));
  else if (Var->isStaticDataMember() &&
           Var->getLexicalDeclContext()->isRecord()) {
    // We need to instantiate the definition of a static data member template,
    // and all we have is the in-class declaration of it. Instantiate a separate
    // declaration of the definition.
    TemplateDeclInstantiator Instantiator(*this, Var->getDeclContext(),
                                          TemplateArgs);
    Var = cast_or_null<VarDecl>(Instantiator.VisitVarTemplateSpecializationDecl(
        VarSpec->getSpecializedTemplate(), Def, 0,
        VarSpec->getTemplateArgsInfo(), VarSpec->getTemplateArgs().asArray()));
    if (Var) {
      llvm::PointerUnion<VarTemplateDecl *,
                         VarTemplatePartialSpecializationDecl *> PatternPtr =
          VarSpec->getSpecializedTemplateOrPartial();
      if (VarTemplatePartialSpecializationDecl *Partial =
          PatternPtr.dyn_cast<VarTemplatePartialSpecializationDecl *>())
        cast<VarTemplateSpecializationDecl>(Var)->setInstantiationOf(
            Partial, &VarSpec->getTemplateInstantiationArgs());

      // Merge the definition with the declaration.
      LookupResult R(*this, Var->getDeclName(), Var->getLocation(),
                     LookupOrdinaryName, ForRedeclaration);
      R.addDecl(OldVar);
      MergeVarDecl(Var, R);

      // Attach the initializer.
      InstantiateVariableInitializer(Var, Def, TemplateArgs);
    }
  } else
    // Complete the existing variable's definition with an appropriately
    // substituted type and initializer.
    Var = CompleteVarTemplateSpecializationDecl(VarSpec, Def, TemplateArgs);

  PreviousContext.pop();

  if (Var) {
    PassToConsumerRAII.Var = Var;
    Var->setTemplateSpecializationKind(OldVar->getTemplateSpecializationKind(),
                                       OldVar->getPointOfInstantiation());
  }

  // This variable may have local implicit instantiations that need to be
  // instantiated within this scope.
  PerformPendingInstantiations(/*LocalOnly=*/true);

  Local.Exit();
  
  if (Recursive) {
    // Define any newly required vtables.
    DefineUsedVTables();

    // Instantiate any pending implicit instantiations found during the
    // instantiation of this template.
    PerformPendingInstantiations();

    // Restore the set of pending vtables.
    assert(VTableUses.empty() &&
           "VTableUses should be empty before it is discarded.");
    VTableUses.swap(SavedVTableUses);

    // Restore the set of pending implicit instantiations.
    assert(PendingInstantiations.empty() &&
           "PendingInstantiations should be empty before it is discarded.");
    PendingInstantiations.swap(SavedPendingInstantiations);
  }
}

void
Sema::InstantiateMemInitializers(CXXConstructorDecl *New,
                                 const CXXConstructorDecl *Tmpl,
                           const MultiLevelTemplateArgumentList &TemplateArgs) {

  SmallVector<CXXCtorInitializer*, 4> NewInits;
  bool AnyErrors = Tmpl->isInvalidDecl();

  // Instantiate all the initializers.
  for (CXXConstructorDecl::init_const_iterator Inits = Tmpl->init_begin(),
                                            InitsEnd = Tmpl->init_end();
       Inits != InitsEnd; ++Inits) {
    CXXCtorInitializer *Init = *Inits;

    // Only instantiate written initializers, let Sema re-construct implicit
    // ones.
    if (!Init->isWritten())
      continue;

    SourceLocation EllipsisLoc;

    if (Init->isPackExpansion()) {
      // This is a pack expansion. We should expand it now.
      TypeLoc BaseTL = Init->getTypeSourceInfo()->getTypeLoc();
      SmallVector<UnexpandedParameterPack, 4> Unexpanded;
      collectUnexpandedParameterPacks(BaseTL, Unexpanded);
      collectUnexpandedParameterPacks(Init->getInit(), Unexpanded);
      bool ShouldExpand = false;
      bool RetainExpansion = false;
      Optional<unsigned> NumExpansions;
      if (CheckParameterPacksForExpansion(Init->getEllipsisLoc(),
                                          BaseTL.getSourceRange(),
                                          Unexpanded,
                                          TemplateArgs, ShouldExpand,
                                          RetainExpansion,
                                          NumExpansions)) {
        AnyErrors = true;
        New->setInvalidDecl();
        continue;
      }
      assert(ShouldExpand && "Partial instantiation of base initializer?");

      // Loop over all of the arguments in the argument pack(s),
      for (unsigned I = 0; I != *NumExpansions; ++I) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(*this, I);

        // Instantiate the initializer.
        ExprResult TempInit = SubstInitializer(Init->getInit(), TemplateArgs,
                                               /*CXXDirectInit=*/true);
        if (TempInit.isInvalid()) {
          AnyErrors = true;
          break;
        }

        // Instantiate the base type.
        TypeSourceInfo *BaseTInfo = SubstType(Init->getTypeSourceInfo(),
                                              TemplateArgs,
                                              Init->getSourceLocation(),
                                              New->getDeclName());
        if (!BaseTInfo) {
          AnyErrors = true;
          break;
        }

        // Build the initializer.
        MemInitResult NewInit = BuildBaseInitializer(BaseTInfo->getType(),
                                                     BaseTInfo, TempInit.take(),
                                                     New->getParent(),
                                                     SourceLocation());
        if (NewInit.isInvalid()) {
          AnyErrors = true;
          break;
        }

        NewInits.push_back(NewInit.get());
      }

      continue;
    }

    // Instantiate the initializer.
    ExprResult TempInit = SubstInitializer(Init->getInit(), TemplateArgs,
                                           /*CXXDirectInit=*/true);
    if (TempInit.isInvalid()) {
      AnyErrors = true;
      continue;
    }

    MemInitResult NewInit;
    if (Init->isDelegatingInitializer() || Init->isBaseInitializer()) {
      TypeSourceInfo *TInfo = SubstType(Init->getTypeSourceInfo(),
                                        TemplateArgs,
                                        Init->getSourceLocation(),
                                        New->getDeclName());
      if (!TInfo) {
        AnyErrors = true;
        New->setInvalidDecl();
        continue;
      }

      if (Init->isBaseInitializer())
        NewInit = BuildBaseInitializer(TInfo->getType(), TInfo, TempInit.take(),
                                       New->getParent(), EllipsisLoc);
      else
        NewInit = BuildDelegatingInitializer(TInfo, TempInit.take(),
                                  cast<CXXRecordDecl>(CurContext->getParent()));
    } else if (Init->isMemberInitializer()) {
      FieldDecl *Member = cast_or_null<FieldDecl>(FindInstantiatedDecl(
                                                     Init->getMemberLocation(),
                                                     Init->getMember(),
                                                     TemplateArgs));
      if (!Member) {
        AnyErrors = true;
        New->setInvalidDecl();
        continue;
      }

      NewInit = BuildMemberInitializer(Member, TempInit.take(),
                                       Init->getSourceLocation());
    } else if (Init->isIndirectMemberInitializer()) {
      IndirectFieldDecl *IndirectMember =
         cast_or_null<IndirectFieldDecl>(FindInstantiatedDecl(
                                 Init->getMemberLocation(),
                                 Init->getIndirectMember(), TemplateArgs));

      if (!IndirectMember) {
        AnyErrors = true;
        New->setInvalidDecl();
        continue;
      }

      NewInit = BuildMemberInitializer(IndirectMember, TempInit.take(),
                                       Init->getSourceLocation());
    }

    if (NewInit.isInvalid()) {
      AnyErrors = true;
      New->setInvalidDecl();
    } else {
      NewInits.push_back(NewInit.get());
    }
  }

  // Assign all the initializers to the new constructor.
  ActOnMemInitializers(New,
                       /*FIXME: ColonLoc */
                       SourceLocation(),
                       NewInits,
                       AnyErrors);
}

// TODO: this could be templated if the various decl types used the
// same method name.
static bool isInstantiationOf(ClassTemplateDecl *Pattern,
                              ClassTemplateDecl *Instance) {
  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromMemberTemplate();
  } while (Instance);

  return false;
}

static bool isInstantiationOf(FunctionTemplateDecl *Pattern,
                              FunctionTemplateDecl *Instance) {
  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromMemberTemplate();
  } while (Instance);

  return false;
}

static bool
isInstantiationOf(ClassTemplatePartialSpecializationDecl *Pattern,
                  ClassTemplatePartialSpecializationDecl *Instance) {
  Pattern
    = cast<ClassTemplatePartialSpecializationDecl>(Pattern->getCanonicalDecl());
  do {
    Instance = cast<ClassTemplatePartialSpecializationDecl>(
                                                Instance->getCanonicalDecl());
    if (Pattern == Instance)
      return true;
    Instance = Instance->getInstantiatedFromMember();
  } while (Instance);

  return false;
}

static bool isInstantiationOf(CXXRecordDecl *Pattern,
                              CXXRecordDecl *Instance) {
  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromMemberClass();
  } while (Instance);

  return false;
}

static bool isInstantiationOf(FunctionDecl *Pattern,
                              FunctionDecl *Instance) {
  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromMemberFunction();
  } while (Instance);

  return false;
}

static bool isInstantiationOf(EnumDecl *Pattern,
                              EnumDecl *Instance) {
  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromMemberEnum();
  } while (Instance);

  return false;
}

static bool isInstantiationOf(UsingShadowDecl *Pattern,
                              UsingShadowDecl *Instance,
                              ASTContext &C) {
  return C.getInstantiatedFromUsingShadowDecl(Instance) == Pattern;
}

static bool isInstantiationOf(UsingDecl *Pattern,
                              UsingDecl *Instance,
                              ASTContext &C) {
  return C.getInstantiatedFromUsingDecl(Instance) == Pattern;
}

static bool isInstantiationOf(UnresolvedUsingValueDecl *Pattern,
                              UsingDecl *Instance,
                              ASTContext &C) {
  return C.getInstantiatedFromUsingDecl(Instance) == Pattern;
}

static bool isInstantiationOf(UnresolvedUsingTypenameDecl *Pattern,
                              UsingDecl *Instance,
                              ASTContext &C) {
  return C.getInstantiatedFromUsingDecl(Instance) == Pattern;
}

static bool isInstantiationOfStaticDataMember(VarDecl *Pattern,
                                              VarDecl *Instance) {
  assert(Instance->isStaticDataMember());

  Pattern = Pattern->getCanonicalDecl();

  do {
    Instance = Instance->getCanonicalDecl();
    if (Pattern == Instance) return true;
    Instance = Instance->getInstantiatedFromStaticDataMember();
  } while (Instance);

  return false;
}

// Other is the prospective instantiation
// D is the prospective pattern
static bool isInstantiationOf(ASTContext &Ctx, NamedDecl *D, Decl *Other) {
  if (D->getKind() != Other->getKind()) {
    if (UnresolvedUsingTypenameDecl *UUD
          = dyn_cast<UnresolvedUsingTypenameDecl>(D)) {
      if (UsingDecl *UD = dyn_cast<UsingDecl>(Other)) {
        return isInstantiationOf(UUD, UD, Ctx);
      }
    }

    if (UnresolvedUsingValueDecl *UUD
          = dyn_cast<UnresolvedUsingValueDecl>(D)) {
      if (UsingDecl *UD = dyn_cast<UsingDecl>(Other)) {
        return isInstantiationOf(UUD, UD, Ctx);
      }
    }

    return false;
  }

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Other))
    return isInstantiationOf(cast<CXXRecordDecl>(D), Record);

  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Other))
    return isInstantiationOf(cast<FunctionDecl>(D), Function);

  if (EnumDecl *Enum = dyn_cast<EnumDecl>(Other))
    return isInstantiationOf(cast<EnumDecl>(D), Enum);

  if (VarDecl *Var = dyn_cast<VarDecl>(Other))
    if (Var->isStaticDataMember())
      return isInstantiationOfStaticDataMember(cast<VarDecl>(D), Var);

  if (ClassTemplateDecl *Temp = dyn_cast<ClassTemplateDecl>(Other))
    return isInstantiationOf(cast<ClassTemplateDecl>(D), Temp);

  if (FunctionTemplateDecl *Temp = dyn_cast<FunctionTemplateDecl>(Other))
    return isInstantiationOf(cast<FunctionTemplateDecl>(D), Temp);

  if (ClassTemplatePartialSpecializationDecl *PartialSpec
        = dyn_cast<ClassTemplatePartialSpecializationDecl>(Other))
    return isInstantiationOf(cast<ClassTemplatePartialSpecializationDecl>(D),
                             PartialSpec);

  if (FieldDecl *Field = dyn_cast<FieldDecl>(Other)) {
    if (!Field->getDeclName()) {
      // This is an unnamed field.
      return Ctx.getInstantiatedFromUnnamedFieldDecl(Field) ==
        cast<FieldDecl>(D);
    }
  }

  if (UsingDecl *Using = dyn_cast<UsingDecl>(Other))
    return isInstantiationOf(cast<UsingDecl>(D), Using, Ctx);

  if (UsingShadowDecl *Shadow = dyn_cast<UsingShadowDecl>(Other))
    return isInstantiationOf(cast<UsingShadowDecl>(D), Shadow, Ctx);

  return D->getDeclName() && isa<NamedDecl>(Other) &&
    D->getDeclName() == cast<NamedDecl>(Other)->getDeclName();
}

template<typename ForwardIterator>
static NamedDecl *findInstantiationOf(ASTContext &Ctx,
                                      NamedDecl *D,
                                      ForwardIterator first,
                                      ForwardIterator last) {
  for (; first != last; ++first)
    if (isInstantiationOf(Ctx, D, *first))
      return cast<NamedDecl>(*first);

  return 0;
}

/// \brief Finds the instantiation of the given declaration context
/// within the current instantiation.
///
/// \returns NULL if there was an error
DeclContext *Sema::FindInstantiatedContext(SourceLocation Loc, DeclContext* DC,
                          const MultiLevelTemplateArgumentList &TemplateArgs) {
  if (NamedDecl *D = dyn_cast<NamedDecl>(DC)) {
    Decl* ID = FindInstantiatedDecl(Loc, D, TemplateArgs);
    return cast_or_null<DeclContext>(ID);
  } else return DC;
}

/// \brief Find the instantiation of the given declaration within the
/// current instantiation.
///
/// This routine is intended to be used when \p D is a declaration
/// referenced from within a template, that needs to mapped into the
/// corresponding declaration within an instantiation. For example,
/// given:
///
/// \code
/// template<typename T>
/// struct X {
///   enum Kind {
///     KnownValue = sizeof(T)
///   };
///
///   bool getKind() const { return KnownValue; }
/// };
///
/// template struct X<int>;
/// \endcode
///
/// In the instantiation of <tt>X<int>::getKind()</tt>, we need to map the
/// \p EnumConstantDecl for \p KnownValue (which refers to
/// <tt>X<T>::<Kind>::KnownValue</tt>) to its instantiation
/// (<tt>X<int>::<Kind>::KnownValue</tt>). \p FindInstantiatedDecl performs
/// this mapping from within the instantiation of <tt>X<int></tt>.
NamedDecl *Sema::FindInstantiatedDecl(SourceLocation Loc, NamedDecl *D,
                          const MultiLevelTemplateArgumentList &TemplateArgs) {
  DeclContext *ParentDC = D->getDeclContext();
  // FIXME: Parmeters of pointer to functions (y below) that are themselves 
  // parameters (p below) can have their ParentDC set to the translation-unit
  // - thus we can not consistently check if the ParentDC of such a parameter 
  // is Dependent or/and a FunctionOrMethod.
  // For e.g. this code, during Template argument deduction tries to 
  // find an instantiated decl for (T y) when the ParentDC for y is
  // the translation unit.  
  //   e.g. template <class T> void Foo(auto (*p)(T y) -> decltype(y())) {} 
  //   float baz(float(*)()) { return 0.0; }
  //   Foo(baz);
  // The better fix here is perhaps to ensure that a ParmVarDecl, by the time
  // it gets here, always has a FunctionOrMethod as its ParentDC??
  // For now:
  //  - as long as we have a ParmVarDecl whose parent is non-dependent and
  //    whose type is not instantiation dependent, do nothing to the decl
  //  - otherwise find its instantiated decl.
  if (isa<ParmVarDecl>(D) && !ParentDC->isDependentContext() &&
      !cast<ParmVarDecl>(D)->getType()->isInstantiationDependentType())
    return D;
  if (isa<ParmVarDecl>(D) || isa<NonTypeTemplateParmDecl>(D) ||
      isa<TemplateTypeParmDecl>(D) || isa<TemplateTemplateParmDecl>(D) ||
      (ParentDC->isFunctionOrMethod() && ParentDC->isDependentContext()) ||
      (isa<CXXRecordDecl>(D) && cast<CXXRecordDecl>(D)->isLambda())) {
    // D is a local of some kind. Look into the map of local
    // declarations to their instantiations.
    typedef LocalInstantiationScope::DeclArgumentPack DeclArgumentPack;
    llvm::PointerUnion<Decl *, DeclArgumentPack *> *Found
      = CurrentInstantiationScope->findInstantiationOf(D);

    if (Found) {
      if (Decl *FD = Found->dyn_cast<Decl *>())
        return cast<NamedDecl>(FD);

      int PackIdx = ArgumentPackSubstitutionIndex;
      assert(PackIdx != -1 && "found declaration pack but not pack expanding");
      return cast<NamedDecl>((*Found->get<DeclArgumentPack *>())[PackIdx]);
    }

    // If we're performing a partial substitution during template argument
    // deduction, we may not have values for template parameters yet. They
    // just map to themselves.
    if (isa<NonTypeTemplateParmDecl>(D) || isa<TemplateTypeParmDecl>(D) ||
        isa<TemplateTemplateParmDecl>(D))
      return D;

    if (D->isInvalidDecl())
      return 0;

    // If we didn't find the decl, then we must have a label decl that hasn't
    // been found yet.  Lazily instantiate it and return it now.
    assert(isa<LabelDecl>(D));

    Decl *Inst = SubstDecl(D, CurContext, TemplateArgs);
    assert(Inst && "Failed to instantiate label??");

    CurrentInstantiationScope->InstantiatedLocal(D, Inst);
    return cast<LabelDecl>(Inst);
  }

  // For variable template specializations, update those that are still
  // type-dependent.
  if (VarTemplateSpecializationDecl *VarSpec =
          dyn_cast<VarTemplateSpecializationDecl>(D)) {
    bool InstantiationDependent = false;
    const TemplateArgumentListInfo &VarTemplateArgs =
        VarSpec->getTemplateArgsInfo();
    if (TemplateSpecializationType::anyDependentTemplateArguments(
            VarTemplateArgs, InstantiationDependent))
      D = cast<NamedDecl>(
          SubstDecl(D, VarSpec->getDeclContext(), TemplateArgs));
    return D;
  }

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D)) {
    if (!Record->isDependentContext())
      return D;

    // Determine whether this record is the "templated" declaration describing
    // a class template or class template partial specialization.
    ClassTemplateDecl *ClassTemplate = Record->getDescribedClassTemplate();
    if (ClassTemplate)
      ClassTemplate = ClassTemplate->getCanonicalDecl();
    else if (ClassTemplatePartialSpecializationDecl *PartialSpec
               = dyn_cast<ClassTemplatePartialSpecializationDecl>(Record))
      ClassTemplate = PartialSpec->getSpecializedTemplate()->getCanonicalDecl();

    // Walk the current context to find either the record or an instantiation of
    // it.
    DeclContext *DC = CurContext;
    while (!DC->isFileContext()) {
      // If we're performing substitution while we're inside the template
      // definition, we'll find our own context. We're done.
      if (DC->Equals(Record))
        return Record;

      if (CXXRecordDecl *InstRecord = dyn_cast<CXXRecordDecl>(DC)) {
        // Check whether we're in the process of instantiating a class template
        // specialization of the template we're mapping.
        if (ClassTemplateSpecializationDecl *InstSpec
                      = dyn_cast<ClassTemplateSpecializationDecl>(InstRecord)){
          ClassTemplateDecl *SpecTemplate = InstSpec->getSpecializedTemplate();
          if (ClassTemplate && isInstantiationOf(ClassTemplate, SpecTemplate))
            return InstRecord;
        }

        // Check whether we're in the process of instantiating a member class.
        if (isInstantiationOf(Record, InstRecord))
          return InstRecord;
      }

      // Move to the outer template scope.
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(DC)) {
        if (FD->getFriendObjectKind() && FD->getDeclContext()->isFileContext()){
          DC = FD->getLexicalDeclContext();
          continue;
        }
      }

      DC = DC->getParent();
    }

    // Fall through to deal with other dependent record types (e.g.,
    // anonymous unions in class templates).
  }

  if (!ParentDC->isDependentContext())
    return D;

  ParentDC = FindInstantiatedContext(Loc, ParentDC, TemplateArgs);
  if (!ParentDC)
    return 0;

  if (ParentDC != D->getDeclContext()) {
    // We performed some kind of instantiation in the parent context,
    // so now we need to look into the instantiated parent context to
    // find the instantiation of the declaration D.

    // If our context used to be dependent, we may need to instantiate
    // it before performing lookup into that context.
    bool IsBeingInstantiated = false;
    if (CXXRecordDecl *Spec = dyn_cast<CXXRecordDecl>(ParentDC)) {
      if (!Spec->isDependentContext()) {
        QualType T = Context.getTypeDeclType(Spec);
        const RecordType *Tag = T->getAs<RecordType>();
        assert(Tag && "type of non-dependent record is not a RecordType");
        if (Tag->isBeingDefined())
          IsBeingInstantiated = true;
        if (!Tag->isBeingDefined() &&
            RequireCompleteType(Loc, T, diag::err_incomplete_type))
          return 0;

        ParentDC = Tag->getDecl();
      }
    }

    NamedDecl *Result = 0;
    if (D->getDeclName()) {
      DeclContext::lookup_result Found = ParentDC->lookup(D->getDeclName());
      Result = findInstantiationOf(Context, D, Found.begin(), Found.end());
    } else {
      // Since we don't have a name for the entity we're looking for,
      // our only option is to walk through all of the declarations to
      // find that name. This will occur in a few cases:
      //
      //   - anonymous struct/union within a template
      //   - unnamed class/struct/union/enum within a template
      //
      // FIXME: Find a better way to find these instantiations!
      Result = findInstantiationOf(Context, D,
                                   ParentDC->decls_begin(),
                                   ParentDC->decls_end());
    }

    if (!Result) {
      if (isa<UsingShadowDecl>(D)) {
        // UsingShadowDecls can instantiate to nothing because of using hiding.
      } else if (Diags.hasErrorOccurred()) {
        // We've already complained about something, so most likely this
        // declaration failed to instantiate. There's no point in complaining
        // further, since this is normal in invalid code.
      } else if (IsBeingInstantiated) {
        // The class in which this member exists is currently being
        // instantiated, and we haven't gotten around to instantiating this
        // member yet. This can happen when the code uses forward declarations
        // of member classes, and introduces ordering dependencies via
        // template instantiation.
        Diag(Loc, diag::err_member_not_yet_instantiated)
          << D->getDeclName()
          << Context.getTypeDeclType(cast<CXXRecordDecl>(ParentDC));
        Diag(D->getLocation(), diag::note_non_instantiated_member_here);
      } else if (EnumConstantDecl *ED = dyn_cast<EnumConstantDecl>(D)) {
        // This enumeration constant was found when the template was defined,
        // but can't be found in the instantiation. This can happen if an
        // unscoped enumeration member is explicitly specialized.
        EnumDecl *Enum = cast<EnumDecl>(ED->getLexicalDeclContext());
        EnumDecl *Spec = cast<EnumDecl>(FindInstantiatedDecl(Loc, Enum,
                                                             TemplateArgs));
        assert(Spec->getTemplateSpecializationKind() ==
                 TSK_ExplicitSpecialization);
        Diag(Loc, diag::err_enumerator_does_not_exist)
          << D->getDeclName()
          << Context.getTypeDeclType(cast<TypeDecl>(Spec->getDeclContext()));
        Diag(Spec->getLocation(), diag::note_enum_specialized_here)
          << Context.getTypeDeclType(Spec);
      } else {
        // We should have found something, but didn't.
        llvm_unreachable("Unable to find instantiation of declaration!");
      }
    }

    D = Result;
  }

  return D;
}

/// \brief Performs template instantiation for all implicit template
/// instantiations we have seen until this point.
void Sema::PerformPendingInstantiations(bool LocalOnly) {
  // Load pending instantiations from the external source.
  if (!LocalOnly && ExternalSource) {
    SmallVector<PendingImplicitInstantiation, 4> Pending;
    ExternalSource->ReadPendingInstantiations(Pending);
    PendingInstantiations.insert(PendingInstantiations.begin(),
                                 Pending.begin(), Pending.end());
  }

  while (!PendingLocalImplicitInstantiations.empty() ||
         (!LocalOnly && !PendingInstantiations.empty())) {
    PendingImplicitInstantiation Inst;

    if (PendingLocalImplicitInstantiations.empty()) {
      Inst = PendingInstantiations.front();
      PendingInstantiations.pop_front();
    } else {
      Inst = PendingLocalImplicitInstantiations.front();
      PendingLocalImplicitInstantiations.pop_front();
    }

    // Instantiate function definitions
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Inst.first)) {
      PrettyDeclStackTraceEntry CrashInfo(*this, Function, SourceLocation(),
                                          "instantiating function definition");
      bool DefinitionRequired = Function->getTemplateSpecializationKind() ==
                                TSK_ExplicitInstantiationDefinition;
      InstantiateFunctionDefinition(/*FIXME:*/Inst.second, Function, true,
                                    DefinitionRequired);
      continue;
    }

    // Instantiate variable definitions
    VarDecl *Var = cast<VarDecl>(Inst.first);

    assert((Var->isStaticDataMember() ||
            isa<VarTemplateSpecializationDecl>(Var)) &&
           "Not a static data member, nor a variable template"
           " specialization?");

    // Don't try to instantiate declarations if the most recent redeclaration
    // is invalid.
    if (Var->getMostRecentDecl()->isInvalidDecl())
      continue;

    // Check if the most recent declaration has changed the specialization kind
    // and removed the need for implicit instantiation.
    switch (Var->getMostRecentDecl()->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
      llvm_unreachable("Cannot instantitiate an undeclared specialization.");
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitSpecialization:
      continue;  // No longer need to instantiate this type.
    case TSK_ExplicitInstantiationDefinition:
      // We only need an instantiation if the pending instantiation *is* the
      // explicit instantiation.
      if (Var != Var->getMostRecentDecl()) continue;
    case TSK_ImplicitInstantiation:
      break;
    }

    PrettyDeclStackTraceEntry CrashInfo(*this, Var, SourceLocation(),
                                        "instantiating variable definition");
    bool DefinitionRequired = Var->getTemplateSpecializationKind() ==
                              TSK_ExplicitInstantiationDefinition;

    // Instantiate static data member definitions or variable template
    // specializations.
    InstantiateVariableDefinition(/*FIXME:*/ Inst.second, Var, true,
                                  DefinitionRequired);
  }
}

void Sema::PerformDependentDiagnostics(const DeclContext *Pattern,
                       const MultiLevelTemplateArgumentList &TemplateArgs) {
  for (DeclContext::ddiag_iterator I = Pattern->ddiag_begin(),
         E = Pattern->ddiag_end(); I != E; ++I) {
    DependentDiagnostic *DD = *I;

    switch (DD->getKind()) {
    case DependentDiagnostic::Access:
      HandleDependentAccessCheck(*DD, TemplateArgs);
      break;
    }
  }
}
