//===--- CTAD.cpp - -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CTAD.h"
#include "TreeTransform.h"
#include "TypeLocBuilder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/ASTStructuralEquivalence.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace clang {

namespace {
/// Tree transform to "extract" a transformed type from a class template's
/// constructor to a deduction guide.
class ExtractTypeForDeductionGuide
    : public TreeTransform<ExtractTypeForDeductionGuide> {
  llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs;

public:
  typedef TreeTransform<ExtractTypeForDeductionGuide> Base;
  ExtractTypeForDeductionGuide(
      Sema &SemaRef,
      llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs)
      : Base(SemaRef), MaterializedTypedefs(MaterializedTypedefs) {}

  TypeSourceInfo *transform(TypeSourceInfo *TSI) { return TransformType(TSI); }

  QualType TransformTypedefType(TypeLocBuilder &TLB, TypedefTypeLoc TL) {
    ASTContext &Context = SemaRef.getASTContext();
    TypedefNameDecl *OrigDecl = TL.getTypedefNameDecl();
    TypedefNameDecl *Decl = OrigDecl;
    // Transform the underlying type of the typedef and clone the Decl only if
    // the typedef has a dependent context.
    if (OrigDecl->getDeclContext()->isDependentContext()) {
      TypeLocBuilder InnerTLB;
      QualType Transformed =
          TransformType(InnerTLB, OrigDecl->getTypeSourceInfo()->getTypeLoc());
      TypeSourceInfo *TSI = InnerTLB.getTypeSourceInfo(Context, Transformed);
      if (isa<TypeAliasDecl>(OrigDecl))
        Decl = TypeAliasDecl::Create(
            Context, Context.getTranslationUnitDecl(), OrigDecl->getBeginLoc(),
            OrigDecl->getLocation(), OrigDecl->getIdentifier(), TSI);
      else {
        assert(isa<TypedefDecl>(OrigDecl) && "Not a Type alias or typedef");
        Decl = TypedefDecl::Create(
            Context, Context.getTranslationUnitDecl(), OrigDecl->getBeginLoc(),
            OrigDecl->getLocation(), OrigDecl->getIdentifier(), TSI);
      }
      MaterializedTypedefs.push_back(Decl);
    }

    QualType TDTy = Context.getTypedefType(Decl);
    TypedefTypeLoc TypedefTL = TLB.push<TypedefTypeLoc>(TDTy);
    TypedefTL.setNameLoc(TL.getNameLoc());

    return TDTy;
  }
};
} // namespace

ParmVarDecl *transformFunctionTypeParam(
    Sema &SemaRef, ParmVarDecl *OldParam, DeclContext *DC,
    MultiLevelTemplateArgumentList &Args,
    llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs) {
  TypeSourceInfo *OldDI = OldParam->getTypeSourceInfo();
  TypeSourceInfo *NewDI;
  if (auto PackTL = OldDI->getTypeLoc().getAs<PackExpansionTypeLoc>()) {
    // Expand out the one and only element in each inner pack.
    Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(SemaRef, 0);
    NewDI = SemaRef.SubstType(PackTL.getPatternLoc(), Args,
                              OldParam->getLocation(), OldParam->getDeclName());
    if (!NewDI)
      return nullptr;
    NewDI = SemaRef.CheckPackExpansion(NewDI, PackTL.getEllipsisLoc(),
                                       PackTL.getTypePtr()->getNumExpansions());
  } else
    NewDI = SemaRef.SubstType(OldDI, Args, OldParam->getLocation(),
                              OldParam->getDeclName());
  if (!NewDI)
    return nullptr;

  // Extract the type. This (for instance) replaces references to typedef
  // members of the current instantiations with the definitions of those
  // typedefs, avoiding triggering instantiation of the deduced type during
  // deduction.
  NewDI = ExtractTypeForDeductionGuide(SemaRef, MaterializedTypedefs)
              .transform(NewDI);

  // Resolving a wording defect, we also inherit default arguments from the
  // constructor.
  ExprResult NewDefArg;
  if (OldParam->hasDefaultArg()) {
    // We don't care what the value is (we won't use it); just create a
    // placeholder to indicate there is a default argument.
    QualType ParamTy = NewDI->getType();
    NewDefArg = new (SemaRef.Context)
        OpaqueValueExpr(OldParam->getDefaultArgRange().getBegin(),
                        ParamTy.getNonLValueExprType(SemaRef.Context),
                        ParamTy->isLValueReferenceType()   ? VK_LValue
                        : ParamTy->isRValueReferenceType() ? VK_XValue
                                                           : VK_PRValue);
  }
  // Handle arrays and functions decay.
  auto NewType = NewDI->getType();
  if (NewType->isArrayType() || NewType->isFunctionType())
    NewType = SemaRef.Context.getDecayedType(NewType);

  ParmVarDecl *NewParam = ParmVarDecl::Create(
      SemaRef.Context, DC, OldParam->getInnerLocStart(),
      OldParam->getLocation(), OldParam->getIdentifier(), NewType, NewDI,
      OldParam->getStorageClass(), NewDefArg.get());
  NewParam->setScopeInfo(OldParam->getFunctionScopeDepth(),
                         OldParam->getFunctionScopeIndex());
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(OldParam, NewParam);
  return NewParam;
}

TemplateTypeParmDecl *
transformTemplateTypeParam(Sema &SemaRef, DeclContext *DC,
                           TemplateTypeParmDecl *TTP,
                           MultiLevelTemplateArgumentList &Args,
                           unsigned NewDepth, unsigned NewIndex) {
  // TemplateTypeParmDecl's index cannot be changed after creation, so
  // substitute it directly.
  auto *NewTTP = TemplateTypeParmDecl::Create(
      SemaRef.Context, DC, TTP->getBeginLoc(), TTP->getLocation(), NewDepth,
      NewIndex, TTP->getIdentifier(), TTP->wasDeclaredWithTypename(),
      TTP->isParameterPack(), TTP->hasTypeConstraint(),
      TTP->isExpandedParameterPack()
          ? std::optional<unsigned>(TTP->getNumExpansionParameters())
          : std::nullopt);
  if (const auto *TC = TTP->getTypeConstraint())
    SemaRef.SubstTypeConstraint(NewTTP, TC, Args,
                                /*EvaluateConstraint*/ true);
  if (TTP->hasDefaultArgument()) {
    TypeSourceInfo *InstantiatedDefaultArg =
        SemaRef.SubstType(TTP->getDefaultArgumentInfo(), Args,
                          TTP->getDefaultArgumentLoc(), TTP->getDeclName());
    if (InstantiatedDefaultArg)
      NewTTP->setDefaultArgument(InstantiatedDefaultArg);
  }
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(TTP, NewTTP);
  return NewTTP;
}

FunctionTemplateDecl *
buildDeductionGuide(Sema &SemaRef, TemplateDecl *OriginalTemplate,
                    TemplateParameterList *TemplateParams,
                    CXXConstructorDecl *Ctor, ExplicitSpecifier ES,
                    TypeSourceInfo *TInfo, SourceLocation LocStart,
                    SourceLocation Loc, SourceLocation LocEnd, bool IsImplicit,
                    llvm::ArrayRef<TypedefNameDecl *> MaterializedTypedefs) {
  DeclContext *DC = OriginalTemplate->getDeclContext();
  auto DeductionGuideName =
      SemaRef.Context.DeclarationNames.getCXXDeductionGuideName(
          OriginalTemplate);

  DeclarationNameInfo Name(DeductionGuideName, Loc);
  ArrayRef<ParmVarDecl *> Params =
      TInfo->getTypeLoc().castAs<FunctionProtoTypeLoc>().getParams();

  // Build the implicit deduction guide template.
  auto *Guide =
      CXXDeductionGuideDecl::Create(SemaRef.Context, DC, LocStart, ES, Name,
                                    TInfo->getType(), TInfo, LocEnd, Ctor);
  Guide->setImplicit(IsImplicit);
  Guide->setParams(Params);

  for (auto *Param : Params)
    Param->setDeclContext(Guide);
  for (auto *TD : MaterializedTypedefs)
    TD->setDeclContext(Guide);

  auto *GuideTemplate = FunctionTemplateDecl::Create(
      SemaRef.Context, DC, Loc, DeductionGuideName, TemplateParams, Guide);
  GuideTemplate->setImplicit(IsImplicit);
  Guide->setDescribedFunctionTemplate(GuideTemplate);

  if (isa<CXXRecordDecl>(DC)) {
    Guide->setAccess(AS_public);
    GuideTemplate->setAccess(AS_public);
  }

  DC->addDecl(GuideTemplate);
  return GuideTemplate;
}

} // namespace clang
