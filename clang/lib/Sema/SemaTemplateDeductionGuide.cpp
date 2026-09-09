//===- SemaTemplateDeductionGude.cpp - Template Argument Deduction---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements deduction guides for C++ class template argument
// deduction.
//
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "TypeLocBuilder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/BuiltinTraits.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <utility>

using namespace clang;
using namespace sema;

namespace {

/// Return true if two associated-constraint sets are semantically equal.
static bool HaveSameAssociatedConstraints(
    Sema &SemaRef, const NamedDecl *Old, ArrayRef<AssociatedConstraint> OldACs,
    const NamedDecl *New, ArrayRef<AssociatedConstraint> NewACs) {
  if (OldACs.size() != NewACs.size())
    return false;
  if (OldACs.empty())
    return true;

  // General case: pairwise compare each associated constraint expression.
  Sema::TemplateCompareNewDeclInfo NewInfo(New);
  for (size_t I = 0, E = OldACs.size(); I != E; ++I)
    if (!SemaRef.AreConstraintExpressionsEqual(
            Old, OldACs[I].ConstraintExpr, NewInfo, NewACs[I].ConstraintExpr))
      return false;

  return true;
}

/// Tree transform to "extract" a transformed type from a class template's
/// constructor to a deduction guide.
class ExtractTypeForDeductionGuide
    : public TreeTransform<ExtractTypeForDeductionGuide> {
  llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs;
  ClassTemplateDecl *NestedPattern;
  const MultiLevelTemplateArgumentList *OuterInstantiationArgs;
  std::optional<TemplateDeclInstantiator> TypedefNameInstantiator;

public:
  typedef TreeTransform<ExtractTypeForDeductionGuide> Base;
  ExtractTypeForDeductionGuide(
      Sema &SemaRef,
      llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs,
      ClassTemplateDecl *NestedPattern = nullptr,
      const MultiLevelTemplateArgumentList *OuterInstantiationArgs = nullptr)
      : Base(SemaRef), MaterializedTypedefs(MaterializedTypedefs),
        NestedPattern(NestedPattern),
        OuterInstantiationArgs(OuterInstantiationArgs) {
    if (OuterInstantiationArgs)
      TypedefNameInstantiator.emplace(
          SemaRef, SemaRef.getASTContext().getTranslationUnitDecl(),
          *OuterInstantiationArgs);
  }

  TypeSourceInfo *transform(TypeSourceInfo *TSI) { return TransformType(TSI); }

  /// Returns true if it's safe to substitute \p Typedef with
  /// \p OuterInstantiationArgs.
  bool mightReferToOuterTemplateParameters(TypedefNameDecl *Typedef) {
    if (!NestedPattern)
      return false;

    static auto WalkUp = [](DeclContext *DC, DeclContext *TargetDC) {
      if (DC->Equals(TargetDC))
        return true;
      while (DC->isRecord()) {
        if (DC->Equals(TargetDC))
          return true;
        DC = DC->getParent();
      }
      return false;
    };

    if (WalkUp(Typedef->getDeclContext(), NestedPattern->getTemplatedDecl()))
      return true;
    if (WalkUp(NestedPattern->getTemplatedDecl(), Typedef->getDeclContext()))
      return true;
    return false;
  }

  QualType RebuildTemplateSpecializationType(
      ElaboratedTypeKeyword Keyword, TemplateName Template,
      SourceLocation TemplateNameLoc, TemplateArgumentListInfo &TemplateArgs) {
    if (!OuterInstantiationArgs ||
        !isa_and_present<TypeAliasTemplateDecl>(Template.getAsTemplateDecl()))
      return Base::RebuildTemplateSpecializationType(
          Keyword, Template, TemplateNameLoc, TemplateArgs);

    auto *TATD = cast<TypeAliasTemplateDecl>(Template.getAsTemplateDecl());
    auto *Pattern = TATD;
    while (Pattern->getInstantiatedFromMemberTemplate())
      Pattern = Pattern->getInstantiatedFromMemberTemplate();
    if (!mightReferToOuterTemplateParameters(Pattern->getTemplatedDecl()))
      return Base::RebuildTemplateSpecializationType(
          Keyword, Template, TemplateNameLoc, TemplateArgs);

    Decl *NewD =
        TypedefNameInstantiator->InstantiateTypeAliasTemplateDecl(TATD);
    if (!NewD)
      return QualType();

    auto *NewTATD = cast<TypeAliasTemplateDecl>(NewD);
    MaterializedTypedefs.push_back(NewTATD->getTemplatedDecl());

    return Base::RebuildTemplateSpecializationType(
        Keyword, TemplateName(NewTATD), TemplateNameLoc, TemplateArgs);
  }

  QualType TransformTypedefType(TypeLocBuilder &TLB, TypedefTypeLoc TL) {
    ASTContext &Context = SemaRef.getASTContext();
    TypedefNameDecl *OrigDecl = TL.getDecl();
    TypedefNameDecl *Decl = OrigDecl;
    const TypedefType *T = TL.getTypePtr();
    // Transform the underlying type of the typedef and clone the Decl only if
    // the typedef has a dependent context.
    bool InDependentContext = OrigDecl->getDeclContext()->isDependentContext();

    // A typedef/alias Decl within the NestedPattern may reference the outer
    // template parameters. They're substituted with corresponding instantiation
    // arguments here and in RebuildTemplateSpecializationType() above.
    // Otherwise, we would have a CTAD guide with "dangling" template
    // parameters.
    // For example,
    //   template <class T> struct Outer {
    //     using Alias = S<T>;
    //     template <class U> struct Inner {
    //       Inner(Alias);
    //     };
    //   };
    if (OuterInstantiationArgs && InDependentContext &&
        T->isInstantiationDependentType()) {
      Decl = cast_if_present<TypedefNameDecl>(
          TypedefNameInstantiator->InstantiateTypedefNameDecl(
              OrigDecl, /*IsTypeAlias=*/isa<TypeAliasDecl>(OrigDecl)));
      if (!Decl)
        return QualType();
      MaterializedTypedefs.push_back(Decl);
    } else if (InDependentContext) {
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

    NestedNameSpecifierLoc QualifierLoc = TL.getQualifierLoc();
    if (QualifierLoc) {
      QualifierLoc = getDerived().TransformNestedNameSpecifierLoc(QualifierLoc);
      if (!QualifierLoc)
        return QualType();
    }

    QualType TDTy = Context.getTypedefType(
        T->getKeyword(), QualifierLoc.getNestedNameSpecifier(), Decl);
    TLB.push<TypedefTypeLoc>(TDTy).set(TL.getElaboratedKeywordLoc(),
                                       QualifierLoc, TL.getNameLoc());
    return TDTy;
  }
};

// Build a deduction guide using the provided information.
//
// A deduction guide can be either a template or a non-template function
// declaration. If \p TemplateParams is null, a non-template function
// declaration will be created.
CXXDeductionGuideDecl *
buildDeductionGuide(Sema &SemaRef, TemplateDecl *OriginalTemplate,
                    TemplateParameterList *TemplateParams,
                    CXXConstructorDecl *Ctor, ExplicitSpecifier ES,
                    TypeSourceInfo *TInfo, SourceLocation LocStart,
                    SourceLocation Loc, SourceLocation LocEnd, bool IsImplicit,
                    llvm::ArrayRef<TypedefNameDecl *> MaterializedTypedefs = {},
                    const AssociatedConstraint &FunctionTrailingRC = {}) {
  DeclContext *DC = OriginalTemplate->getDeclContext();
  auto DeductionGuideName =
      SemaRef.Context.DeclarationNames.getCXXDeductionGuideName(
          OriginalTemplate);

  DeclarationNameInfo Name(DeductionGuideName, Loc);
  ArrayRef<ParmVarDecl *> Params =
      TInfo->getTypeLoc().castAs<FunctionProtoTypeLoc>().getParams();

  // Build the implicit deduction guide template.
  QualType GuideType = TInfo->getType();

  // In CUDA/HIP mode, avoid duplicate implicit guides that differ only in CUDA
  // target attributes (same constructor signature and constraints).
  if (IsImplicit && Ctor && SemaRef.getLangOpts().CUDA) {
    SmallVector<AssociatedConstraint, 4> NewACs;
    Ctor->getAssociatedConstraints(NewACs);

    for (NamedDecl *Existing : DC->lookup(DeductionGuideName)) {
      auto *ExistingFT = dyn_cast<FunctionTemplateDecl>(Existing);
      auto *ExistingGuide =
          ExistingFT
              ? dyn_cast<CXXDeductionGuideDecl>(ExistingFT->getTemplatedDecl())
              : dyn_cast<CXXDeductionGuideDecl>(Existing);
      if (!ExistingGuide)
        continue;

      // Only consider guides that were also synthesized from a constructor.
      auto *ExistingCtor = ExistingGuide->getCorrespondingConstructor();
      if (!ExistingCtor)
        continue;

      // If the underlying constructors are overloads (different signatures once
      // CUDA attributes are ignored), they should each get their own guides.
      if (SemaRef.IsOverload(Ctor, ExistingCtor,
                             /*UseMemberUsingDeclRules=*/false,
                             /*ConsiderCudaAttrs=*/false))
        continue;

      // At this point, the constructors have the same signature ignoring CUDA
      // attributes. Decide whether their associated constraints are also the
      // same; only in that case do we treat one guide as a duplicate of the
      // other.
      SmallVector<AssociatedConstraint, 4> ExistingACs;
      ExistingCtor->getAssociatedConstraints(ExistingACs);

      if (HaveSameAssociatedConstraints(SemaRef, ExistingCtor, ExistingACs,
                                        Ctor, NewACs))
        return ExistingGuide;
    }
  }

  auto *Guide = CXXDeductionGuideDecl::Create(
      SemaRef.Context, DC, LocStart, ES, Name, GuideType, TInfo, LocEnd, Ctor,
      DeductionCandidate::Normal, FunctionTrailingRC);
  Guide->setImplicit(IsImplicit);
  Guide->setParams(Params);

  for (auto *Param : Params)
    Param->setDeclContext(Guide);
  for (auto *TD : MaterializedTypedefs)
    TD->setDeclContext(Guide);
  if (isa<CXXRecordDecl>(DC))
    Guide->setAccess(AS_public);

  if (!TemplateParams) {
    DC->addDecl(Guide);
    return Guide;
  }

  auto *GuideTemplate = FunctionTemplateDecl::Create(
      SemaRef.Context, DC, Loc, DeductionGuideName, TemplateParams, Guide);
  GuideTemplate->setImplicit(IsImplicit);
  Guide->setDescribedFunctionTemplate(GuideTemplate);

  if (isa<CXXRecordDecl>(DC))
    GuideTemplate->setAccess(AS_public);

  DC->addDecl(GuideTemplate);
  return Guide;
}

// Transform a given template type parameter `TTP`.
TemplateTypeParmDecl *
transformTemplateParam(Sema &SemaRef, DeclContext *DC,
                       TemplateTypeParmDecl *TTP,
                       MultiLevelTemplateArgumentList &Args, unsigned NewDepth,
                       unsigned NewIndex, bool EvaluateConstraint) {
  // TemplateTypeParmDecl's index cannot be changed after creation, so
  // substitute it directly.
  auto *NewTTP = TemplateTypeParmDecl::Create(
      SemaRef.Context, DC, TTP->getBeginLoc(), TTP->getLocation(), NewDepth,
      NewIndex, TTP->getIdentifier(), TTP->wasDeclaredWithTypename(),
      TTP->isParameterPack(), TTP->hasTypeConstraint(),
      TTP->getNumExpansionParameters());
  if (const auto *TC = TTP->getTypeConstraint())
    SemaRef.SubstTypeConstraint(NewTTP, TC, Args,
                                /*EvaluateConstraint=*/EvaluateConstraint);
  if (TTP->hasDefaultArgument()) {
    TemplateArgumentLoc InstantiatedDefaultArg;
    if (!SemaRef.SubstTemplateArgument(
            TTP->getDefaultArgument(), Args, InstantiatedDefaultArg,
            TTP->getDefaultArgumentLoc(), TTP->getDeclName()))
      NewTTP->setDefaultArgument(SemaRef.Context, InstantiatedDefaultArg);
  }
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(TTP, NewTTP);
  return NewTTP;
}

// Transform a given non-type template parameter `TTP`. Returns null if its type
// becomes invalid after substitution.
NonTypeTemplateParmDecl *
transformTemplateParam(Sema &SemaRef, DeclContext *DC,
                       NonTypeTemplateParmDecl *TTP, unsigned NewDepth,
                       unsigned NewIndex,
                       MultiLevelTemplateArgumentList &Args) {
  NonTypeTemplateParmDecl *NewTTP;
  if (TTP->isExpandedParameterPack()) {
    SmallVector<TypeSourceInfo *, 4> ExpandedTypeSourceInfos(
        TTP->getNumExpansionTypes());
    SmallVector<QualType, 4> ExpandedTypes(TTP->getNumExpansionTypes());
    for (unsigned I = 0, N = TTP->getNumExpansionTypes(); I != N; ++I) {
      TypeSourceInfo *NewTSI =
          SemaRef.SubstType(TTP->getExpansionTypeSourceInfo(I), Args,
                            TTP->getLocation(), TTP->getDeclName());
      if (!NewTSI)
        return nullptr;

      QualType NewT =
          SemaRef.CheckNonTypeTemplateParameterType(NewTSI, TTP->getLocation());
      if (NewT.isNull())
        return nullptr;

      ExpandedTypeSourceInfos[I] = NewTSI;
      ExpandedTypes[I] = NewT;
    }
    NewTTP = NonTypeTemplateParmDecl::Create(
        SemaRef.Context, DC, TTP->getBeginLoc(), TTP->getLocation(), NewDepth,
        NewIndex, TTP->getIdentifier(), TTP->getType(),
        TTP->getTypeSourceInfo(), ExpandedTypes, ExpandedTypeSourceInfos);
  } else {
    TypeSourceInfo *NewTSI = SemaRef.SubstType(
        TTP->getTypeSourceInfo(), Args, TTP->getLocation(), TTP->getDeclName());
    if (!NewTSI)
      return nullptr;

    QualType NewT =
        SemaRef.CheckNonTypeTemplateParameterType(NewTSI, TTP->getLocation());
    if (NewT.isNull())
      return nullptr;

    NewTTP = NonTypeTemplateParmDecl::Create(
        SemaRef.Context, DC, TTP->getBeginLoc(), TTP->getLocation(), NewDepth,
        NewIndex, TTP->getIdentifier(), NewT, TTP->isParameterPack(), NewTSI);
  }

  if (TypeSourceInfo *TSI = TTP->getTypeSourceInfo();
      AutoTypeLoc AutoLoc = TSI->getTypeLoc().getContainedAutoTypeLoc()) {
    if (AutoLoc.isConstrained()) {
      SourceLocation EllipsisLoc;
      if (TTP->isExpandedParameterPack())
        EllipsisLoc =
            TSI->getTypeLoc().getAs<PackExpansionTypeLoc>().getEllipsisLoc();
      else if (auto *Constraint = dyn_cast_if_present<CXXFoldExpr>(
                   TTP->getPlaceholderTypeConstraint()))
        EllipsisLoc = Constraint->getEllipsisLoc();
      // Note: We attach the non-instantiated constraint here, so that it can be
      // instantiated relative to the top level, like all our other
      // constraints.
      if (SemaRef.AttachTypeConstraint(AutoLoc, /*NewConstrainedParm=*/NewTTP,
                                       /*OrigConstrainedParm=*/TTP,
                                       EllipsisLoc))
        llvm_unreachable("unexpected failure attaching type constraint");
    }
  }

  NewTTP->setAccess(AS_public);
  NewTTP->setImplicit(TTP->isImplicit());

  if (TTP->hasDefaultArgument()) {
    TemplateArgumentLoc InstantiatedDefaultArg;
    if (!SemaRef.SubstTemplateArgument(
            TTP->getDefaultArgument(), Args, InstantiatedDefaultArg,
            TTP->getDefaultArgumentLoc(), TTP->getDeclName()))
      NewTTP->setDefaultArgument(SemaRef.Context, InstantiatedDefaultArg);
  }

  SemaRef.CurrentInstantiationScope->InstantiatedLocal(TTP, NewTTP);
  return NewTTP;
}

TemplateParameterList *
transformTemplateParameters(Sema &SemaRef, DeclContext *DC,
                            TemplateParameterList *TPL,
                            MultiLevelTemplateArgumentList &Args,
                            unsigned NewDepth, bool EvaluateConstraint);

TemplateTemplateParmDecl *
transformTemplateParam(Sema &SemaRef, DeclContext *DC,
                       TemplateTemplateParmDecl *TTP, unsigned NewDepth,
                       unsigned NewIndex, MultiLevelTemplateArgumentList &Args,
                       bool EvaluateConstraint) {
  TemplateTemplateParmDecl *NewTTP;
  if (TTP->isExpandedParameterPack()) {
    SmallVector<TemplateParameterList *, 4> ExpandedTPLs(
        TTP->getNumExpansionTemplateParameters());
    for (unsigned I = 0, N = TTP->getNumExpansionTemplateParameters(); I != N;
         ++I)
      ExpandedTPLs[I] = transformTemplateParameters(
          SemaRef, DC, TTP->getExpansionTemplateParameters(I), Args,
          NewDepth + 1, EvaluateConstraint);
    NewTTP = TemplateTemplateParmDecl::Create(
        SemaRef.Context, DC, TTP->getLocation(), NewDepth, NewIndex,
        TTP->getIdentifier(), TTP->templateParameterKind(),
        TTP->wasDeclaredWithTypename(), TTP->getTemplateParameters(),
        ExpandedTPLs);
  } else {
    TemplateParameterList *NewTPL =
        transformTemplateParameters(SemaRef, DC, TTP->getTemplateParameters(),
                                    Args, NewDepth + 1, EvaluateConstraint);
    NewTTP = TemplateTemplateParmDecl::Create(
        SemaRef.Context, DC, TTP->getLocation(), NewDepth, NewIndex,
        TTP->isParameterPack(), TTP->getIdentifier(),
        TTP->templateParameterKind(), TTP->wasDeclaredWithTypename(), NewTPL);
  }

  NewTTP->setAccess(AS_public);
  NewTTP->setImplicit(TTP->isImplicit());

  if (TTP->hasDefaultArgument()) {
    TemplateArgumentLoc InstantiatedDefaultArg;
    if (!SemaRef.SubstTemplateArgument(
            TTP->getDefaultArgument(), Args, InstantiatedDefaultArg,
            TTP->getDefaultArgumentLoc(), TTP->getDeclName()))
      NewTTP->setDefaultArgument(SemaRef.Context, InstantiatedDefaultArg);
  }

  SemaRef.CurrentInstantiationScope->InstantiatedLocal(TTP, NewTTP);
  return NewTTP;
}

NamedDecl *transformTemplateParameter(Sema &SemaRef, DeclContext *DC,
                                      NamedDecl *TemplateParam,
                                      MultiLevelTemplateArgumentList &Args,
                                      unsigned NewIndex, unsigned NewDepth,
                                      bool EvaluateConstraint = true) {
  if (auto *TTP = dyn_cast<TemplateTypeParmDecl>(TemplateParam))
    return transformTemplateParam(SemaRef, DC, TTP, Args, NewDepth, NewIndex,
                                  EvaluateConstraint);
  if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(TemplateParam))
    return transformTemplateParam(SemaRef, DC, NTTP, NewDepth, NewIndex, Args);
  if (auto *TTP = dyn_cast<TemplateTemplateParmDecl>(TemplateParam))
    return transformTemplateParam(SemaRef, DC, TTP, NewDepth, NewIndex, Args,
                                  EvaluateConstraint);
  llvm_unreachable("Unhandled template parameter types");
}

TemplateParameterList *
transformTemplateParameters(Sema &SemaRef, DeclContext *DC,
                            TemplateParameterList *TPL,
                            MultiLevelTemplateArgumentList &Args,
                            unsigned NewDepth, bool EvaluateConstraint) {
  SmallVector<NamedDecl *, 4> Params(TPL->size());
  for (unsigned I = 0, E = TPL->size(); I < E; ++I) {
    Params[I] = transformTemplateParameter(SemaRef, DC, TPL->getParam(I), Args,
                                           /*NewIndex=*/I, NewDepth,
                                           EvaluateConstraint);
  }
  return TemplateParameterList::Create(
      SemaRef.Context, TPL->getTemplateLoc(), TPL->getLAngleLoc(), Params,
      TPL->getRAngleLoc(), TPL->getRequiresClause());
}

/// Transform to convert portions of a constructor declaration into the
/// corresponding deduction guide, per C++1z [over.match.class.deduct]p1.
struct ConvertConstructorToDeductionGuideTransform {
  ConvertConstructorToDeductionGuideTransform(Sema &S,
                                              ClassTemplateDecl *Template)
      : SemaRef(S), Template(Template) {
    // If the template is nested, then we need to use the original
    // pattern to iterate over the constructors.
    ClassTemplateDecl *Pattern = Template;
    while (Pattern->getInstantiatedFromMemberTemplate()) {
      if (Pattern->isMemberSpecialization())
        break;
      Pattern = Pattern->getInstantiatedFromMemberTemplate();
      NestedPattern = Pattern;
    }

    if (NestedPattern)
      OuterInstantiationArgs = SemaRef.getTemplateInstantiationArgs(Template);
  }

  Sema &SemaRef;
  ClassTemplateDecl *Template;
  ClassTemplateDecl *NestedPattern = nullptr;

  DeclContext *DC = Template->getDeclContext();
  CXXRecordDecl *Primary = Template->getTemplatedDecl();
  DeclarationName DeductionGuideName =
      SemaRef.Context.DeclarationNames.getCXXDeductionGuideName(Template);

  QualType DeducedType = SemaRef.Context.getCanonicalTagType(Primary);

  // Index adjustment to apply to convert depth-1 template parameters into
  // depth-0 template parameters.
  unsigned Depth1IndexAdjustment = Template->getTemplateParameters()->size();

  // Instantiation arguments for the outermost depth-1 templates
  // when the template is nested
  MultiLevelTemplateArgumentList OuterInstantiationArgs;

  /// Transform a constructor declaration into a deduction guide.
  NamedDecl *transformConstructor(FunctionTemplateDecl *FTD,
                                  CXXConstructorDecl *CD) {
    SmallVector<TemplateArgument, 16> SubstArgs;

    LocalInstantiationScope Scope(SemaRef);

    // C++ [over.match.class.deduct]p1:
    // -- For each constructor of the class template designated by the
    //    template-name, a function template with the following properties:

    //    -- The template parameters are the template parameters of the class
    //       template followed by the template parameters (including default
    //       template arguments) of the constructor, if any.
    TemplateParameterList *TemplateParams =
        SemaRef.GetTemplateParameterList(Template);
    SmallVector<TemplateArgument, 16> Depth1Args;
    AssociatedConstraint OuterRC(TemplateParams->getRequiresClause());
    if (FTD) {
      TemplateParameterList *InnerParams = FTD->getTemplateParameters();
      SmallVector<NamedDecl *, 16> AllParams;
      AllParams.reserve(TemplateParams->size() + InnerParams->size());
      AllParams.insert(AllParams.begin(), TemplateParams->begin(),
                       TemplateParams->end());
      SubstArgs.reserve(InnerParams->size());
      Depth1Args.reserve(InnerParams->size());

      // Later template parameters could refer to earlier ones, so build up
      // a list of substituted template arguments as we go.
      for (NamedDecl *Param : *InnerParams) {
        MultiLevelTemplateArgumentList Args;
        Args.setKind(TemplateSubstitutionKind::Rewrite);
        Args.addOuterTemplateArguments(Depth1Args);
        Args.addOuterRetainedLevel();
        if (NestedPattern)
          Args.addOuterRetainedLevels(NestedPattern->getTemplateDepth());
        auto [Depth, Index] = getDepthAndIndex(Param);
        // Depth can be 0 if FTD belongs to a non-template class/a class
        // template specialization with an empty template parameter list. In
        // that case, we don't want the NewDepth to overflow, and it should
        // remain 0.
        NamedDecl *NewParam = transformTemplateParameter(
            SemaRef, DC, Param, Args, Index + Depth1IndexAdjustment,
            Depth ? Depth - 1 : 0);
        if (!NewParam)
          return nullptr;
        // Constraints require that we substitute depth-1 arguments
        // to match depths when substituted for evaluation later
        Depth1Args.push_back(SemaRef.Context.getInjectedTemplateArg(NewParam));

        if (NestedPattern) {
          auto [Depth, Index] = getDepthAndIndex(NewParam);
          NewParam = transformTemplateParameter(
              SemaRef, DC, NewParam, OuterInstantiationArgs, Index,
              Depth - OuterInstantiationArgs.getNumSubstitutedLevels(),
              /*EvaluateConstraint=*/false);
        }

        assert(getDepthAndIndex(NewParam).first == 0 &&
               "Unexpected template parameter depth");

        AllParams.push_back(NewParam);
        SubstArgs.push_back(SemaRef.Context.getInjectedTemplateArg(NewParam));
      }

      // Substitute new template parameters into requires-clause if present.
      Expr *RequiresClause = nullptr;
      if (Expr *InnerRC = InnerParams->getRequiresClause()) {
        MultiLevelTemplateArgumentList Args;
        Args.setKind(TemplateSubstitutionKind::Rewrite);
        Args.addOuterTemplateArguments(Depth1Args);
        Args.addOuterRetainedLevel();
        if (NestedPattern)
          Args.addOuterRetainedLevels(NestedPattern->getTemplateDepth());
        ExprResult E =
            SemaRef.SubstConstraintExprWithoutSatisfaction(InnerRC, Args);
        if (!E.isUsable())
          return nullptr;
        RequiresClause = E.get();
      }

      TemplateParams = TemplateParameterList::Create(
          SemaRef.Context, InnerParams->getTemplateLoc(),
          InnerParams->getLAngleLoc(), AllParams, InnerParams->getRAngleLoc(),
          RequiresClause);
    }

    // If we built a new template-parameter-list, track that we need to
    // substitute references to the old parameters into references to the
    // new ones.
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    if (FTD) {
      Args.addOuterTemplateArguments(SubstArgs);
      Args.addOuterRetainedLevel();
    }

    FunctionProtoTypeLoc FPTL = CD->getTypeSourceInfo()
                                    ->getTypeLoc()
                                    .getAsAdjusted<FunctionProtoTypeLoc>();
    assert(FPTL && "no prototype for constructor declaration");

    // Transform the type of the function, adjusting the return type and
    // replacing references to the old parameters with references to the
    // new ones.
    TypeLocBuilder TLB;
    SmallVector<ParmVarDecl *, 8> Params;
    SmallVector<TypedefNameDecl *, 4> MaterializedTypedefs;
    QualType NewType = transformFunctionProtoType(TLB, FPTL, Params, Args,
                                                  MaterializedTypedefs);
    if (NewType.isNull())
      return nullptr;
    TypeSourceInfo *NewTInfo = TLB.getTypeSourceInfo(SemaRef.Context, NewType);

    // At this point, the function parameters are already 'instantiated' in the
    // current scope. Substitute into the constructor's trailing
    // requires-clause, if any.
    AssociatedConstraint FunctionTrailingRC;
    if (const AssociatedConstraint &RC = CD->getTrailingRequiresClause()) {
      MultiLevelTemplateArgumentList Args;
      Args.setKind(TemplateSubstitutionKind::Rewrite);
      Args.addOuterTemplateArguments(Depth1Args);
      Args.addOuterRetainedLevel();
      if (NestedPattern)
        Args.addOuterRetainedLevels(NestedPattern->getTemplateDepth());
      ExprResult E = SemaRef.SubstConstraintExprWithoutSatisfaction(
          const_cast<Expr *>(RC.ConstraintExpr), Args);
      if (!E.isUsable())
        return nullptr;
      FunctionTrailingRC = AssociatedConstraint(E.get(), RC.ArgPackSubstIndex);
    }

    // C++ [over.match.class.deduct]p1:
    // If C is defined, for each constructor of C, a function template with
    // the following properties:
    // [...]
    // - The associated constraints are the conjunction of the associated
    // constraints of C and the associated constraints of the constructor, if
    // any.
    if (OuterRC) {
      // The outer template parameters are not transformed, so their
      // associated constraints don't need substitution.
      // FIXME: Should simply add another field for the OuterRC, instead of
      // combining them like this.
      if (!FunctionTrailingRC)
        FunctionTrailingRC = OuterRC;
      else
        FunctionTrailingRC = AssociatedConstraint(
            BinaryOperator::Create(
                SemaRef.Context,
                /*lhs=*/const_cast<Expr *>(OuterRC.ConstraintExpr),
                /*rhs=*/const_cast<Expr *>(FunctionTrailingRC.ConstraintExpr),
                BO_LAnd, SemaRef.Context.BoolTy, VK_PRValue, OK_Ordinary,
                TemplateParams->getTemplateLoc(), FPOptionsOverride()),
            FunctionTrailingRC.ArgPackSubstIndex);
    }

    return buildDeductionGuide(
        SemaRef, Template, TemplateParams, CD, CD->getExplicitSpecifier(),
        NewTInfo, CD->getBeginLoc(), CD->getLocation(), CD->getEndLoc(),
        /*IsImplicit=*/true, MaterializedTypedefs, FunctionTrailingRC);
  }

  /// Build a deduction guide with the specified parameter types.
  CXXDeductionGuideDecl *
  buildSimpleDeductionGuide(MutableArrayRef<QualType> ParamTypes) {
    SourceLocation Loc = Template->getLocation();

    // Build the requested type.
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.HasTrailingReturn = true;
    QualType Result = SemaRef.BuildFunctionType(DeducedType, ParamTypes, Loc,
                                                DeductionGuideName, EPI);
    TypeSourceInfo *TSI = SemaRef.Context.getTrivialTypeSourceInfo(Result, Loc);
    if (NestedPattern)
      TSI = SemaRef.SubstType(TSI, OuterInstantiationArgs, Loc,
                              DeductionGuideName);

    if (!TSI)
      return nullptr;

    FunctionProtoTypeLoc FPTL =
        TSI->getTypeLoc().castAs<FunctionProtoTypeLoc>();

    // Build the parameters, needed during deduction / substitution.
    SmallVector<ParmVarDecl *, 4> Params;
    for (auto T : ParamTypes) {
      auto *TSI = SemaRef.Context.getTrivialTypeSourceInfo(T, Loc);
      if (NestedPattern)
        TSI = SemaRef.SubstType(TSI, OuterInstantiationArgs, Loc,
                                DeclarationName());
      if (!TSI)
        return nullptr;

      ParmVarDecl *NewParam =
          ParmVarDecl::Create(SemaRef.Context, DC, Loc, Loc, nullptr,
                              TSI->getType(), TSI, SC_None, nullptr);
      NewParam->setScopeInfo(0, Params.size());
      FPTL.setParam(Params.size(), NewParam);
      Params.push_back(NewParam);
    }

    return buildDeductionGuide(
        SemaRef, Template, SemaRef.GetTemplateParameterList(Template), nullptr,
        ExplicitSpecifier(), TSI, Loc, Loc, Loc, /*IsImplicit=*/true);
  }

private:
  QualType transformFunctionProtoType(
      TypeLocBuilder &TLB, FunctionProtoTypeLoc TL,
      SmallVectorImpl<ParmVarDecl *> &Params,
      MultiLevelTemplateArgumentList &Args,
      SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs) {
    SmallVector<QualType, 4> ParamTypes;
    const FunctionProtoType *T = TL.getTypePtr();

    //    -- The types of the function parameters are those of the constructor.
    for (auto *OldParam : TL.getParams()) {
      ParmVarDecl *NewParam = OldParam;
      // Given
      //   template <class T> struct C {
      //     template <class U> struct D {
      //       template <class V> D(U, V);
      //     };
      //   };
      // First, transform all the references to template parameters that are
      // defined outside of the surrounding class template. That is T in the
      // above example.
      if (NestedPattern) {
        NewParam = transformFunctionTypeParam(
            NewParam, OuterInstantiationArgs, MaterializedTypedefs,
            /*TransformingOuterPatterns=*/true);
        if (!NewParam)
          return QualType();
      }
      // Then, transform all the references to template parameters that are
      // defined at the class template and the constructor. In this example,
      // they're U and V, respectively.
      NewParam =
          transformFunctionTypeParam(NewParam, Args, MaterializedTypedefs,
                                     /*TransformingOuterPatterns=*/false);
      if (!NewParam)
        return QualType();
      ParamTypes.push_back(NewParam->getType());
      Params.push_back(NewParam);
    }

    //    -- The return type is the class template specialization designated by
    //       the template-name and template arguments corresponding to the
    //       template parameters obtained from the class template.
    //
    // We use the injected-class-name type of the primary template instead.
    // This has the convenient property that it is different from any type that
    // the user can write in a deduction-guide (because they cannot enter the
    // context of the template), so implicit deduction guides can never collide
    // with explicit ones.
    QualType ReturnType = DeducedType;
    auto TTL = TLB.push<TagTypeLoc>(ReturnType);
    TTL.setElaboratedKeywordLoc(SourceLocation());
    TTL.setQualifierLoc(NestedNameSpecifierLoc());
    TTL.setNameLoc(Primary->getLocation());

    // Resolving a wording defect, we also inherit the variadicness of the
    // constructor.
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.Variadic = T->isVariadic();
    EPI.HasTrailingReturn = true;

    QualType Result = SemaRef.BuildFunctionType(
        ReturnType, ParamTypes, TL.getBeginLoc(), DeductionGuideName, EPI);
    if (Result.isNull())
      return QualType();

    FunctionProtoTypeLoc NewTL = TLB.push<FunctionProtoTypeLoc>(Result);
    NewTL.setLocalRangeBegin(TL.getLocalRangeBegin());
    NewTL.setLParenLoc(TL.getLParenLoc());
    NewTL.setRParenLoc(TL.getRParenLoc());
    NewTL.setExceptionSpecRange(SourceRange());
    NewTL.setLocalRangeEnd(TL.getLocalRangeEnd());
    for (unsigned I = 0, E = NewTL.getNumParams(); I != E; ++I)
      NewTL.setParam(I, Params[I]);

    return Result;
  }

  ParmVarDecl *transformFunctionTypeParam(
      ParmVarDecl *OldParam, MultiLevelTemplateArgumentList &Args,
      llvm::SmallVectorImpl<TypedefNameDecl *> &MaterializedTypedefs,
      bool TransformingOuterPatterns) {
    TypeSourceInfo *OldTSI = OldParam->getTypeSourceInfo();
    TypeSourceInfo *NewTSI;
    if (auto PackTL = OldTSI->getTypeLoc().getAs<PackExpansionTypeLoc>()) {
      // Expand out the one and only element in each inner pack.
      Sema::ArgPackSubstIndexRAII SubstIndex(SemaRef, 0u);
      NewTSI =
          SemaRef.SubstType(PackTL.getPatternLoc(), Args,
                            OldParam->getLocation(), OldParam->getDeclName());
      if (!NewTSI)
        return nullptr;
      NewTSI =
          SemaRef.CheckPackExpansion(NewTSI, PackTL.getEllipsisLoc(),
                                     PackTL.getTypePtr()->getNumExpansions());
    } else
      NewTSI = SemaRef.SubstType(OldTSI, Args, OldParam->getLocation(),
                                 OldParam->getDeclName());
    if (!NewTSI)
      return nullptr;

    // Extract the type. This (for instance) replaces references to typedef
    // members of the current instantiations with the definitions of those
    // typedefs, avoiding triggering instantiation of the deduced type during
    // deduction.
    NewTSI = ExtractTypeForDeductionGuide(
                 SemaRef, MaterializedTypedefs, NestedPattern,
                 TransformingOuterPatterns ? &Args : nullptr)
                 .transform(NewTSI);
    if (!NewTSI)
      return nullptr;
    // Resolving a wording defect, we also inherit default arguments from the
    // constructor.
    ExprResult NewDefArg;
    if (OldParam->hasDefaultArg()) {
      // We don't care what the value is (we won't use it); just create a
      // placeholder to indicate there is a default argument.
      QualType ParamTy = NewTSI->getType();
      NewDefArg = new (SemaRef.Context)
          OpaqueValueExpr(OldParam->getDefaultArgRange().getBegin(),
                          ParamTy.getNonLValueExprType(SemaRef.Context),
                          ParamTy->isLValueReferenceType()   ? VK_LValue
                          : ParamTy->isRValueReferenceType() ? VK_XValue
                                                             : VK_PRValue);
    }
    // Handle arrays and functions decay.
    auto NewType = NewTSI->getType();
    if (NewType->isArrayType() || NewType->isFunctionType())
      NewType = SemaRef.Context.getDecayedType(NewType);

    ParmVarDecl *NewParam = ParmVarDecl::Create(
        SemaRef.Context, DC, OldParam->getInnerLocStart(),
        OldParam->getLocation(), OldParam->getIdentifier(), NewType, NewTSI,
        OldParam->getStorageClass(), NewDefArg.get());
    NewParam->setScopeInfo(OldParam->getFunctionScopeDepth(),
                           OldParam->getFunctionScopeIndex());
    SemaRef.CurrentInstantiationScope->InstantiatedLocal(OldParam, NewParam);
    return NewParam;
  }
};

// Returns the default template argument of the given template parameter, or
// null if it doesn't have one.
static const TemplateArgumentLoc *getDefaultArgument(const NamedDecl *Param) {
  auto Get = [](const auto *P) -> const TemplateArgumentLoc * {
    return P->hasDefaultArgument() ? &P->getDefaultArgument() : nullptr;
  };
  if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
    return Get(TTP);
  if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param))
    return Get(NTTP);
  return Get(cast<TemplateTemplateParmDecl>(Param));
}

// Find all template parameters that appear in the given DeducedArgs.
// Return the indices of the template parameters in the TemplateParams.
SmallVector<unsigned> TemplateParamsReferencedInTemplateArgumentList(
    Sema &SemaRef, const TemplateParameterList *TemplateParamsList,
    ArrayRef<TemplateArgument> DeducedArgs) {

  llvm::SmallBitVector ReferencedTemplateParams(TemplateParamsList->size());
  SemaRef.MarkUsedTemplateParameters(DeducedArgs, /*OnlyDeduced=*/false,
                                     TemplateParamsList->getDepth(),
                                     ReferencedTemplateParams);

  for (unsigned Index = 0; Index < TemplateParamsList->size(); ++Index) {
    if (!ReferencedTemplateParams[Index])
      continue;
    if (const TemplateArgumentLoc *Default =
            getDefaultArgument(TemplateParamsList->getParam(Index)))
      SemaRef.MarkUsedTemplateParameters(
          Default->getArgument(), /*OnlyDeduced=*/false,
          TemplateParamsList->getDepth(), ReferencedTemplateParams);
  }

  SmallVector<unsigned> Results;
  for (unsigned Index = 0; Index < TemplateParamsList->size(); ++Index) {
    if (ReferencedTemplateParams[Index])
      Results.push_back(Index);
  }
  return Results;
}

bool hasDeclaredDeductionGuides(DeclarationName Name, DeclContext *DC) {
  // Check whether we've already declared deduction guides for this template.
  // FIXME: Consider storing a flag on the template to indicate this.
  assert(Name.getNameKind() ==
             DeclarationName::NameKind::CXXDeductionGuideName &&
         "name must be a deduction guide name");
  auto Existing = DC->lookup(Name);
  for (auto *D : Existing)
    if (D->isImplicit())
      return true;
  return false;
}

// Returns all source deduction guides associated with the declared
// deduction guides that have the specified deduction guide name.
llvm::DenseSet<const NamedDecl *> getSourceDeductionGuides(DeclarationName Name,
                                                           DeclContext *DC) {
  assert(Name.getNameKind() ==
             DeclarationName::NameKind::CXXDeductionGuideName &&
         "name must be a deduction guide name");
  llvm::DenseSet<const NamedDecl *> Result;
  for (auto *D : DC->lookup(Name)) {
    if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(D))
      D = FTD->getTemplatedDecl();

    if (const auto *GD = dyn_cast<CXXDeductionGuideDecl>(D)) {
      assert(GD->getSourceDeductionGuide() &&
             "deduction guide for alias template must have a source deduction "
             "guide");
      Result.insert(GD->getSourceDeductionGuide());
    }
  }
  return Result;
}

// Marker for a template parameter that doesn't appear in the synthesized
// deduction guide f' of an alias template.
constexpr unsigned InvalidFPrimeIndex = -1;

// A template parameter of the synthesized deduction guide f' of an alias
// template A, before it is created.
struct FPrimeTemplateParamRef {
  // Whether this is a template parameter of A, as opposed to a non-deduced
  // template parameter of the underlying deduction guide f.
  bool IsAliasParam;
  // The index of the template parameter in the template parameter list of A or
  // f, respectively.
  unsigned Index;
};

static void setDefaultArgument(ASTContext &Context, NamedDecl *Param,
                               const TemplateArgumentLoc &DefArg) {
  if (auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
    TTP->setDefaultArgument(Context, DefArg);
  else if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param))
    NTTP->setDefaultArgument(Context, DefArg);
  else
    cast<TemplateTemplateParmDecl>(Param)->setDefaultArgument(Context, DefArg);
}

// Build the associated constraints for the alias deduction guides.
// C++ [over.match.class.deduct]p3.3:
//   The associated constraints ([temp.constr.decl]) are the conjunction of the
//   associated constraints of g and a constraint that is satisfied if and only
//   if the arguments of A are deducible (see below) from the return type.
//
// AliasParamFPrimeIndex and FParamFPrimeIndex give the index in f' of the
// template parameters of the alias template and of the non-deduced template
// parameters of F, respectively (InvalidFPrimeIndex for those not in f').
//
// The return result is expected to be the require-clause for the synthesized
// alias deduction guide.
Expr *
buildAssociatedConstraints(Sema &SemaRef, FunctionTemplateDecl *F,
                           TypeAliasTemplateDecl *AliasTemplate,
                           ArrayRef<DeducedTemplateArgument> DeduceResults,
                           ArrayRef<unsigned> AliasParamFPrimeIndex,
                           ArrayRef<unsigned> FParamFPrimeIndex,
                           Expr *IsDeducible) {
  Expr *RC = F->getTemplateParameters()->getRequiresClause();
  if (!RC)
    return IsDeducible;

  ASTContext &Context = SemaRef.Context;
  LocalInstantiationScope Scope(SemaRef);

  // In the clang AST, constraint nodes are deliberately not instantiated unless
  // they are actively being evaluated. Consequently, occurrences of template
  // parameters in the require-clause expression have a subtle "depth"
  // difference compared to normal occurrences in places, such as function
  // parameters. When transforming the require-clause, we must take this
  // distinction into account:
  //
  //   1) In the transformed require-clause, occurrences of template parameters
  //   must use the "uninstantiated" depth;
  //   2) When substituting on the require-clause expr of the underlying
  //   deduction guide, we must use the entire set of template argument lists;
  //
  // It's important to note that we're performing this transformation on an
  // *instantiated* AliasTemplate.

  // For 1), if the alias template is nested within a class template, we
  // calcualte the 'uninstantiated' depth by adding the substitution level back.
  unsigned AdjustDepth = 0;
  if (auto *PrimaryTemplate =
          AliasTemplate->getInstantiatedFromMemberTemplate())
    AdjustDepth = PrimaryTemplate->getTemplateDepth();

  // We rebuild all template parameters with the uninstantiated depth, and
  // build template arguments refer to them.
  SmallVector<TemplateArgument> AdjustedAliasTemplateArgs;

  for (auto [Index, TP] :
       llvm::enumerate(*AliasTemplate->getTemplateParameters())) {
    // Rebuild any internal references to earlier parameters and reindex
    // as we go. Template parameters of the alias that don't appear in f' are
    // not referred to by the deduced template arguments; keep their index.
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    Args.addOuterTemplateArguments(AdjustedAliasTemplateArgs);
    unsigned NewIndex = AliasParamFPrimeIndex[Index] != InvalidFPrimeIndex
                            ? AliasParamFPrimeIndex[Index]
                            : Index;
    NamedDecl *NewParam = transformTemplateParameter(
        SemaRef, AliasTemplate->getDeclContext(), TP, Args, NewIndex,
        getDepthAndIndex(TP).first + AdjustDepth);

    TemplateArgument NewTemplateArgument =
        Context.getInjectedTemplateArg(NewParam);
    AdjustedAliasTemplateArgs.push_back(NewTemplateArgument);
  }
  // Template arguments used to transform the template arguments in
  // DeducedResults.
  SmallVector<TemplateArgument> TemplateArgsForBuildingRC(
      F->getTemplateParameters()->size());
  // Transform the transformed template args
  MultiLevelTemplateArgumentList Args;
  Args.setKind(TemplateSubstitutionKind::Rewrite);
  Args.addOuterTemplateArguments(AdjustedAliasTemplateArgs);

  for (unsigned Index = 0; Index < DeduceResults.size(); ++Index) {
    const auto &D = DeduceResults[Index];
    if (D.isNull()) { // non-deduced template parameters of f
      NamedDecl *TP = F->getTemplateParameters()->getParam(Index);
      MultiLevelTemplateArgumentList Args;
      Args.setKind(TemplateSubstitutionKind::Rewrite);
      Args.addOuterTemplateArguments(TemplateArgsForBuildingRC);
      // Rebuild the template parameter with updated depth and index.
      NamedDecl *NewParam =
          transformTemplateParameter(SemaRef, F->getDeclContext(), TP, Args,
                                     /*NewIndex=*/FParamFPrimeIndex[Index],
                                     getDepthAndIndex(TP).first + AdjustDepth);
      assert(TemplateArgsForBuildingRC[Index].isNull());
      TemplateArgsForBuildingRC[Index] =
          Context.getInjectedTemplateArg(NewParam);
      continue;
    }
    TemplateArgumentLoc Input =
        SemaRef.getTrivialTemplateArgumentLoc(D, QualType(), SourceLocation{});
    TemplateArgumentLoc Output;
    if (!SemaRef.SubstTemplateArgument(Input, Args, Output)) {
      assert(TemplateArgsForBuildingRC[Index].isNull() &&
             "InstantiatedArgs must be null before setting");
      TemplateArgsForBuildingRC[Index] = Output.getArgument();
    }
  }

  // A list of template arguments for transforming the require-clause of F.
  // It must contain the entire set of template argument lists.
  MultiLevelTemplateArgumentList ArgsForBuildingRC;
  ArgsForBuildingRC.setKind(clang::TemplateSubstitutionKind::Rewrite);
  ArgsForBuildingRC.addOuterTemplateArguments(TemplateArgsForBuildingRC);
  // For 2), if the underlying deduction guide F is nested in a class template,
  // we need the entire template argument list, as the constraint AST in the
  // require-clause of F remains completely uninstantiated.
  //
  // For example:
  //   template <typename T> // depth 0
  //   struct Outer {
  //      template <typename U>
  //      struct Foo { Foo(U); };
  //
  //      template <typename U> // depth 1
  //      requires C<U>
  //      Foo(U) -> Foo<int>;
  //   };
  //   template <typename U>
  //   using AFoo = Outer<int>::Foo<U>;
  //
  // In this scenario, the deduction guide for `Foo` inside `Outer<int>`:
  //   - The occurrence of U in the require-expression is [depth:1, index:0]
  //   - The occurrence of U in the function parameter is [depth:0, index:0]
  //   - The template parameter of U is [depth:0, index:0]
  //
  // We add the outer template arguments which is [int] to the multi-level arg
  // list to ensure that the occurrence U in `C<U>` will be replaced with int
  // during the substitution.
  //
  // NOTE: The underlying deduction guide F is instantiated -- either from an
  // explicitly-written deduction guide member, or from a constructor.
  // getInstantiatedFromMemberTemplate() can only handle the former case, so we
  // check the DeclContext kind.
  if (F->getLexicalDeclContext()->getDeclKind() ==
      clang::Decl::ClassTemplateSpecialization) {
    auto OuterLevelArgs = SemaRef.getTemplateInstantiationArgs(
        F, F->getLexicalDeclContext(),
        /*Final=*/false, /*Innermost=*/std::nullopt,
        /*RelativeToPrimary=*/true,
        /*Pattern=*/nullptr,
        /*ForConstraintInstantiation=*/true);
    for (auto It : OuterLevelArgs)
      ArgsForBuildingRC.addOuterTemplateArguments(It.Args);
  }

  ExprResult E = SemaRef.SubstExpr(RC, ArgsForBuildingRC);
  if (E.isInvalid())
    return nullptr;

  auto Conjunction =
      SemaRef.BuildBinOp(SemaRef.getCurScope(), SourceLocation{},
                         BinaryOperatorKind::BO_LAnd, E.get(), IsDeducible);
  if (Conjunction.isInvalid())
    return nullptr;
  return Conjunction.getAs<Expr>();
}
// Build the is_deducible constraint for the alias deduction guides.
// [over.match.class.deduct]p3.3:
//    ... and a constraint that is satisfied if and only if the arguments
//    of A are deducible (see below) from the return type.
Expr *buildIsDeducibleConstraint(Sema &SemaRef,
                                 TypeAliasTemplateDecl *AliasTemplate,
                                 QualType ReturnType,
                                 SmallVector<NamedDecl *> TemplateParams) {
  ASTContext &Context = SemaRef.Context;
  // Constraint AST nodes must use uninstantiated depth.
  if (auto *PrimaryTemplate =
          AliasTemplate->getInstantiatedFromMemberTemplate();
      PrimaryTemplate && TemplateParams.size() > 0) {
    LocalInstantiationScope Scope(SemaRef);

    // Adjust the depth for TemplateParams.
    unsigned AdjustDepth = PrimaryTemplate->getTemplateDepth();
    SmallVector<TemplateArgument> TransformedTemplateArgs;
    for (auto *TP : TemplateParams) {
      // Rebuild any internal references to earlier parameters and reindex
      // as we go.
      MultiLevelTemplateArgumentList Args;
      Args.setKind(TemplateSubstitutionKind::Rewrite);
      Args.addOuterTemplateArguments(TransformedTemplateArgs);
      NamedDecl *NewParam = transformTemplateParameter(
          SemaRef, AliasTemplate->getDeclContext(), TP, Args,
          /*NewIndex=*/TransformedTemplateArgs.size(),
          getDepthAndIndex(TP).first + AdjustDepth);

      TemplateArgument NewTemplateArgument =
          Context.getInjectedTemplateArg(NewParam);
      TransformedTemplateArgs.push_back(NewTemplateArgument);
    }
    // Transformed the ReturnType to restore the uninstantiated depth.
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    Args.addOuterTemplateArguments(TransformedTemplateArgs);
    ReturnType = SemaRef.SubstType(
        ReturnType, Args, AliasTemplate->getLocation(),
        Context.DeclarationNames.getCXXDeductionGuideName(AliasTemplate));
  }

  SmallVector<TypeSourceInfo *> IsDeducibleTypeTraitArgs = {
      Context.getTrivialTypeSourceInfo(
          Context.getDeducedTemplateSpecializationType(
              DeducedKind::DeducedAsDependent,
              /*DeducedAsType=*/QualType(), ElaboratedTypeKeyword::None,
              TemplateName(AliasTemplate)),
          AliasTemplate->getLocation()), // template specialization type whose
                                         // arguments will be deduced.
      Context.getTrivialTypeSourceInfo(
          ReturnType, AliasTemplate->getLocation()), // type from which template
                                                     // arguments are deduced.
  };
  return TypeTraitExpr::Create(
      Context, Context.getLogicalOperationType(), AliasTemplate->getLocation(),
      TypeTrait::BTT_IsDeducible, IsDeducibleTypeTraitArgs,
      AliasTemplate->getLocation(), /*Value*/ false);
}

std::pair<TemplateDecl *, llvm::ArrayRef<TemplateArgument>>
getRHSTemplateDeclAndArgs(Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate) {
  auto RhsType = AliasTemplate->getTemplatedDecl()->getUnderlyingType();
  TemplateDecl *Template = nullptr;
  llvm::ArrayRef<TemplateArgument> AliasRhsTemplateArgs;
  const auto *TST = RhsType->getAs<TemplateSpecializationType>();

  // The RHS of the alias may name another alias template that can never have
  // deduction guides of its own, because its defining-type-id is not of the
  // form
  //   [typename] [nested-name-specifier] [template] simple-template-id
  // as required by [over.match.class.deduct]p3. e.g.
  //   template <typename T>
  //   using Identity = T;
  //   template <typename T>
  //   using C = Identity<Foo<T>>;
  // Per [temp.alias]p2, Identity<Foo<T>> is equivalent to Foo<T>, so step
  // through such aliases and derive the deduction guides from the first
  // template that can actually have them (GH125821).
  while (TST) {
    auto *RhsAlias = dyn_cast_or_null<TypeAliasTemplateDecl>(
        TST->getTemplateName().getAsTemplateDecl());
    if (!RhsAlias || getRHSTemplateDeclAndArgs(SemaRef, RhsAlias).first)
      break;
    RhsType = TST->desugar();
    TST = RhsType->getAs<TemplateSpecializationType>();
  }

  if (TST) {
    // Cases where the RHS of the alias is dependent. e.g.
    //   template<typename T>
    //   using AliasFoo1 = Foo<T>; // a class/type alias template specialization
    // The RHS may not desugar to a template specialization at all (e.g. an
    // alias of the form 'T*' whose specialization ends up being a pointer);
    // in that case, there is no template to derive the guides from.
    if (const auto *RhsTST = TST->getAsNonAliasTemplateSpecializationType()) {
      Template = TST->getTemplateName().getAsTemplateDecl();
      AliasRhsTemplateArgs = RhsTST->template_arguments();
    }
  } else if (const auto *RT = RhsType->getAs<RecordType>()) {
    // Cases where template arguments in the RHS of the alias are not
    // dependent. e.g.
    //   using AliasFoo = Foo<bool>;
    if (const auto *CTSD =
            dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl())) {
      Template = CTSD->getSpecializedTemplate();
      AliasRhsTemplateArgs = CTSD->getTemplateArgs().asArray();
    }
  }
  return {Template, AliasRhsTemplateArgs};
}

bool IsNonDeducedArgument(const TemplateArgument &TA) {
  // The following cases indicate the template argument is non-deducible:
  //   1. The result is null. E.g. When it comes from a default template
  //   argument that doesn't appear in the alias declaration.
  //   2. The template parameter is a pack and that cannot be deduced from
  //   the arguments within the alias declaration.
  // Non-deducible template parameters will persist in the transformed
  // deduction guide.
  return TA.isNull() ||
         (TA.getKind() == TemplateArgument::Pack &&
          llvm::any_of(TA.pack_elements(), IsNonDeducedArgument));
}

// Synthesize default template arguments for the template parameters of the
// alias template A that appear in the deduction guide f' without having a
// default template argument.
//
// Such a template parameter cannot be deduced from the function parameters of
// f' if it only appears in f' through the default template arguments of other
// template parameters of A, e.g.
//
//   template <class Key, class Hash = std::hash<Key>>
//   using MySet = std::unordered_set<Key, Hash>;
//
// with the deduction guide
//
//   template <class It, class H = std::hash<iter_value_t<It>>>
//   unordered_set(It, It, H = H()) -> unordered_set<iter_value_t<It>, H>;
//
// Deducing the return type of the guide from the defining-type-id of MySet
// gives H = Hash, so Hash and, through its default template argument, Key are
// template parameters of f':
//
//   template <class Key, class Hash = std::hash<Key>, class It>
//   MySet(It, It, Hash) -> unordered_set<iter_value_t<It>, Hash>;
//
// and `MySet(first, last)` fails, as Key cannot be deduced. However, Key
// corresponds to `iter_value_t<It>` in the return type of the guide: deducing
// the template arguments of A from the return type of f gives
// Key = iter_value_t<It>, which we use as the default template argument of Key
// in f' (see orderFPrimeTemplateParameters for the resulting order):
//
//   template <class It, class Key = iter_value_t<It>,
//             class Hash = std::hash<Key>>
//   MySet(It, It, Hash) -> unordered_set<iter_value_t<It>, Hash>;
//
// The result has an entry for each template parameter of A, which is null for
// those that don't get a synthesized default template argument. The synthesized
// default template arguments refer to the (non-deduced) template parameters of
// f.
static SmallVector<TemplateArgument> synthesizeDefaultArgumentsForFPrime(
    Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate,
    FunctionTemplateDecl *F, ArrayRef<TemplateArgument> AliasRhsTemplateArgs,
    ArrayRef<TemplateArgument> FReturnTemplateArgs,
    ArrayRef<DeducedTemplateArgument> DeduceResults,
    ArrayRef<unsigned> AliasParamsInFPrime, SourceLocation Loc) {
  TemplateParameterList *AliasParams = AliasTemplate->getTemplateParameters();
  TemplateParameterList *FParams = F->getTemplateParameters();
  SmallVector<TemplateArgument> Result(AliasParams->size());

  auto NeedsDefaultArgument = [&](unsigned Index) {
    NamedDecl *Param = AliasParams->getParam(Index);
    return !Param->isTemplateParameterPack() && !getDefaultArgument(Param);
  };

  if (llvm::none_of(AliasParamsInFPrime, NeedsDefaultArgument))
    return Result;

  // Deduce the template arguments of A from the return type of f, the reverse
  // of the deduction of the template arguments of f from the defining-type-id
  // of A.
  sema::TemplateDeductionInfo Info(Loc, AliasParams->getDepth());
  SmallVector<DeducedTemplateArgument> Deduced(AliasParams->size());
  SemaRef.DeduceTemplateArguments(AliasParams, AliasRhsTemplateArgs,
                                  FReturnTemplateArgs, Info, Deduced,
                                  /*NumberOfArgumentsMustMatch=*/false);

  for (unsigned Index : AliasParamsInFPrime) {
    if (!NeedsDefaultArgument(Index))
      continue;
    const TemplateArgument &D = Deduced[Index];

    if (D.isNull() || D.isPackExpansion())
      continue;

    NamedDecl *Param = AliasParams->getParam(Index);

    bool KindMatches = [&] {
      switch (D.getKind()) {
      case TemplateArgument::Type:
        return isa<TemplateTypeParmDecl>(Param);
      case TemplateArgument::Template:
        return isa<TemplateTemplateParmDecl>(Param);
      case TemplateArgument::Expression:
        return isa<NonTypeTemplateParmDecl>(Param);
      default:
        return false;
      }
    }();

    if (!KindMatches)
      continue;

    // The deduced argument may only refer to the non-deduced template
    // parameters of f. The deduced ones are replaced in f' by the template
    // parameters of A they were deduced to, whose default template arguments
    // may in turn refer to this template parameter.
    llvm::SmallBitVector UsedFParams(FParams->size());
    SemaRef.MarkUsedTemplateParameters(D, /*OnlyDeduced=*/false,
                                       FParams->getDepth(), UsedFParams);
    if (llvm::any_of(UsedFParams.set_bits(), [&](unsigned FIndex) {
          return !IsNonDeducedArgument(DeduceResults[FIndex]);
        }))
      continue;
    Result[Index] = D;
  }
  return Result;
}

// Mark the template parameters at the given depth that the given template
// parameter refers to: through its default template argument, its type (for a
// non-type template parameter), its type-constraint (for a type template
// parameter) or its template parameter list (for a template template
// parameter).
static void markReferencedTemplateParams(Sema &SemaRef, const NamedDecl *Param,
                                         unsigned Depth,
                                         llvm::SmallBitVector &Used) {
  if (const TemplateArgumentLoc *Default = getDefaultArgument(Param))
    SemaRef.MarkUsedTemplateParameters(Default->getArgument(),
                                       /*OnlyDeduced=*/false, Depth, Used);
  if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
    if (const TypeConstraint *TC = TTP->getTypeConstraint())
      if (const Expr *E = TC->getImmediatelyDeclaredConstraint())
        SemaRef.MarkUsedTemplateParameters(E, /*OnlyDeduced=*/false, Depth,
                                           Used);
  } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
    SemaRef.MarkUsedTemplateParameters(TemplateArgument(NTTP->getType()),
                                       /*OnlyDeduced=*/false, Depth, Used);
  } else {
    TemplateParameterList *TPL =
        cast<TemplateTemplateParmDecl>(Param)->getTemplateParameters();
    for (const NamedDecl *P : *TPL)
      markReferencedTemplateParams(SemaRef, P, Depth, Used);
    if (const Expr *RC = TPL->getRequiresClause())
      SemaRef.MarkUsedTemplateParameters(RC, /*OnlyDeduced=*/false, Depth,
                                         Used);
  }
}

// Reorder the template parameters of the deduction guide f' of the alias
// template A, given in the standard's order (the template parameters of A,
// followed by the non-deduced template parameters of f), when some of them got
// a synthesized default template argument (see
// synthesizeDefaultArgumentsForFPrime).
//
// A default template argument can only refer to preceding template parameters.
// The standard's order doesn't satisfy that for the synthesized default
// template arguments, which refer to non-deduced template parameters of f.
// Instead, order the template parameters of f' such that each of them follows
// the ones its default template argument, its type (for a non-type template
// parameter) and its type-constraint refer to, staying as close to the
// standard's order as possible.
//
// AliasParamsUsedByDeducedArg[I] holds the template parameters of A that the
// deduced template argument for the I-th template parameter of f refers to.
//
// Returns false, leaving Params unchanged, if there is no such order.
static bool reorderFPrimeTemplateParameters(
    Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate,
    FunctionTemplateDecl *F,
    ArrayRef<llvm::SmallBitVector> AliasParamsUsedByDeducedArg,
    ArrayRef<TemplateArgument> SynthesizedDefaultArgs,
    SmallVectorImpl<FPrimeTemplateParamRef> &Params) {
  TemplateParameterList *AliasParams = AliasTemplate->getTemplateParameters();
  TemplateParameterList *FParams = F->getTemplateParameters();
  unsigned NumParams = Params.size();

  // The position in Params of each template parameter of A / non-deduced
  // template parameter of f.
  SmallVector<unsigned> AliasParamPos(AliasParams->size(), InvalidFPrimeIndex);
  SmallVector<unsigned> FParamPos(FParams->size(), InvalidFPrimeIndex);
  for (auto [Pos, P] : llvm::enumerate(Params)) {
    if (P.IsAliasParam)
      AliasParamPos[P.Index] = Pos;
    else
      FParamPos[P.Index] = Pos;
  }

  // Deps[I] holds the positions of the template parameters that must precede
  // the I-th template parameter.
  SmallVector<llvm::SmallBitVector> Deps(NumParams,
                                         llvm::SmallBitVector(NumParams));
  for (auto [Pos, P] : llvm::enumerate(Params)) {
    if (P.IsAliasParam) {
      llvm::SmallBitVector UsedAliasParams(AliasParams->size());
      markReferencedTemplateParams(SemaRef, AliasParams->getParam(P.Index),
                                   AliasParams->getDepth(), UsedAliasParams);
      for (unsigned Index : UsedAliasParams.set_bits())
        if (Index != P.Index && AliasParamPos[Index] != InvalidFPrimeIndex)
          Deps[Pos].set(AliasParamPos[Index]);
      // The synthesized default template argument refers to non-deduced
      // template parameters of f.
      if (!SynthesizedDefaultArgs[P.Index].isNull()) {
        llvm::SmallBitVector UsedFParams(FParams->size());
        SemaRef.MarkUsedTemplateParameters(SynthesizedDefaultArgs[P.Index],
                                           /*OnlyDeduced=*/false,
                                           FParams->getDepth(), UsedFParams);
        for (unsigned Index : UsedFParams.set_bits()) {
          assert(FParamPos[Index] != InvalidFPrimeIndex &&
                 "synthesized default argument refers to a deduced parameter");
          Deps[Pos].set(FParamPos[Index]);
        }
      }
      continue;
    }

    llvm::SmallBitVector UsedFParams(FParams->size());
    markReferencedTemplateParams(SemaRef, FParams->getParam(P.Index),
                                 FParams->getDepth(), UsedFParams);
    for (unsigned Index : UsedFParams.set_bits()) {
      if (Index == P.Index)
        continue;
      if (FParamPos[Index] != InvalidFPrimeIndex) {
        Deps[Pos].set(FParamPos[Index]);
        continue;
      }
      // A deduced template parameter of f, which is replaced in f' by the
      // template parameters of A that its deduced argument refers to.
      for (unsigned AliasIndex : AliasParamsUsedByDeducedArg[Index].set_bits())
        if (AliasParamPos[AliasIndex] != InvalidFPrimeIndex)
          Deps[Pos].set(AliasParamPos[AliasIndex]);
    }
  }

  // Repeatedly pick the first template parameter all of whose dependencies
  // have been placed.
  SmallVector<FPrimeTemplateParamRef> Order;
  llvm::SmallBitVector Placed(NumParams);
  while (Order.size() < NumParams) {
    unsigned Next = NumParams;
    for (unsigned Pos = 0; Pos != NumParams && Next == NumParams; ++Pos)
      if (!Placed[Pos] && !Deps[Pos].test(Placed))
        Next = Pos;
    if (Next == NumParams) // The dependencies are circular.
      return false;
    Placed.set(Next);
    Order.push_back(Params[Next]);
  }
  Params = std::move(Order);
  return true;
}

// Build deduction guides for a type alias template from the given underlying
// source deduction guide.
CXXDeductionGuideDecl *BuildDeductionGuideForTypeAlias(
    Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate,
    CXXDeductionGuideDecl *SourceDeductionGuide, SourceLocation Loc) {
  FunctionTemplateDecl *F =
      SourceDeductionGuide->getDescribedFunctionTemplate();
  assert(F && "deduction guide for alias template must be a function template");
  TemplateParameterList *AliasParams = AliasTemplate->getTemplateParameters();
  TemplateParameterList *FParams = F->getTemplateParameters();

  LocalInstantiationScope Scope(SemaRef);
  Sema::NonSFINAEContext _1(SemaRef);
  Sema::InstantiatingTemplate BuildingDeductionGuides(
      SemaRef, AliasTemplate->getLocation(), F,
      Sema::InstantiatingTemplate::BuildingDeductionGuidesTag{});
  if (BuildingDeductionGuides.isInvalid())
    return nullptr;

  auto &Context = SemaRef.Context;
  auto [Template, AliasRhsTemplateArgs] =
      getRHSTemplateDeclAndArgs(SemaRef, AliasTemplate);

  // We need both types desugared, before we continue to perform type deduction.
  // The intent is to get the template argument list 'matched', e.g. in the
  // following case:
  //
  //
  //  template <class T>
  //  struct A {};
  //  template <class T>
  //  using Foo = A<A<T>>;
  //  template <class U = int>
  //  using Bar = Foo<U>;
  //
  // In terms of Bar, we want U (which has the default argument) to appear in
  // the synthesized deduction guide, but U would remain undeduced if we deduced
  // A<A<T>> using Foo<U> directly.
  //
  // Instead, we need to canonicalize both against A, i.e. A<A<T>> and A<A<U>>,
  // such that T can be deduced as U.
  auto RType = SourceDeductionGuide->getReturnType();
  // The (trailing) return type of the deduction guide.
  const auto *FReturnType = RType->getAs<TemplateSpecializationType>();
  if (const auto *ICNT = RType->getAsCanonical<InjectedClassNameType>())
    // implicitly-generated deduction guide.
    FReturnType = cast<TemplateSpecializationType>(
        ICNT->getDecl()->getCanonicalTemplateSpecializationType(
            SemaRef.Context));

  ArrayRef<TemplateArgument> FReturnTemplateArgs;
  if (FReturnType) {
    FReturnTemplateArgs = FReturnType->template_arguments();
  } else if (const auto *RT = RType->getAs<RecordType>()) {
    // If the return type is a non-dependent class template specialization,
    // it might be resolved to a RecordType.
    if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl()))
      FReturnTemplateArgs = CTSD->getTemplateArgs().asArray();
  }
  assert(!FReturnTemplateArgs.empty() && "expected to see template arguments");

  // Deduce template arguments of the deduction guide f from the RHS of
  // the alias.
  //
  // C++ [over.match.class.deduct]p3: ...For each function or function
  // template f in the guides of the template named by the
  // simple-template-id of the defining-type-id, the template arguments
  // of the return type of f are deduced from the defining-type-id of A
  // according to the process in [temp.deduct.type] with the exception
  // that deduction does not fail if not all template arguments are
  // deduced.
  //
  //
  //  template<typename X, typename Y>
  //  f(X, Y) -> f<Y, X>;
  //
  //  template<typename U>
  //  using alias = f<int, U>;
  //
  // The RHS of alias is f<int, U>, we deduced the template arguments of
  // the return type of the deduction guide from it: Y->int, X->U
  sema::TemplateDeductionInfo TDeduceInfo(Loc);
  // Must initialize n elements, this is required by DeduceTemplateArguments.
  SmallVector<DeducedTemplateArgument> DeduceResults(FParams->size());

  // FIXME: DeduceTemplateArguments stops immediately at the first
  // non-deducible template argument. However, this doesn't seem to cause
  // issues for practice cases, we probably need to extend it to continue
  // performing deduction for rest of arguments to align with the C++
  // standard.
  SemaRef.DeduceTemplateArguments(FParams, FReturnTemplateArgs,
                                  AliasRhsTemplateArgs, TDeduceInfo,
                                  DeduceResults,
                                  /*NumberOfArgumentsMustMatch=*/false);

  SmallVector<TemplateArgument> DeducedArgs;
  SmallVector<unsigned> NonDeducedTemplateParamsInFIndex;
  // The template parameters of A that the deduced template argument for each
  // template parameter of f refers to (none for the non-deduced ones).
  SmallVector<llvm::SmallBitVector> AliasParamsUsedByDeducedArg(
      FParams->size(), llvm::SmallBitVector(AliasParams->size()));
  // !!NOTE: DeduceResults respects the sequence of template parameters of
  // the deduction guide f.
  for (unsigned Index = 0; Index < DeduceResults.size(); ++Index) {
    const TemplateArgument &D = DeduceResults[Index];
    if (IsNonDeducedArgument(D)) {
      NonDeducedTemplateParamsInFIndex.push_back(Index);
      continue;
    }
    DeducedArgs.push_back(D);
    SemaRef.MarkUsedTemplateParameters(D, /*OnlyDeduced=*/false,
                                       AliasParams->getDepth(),
                                       AliasParamsUsedByDeducedArg[Index]);
  }
  auto DeducedAliasTemplateParams =
      TemplateParamsReferencedInTemplateArgumentList(SemaRef, AliasParams,
                                                     DeducedArgs);
  // All template arguments null by default.
  SmallVector<TemplateArgument> TemplateArgsForBuildingFPrime(FParams->size());
  // The same template arguments, but with the deduced non-type template
  // arguments as (not yet converted) expressions, for rewriting the template
  // parameters of f in terms of those of f'. The rewrite requires expressions,
  // while instantiating f requires converted template arguments, e.g. a
  // template parameter `bool B` of f deduced as `false` from the alias must be
  // rewritten as the expression `false` in the type of a template parameter
  // `std::enable_if_t<!B, int> = 0` of f, but instantiated as the integral
  // value.
  SmallVector<TemplateArgument> TemplateArgsForRewritingFPrime(FParams->size());

  // Create a template parameter list for the synthesized deduction guide f'.
  //
  // C++ [over.match.class.deduct]p3.2:
  //   If f is a function template, f' is a function template whose template
  //   parameter list consists of all the template parameters of A
  //   (including their default template arguments) that appear in the above
  //   deductions or (recursively) in their default template arguments,
  //   followed by the template parameters of f that were not deduced
  //   (including their default template arguments)
  SmallVector<NamedDecl *> FPrimeTemplateParams;
  // Store template arguments that refer to the newly-created template
  // parameters, used for building `TemplateArgsForBuildingFPrime`.
  SmallVector<TemplateArgument, 16> TransformedDeducedAliasArgs(
      AliasParams->size());
  // The index in f' of the template parameters of A, and of the non-deduced
  // template parameters of f, that appear in f'.
  SmallVector<unsigned> AliasParamFPrimeIndex(AliasParams->size(),
                                              InvalidFPrimeIndex);
  SmallVector<unsigned> FParamFPrimeIndex(FParams->size(), InvalidFPrimeIndex);

  // The template parameters of f', in the standard's order: the template
  // parameters of A that appear in the deductions, followed by the non-deduced
  // template parameters of f.
  SmallVector<FPrimeTemplateParamRef> FPrimeParamOrder;
  for (unsigned Index : DeducedAliasTemplateParams)
    FPrimeParamOrder.push_back({/*IsAliasParam=*/true, Index});
  for (unsigned Index : NonDeducedTemplateParamsInFIndex)
    FPrimeParamOrder.push_back({/*IsAliasParam=*/false, Index});

  // Template parameters of A that appear in f' without a default template
  // argument, and that cannot be deduced from the function parameters of f',
  // get a default template argument synthesized from the return type of f.
  SmallVector<TemplateArgument> SynthesizedDefaultArgs =
      synthesizeDefaultArgumentsForFPrime(
          SemaRef, AliasTemplate, F, AliasRhsTemplateArgs, FReturnTemplateArgs,
          DeduceResults, DeducedAliasTemplateParams, Loc);
  // Those refer to template parameters of f, which the standard's order places
  // after the template parameters of A; reorder the template parameters of f'
  // so that default template arguments only refer to preceding template
  // parameters. If that is not possible, don't synthesize any.
  if (llvm::any_of(SynthesizedDefaultArgs,
                   [](const TemplateArgument &TA) { return !TA.isNull(); }) &&
      !reorderFPrimeTemplateParameters(
          SemaRef, AliasTemplate, F, AliasParamsUsedByDeducedArg,
          SynthesizedDefaultArgs, FPrimeParamOrder))
    llvm::fill(SynthesizedDefaultArgs, TemplateArgument());

  // We might be already within a pack expansion, but rewriting template
  // parameters is independent of that. (We may or may not expand new packs
  // when rewriting. So clear the state)
  Sema::ArgPackSubstIndexRAII PackSubstReset(SemaRef, std::nullopt);

  // Add the template parameter of A at the given index to f'.
  auto AddAliasTemplateParam = [&](unsigned AliasTemplateParamIdx) -> bool {
    auto *TP = AliasParams->getParam(AliasTemplateParamIdx);
    // Rebuild any internal references to earlier parameters and reindex as
    // we go.
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    Args.addOuterTemplateArguments(TransformedDeducedAliasArgs);
    NamedDecl *NewParam = transformTemplateParameter(
        SemaRef, AliasTemplate->getDeclContext(), TP, Args,
        /*NewIndex=*/FPrimeTemplateParams.size(), getDepthAndIndex(TP).first);
    if (!NewParam)
      return false;
    if (const TemplateArgument &Default =
            SynthesizedDefaultArgs[AliasTemplateParamIdx];
        !Default.isNull()) {
      // The synthesized default template argument refers to template
      // parameters of f; rewrite it in terms of the corresponding
      // (already created) template parameters of f'.
      QualType NTTPType;
      if (auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(NewParam))
        NTTPType = NTTP->getType();
      MultiLevelTemplateArgumentList FArgs;
      FArgs.setKind(TemplateSubstitutionKind::Rewrite);
      FArgs.addOuterTemplateArguments(TemplateArgsForRewritingFPrime);
      TemplateArgumentLoc Output;
      if (SemaRef.SubstTemplateArgument(
              SemaRef.getTrivialTemplateArgumentLoc(Default, NTTPType, Loc),
              FArgs, Output, Loc, NewParam->getDeclName()))
        return false;
      setDefaultArgument(Context, NewParam, Output);
    }
    AliasParamFPrimeIndex[AliasTemplateParamIdx] = FPrimeTemplateParams.size();
    FPrimeTemplateParams.push_back(NewParam);
    TransformedDeducedAliasArgs[AliasTemplateParamIdx] =
        Context.getInjectedTemplateArg(NewParam);
    return true;
  };

  // To form a deduction guide f' from f, we leverage clang's instantiation
  // mechanism, we construct a template argument list where the template
  // arguments refer to the newly-created template parameters of f', and
  // then apply instantiation on this template argument list to instantiate
  // f, this ensures all template parameter occurrences are updated
  // correctly.
  //
  // The template argument list is formed, in order, from
  //   1) For the template parameters of the alias, the corresponding deduced
  //      template arguments
  //   2) For the non-deduced template parameters of f. the
  //      (rebuilt) template arguments corresponding.
  //
  // Note: the non-deduced template arguments of `f` might refer to arguments
  // deduced in 1), as in a type constraint.
  //
  // Substitute the template parameters of f' into the deduced template
  // argument for the template parameter of f at the given index (1). The
  // template parameters of A it refers to must have been added to f' already.
  auto SubstDeducedTemplateArg = [&](unsigned Index) -> bool {
    const auto &D = DeduceResults[Index];
    auto *TP = FParams->getParam(Index);
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    Args.addOuterTemplateArguments(TransformedDeducedAliasArgs);
    TemplateArgumentLoc Input =
        SemaRef.getTrivialTemplateArgumentLoc(D, QualType(), SourceLocation{});
    TemplateArgumentListInfo Output;
    if (SemaRef.SubstTemplateArguments(Input, Args, Output))
      return false;
    assert(TemplateArgsForBuildingFPrime[Index].isNull() &&
           "InstantiatedArgs must be null before setting");
    // CheckTemplateArgument is necessary for NTTP initializations.
    // FIXME: We may want to call CheckTemplateArguments instead, but we cannot
    // match packs as usual, since packs can appear in the middle of the
    // parameter list of a synthesized CTAD guide. See also the FIXME in
    // test/SemaCXX/cxx20-ctad-type-alias.cpp:test25.
    Sema::CheckTemplateArgumentInfo CTAI;
    for (auto TA : Output.arguments())
      if (SemaRef.CheckTemplateArgument(
              TP, TA, F, F->getLocation(), F->getLocation(),
              /*ArgumentPackIndex=*/-1, CTAI,
              Sema::CheckTemplateArgumentKind::CTAK_Specified))
        return false;
    SmallVector<TemplateArgument> OutputArgs;
    for (const TemplateArgumentLoc &TA : Output.arguments())
      OutputArgs.push_back(TA.getArgument());
    if (Input.getArgument().getKind() == TemplateArgument::Pack) {
      // We will substitute the non-deduced template arguments with these
      // transformed (unpacked at this point) arguments, where that substitution
      // requires a pack for the corresponding parameter packs.
      TemplateArgsForBuildingFPrime[Index] =
          TemplateArgument::CreatePackCopy(Context, CTAI.SugaredConverted);
      TemplateArgsForRewritingFPrime[Index] =
          TemplateArgument::CreatePackCopy(Context, OutputArgs);
    } else {
      assert(Output.arguments().size() == 1);
      TemplateArgsForBuildingFPrime[Index] = CTAI.SugaredConverted[0];
      TemplateArgsForRewritingFPrime[Index] = OutputArgs[0];
    }
    return true;
  };

  // Add the non-deduced template parameter of f at the given index to f' (2).
  auto AddFTemplateParam = [&](unsigned FTemplateParamIdx) -> bool {
    auto *TP = FParams->getParam(FTemplateParamIdx);
    MultiLevelTemplateArgumentList Args;
    Args.setKind(TemplateSubstitutionKind::Rewrite);
    Args.addOuterTemplateArguments(TemplateArgsForRewritingFPrime);
    // Substituting the deduced template arguments into the template parameter
    // may fail, e.g. for a template parameter `std::enable_if_t<!B, int> = 0`
    // of f whose template parameter B was deduced as `true` from the alias.
    // Then f' can't be formed; it is not viable, but forming it is not an
    // error either. Don't diagnose the failure and don't form f'.
    Sema::SFINAETrap Trap(SemaRef);
    NamedDecl *NewParam = transformTemplateParameter(
        SemaRef, F->getDeclContext(), TP, Args, FPrimeTemplateParams.size(),
        getDepthAndIndex(TP).first);
    if (!NewParam || Trap.hasErrorOccurred())
      return false;
    FParamFPrimeIndex[FTemplateParamIdx] = FPrimeTemplateParams.size();
    FPrimeTemplateParams.push_back(NewParam);

    assert(TemplateArgsForBuildingFPrime[FTemplateParamIdx].isNull() &&
           "The argument must be null before setting");
    TemplateArgsForBuildingFPrime[FTemplateParamIdx] =
        Context.getInjectedTemplateArg(NewParam);
    TemplateArgsForRewritingFPrime[FTemplateParamIdx] =
        TemplateArgsForBuildingFPrime[FTemplateParamIdx];
    return true;
  };

  // Substitute the deduced template arguments (1) that haven't been substituted
  // yet and whose template parameters of A have all been added to f' already.
  auto SubstDeducedTemplateArgs = [&]() -> bool {
    for (unsigned Index = 0; Index < DeduceResults.size(); ++Index) {
      if (IsNonDeducedArgument(DeduceResults[Index]) ||
          !TemplateArgsForBuildingFPrime[Index].isNull())
        continue;
      if (llvm::any_of(AliasParamsUsedByDeducedArg[Index].set_bits(),
                       [&](unsigned AliasIndex) {
                         return AliasParamFPrimeIndex[AliasIndex] ==
                                InvalidFPrimeIndex;
                       }))
        continue;
      if (!SubstDeducedTemplateArg(Index))
        return false;
    }
    return true;
  };

  for (const FPrimeTemplateParamRef &P : FPrimeParamOrder) {
    if (P.IsAliasParam) {
      if (!AddAliasTemplateParam(P.Index))
        return nullptr;
      continue;
    }
    // A non-deduced template parameter of f may refer to deduced template
    // parameters of f, whose deduced template arguments must have been
    // substituted by then. In the standard's order, all of them can be, as all
    // the template parameters of A precede the template parameters of f.
    if (!SubstDeducedTemplateArgs() || !AddFTemplateParam(P.Index))
      return nullptr;
  }
  // Substitute the deduced template arguments that no template parameter of f'
  // needed.
  if (!SubstDeducedTemplateArgs())
    return nullptr;

  auto *TemplateArgListForBuildingFPrime =
      TemplateArgumentList::CreateCopy(Context, TemplateArgsForBuildingFPrime);
  // Form the f' by substituting the template arguments into f.
  if (auto *FPrime = SemaRef.InstantiateFunctionDeclaration(
          F, TemplateArgListForBuildingFPrime, AliasTemplate->getLocation(),
          Sema::CodeSynthesisContext::BuildingDeductionGuides)) {
    auto *GG = cast<CXXDeductionGuideDecl>(FPrime);

    Expr *IsDeducible = buildIsDeducibleConstraint(
        SemaRef, AliasTemplate, FPrime->getReturnType(), FPrimeTemplateParams);
    Expr *RequiresClause = buildAssociatedConstraints(
        SemaRef, F, AliasTemplate, DeduceResults, AliasParamFPrimeIndex,
        FParamFPrimeIndex, IsDeducible);

    TemplateParameterList *FPrimeTemplateParamList = nullptr;
    if (!FPrimeTemplateParams.empty())
      FPrimeTemplateParamList = TemplateParameterList::Create(
          Context, AliasParams->getTemplateLoc(), AliasParams->getLAngleLoc(),
          FPrimeTemplateParams, AliasParams->getRAngleLoc(),
          /*RequiresClause=*/RequiresClause);

    auto *DGuide = buildDeductionGuide(
        SemaRef, AliasTemplate, FPrimeTemplateParamList,
        GG->getCorrespondingConstructor(), GG->getExplicitSpecifier(),
        GG->getTypeSourceInfo(), AliasTemplate->getBeginLoc(),
        AliasTemplate->getLocation(), AliasTemplate->getEndLoc(),
        F->isImplicit());
    DGuide->setDeductionCandidateKind(GG->getDeductionCandidateKind());
    DGuide->setSourceDeductionGuide(SourceDeductionGuide);
    DGuide->setSourceDeductionGuideKind(
        CXXDeductionGuideDecl::SourceDeductionGuideKind::Alias);
    return DGuide;
  }
  return nullptr;
}

void DeclareImplicitDeductionGuidesForTypeAlias(
    Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate, SourceLocation Loc) {
  if (AliasTemplate->isInvalidDecl())
    return;
  auto &Context = SemaRef.Context;
  auto [Template, AliasRhsTemplateArgs] =
      getRHSTemplateDeclAndArgs(SemaRef, AliasTemplate);
  if (!Template)
    return;
  auto SourceDeductionGuides = getSourceDeductionGuides(
      Context.DeclarationNames.getCXXDeductionGuideName(AliasTemplate),
      AliasTemplate->getDeclContext());

  DeclarationNameInfo NameInfo(
      Context.DeclarationNames.getCXXDeductionGuideName(Template), Loc);
  LookupResult Guides(SemaRef, NameInfo, clang::Sema::LookupOrdinaryName);
  SemaRef.LookupQualifiedName(Guides, Template->getDeclContext());
  Guides.suppressDiagnostics();

  for (auto *G : Guides) {
    if (auto *DG = dyn_cast<CXXDeductionGuideDecl>(G)) {
      if (SourceDeductionGuides.contains(DG))
        continue;
      // The deduction guide is a non-template function decl, we just clone it.
      auto *FunctionType =
          SemaRef.Context.getTrivialTypeSourceInfo(DG->getType());
      FunctionProtoTypeLoc FPTL =
          FunctionType->getTypeLoc().castAs<FunctionProtoTypeLoc>();

      // Clone the parameters.
      for (unsigned I = 0, N = DG->getNumParams(); I != N; ++I) {
        const auto *P = DG->getParamDecl(I);
        auto *TSI = SemaRef.Context.getTrivialTypeSourceInfo(P->getType());
        ParmVarDecl *NewParam = ParmVarDecl::Create(
            SemaRef.Context, G->getDeclContext(),
            DG->getParamDecl(I)->getBeginLoc(), P->getLocation(), nullptr,
            TSI->getType(), TSI, SC_None, nullptr);
        NewParam->setScopeInfo(0, I);
        FPTL.setParam(I, NewParam);
      }
      auto *Transformed = cast<CXXDeductionGuideDecl>(buildDeductionGuide(
          SemaRef, AliasTemplate, /*TemplateParams=*/nullptr,
          /*Constructor=*/nullptr, DG->getExplicitSpecifier(), FunctionType,
          AliasTemplate->getBeginLoc(), AliasTemplate->getLocation(),
          AliasTemplate->getEndLoc(), DG->isImplicit()));
      Transformed->setSourceDeductionGuide(DG);
      Transformed->setSourceDeductionGuideKind(
          CXXDeductionGuideDecl::SourceDeductionGuideKind::Alias);

      // FIXME: Here the synthesized deduction guide is not a templated
      // function. Per [dcl.decl]p4, the requires-clause shall be present only
      // if the declarator declares a templated function, a bug in standard?
      AssociatedConstraint Constraint(buildIsDeducibleConstraint(
          SemaRef, AliasTemplate, Transformed->getReturnType(), {}));
      if (const AssociatedConstraint &RC = DG->getTrailingRequiresClause()) {
        auto Conjunction = SemaRef.BuildBinOp(
            SemaRef.getCurScope(), SourceLocation{},
            BinaryOperatorKind::BO_LAnd, const_cast<Expr *>(RC.ConstraintExpr),
            const_cast<Expr *>(Constraint.ConstraintExpr));
        if (!Conjunction.isInvalid()) {
          Constraint.ConstraintExpr = Conjunction.getAs<Expr>();
          Constraint.ArgPackSubstIndex = RC.ArgPackSubstIndex;
        }
      }
      Transformed->setTrailingRequiresClause(Constraint);
      continue;
    }
    FunctionTemplateDecl *F = dyn_cast<FunctionTemplateDecl>(G);
    if (!F || SourceDeductionGuides.contains(F->getTemplatedDecl()))
      continue;
    // The **aggregate** deduction guides are handled in a different code path
    // (DeclareAggregateDeductionGuideFromInitList), which involves the tricky
    // cache.
    auto *DGuide = cast<CXXDeductionGuideDecl>(F->getTemplatedDecl());
    if (DGuide->getDeductionCandidateKind() == DeductionCandidate::Aggregate)
      continue;

    BuildDeductionGuideForTypeAlias(SemaRef, AliasTemplate, DGuide, Loc);
  }
}

// Build an aggregate deduction guide for a type alias template.
CXXDeductionGuideDecl *DeclareAggregateDeductionGuideForTypeAlias(
    Sema &SemaRef, TypeAliasTemplateDecl *AliasTemplate,
    MutableArrayRef<QualType> ParamTypes, SourceLocation Loc) {
  TemplateDecl *RHSTemplate =
      getRHSTemplateDeclAndArgs(SemaRef, AliasTemplate).first;
  if (!RHSTemplate)
    return nullptr;

  llvm::SmallVector<TypedefNameDecl *> TypedefDecls;
  llvm::SmallVector<QualType> NewParamTypes;
  ExtractTypeForDeductionGuide TypeAliasTransformer(SemaRef, TypedefDecls);
  for (QualType P : ParamTypes) {
    QualType Type = TypeAliasTransformer.TransformType(P);
    if (Type.isNull())
      return nullptr;
    NewParamTypes.push_back(Type);
  }

  auto *RHSDeductionGuide = SemaRef.DeclareAggregateDeductionGuideFromInitList(
      RHSTemplate, NewParamTypes, Loc);
  if (!RHSDeductionGuide)
    return nullptr;

  for (TypedefNameDecl *TD : TypedefDecls)
    TD->setDeclContext(RHSDeductionGuide);

  return BuildDeductionGuideForTypeAlias(SemaRef, AliasTemplate,
                                         RHSDeductionGuide, Loc);
}

} // namespace

CXXDeductionGuideDecl *Sema::DeclareAggregateDeductionGuideFromInitList(
    TemplateDecl *Template, MutableArrayRef<QualType> ParamTypes,
    SourceLocation Loc) {
  llvm::FoldingSetNodeID ID;
  ID.AddPointer(Template);
  for (auto &T : ParamTypes)
    T.getCanonicalType().Profile(ID);
  unsigned Hash = ID.computeHash();

  auto Found = AggregateDeductionCandidates.find(Hash);
  if (Found != AggregateDeductionCandidates.end())
    return Found->getSecond();

  if (auto *AliasTemplate = llvm::dyn_cast<TypeAliasTemplateDecl>(Template)) {
    if (auto *GD = DeclareAggregateDeductionGuideForTypeAlias(
            *this, AliasTemplate, ParamTypes, Loc)) {
      GD->setDeductionCandidateKind(DeductionCandidate::Aggregate);
      AggregateDeductionCandidates[Hash] = GD;
      return GD;
    }
    return nullptr;
  }

  if (CXXRecordDecl *DefRecord =
          cast<CXXRecordDecl>(Template->getTemplatedDecl())->getDefinition()) {
    if (TemplateDecl *DescribedTemplate =
            DefRecord->getDescribedClassTemplate())
      Template = DescribedTemplate;
  }

  DeclContext *DC = Template->getDeclContext();
  if (DC->isDependentContext())
    return nullptr;

  ConvertConstructorToDeductionGuideTransform Transform(
      *this, cast<ClassTemplateDecl>(Template));
  if (!isCompleteType(Loc, Transform.DeducedType))
    return nullptr;

  // In case we were expanding a pack when we attempted to declare deduction
  // guides, turn off pack expansion for everything we're about to do.
  ArgPackSubstIndexRAII SubstIndex(*this, std::nullopt);
  // Create a template instantiation record to track the "instantiation" of
  // constructors into deduction guides.
  InstantiatingTemplate BuildingDeductionGuides(
      *this, Loc, Template,
      Sema::InstantiatingTemplate::BuildingDeductionGuidesTag{});
  if (BuildingDeductionGuides.isInvalid())
    return nullptr;

  ClassTemplateDecl *Pattern =
      Transform.NestedPattern ? Transform.NestedPattern : Transform.Template;
  ContextRAII SavedContext(*this, Pattern->getTemplatedDecl());

  CXXDeductionGuideDecl *GD = Transform.buildSimpleDeductionGuide(ParamTypes);
  SavedContext.pop();
  GD->setDeductionCandidateKind(DeductionCandidate::Aggregate);
  AggregateDeductionCandidates[Hash] = GD;
  return GD;
}

void Sema::DeclareImplicitDeductionGuides(TemplateDecl *Template,
                                          SourceLocation Loc) {
  if (auto *AliasTemplate = llvm::dyn_cast<TypeAliasTemplateDecl>(Template)) {
    DeclareImplicitDeductionGuidesForTypeAlias(*this, AliasTemplate, Loc);
    return;
  }
  CXXRecordDecl *DefRecord =
      dyn_cast_or_null<CXXRecordDecl>(Template->getTemplatedDecl());
  if (!DefRecord)
    return;
  if (const CXXRecordDecl *Definition = DefRecord->getDefinition()) {
    if (TemplateDecl *DescribedTemplate =
            Definition->getDescribedClassTemplate())
      Template = DescribedTemplate;
  }

  DeclContext *DC = Template->getDeclContext();
  if (DC->isDependentContext())
    return;

  ConvertConstructorToDeductionGuideTransform Transform(
      *this, cast<ClassTemplateDecl>(Template));
  if (!isCompleteType(Loc, Transform.DeducedType))
    return;

  if (hasDeclaredDeductionGuides(Transform.DeductionGuideName, DC))
    return;

  // In case we were expanding a pack when we attempted to declare deduction
  // guides, turn off pack expansion for everything we're about to do.
  ArgPackSubstIndexRAII SubstIndex(*this, std::nullopt);
  // Create a template instantiation record to track the "instantiation" of
  // constructors into deduction guides.
  InstantiatingTemplate BuildingDeductionGuides(
      *this, Loc, Template,
      Sema::InstantiatingTemplate::BuildingDeductionGuidesTag{});
  if (BuildingDeductionGuides.isInvalid())
    return;

  // Convert declared constructors into deduction guide templates.
  // FIXME: Skip constructors for which deduction must necessarily fail (those
  // for which some class template parameter without a default argument never
  // appears in a deduced context).
  ClassTemplateDecl *Pattern =
      Transform.NestedPattern ? Transform.NestedPattern : Transform.Template;
  ContextRAII SavedContext(*this, Pattern->getTemplatedDecl());
  llvm::SmallPtrSet<NamedDecl *, 8> ProcessedCtors;
  bool AddedAny = false;
  for (NamedDecl *D : LookupConstructors(Pattern->getTemplatedDecl())) {
    D = D->getUnderlyingDecl();
    if (D->isInvalidDecl() || D->isImplicit())
      continue;

    D = cast<NamedDecl>(D->getCanonicalDecl());

    // Within C++20 modules, we may have multiple same constructors in
    // multiple same RecordDecls. And it doesn't make sense to create
    // duplicated deduction guides for the duplicated constructors.
    if (ProcessedCtors.count(D))
      continue;

    auto *FTD = dyn_cast<FunctionTemplateDecl>(D);
    auto *CD =
        dyn_cast_or_null<CXXConstructorDecl>(FTD ? FTD->getTemplatedDecl() : D);
    // Class-scope explicit specializations (MS extension) do not result in
    // deduction guides.
    if (!CD || (!FTD && CD->isFunctionTemplateSpecialization()))
      continue;

    // Cannot make a deduction guide when unparsed arguments are present.
    if (llvm::any_of(CD->parameters(), [](ParmVarDecl *P) {
          return !P || P->hasUnparsedDefaultArg();
        }))
      continue;

    ProcessedCtors.insert(D);
    Transform.transformConstructor(FTD, CD);
    AddedAny = true;
  }

  // C++17 [over.match.class.deduct]
  //    --  If C is not defined or does not declare any constructors, an
  //    additional function template derived as above from a hypothetical
  //    constructor C().
  if (!AddedAny)
    Transform.buildSimpleDeductionGuide({});

  //    -- An additional function template derived as above from a hypothetical
  //    constructor C(C), called the copy deduction candidate.
  Transform.buildSimpleDeductionGuide(Transform.DeducedType)
      ->setDeductionCandidateKind(DeductionCandidate::Copy);

  SavedContext.pop();
}
