//===- ExprCXX.cpp - (C++) Expression AST Node Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Expr class declared in ExprCXX.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ComputeDependence.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DependenceFlags.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <string>
#include <utility>

using namespace clang;

ConceptSpecializationExpr::ConceptSpecializationExpr(
    const ASTContext &C, NestedNameSpecifierLoc NNS,
    SourceLocation TemplateKWLoc, DeclarationNameInfo ConceptNameInfo,
    NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
    const ASTTemplateArgumentListInfo *ArgsAsWritten,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      ConceptReference(NNS, TemplateKWLoc, ConceptNameInfo, FoundDecl,
                       NamedConcept, ArgsAsWritten),
      SpecDecl(SpecDecl),
      Satisfaction(Satisfaction
                       ? ASTConstraintSatisfaction::Create(C, *Satisfaction)
                       : nullptr) {
  setDependence(computeDependence(this, /*ValueDependent=*/!Satisfaction));

  // Currently guaranteed by the fact concepts can only be at namespace-scope.
  assert(!NestedNameSpec ||
         (!NestedNameSpec.getNestedNameSpecifier()->isInstantiationDependent() &&
          !NestedNameSpec.getNestedNameSpecifier()
              ->containsUnexpandedParameterPack()));
  assert((!isValueDependent() || isInstantiationDependent()) &&
         "should not be value-dependent");
}

ConceptSpecializationExpr::ConceptSpecializationExpr(EmptyShell Empty)
    : Expr(ConceptSpecializationExprClass, Empty) {}

ConceptSpecializationExpr *ConceptSpecializationExpr::Create(
    const ASTContext &C, NestedNameSpecifierLoc NNS,
    SourceLocation TemplateKWLoc, DeclarationNameInfo ConceptNameInfo,
    NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
    const ASTTemplateArgumentListInfo *ArgsAsWritten,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction) {
  return new (C) ConceptSpecializationExpr(
      C, NNS, TemplateKWLoc, ConceptNameInfo, FoundDecl, NamedConcept,
      ArgsAsWritten, SpecDecl, Satisfaction);
}

ConceptSpecializationExpr::ConceptSpecializationExpr(
    const ASTContext &C, ConceptDecl *NamedConcept,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction, bool Dependent,
    bool ContainsUnexpandedParameterPack)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      ConceptReference(NestedNameSpecifierLoc(), SourceLocation(),
                       DeclarationNameInfo(), NamedConcept, NamedConcept,
                       nullptr),
      SpecDecl(SpecDecl),
      Satisfaction(Satisfaction
                       ? ASTConstraintSatisfaction::Create(C, *Satisfaction)
                       : nullptr) {
  ExprDependence D = ExprDependence::None;
  if (!Satisfaction)
    D |= ExprDependence::Value;
  if (Dependent)
    D |= ExprDependence::Instantiation;
  if (ContainsUnexpandedParameterPack)
    D |= ExprDependence::UnexpandedPack;
  setDependence(D);
}

ConceptSpecializationExpr *ConceptSpecializationExpr::Create(
    const ASTContext &C, ConceptDecl *NamedConcept,
    ImplicitConceptSpecializationDecl *SpecDecl,
    const ConstraintSatisfaction *Satisfaction, bool Dependent,
    bool ContainsUnexpandedParameterPack) {
  return new (C)
      ConceptSpecializationExpr(C, NamedConcept, SpecDecl, Satisfaction,
                                Dependent, ContainsUnexpandedParameterPack);
}

const TypeConstraint *
concepts::ExprRequirement::ReturnTypeRequirement::getTypeConstraint() const {
  assert(isTypeConstraint());
  auto TPL =
      TypeConstraintInfo.getPointer().get<TemplateParameterList *>();
  return cast<TemplateTypeParmDecl>(TPL->getParam(0))
      ->getTypeConstraint();
}

RequiresExpr::RequiresExpr(ASTContext &C, SourceLocation RequiresKWLoc,
                           RequiresExprBodyDecl *Body,
                           ArrayRef<ParmVarDecl *> LocalParameters,
                           ArrayRef<concepts::Requirement *> Requirements,
                           SourceLocation RBraceLoc)
    : Expr(RequiresExprClass, C.BoolTy, VK_PRValue, OK_Ordinary),
      NumLocalParameters(LocalParameters.size()),
      NumRequirements(Requirements.size()), Body(Body), RBraceLoc(RBraceLoc) {
  RequiresExprBits.IsSatisfied = false;
  RequiresExprBits.RequiresKWLoc = RequiresKWLoc;
  bool Dependent = false;
  bool ContainsUnexpandedParameterPack = false;
  for (ParmVarDecl *P : LocalParameters) {
    Dependent |= P->getType()->isInstantiationDependentType();
    ContainsUnexpandedParameterPack |=
        P->getType()->containsUnexpandedParameterPack();
  }
  RequiresExprBits.IsSatisfied = true;
  for (concepts::Requirement *R : Requirements) {
    Dependent |= R->isDependent();
    ContainsUnexpandedParameterPack |= R->containsUnexpandedParameterPack();
    if (!Dependent) {
      RequiresExprBits.IsSatisfied = R->isSatisfied();
      if (!RequiresExprBits.IsSatisfied)
        break;
    }
  }
  std::copy(LocalParameters.begin(), LocalParameters.end(),
            getTrailingObjects<ParmVarDecl *>());
  std::copy(Requirements.begin(), Requirements.end(),
            getTrailingObjects<concepts::Requirement *>());
  RequiresExprBits.IsSatisfied |= Dependent;
  // FIXME: move the computing dependency logic to ComputeDependence.h
  if (ContainsUnexpandedParameterPack)
    setDependence(getDependence() | ExprDependence::UnexpandedPack);
  // FIXME: this is incorrect for cases where we have a non-dependent
  // requirement, but its parameters are instantiation-dependent. RequiresExpr
  // should be instantiation-dependent if it has instantiation-dependent
  // parameters.
  if (Dependent)
    setDependence(getDependence() | ExprDependence::ValueInstantiation);
}

RequiresExpr::RequiresExpr(ASTContext &C, EmptyShell Empty,
                           unsigned NumLocalParameters,
                           unsigned NumRequirements)
  : Expr(RequiresExprClass, Empty), NumLocalParameters(NumLocalParameters),
    NumRequirements(NumRequirements) { }

RequiresExpr *
RequiresExpr::Create(ASTContext &C, SourceLocation RequiresKWLoc,
                     RequiresExprBodyDecl *Body,
                     ArrayRef<ParmVarDecl *> LocalParameters,
                     ArrayRef<concepts::Requirement *> Requirements,
                     SourceLocation RBraceLoc) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<ParmVarDecl *, concepts::Requirement *>(
                     LocalParameters.size(), Requirements.size()),
                 alignof(RequiresExpr));
  return new (Mem) RequiresExpr(C, RequiresKWLoc, Body, LocalParameters,
                                Requirements, RBraceLoc);
}

RequiresExpr *
RequiresExpr::Create(ASTContext &C, EmptyShell Empty,
                     unsigned NumLocalParameters, unsigned NumRequirements) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<ParmVarDecl *, concepts::Requirement *>(
                     NumLocalParameters, NumRequirements),
                 alignof(RequiresExpr));
  return new (Mem) RequiresExpr(C, Empty, NumLocalParameters, NumRequirements);
}
