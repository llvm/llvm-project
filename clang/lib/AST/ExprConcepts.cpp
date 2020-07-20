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
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConcept.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <utility>
#include <string>

using namespace clang;

ConceptSpecializationExpr::ConceptSpecializationExpr(const ASTContext &C,
    NestedNameSpecifierLoc NNS, SourceLocation TemplateKWLoc,
    DeclarationNameInfo ConceptNameInfo, NamedDecl *FoundDecl,
    ConceptDecl *NamedConcept, const ASTTemplateArgumentListInfo *ArgsAsWritten,
    ArrayRef<TemplateArgument> ConvertedArgs,
    const ConstraintSatisfaction *Satisfaction)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_RValue, OK_Ordinary,
           /*TypeDependent=*/false,
           // All the flags below are set in setTemplateArguments.
           /*ValueDependent=*/!Satisfaction, /*InstantiationDependent=*/false,
           /*ContainsUnexpandedParameterPacks=*/false),
      ConceptReference(NNS, TemplateKWLoc, ConceptNameInfo, FoundDecl,
                       NamedConcept, ArgsAsWritten),
      NumTemplateArgs(ConvertedArgs.size()),
      Satisfaction(Satisfaction ?
                   ASTConstraintSatisfaction::Create(C, *Satisfaction) :
                   nullptr) {
  setTemplateArguments(ConvertedArgs);
  bool IsInstantiationDependent = false;
  bool ContainsUnexpandedParameterPack = false;
  for (const TemplateArgumentLoc& ArgLoc : ArgsAsWritten->arguments()) {
    if (ArgLoc.getArgument().isInstantiationDependent())
      IsInstantiationDependent = true;
    if (ArgLoc.getArgument().containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;
    if (ContainsUnexpandedParameterPack && IsInstantiationDependent)
      break;
  }

  // Currently guaranteed by the fact concepts can only be at namespace-scope.
  assert(!NestedNameSpec ||
         (!NestedNameSpec.getNestedNameSpecifier()->isInstantiationDependent() &&
          !NestedNameSpec.getNestedNameSpecifier()
              ->containsUnexpandedParameterPack()));
  setInstantiationDependent(IsInstantiationDependent);
  setContainsUnexpandedParameterPack(ContainsUnexpandedParameterPack);
  assert((!isValueDependent() || isInstantiationDependent()) &&
         "should not be value-dependent");
}

ConceptSpecializationExpr::ConceptSpecializationExpr(EmptyShell Empty,
    unsigned NumTemplateArgs)
    : Expr(ConceptSpecializationExprClass, Empty), ConceptReference(),
      NumTemplateArgs(NumTemplateArgs) { }

void ConceptSpecializationExpr::setTemplateArguments(
    ArrayRef<TemplateArgument> Converted) {
  assert(Converted.size() == NumTemplateArgs);
  std::uninitialized_copy(Converted.begin(), Converted.end(),
                          getTrailingObjects<TemplateArgument>());
}

ConceptSpecializationExpr *
ConceptSpecializationExpr::Create(const ASTContext &C,
                                  NestedNameSpecifierLoc NNS,
                                  SourceLocation TemplateKWLoc,
                                  DeclarationNameInfo ConceptNameInfo,
                                  NamedDecl *FoundDecl,
                                  ConceptDecl *NamedConcept,
                               const ASTTemplateArgumentListInfo *ArgsAsWritten,
                                  ArrayRef<TemplateArgument> ConvertedArgs,
                                  const ConstraintSatisfaction *Satisfaction) {
  void *Buffer = C.Allocate(totalSizeToAlloc<TemplateArgument>(
                                ConvertedArgs.size()));
  return new (Buffer) ConceptSpecializationExpr(C, NNS, TemplateKWLoc,
                                                ConceptNameInfo, FoundDecl,
                                                NamedConcept, ArgsAsWritten,
                                                ConvertedArgs, Satisfaction);
}

ConceptSpecializationExpr::ConceptSpecializationExpr(
    const ASTContext &C, ConceptDecl *NamedConcept,
    ArrayRef<TemplateArgument> ConvertedArgs,
    const ConstraintSatisfaction *Satisfaction, bool Dependent,
    bool ContainsUnexpandedParameterPack)
    : Expr(ConceptSpecializationExprClass, C.BoolTy, VK_RValue, OK_Ordinary,
           /*TypeDependent=*/false,
           /*ValueDependent=*/!Satisfaction, Dependent,
           ContainsUnexpandedParameterPack),
      ConceptReference(NestedNameSpecifierLoc(), SourceLocation(),
                       DeclarationNameInfo(), NamedConcept,
                       NamedConcept, nullptr),
      NumTemplateArgs(ConvertedArgs.size()),
      Satisfaction(Satisfaction ?
                   ASTConstraintSatisfaction::Create(C, *Satisfaction) :
                   nullptr) {
  setTemplateArguments(ConvertedArgs);
}

ConceptSpecializationExpr *
ConceptSpecializationExpr::Create(const ASTContext &C,
                                  ConceptDecl *NamedConcept,
                                  ArrayRef<TemplateArgument> ConvertedArgs,
                                  const ConstraintSatisfaction *Satisfaction,
                                  bool Dependent,
                                  bool ContainsUnexpandedParameterPack) {
  void *Buffer = C.Allocate(totalSizeToAlloc<TemplateArgument>(
                                ConvertedArgs.size()));
  return new (Buffer) ConceptSpecializationExpr(
      C, NamedConcept, ConvertedArgs, Satisfaction, Dependent,
      ContainsUnexpandedParameterPack);
}

ConceptSpecializationExpr *
ConceptSpecializationExpr::Create(ASTContext &C, EmptyShell Empty,
                                  unsigned NumTemplateArgs) {
  void *Buffer = C.Allocate(totalSizeToAlloc<TemplateArgument>(
                                NumTemplateArgs));
  return new (Buffer) ConceptSpecializationExpr(Empty, NumTemplateArgs);
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
  : Expr(RequiresExprClass, C.BoolTy, VK_RValue, OK_Ordinary,
         /*TD=*/false, /*VD=*/false, /*ID=*/false,
         /*ContainsUnexpandedParameterPack=*/false),
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
  setValueDependent(Dependent);
  setInstantiationDependent(Dependent);
  setContainsUnexpandedParameterPack(ContainsUnexpandedParameterPack);
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
