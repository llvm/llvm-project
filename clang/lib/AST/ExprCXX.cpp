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

#include "clang/AST/ExprCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/LambdaCapture.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

using namespace clang;

//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

bool CXXOperatorCallExpr::isInfixBinaryOp() const {
  // An infix binary operator is any operator with two arguments other than
  // operator() and operator[]. Note that none of these operators can have
  // default arguments, so it suffices to check the number of argument
  // expressions.
  if (getNumArgs() != 2)
    return false;

  switch (getOperator()) {
  case OO_Call: case OO_Subscript:
    return false;
  default:
    return true;
  }
}

CXXRewrittenBinaryOperator::DecomposedForm
CXXRewrittenBinaryOperator::getDecomposedForm() const {
  DecomposedForm Result = {};
  const Expr *E = getSemanticForm()->IgnoreImplicit();

  // Remove an outer '!' if it exists (only happens for a '!=' rewrite).
  bool SkippedNot = false;
  if (auto *NotEq = dyn_cast<UnaryOperator>(E)) {
    assert(NotEq->getOpcode() == UO_LNot);
    E = NotEq->getSubExpr()->IgnoreImplicit();
    SkippedNot = true;
  }

  // Decompose the outer binary operator.
  if (auto *BO = dyn_cast<BinaryOperator>(E)) {
    assert(!SkippedNot || BO->getOpcode() == BO_EQ);
    Result.Opcode = SkippedNot ? BO_NE : BO->getOpcode();
    Result.LHS = BO->getLHS();
    Result.RHS = BO->getRHS();
    Result.InnerBinOp = BO;
  } else if (auto *BO = dyn_cast<CXXOperatorCallExpr>(E)) {
    assert(!SkippedNot || BO->getOperator() == OO_EqualEqual);
    assert(BO->isInfixBinaryOp());
    switch (BO->getOperator()) {
    case OO_Less: Result.Opcode = BO_LT; break;
    case OO_LessEqual: Result.Opcode = BO_LE; break;
    case OO_Greater: Result.Opcode = BO_GT; break;
    case OO_GreaterEqual: Result.Opcode = BO_GE; break;
    case OO_Spaceship: Result.Opcode = BO_Cmp; break;
    case OO_EqualEqual: Result.Opcode = SkippedNot ? BO_NE : BO_EQ; break;
    default: llvm_unreachable("unexpected binop in rewritten operator expr");
    }
    Result.LHS = BO->getArg(0);
    Result.RHS = BO->getArg(1);
    Result.InnerBinOp = BO;
  } else {
    llvm_unreachable("unexpected rewritten operator form");
  }

  // Put the operands in the right order for == and !=, and canonicalize the
  // <=> subexpression onto the LHS for all other forms.
  if (isReversed())
    std::swap(Result.LHS, Result.RHS);

  // If this isn't a spaceship rewrite, we're done.
  if (Result.Opcode == BO_EQ || Result.Opcode == BO_NE)
    return Result;

  // Otherwise, we expect a <=> to now be on the LHS.
  E = Result.LHS->IgnoreImplicitAsWritten();
  if (auto *BO = dyn_cast<BinaryOperator>(E)) {
    assert(BO->getOpcode() == BO_Cmp);
    Result.LHS = BO->getLHS();
    Result.RHS = BO->getRHS();
    Result.InnerBinOp = BO;
  } else if (auto *BO = dyn_cast<CXXOperatorCallExpr>(E)) {
    assert(BO->getOperator() == OO_Spaceship);
    Result.LHS = BO->getArg(0);
    Result.RHS = BO->getArg(1);
    Result.InnerBinOp = BO;
  } else {
    llvm_unreachable("unexpected rewritten operator form");
  }

  // Put the comparison operands in the right order.
  if (isReversed())
    std::swap(Result.LHS, Result.RHS);
  return Result;
}

bool CXXTypeidExpr::isPotentiallyEvaluated() const {
  if (isTypeOperand())
    return false;

  // C++11 [expr.typeid]p3:
  //   When typeid is applied to an expression other than a glvalue of
  //   polymorphic class type, [...] the expression is an unevaluated operand.
  const Expr *E = getExprOperand();
  if (const CXXRecordDecl *RD = E->getType()->getAsCXXRecordDecl())
    if (RD->isPolymorphic() && E->isGLValue())
      return true;

  return false;
}

QualType CXXTypeidExpr::getTypeOperand(ASTContext &Context) const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for typeid(expr)");
  Qualifiers Quals;
  return Context.getUnqualifiedArrayType(
      Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType(), Quals);
}

QualType CXXUuidofExpr::getTypeOperand(ASTContext &Context) const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for __uuidof(expr)");
  Qualifiers Quals;
  return Context.getUnqualifiedArrayType(
      Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType(), Quals);
}

// CXXScalarValueInitExpr
SourceLocation CXXScalarValueInitExpr::getBeginLoc() const {
  return TypeInfo ? TypeInfo->getTypeLoc().getBeginLoc() : getRParenLoc();
}

// CXXNewExpr
CXXNewExpr::CXXNewExpr(bool IsGlobalNew, FunctionDecl *OperatorNew,
                       FunctionDecl *OperatorDelete, bool ShouldPassAlignment,
                       bool UsualArrayDeleteWantsSize,
                       ArrayRef<Expr *> PlacementArgs, SourceRange TypeIdParens,
                       Optional<Expr *> ArraySize,
                       InitializationStyle InitializationStyle,
                       Expr *Initializer, QualType Ty,
                       TypeSourceInfo *AllocatedTypeInfo, SourceRange Range,
                       SourceRange DirectInitRange)
    : Expr(CXXNewExprClass, Ty, VK_RValue, OK_Ordinary, Ty->isDependentType(),
           Ty->isDependentType(), Ty->isInstantiationDependentType(),
           Ty->containsUnexpandedParameterPack()),
      OperatorNew(OperatorNew), OperatorDelete(OperatorDelete),
      AllocatedTypeInfo(AllocatedTypeInfo), Range(Range),
      DirectInitRange(DirectInitRange) {

  assert((Initializer != nullptr || InitializationStyle == NoInit) &&
         "Only NoInit can have no initializer!");

  CXXNewExprBits.IsGlobalNew = IsGlobalNew;
  CXXNewExprBits.IsArray = ArraySize.hasValue();
  CXXNewExprBits.ShouldPassAlignment = ShouldPassAlignment;
  CXXNewExprBits.UsualArrayDeleteWantsSize = UsualArrayDeleteWantsSize;
  CXXNewExprBits.StoredInitializationStyle =
      Initializer ? InitializationStyle + 1 : 0;
  bool IsParenTypeId = TypeIdParens.isValid();
  CXXNewExprBits.IsParenTypeId = IsParenTypeId;
  CXXNewExprBits.NumPlacementArgs = PlacementArgs.size();

  if (ArraySize) {
    if (Expr *SizeExpr = *ArraySize) {
      if (SizeExpr->isValueDependent())
        ExprBits.ValueDependent = true;
      if (SizeExpr->isInstantiationDependent())
        ExprBits.InstantiationDependent = true;
      if (SizeExpr->containsUnexpandedParameterPack())
        ExprBits.ContainsUnexpandedParameterPack = true;
    }

    getTrailingObjects<Stmt *>()[arraySizeOffset()] = *ArraySize;
  }

  if (Initializer) {
    if (Initializer->isValueDependent())
      ExprBits.ValueDependent = true;
    if (Initializer->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;
    if (Initializer->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    getTrailingObjects<Stmt *>()[initExprOffset()] = Initializer;
  }

  for (unsigned I = 0; I != PlacementArgs.size(); ++I) {
    if (PlacementArgs[I]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (PlacementArgs[I]->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;
    if (PlacementArgs[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    getTrailingObjects<Stmt *>()[placementNewArgsOffset() + I] =
        PlacementArgs[I];
  }

  if (IsParenTypeId)
    getTrailingObjects<SourceRange>()[0] = TypeIdParens;

  switch (getInitializationStyle()) {
  case CallInit:
    this->Range.setEnd(DirectInitRange.getEnd());
    break;
  case ListInit:
    this->Range.setEnd(getInitializer()->getSourceRange().getEnd());
    break;
  default:
    if (IsParenTypeId)
      this->Range.setEnd(TypeIdParens.getEnd());
    break;
  }
}

CXXNewExpr::CXXNewExpr(EmptyShell Empty, bool IsArray,
                       unsigned NumPlacementArgs, bool IsParenTypeId)
    : Expr(CXXNewExprClass, Empty) {
  CXXNewExprBits.IsArray = IsArray;
  CXXNewExprBits.NumPlacementArgs = NumPlacementArgs;
  CXXNewExprBits.IsParenTypeId = IsParenTypeId;
}

CXXNewExpr *
CXXNewExpr::Create(const ASTContext &Ctx, bool IsGlobalNew,
                   FunctionDecl *OperatorNew, FunctionDecl *OperatorDelete,
                   bool ShouldPassAlignment, bool UsualArrayDeleteWantsSize,
                   ArrayRef<Expr *> PlacementArgs, SourceRange TypeIdParens,
                   Optional<Expr *> ArraySize,
                   InitializationStyle InitializationStyle, Expr *Initializer,
                   QualType Ty, TypeSourceInfo *AllocatedTypeInfo,
                   SourceRange Range, SourceRange DirectInitRange) {
  bool IsArray = ArraySize.hasValue();
  bool HasInit = Initializer != nullptr;
  unsigned NumPlacementArgs = PlacementArgs.size();
  bool IsParenTypeId = TypeIdParens.isValid();
  void *Mem =
      Ctx.Allocate(totalSizeToAlloc<Stmt *, SourceRange>(
                       IsArray + HasInit + NumPlacementArgs, IsParenTypeId),
                   alignof(CXXNewExpr));
  return new (Mem)
      CXXNewExpr(IsGlobalNew, OperatorNew, OperatorDelete, ShouldPassAlignment,
                 UsualArrayDeleteWantsSize, PlacementArgs, TypeIdParens,
                 ArraySize, InitializationStyle, Initializer, Ty,
                 AllocatedTypeInfo, Range, DirectInitRange);
}

CXXNewExpr *CXXNewExpr::CreateEmpty(const ASTContext &Ctx, bool IsArray,
                                    bool HasInit, unsigned NumPlacementArgs,
                                    bool IsParenTypeId) {
  void *Mem =
      Ctx.Allocate(totalSizeToAlloc<Stmt *, SourceRange>(
                       IsArray + HasInit + NumPlacementArgs, IsParenTypeId),
                   alignof(CXXNewExpr));
  return new (Mem)
      CXXNewExpr(EmptyShell(), IsArray, NumPlacementArgs, IsParenTypeId);
}

bool CXXNewExpr::shouldNullCheckAllocation() const {
  return getOperatorNew()
             ->getType()
             ->castAs<FunctionProtoType>()
             ->isNothrow() &&
         !getOperatorNew()->isReservedGlobalPlacementOperator();
}

// CXXDeleteExpr
QualType CXXDeleteExpr::getDestroyedType() const {
  const Expr *Arg = getArgument();

  // For a destroying operator delete, we may have implicitly converted the
  // pointer type to the type of the parameter of the 'operator delete'
  // function.
  while (const auto *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
    if (ICE->getCastKind() == CK_DerivedToBase ||
        ICE->getCastKind() == CK_UncheckedDerivedToBase ||
        ICE->getCastKind() == CK_NoOp) {
      assert((ICE->getCastKind() == CK_NoOp ||
              getOperatorDelete()->isDestroyingOperatorDelete()) &&
             "only a destroying operator delete can have a converted arg");
      Arg = ICE->getSubExpr();
    } else
      break;
  }

  // The type-to-delete may not be a pointer if it's a dependent type.
  const QualType ArgType = Arg->getType();

  if (ArgType->isDependentType() && !ArgType->isPointerType())
    return QualType();

  return ArgType->castAs<PointerType>()->getPointeeType();
}

// CXXPseudoDestructorExpr
PseudoDestructorTypeStorage::PseudoDestructorTypeStorage(TypeSourceInfo *Info)
    : Type(Info) {
  Location = Info->getTypeLoc().getLocalSourceRange().getBegin();
}

CXXPseudoDestructorExpr::CXXPseudoDestructorExpr(const ASTContext &Context,
                Expr *Base, bool isArrow, SourceLocation OperatorLoc,
                NestedNameSpecifierLoc QualifierLoc, TypeSourceInfo *ScopeType,
                SourceLocation ColonColonLoc, SourceLocation TildeLoc,
                PseudoDestructorTypeStorage DestroyedType)
  : Expr(CXXPseudoDestructorExprClass,
         Context.BoundMemberTy,
         VK_RValue, OK_Ordinary,
         /*isTypeDependent=*/(Base->isTypeDependent() ||
           (DestroyedType.getTypeSourceInfo() &&
            DestroyedType.getTypeSourceInfo()->getType()->isDependentType())),
         /*isValueDependent=*/Base->isValueDependent(),
         (Base->isInstantiationDependent() ||
          (QualifierLoc &&
           QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent()) ||
          (ScopeType &&
           ScopeType->getType()->isInstantiationDependentType()) ||
          (DestroyedType.getTypeSourceInfo() &&
           DestroyedType.getTypeSourceInfo()->getType()
                                             ->isInstantiationDependentType())),
         // ContainsUnexpandedParameterPack
         (Base->containsUnexpandedParameterPack() ||
          (QualifierLoc &&
           QualifierLoc.getNestedNameSpecifier()
                                        ->containsUnexpandedParameterPack()) ||
          (ScopeType &&
           ScopeType->getType()->containsUnexpandedParameterPack()) ||
          (DestroyedType.getTypeSourceInfo() &&
           DestroyedType.getTypeSourceInfo()->getType()
                                   ->containsUnexpandedParameterPack()))),
    Base(static_cast<Stmt *>(Base)), IsArrow(isArrow),
    OperatorLoc(OperatorLoc), QualifierLoc(QualifierLoc),
    ScopeType(ScopeType), ColonColonLoc(ColonColonLoc), TildeLoc(TildeLoc),
    DestroyedType(DestroyedType) {}

QualType CXXPseudoDestructorExpr::getDestroyedType() const {
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    return TInfo->getType();

  return QualType();
}

SourceLocation CXXPseudoDestructorExpr::getEndLoc() const {
  SourceLocation End = DestroyedType.getLocation();
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    End = TInfo->getTypeLoc().getLocalSourceRange().getEnd();
  return End;
}

// UnresolvedLookupExpr
UnresolvedLookupExpr::UnresolvedLookupExpr(
    const ASTContext &Context, CXXRecordDecl *NamingClass,
    NestedNameSpecifierLoc QualifierLoc, SourceLocation TemplateKWLoc,
    const DeclarationNameInfo &NameInfo, bool RequiresADL, bool Overloaded,
    const TemplateArgumentListInfo *TemplateArgs, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End)
    : OverloadExpr(UnresolvedLookupExprClass, Context, QualifierLoc,
                   TemplateKWLoc, NameInfo, TemplateArgs, Begin, End, false,
                   false, false),
      NamingClass(NamingClass) {
  UnresolvedLookupExprBits.RequiresADL = RequiresADL;
  UnresolvedLookupExprBits.Overloaded = Overloaded;
}

UnresolvedLookupExpr::UnresolvedLookupExpr(EmptyShell Empty,
                                           unsigned NumResults,
                                           bool HasTemplateKWAndArgsInfo)
    : OverloadExpr(UnresolvedLookupExprClass, Empty, NumResults,
                   HasTemplateKWAndArgsInfo) {}

UnresolvedLookupExpr *UnresolvedLookupExpr::Create(
    const ASTContext &Context, CXXRecordDecl *NamingClass,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    bool RequiresADL, bool Overloaded, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End) {
  unsigned NumResults = End - Begin;
  unsigned Size = totalSizeToAlloc<DeclAccessPair, ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc>(NumResults, 0, 0);
  void *Mem = Context.Allocate(Size, alignof(UnresolvedLookupExpr));
  return new (Mem) UnresolvedLookupExpr(Context, NamingClass, QualifierLoc,
                                        SourceLocation(), NameInfo, RequiresADL,
                                        Overloaded, nullptr, Begin, End);
}

UnresolvedLookupExpr *UnresolvedLookupExpr::Create(
    const ASTContext &Context, CXXRecordDecl *NamingClass,
    NestedNameSpecifierLoc QualifierLoc, SourceLocation TemplateKWLoc,
    const DeclarationNameInfo &NameInfo, bool RequiresADL,
    const TemplateArgumentListInfo *Args, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End) {
  assert(Args || TemplateKWLoc.isValid());
  unsigned NumResults = End - Begin;
  unsigned NumTemplateArgs = Args ? Args->size() : 0;
  unsigned Size =
      totalSizeToAlloc<DeclAccessPair, ASTTemplateKWAndArgsInfo,
                       TemplateArgumentLoc>(NumResults, 1, NumTemplateArgs);
  void *Mem = Context.Allocate(Size, alignof(UnresolvedLookupExpr));
  return new (Mem) UnresolvedLookupExpr(Context, NamingClass, QualifierLoc,
                                        TemplateKWLoc, NameInfo, RequiresADL,
                                        /*Overloaded*/ true, Args, Begin, End);
}

UnresolvedLookupExpr *UnresolvedLookupExpr::CreateEmpty(
    const ASTContext &Context, unsigned NumResults,
    bool HasTemplateKWAndArgsInfo, unsigned NumTemplateArgs) {
  assert(NumTemplateArgs == 0 || HasTemplateKWAndArgsInfo);
  unsigned Size = totalSizeToAlloc<DeclAccessPair, ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc>(
      NumResults, HasTemplateKWAndArgsInfo, NumTemplateArgs);
  void *Mem = Context.Allocate(Size, alignof(UnresolvedLookupExpr));
  return new (Mem)
      UnresolvedLookupExpr(EmptyShell(), NumResults, HasTemplateKWAndArgsInfo);
}

OverloadExpr::OverloadExpr(StmtClass SC, const ASTContext &Context,
                           NestedNameSpecifierLoc QualifierLoc,
                           SourceLocation TemplateKWLoc,
                           const DeclarationNameInfo &NameInfo,
                           const TemplateArgumentListInfo *TemplateArgs,
                           UnresolvedSetIterator Begin,
                           UnresolvedSetIterator End, bool KnownDependent,
                           bool KnownInstantiationDependent,
                           bool KnownContainsUnexpandedParameterPack)
    : Expr(
          SC, Context.OverloadTy, VK_LValue, OK_Ordinary, KnownDependent,
          KnownDependent,
          (KnownInstantiationDependent || NameInfo.isInstantiationDependent() ||
           (QualifierLoc &&
            QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent())),
          (KnownContainsUnexpandedParameterPack ||
           NameInfo.containsUnexpandedParameterPack() ||
           (QualifierLoc && QualifierLoc.getNestedNameSpecifier()
                                ->containsUnexpandedParameterPack()))),
      NameInfo(NameInfo), QualifierLoc(QualifierLoc) {
  unsigned NumResults = End - Begin;
  OverloadExprBits.NumResults = NumResults;
  OverloadExprBits.HasTemplateKWAndArgsInfo =
      (TemplateArgs != nullptr ) || TemplateKWLoc.isValid();

  if (NumResults) {
    // Determine whether this expression is type-dependent.
    for (UnresolvedSetImpl::const_iterator I = Begin; I != End; ++I) {
      if ((*I)->getDeclContext()->isDependentContext() ||
          isa<UnresolvedUsingValueDecl>(*I)) {
        ExprBits.TypeDependent = true;
        ExprBits.ValueDependent = true;
        ExprBits.InstantiationDependent = true;
      }
    }

    // Copy the results to the trailing array past UnresolvedLookupExpr
    // or UnresolvedMemberExpr.
    DeclAccessPair *Results = getTrailingResults();
    memcpy(Results, Begin.I, NumResults * sizeof(DeclAccessPair));
  }

  // If we have explicit template arguments, check for dependent
  // template arguments and whether they contain any unexpanded pack
  // expansions.
  if (TemplateArgs) {
    bool Dependent = false;
    bool InstantiationDependent = false;
    bool ContainsUnexpandedParameterPack = false;
    getTrailingASTTemplateKWAndArgsInfo()->initializeFrom(
        TemplateKWLoc, *TemplateArgs, getTrailingTemplateArgumentLoc(),
        Dependent, InstantiationDependent, ContainsUnexpandedParameterPack);

    if (Dependent) {
      ExprBits.TypeDependent = true;
      ExprBits.ValueDependent = true;
    }
    if (InstantiationDependent)
      ExprBits.InstantiationDependent = true;
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  } else if (TemplateKWLoc.isValid()) {
    getTrailingASTTemplateKWAndArgsInfo()->initializeFrom(TemplateKWLoc);
  }

  if (isTypeDependent())
    setType(Context.DependentTy);
}

OverloadExpr::OverloadExpr(StmtClass SC, EmptyShell Empty, unsigned NumResults,
                           bool HasTemplateKWAndArgsInfo)
    : Expr(SC, Empty) {
  OverloadExprBits.NumResults = NumResults;
  OverloadExprBits.HasTemplateKWAndArgsInfo = HasTemplateKWAndArgsInfo;
}

// DependentScopeDeclRefExpr
DependentScopeDeclRefExpr::DependentScopeDeclRefExpr(
    QualType Ty, NestedNameSpecifierLoc QualifierLoc,
    SourceLocation TemplateKWLoc, const DeclarationNameInfo &NameInfo,
    const TemplateArgumentListInfo *Args)
    : Expr(
          DependentScopeDeclRefExprClass, Ty, VK_LValue, OK_Ordinary, true,
          true,
          (NameInfo.isInstantiationDependent() ||
           (QualifierLoc &&
            QualifierLoc.getNestedNameSpecifier()->isInstantiationDependent())),
          (NameInfo.containsUnexpandedParameterPack() ||
           (QualifierLoc && QualifierLoc.getNestedNameSpecifier()
                                ->containsUnexpandedParameterPack()))),
      QualifierLoc(QualifierLoc), NameInfo(NameInfo) {
  DependentScopeDeclRefExprBits.HasTemplateKWAndArgsInfo =
      (Args != nullptr) || TemplateKWLoc.isValid();
  if (Args) {
    bool Dependent = true;
    bool InstantiationDependent = true;
    bool ContainsUnexpandedParameterPack
      = ExprBits.ContainsUnexpandedParameterPack;
    getTrailingObjects<ASTTemplateKWAndArgsInfo>()->initializeFrom(
        TemplateKWLoc, *Args, getTrailingObjects<TemplateArgumentLoc>(),
        Dependent, InstantiationDependent, ContainsUnexpandedParameterPack);
    ExprBits.ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;
  } else if (TemplateKWLoc.isValid()) {
    getTrailingObjects<ASTTemplateKWAndArgsInfo>()->initializeFrom(
        TemplateKWLoc);
  }
}

DependentScopeDeclRefExpr *DependentScopeDeclRefExpr::Create(
    const ASTContext &Context, NestedNameSpecifierLoc QualifierLoc,
    SourceLocation TemplateKWLoc, const DeclarationNameInfo &NameInfo,
    const TemplateArgumentListInfo *Args) {
  assert(QualifierLoc && "should be created for dependent qualifiers");
  bool HasTemplateKWAndArgsInfo = Args || TemplateKWLoc.isValid();
  std::size_t Size =
      totalSizeToAlloc<ASTTemplateKWAndArgsInfo, TemplateArgumentLoc>(
          HasTemplateKWAndArgsInfo, Args ? Args->size() : 0);
  void *Mem = Context.Allocate(Size);
  return new (Mem) DependentScopeDeclRefExpr(Context.DependentTy, QualifierLoc,
                                             TemplateKWLoc, NameInfo, Args);
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::CreateEmpty(const ASTContext &Context,
                                       bool HasTemplateKWAndArgsInfo,
                                       unsigned NumTemplateArgs) {
  assert(NumTemplateArgs == 0 || HasTemplateKWAndArgsInfo);
  std::size_t Size =
      totalSizeToAlloc<ASTTemplateKWAndArgsInfo, TemplateArgumentLoc>(
          HasTemplateKWAndArgsInfo, NumTemplateArgs);
  void *Mem = Context.Allocate(Size);
  auto *E = new (Mem) DependentScopeDeclRefExpr(
      QualType(), NestedNameSpecifierLoc(), SourceLocation(),
      DeclarationNameInfo(), nullptr);
  E->DependentScopeDeclRefExprBits.HasTemplateKWAndArgsInfo =
      HasTemplateKWAndArgsInfo;
  return E;
}

SourceLocation CXXConstructExpr::getBeginLoc() const {
  if (isa<CXXTemporaryObjectExpr>(this))
    return cast<CXXTemporaryObjectExpr>(this)->getBeginLoc();
  return getLocation();
}

SourceLocation CXXConstructExpr::getEndLoc() const {
  if (isa<CXXTemporaryObjectExpr>(this))
    return cast<CXXTemporaryObjectExpr>(this)->getEndLoc();

  if (ParenOrBraceRange.isValid())
    return ParenOrBraceRange.getEnd();

  SourceLocation End = getLocation();
  for (unsigned I = getNumArgs(); I > 0; --I) {
    const Expr *Arg = getArg(I-1);
    if (!Arg->isDefaultArgument()) {
      SourceLocation NewEnd = Arg->getEndLoc();
      if (NewEnd.isValid()) {
        End = NewEnd;
        break;
      }
    }
  }

  return End;
}

CXXOperatorCallExpr::CXXOperatorCallExpr(OverloadedOperatorKind OpKind,
                                         Expr *Fn, ArrayRef<Expr *> Args,
                                         QualType Ty, ExprValueKind VK,
                                         SourceLocation OperatorLoc,
                                         FPOptions FPFeatures,
                                         ADLCallKind UsesADL)
    : CallExpr(CXXOperatorCallExprClass, Fn, /*PreArgs=*/{}, Args, Ty, VK,
               OperatorLoc, /*MinNumArgs=*/0, UsesADL) {
  CXXOperatorCallExprBits.OperatorKind = OpKind;
  CXXOperatorCallExprBits.FPFeatures = FPFeatures.getInt();
  assert(
      (CXXOperatorCallExprBits.OperatorKind == static_cast<unsigned>(OpKind)) &&
      "OperatorKind overflow!");
  assert((CXXOperatorCallExprBits.FPFeatures == FPFeatures.getInt()) &&
         "FPFeatures overflow!");
  Range = getSourceRangeImpl();
}

CXXOperatorCallExpr::CXXOperatorCallExpr(unsigned NumArgs, EmptyShell Empty)
    : CallExpr(CXXOperatorCallExprClass, /*NumPreArgs=*/0, NumArgs, Empty) {}

CXXOperatorCallExpr *CXXOperatorCallExpr::Create(
    const ASTContext &Ctx, OverloadedOperatorKind OpKind, Expr *Fn,
    ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK,
    SourceLocation OperatorLoc, FPOptions FPFeatures, ADLCallKind UsesADL) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned NumArgs = Args.size();
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CXXOperatorCallExpr) + SizeOfTrailingObjects,
                           alignof(CXXOperatorCallExpr));
  return new (Mem) CXXOperatorCallExpr(OpKind, Fn, Args, Ty, VK, OperatorLoc,
                                       FPFeatures, UsesADL);
}

CXXOperatorCallExpr *CXXOperatorCallExpr::CreateEmpty(const ASTContext &Ctx,
                                                      unsigned NumArgs,
                                                      EmptyShell Empty) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CXXOperatorCallExpr) + SizeOfTrailingObjects,
                           alignof(CXXOperatorCallExpr));
  return new (Mem) CXXOperatorCallExpr(NumArgs, Empty);
}

SourceRange CXXOperatorCallExpr::getSourceRangeImpl() const {
  OverloadedOperatorKind Kind = getOperator();
  if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
    if (getNumArgs() == 1)
      // Prefix operator
      return SourceRange(getOperatorLoc(), getArg(0)->getEndLoc());
    else
      // Postfix operator
      return SourceRange(getArg(0)->getBeginLoc(), getOperatorLoc());
  } else if (Kind == OO_Arrow) {
    return getArg(0)->getSourceRange();
  } else if (Kind == OO_Call) {
    return SourceRange(getArg(0)->getBeginLoc(), getRParenLoc());
  } else if (Kind == OO_Subscript) {
    return SourceRange(getArg(0)->getBeginLoc(), getRParenLoc());
  } else if (getNumArgs() == 1) {
    return SourceRange(getOperatorLoc(), getArg(0)->getEndLoc());
  } else if (getNumArgs() == 2) {
    return SourceRange(getArg(0)->getBeginLoc(), getArg(1)->getEndLoc());
  } else {
    return getOperatorLoc();
  }
}

CXXMemberCallExpr::CXXMemberCallExpr(Expr *Fn, ArrayRef<Expr *> Args,
                                     QualType Ty, ExprValueKind VK,
                                     SourceLocation RP, unsigned MinNumArgs)
    : CallExpr(CXXMemberCallExprClass, Fn, /*PreArgs=*/{}, Args, Ty, VK, RP,
               MinNumArgs, NotADL) {}

CXXMemberCallExpr::CXXMemberCallExpr(unsigned NumArgs, EmptyShell Empty)
    : CallExpr(CXXMemberCallExprClass, /*NumPreArgs=*/0, NumArgs, Empty) {}

CXXMemberCallExpr *CXXMemberCallExpr::Create(const ASTContext &Ctx, Expr *Fn,
                                             ArrayRef<Expr *> Args, QualType Ty,
                                             ExprValueKind VK,
                                             SourceLocation RP,
                                             unsigned MinNumArgs) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned NumArgs = std::max<unsigned>(Args.size(), MinNumArgs);
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CXXMemberCallExpr) + SizeOfTrailingObjects,
                           alignof(CXXMemberCallExpr));
  return new (Mem) CXXMemberCallExpr(Fn, Args, Ty, VK, RP, MinNumArgs);
}

CXXMemberCallExpr *CXXMemberCallExpr::CreateEmpty(const ASTContext &Ctx,
                                                  unsigned NumArgs,
                                                  EmptyShell Empty) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CXXMemberCallExpr) + SizeOfTrailingObjects,
                           alignof(CXXMemberCallExpr));
  return new (Mem) CXXMemberCallExpr(NumArgs, Empty);
}

Expr *CXXMemberCallExpr::getImplicitObjectArgument() const {
  const Expr *Callee = getCallee()->IgnoreParens();
  if (const auto *MemExpr = dyn_cast<MemberExpr>(Callee))
    return MemExpr->getBase();
  if (const auto *BO = dyn_cast<BinaryOperator>(Callee))
    if (BO->getOpcode() == BO_PtrMemD || BO->getOpcode() == BO_PtrMemI)
      return BO->getLHS();

  // FIXME: Will eventually need to cope with member pointers.
  return nullptr;
}

QualType CXXMemberCallExpr::getObjectType() const {
  QualType Ty = getImplicitObjectArgument()->getType();
  if (Ty->isPointerType())
    Ty = Ty->getPointeeType();
  return Ty;
}

CXXMethodDecl *CXXMemberCallExpr::getMethodDecl() const {
  if (const auto *MemExpr = dyn_cast<MemberExpr>(getCallee()->IgnoreParens()))
    return cast<CXXMethodDecl>(MemExpr->getMemberDecl());

  // FIXME: Will eventually need to cope with member pointers.
  return nullptr;
}

CXXRecordDecl *CXXMemberCallExpr::getRecordDecl() const {
  Expr* ThisArg = getImplicitObjectArgument();
  if (!ThisArg)
    return nullptr;

  if (ThisArg->getType()->isAnyPointerType())
    return ThisArg->getType()->getPointeeType()->getAsCXXRecordDecl();

  return ThisArg->getType()->getAsCXXRecordDecl();
}

//===----------------------------------------------------------------------===//
//  Named casts
//===----------------------------------------------------------------------===//

/// getCastName - Get the name of the C++ cast being used, e.g.,
/// "static_cast", "dynamic_cast", "reinterpret_cast", or
/// "const_cast". The returned pointer must not be freed.
const char *CXXNamedCastExpr::getCastName() const {
  switch (getStmtClass()) {
  case CXXStaticCastExprClass:      return "static_cast";
  case CXXDynamicCastExprClass:     return "dynamic_cast";
  case CXXReinterpretCastExprClass: return "reinterpret_cast";
  case CXXConstCastExprClass:       return "const_cast";
  default:                          return "<invalid cast>";
  }
}

CXXStaticCastExpr *CXXStaticCastExpr::Create(const ASTContext &C, QualType T,
                                             ExprValueKind VK,
                                             CastKind K, Expr *Op,
                                             const CXXCastPath *BasePath,
                                             TypeSourceInfo *WrittenTy,
                                             SourceLocation L,
                                             SourceLocation RParenLoc,
                                             SourceRange AngleBrackets) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  auto *E =
      new (Buffer) CXXStaticCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                     RParenLoc, AngleBrackets);
  if (PathSize)
    std::uninitialized_copy_n(BasePath->data(), BasePath->size(),
                              E->getTrailingObjects<CXXBaseSpecifier *>());
  return E;
}

CXXStaticCastExpr *CXXStaticCastExpr::CreateEmpty(const ASTContext &C,
                                                  unsigned PathSize) {
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  return new (Buffer) CXXStaticCastExpr(EmptyShell(), PathSize);
}

CXXDynamicCastExpr *CXXDynamicCastExpr::Create(const ASTContext &C, QualType T,
                                               ExprValueKind VK,
                                               CastKind K, Expr *Op,
                                               const CXXCastPath *BasePath,
                                               TypeSourceInfo *WrittenTy,
                                               SourceLocation L,
                                               SourceLocation RParenLoc,
                                               SourceRange AngleBrackets) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  auto *E =
      new (Buffer) CXXDynamicCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                      RParenLoc, AngleBrackets);
  if (PathSize)
    std::uninitialized_copy_n(BasePath->data(), BasePath->size(),
                              E->getTrailingObjects<CXXBaseSpecifier *>());
  return E;
}

CXXDynamicCastExpr *CXXDynamicCastExpr::CreateEmpty(const ASTContext &C,
                                                    unsigned PathSize) {
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  return new (Buffer) CXXDynamicCastExpr(EmptyShell(), PathSize);
}

/// isAlwaysNull - Return whether the result of the dynamic_cast is proven
/// to always be null. For example:
///
/// struct A { };
/// struct B final : A { };
/// struct C { };
///
/// C *f(B* b) { return dynamic_cast<C*>(b); }
bool CXXDynamicCastExpr::isAlwaysNull() const
{
  QualType SrcType = getSubExpr()->getType();
  QualType DestType = getType();

  if (const auto *SrcPTy = SrcType->getAs<PointerType>()) {
    SrcType = SrcPTy->getPointeeType();
    DestType = DestType->castAs<PointerType>()->getPointeeType();
  }

  if (DestType->isVoidType())
    return false;

  const auto *SrcRD =
      cast<CXXRecordDecl>(SrcType->castAs<RecordType>()->getDecl());

  if (!SrcRD->hasAttr<FinalAttr>())
    return false;

  const auto *DestRD =
      cast<CXXRecordDecl>(DestType->castAs<RecordType>()->getDecl());

  return !DestRD->isDerivedFrom(SrcRD);
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::Create(const ASTContext &C, QualType T,
                               ExprValueKind VK, CastKind K, Expr *Op,
                               const CXXCastPath *BasePath,
                               TypeSourceInfo *WrittenTy, SourceLocation L,
                               SourceLocation RParenLoc,
                               SourceRange AngleBrackets) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  auto *E =
      new (Buffer) CXXReinterpretCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                          RParenLoc, AngleBrackets);
  if (PathSize)
    std::uninitialized_copy_n(BasePath->data(), BasePath->size(),
                              E->getTrailingObjects<CXXBaseSpecifier *>());
  return E;
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::CreateEmpty(const ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  return new (Buffer) CXXReinterpretCastExpr(EmptyShell(), PathSize);
}

CXXConstCastExpr *CXXConstCastExpr::Create(const ASTContext &C, QualType T,
                                           ExprValueKind VK, Expr *Op,
                                           TypeSourceInfo *WrittenTy,
                                           SourceLocation L,
                                           SourceLocation RParenLoc,
                                           SourceRange AngleBrackets) {
  return new (C) CXXConstCastExpr(T, VK, Op, WrittenTy, L, RParenLoc, AngleBrackets);
}

CXXConstCastExpr *CXXConstCastExpr::CreateEmpty(const ASTContext &C) {
  return new (C) CXXConstCastExpr(EmptyShell());
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::Create(const ASTContext &C, QualType T, ExprValueKind VK,
                              TypeSourceInfo *Written, CastKind K, Expr *Op,
                              const CXXCastPath *BasePath,
                              SourceLocation L, SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  auto *E =
      new (Buffer) CXXFunctionalCastExpr(T, VK, Written, K, Op, PathSize, L, R);
  if (PathSize)
    std::uninitialized_copy_n(BasePath->data(), BasePath->size(),
                              E->getTrailingObjects<CXXBaseSpecifier *>());
  return E;
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::CreateEmpty(const ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(totalSizeToAlloc<CXXBaseSpecifier *>(PathSize));
  return new (Buffer) CXXFunctionalCastExpr(EmptyShell(), PathSize);
}

SourceLocation CXXFunctionalCastExpr::getBeginLoc() const {
  return getTypeInfoAsWritten()->getTypeLoc().getBeginLoc();
}

SourceLocation CXXFunctionalCastExpr::getEndLoc() const {
  return RParenLoc.isValid() ? RParenLoc : getSubExpr()->getEndLoc();
}

UserDefinedLiteral::UserDefinedLiteral(Expr *Fn, ArrayRef<Expr *> Args,
                                       QualType Ty, ExprValueKind VK,
                                       SourceLocation LitEndLoc,
                                       SourceLocation SuffixLoc)
    : CallExpr(UserDefinedLiteralClass, Fn, /*PreArgs=*/{}, Args, Ty, VK,
               LitEndLoc, /*MinNumArgs=*/0, NotADL),
      UDSuffixLoc(SuffixLoc) {}

UserDefinedLiteral::UserDefinedLiteral(unsigned NumArgs, EmptyShell Empty)
    : CallExpr(UserDefinedLiteralClass, /*NumPreArgs=*/0, NumArgs, Empty) {}

UserDefinedLiteral *UserDefinedLiteral::Create(const ASTContext &Ctx, Expr *Fn,
                                               ArrayRef<Expr *> Args,
                                               QualType Ty, ExprValueKind VK,
                                               SourceLocation LitEndLoc,
                                               SourceLocation SuffixLoc) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned NumArgs = Args.size();
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(UserDefinedLiteral) + SizeOfTrailingObjects,
                           alignof(UserDefinedLiteral));
  return new (Mem) UserDefinedLiteral(Fn, Args, Ty, VK, LitEndLoc, SuffixLoc);
}

UserDefinedLiteral *UserDefinedLiteral::CreateEmpty(const ASTContext &Ctx,
                                                    unsigned NumArgs,
                                                    EmptyShell Empty) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/0, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(UserDefinedLiteral) + SizeOfTrailingObjects,
                           alignof(UserDefinedLiteral));
  return new (Mem) UserDefinedLiteral(NumArgs, Empty);
}

UserDefinedLiteral::LiteralOperatorKind
UserDefinedLiteral::getLiteralOperatorKind() const {
  if (getNumArgs() == 0)
    return LOK_Template;
  if (getNumArgs() == 2)
    return LOK_String;

  assert(getNumArgs() == 1 && "unexpected #args in literal operator call");
  QualType ParamTy =
    cast<FunctionDecl>(getCalleeDecl())->getParamDecl(0)->getType();
  if (ParamTy->isPointerType())
    return LOK_Raw;
  if (ParamTy->isAnyCharacterType())
    return LOK_Character;
  if (ParamTy->isIntegerType())
    return LOK_Integer;
  if (ParamTy->isFloatingType())
    return LOK_Floating;

  llvm_unreachable("unknown kind of literal operator");
}

Expr *UserDefinedLiteral::getCookedLiteral() {
#ifndef NDEBUG
  LiteralOperatorKind LOK = getLiteralOperatorKind();
  assert(LOK != LOK_Template && LOK != LOK_Raw && "not a cooked literal");
#endif
  return getArg(0);
}

const IdentifierInfo *UserDefinedLiteral::getUDSuffix() const {
  return cast<FunctionDecl>(getCalleeDecl())->getLiteralIdentifier();
}

CXXDefaultInitExpr::CXXDefaultInitExpr(const ASTContext &Ctx, SourceLocation Loc,
                                       FieldDecl *Field, QualType Ty,
                                       DeclContext *UsedContext)
    : Expr(CXXDefaultInitExprClass, Ty.getNonLValueExprType(Ctx),
           Ty->isLValueReferenceType() ? VK_LValue : Ty->isRValueReferenceType()
                                                        ? VK_XValue
                                                        : VK_RValue,
           /*FIXME*/ OK_Ordinary, false, false, false, false),
      Field(Field), UsedContext(UsedContext) {
  CXXDefaultInitExprBits.Loc = Loc;
  assert(Field->hasInClassInitializer());
}

CXXTemporary *CXXTemporary::Create(const ASTContext &C,
                                   const CXXDestructorDecl *Destructor) {
  return new (C) CXXTemporary(Destructor);
}

CXXBindTemporaryExpr *CXXBindTemporaryExpr::Create(const ASTContext &C,
                                                   CXXTemporary *Temp,
                                                   Expr* SubExpr) {
  assert((SubExpr->getType()->isRecordType() ||
          SubExpr->getType()->isArrayType()) &&
         "Expression bound to a temporary must have record or array type!");

  return new (C) CXXBindTemporaryExpr(Temp, SubExpr);
}

CXXTemporaryObjectExpr::CXXTemporaryObjectExpr(
    CXXConstructorDecl *Cons, QualType Ty, TypeSourceInfo *TSI,
    ArrayRef<Expr *> Args, SourceRange ParenOrBraceRange,
    bool HadMultipleCandidates, bool ListInitialization,
    bool StdInitListInitialization, bool ZeroInitialization)
    : CXXConstructExpr(
          CXXTemporaryObjectExprClass, Ty, TSI->getTypeLoc().getBeginLoc(),
          Cons, /* Elidable=*/false, Args, HadMultipleCandidates,
          ListInitialization, StdInitListInitialization, ZeroInitialization,
          CXXConstructExpr::CK_Complete, ParenOrBraceRange),
      TSI(TSI) {}

CXXTemporaryObjectExpr::CXXTemporaryObjectExpr(EmptyShell Empty,
                                               unsigned NumArgs)
    : CXXConstructExpr(CXXTemporaryObjectExprClass, Empty, NumArgs) {}

CXXTemporaryObjectExpr *CXXTemporaryObjectExpr::Create(
    const ASTContext &Ctx, CXXConstructorDecl *Cons, QualType Ty,
    TypeSourceInfo *TSI, ArrayRef<Expr *> Args, SourceRange ParenOrBraceRange,
    bool HadMultipleCandidates, bool ListInitialization,
    bool StdInitListInitialization, bool ZeroInitialization) {
  unsigned SizeOfTrailingObjects = sizeOfTrailingObjects(Args.size());
  void *Mem =
      Ctx.Allocate(sizeof(CXXTemporaryObjectExpr) + SizeOfTrailingObjects,
                   alignof(CXXTemporaryObjectExpr));
  return new (Mem) CXXTemporaryObjectExpr(
      Cons, Ty, TSI, Args, ParenOrBraceRange, HadMultipleCandidates,
      ListInitialization, StdInitListInitialization, ZeroInitialization);
}

CXXTemporaryObjectExpr *
CXXTemporaryObjectExpr::CreateEmpty(const ASTContext &Ctx, unsigned NumArgs) {
  unsigned SizeOfTrailingObjects = sizeOfTrailingObjects(NumArgs);
  void *Mem =
      Ctx.Allocate(sizeof(CXXTemporaryObjectExpr) + SizeOfTrailingObjects,
                   alignof(CXXTemporaryObjectExpr));
  return new (Mem) CXXTemporaryObjectExpr(EmptyShell(), NumArgs);
}

SourceLocation CXXTemporaryObjectExpr::getBeginLoc() const {
  return getTypeSourceInfo()->getTypeLoc().getBeginLoc();
}

SourceLocation CXXTemporaryObjectExpr::getEndLoc() const {
  SourceLocation Loc = getParenOrBraceRange().getEnd();
  if (Loc.isInvalid() && getNumArgs())
    Loc = getArg(getNumArgs() - 1)->getEndLoc();
  return Loc;
}

CXXConstructExpr *CXXConstructExpr::Create(
    const ASTContext &Ctx, QualType Ty, SourceLocation Loc,
    CXXConstructorDecl *Ctor, bool Elidable, ArrayRef<Expr *> Args,
    bool HadMultipleCandidates, bool ListInitialization,
    bool StdInitListInitialization, bool ZeroInitialization,
    ConstructionKind ConstructKind, SourceRange ParenOrBraceRange) {
  unsigned SizeOfTrailingObjects = sizeOfTrailingObjects(Args.size());
  void *Mem = Ctx.Allocate(sizeof(CXXConstructExpr) + SizeOfTrailingObjects,
                           alignof(CXXConstructExpr));
  return new (Mem) CXXConstructExpr(
      CXXConstructExprClass, Ty, Loc, Ctor, Elidable, Args,
      HadMultipleCandidates, ListInitialization, StdInitListInitialization,
      ZeroInitialization, ConstructKind, ParenOrBraceRange);
}

CXXConstructExpr *CXXConstructExpr::CreateEmpty(const ASTContext &Ctx,
                                                unsigned NumArgs) {
  unsigned SizeOfTrailingObjects = sizeOfTrailingObjects(NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CXXConstructExpr) + SizeOfTrailingObjects,
                           alignof(CXXConstructExpr));
  return new (Mem)
      CXXConstructExpr(CXXConstructExprClass, EmptyShell(), NumArgs);
}

CXXConstructExpr::CXXConstructExpr(
    StmtClass SC, QualType Ty, SourceLocation Loc, CXXConstructorDecl *Ctor,
    bool Elidable, ArrayRef<Expr *> Args, bool HadMultipleCandidates,
    bool ListInitialization, bool StdInitListInitialization,
    bool ZeroInitialization, ConstructionKind ConstructKind,
    SourceRange ParenOrBraceRange)
    : Expr(SC, Ty, VK_RValue, OK_Ordinary, Ty->isDependentType(),
           Ty->isDependentType(), Ty->isInstantiationDependentType(),
           Ty->containsUnexpandedParameterPack()),
      Constructor(Ctor), ParenOrBraceRange(ParenOrBraceRange),
      NumArgs(Args.size()) {
  CXXConstructExprBits.Elidable = Elidable;
  CXXConstructExprBits.HadMultipleCandidates = HadMultipleCandidates;
  CXXConstructExprBits.ListInitialization = ListInitialization;
  CXXConstructExprBits.StdInitListInitialization = StdInitListInitialization;
  CXXConstructExprBits.ZeroInitialization = ZeroInitialization;
  CXXConstructExprBits.ConstructionKind = ConstructKind;
  CXXConstructExprBits.Loc = Loc;

  Stmt **TrailingArgs = getTrailingArgs();
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    assert(Args[I] && "NULL argument in CXXConstructExpr!");

    if (Args[I]->isValueDependent())
      ExprBits.ValueDependent = true;
    if (Args[I]->isInstantiationDependent())
      ExprBits.InstantiationDependent = true;
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    TrailingArgs[I] = Args[I];
  }
}

CXXConstructExpr::CXXConstructExpr(StmtClass SC, EmptyShell Empty,
                                   unsigned NumArgs)
    : Expr(SC, Empty), NumArgs(NumArgs) {}

LambdaCapture::LambdaCapture(SourceLocation Loc, bool Implicit,
                             LambdaCaptureKind Kind, VarDecl *Var,
                             SourceLocation EllipsisLoc)
    : DeclAndBits(Var, 0), Loc(Loc), EllipsisLoc(EllipsisLoc) {
  unsigned Bits = 0;
  if (Implicit)
    Bits |= Capture_Implicit;

  switch (Kind) {
  case LCK_StarThis:
    Bits |= Capture_ByCopy;
    LLVM_FALLTHROUGH;
  case LCK_This:
    assert(!Var && "'this' capture cannot have a variable!");
    Bits |= Capture_This;
    break;

  case LCK_ByCopy:
    Bits |= Capture_ByCopy;
    LLVM_FALLTHROUGH;
  case LCK_ByRef:
    assert(Var && "capture must have a variable!");
    break;
  case LCK_VLAType:
    assert(!Var && "VLA type capture cannot have a variable!");
    break;
  }
  DeclAndBits.setInt(Bits);
}

LambdaCaptureKind LambdaCapture::getCaptureKind() const {
  if (capturesVLAType())
    return LCK_VLAType;
  bool CapByCopy = DeclAndBits.getInt() & Capture_ByCopy;
  if (capturesThis())
    return CapByCopy ? LCK_StarThis : LCK_This;
  return CapByCopy ? LCK_ByCopy : LCK_ByRef;
}

LambdaExpr::LambdaExpr(QualType T, SourceRange IntroducerRange,
                       LambdaCaptureDefault CaptureDefault,
                       SourceLocation CaptureDefaultLoc,
                       ArrayRef<LambdaCapture> Captures, bool ExplicitParams,
                       bool ExplicitResultType, ArrayRef<Expr *> CaptureInits,
                       SourceLocation ClosingBrace,
                       bool ContainsUnexpandedParameterPack)
    : Expr(LambdaExprClass, T, VK_RValue, OK_Ordinary, T->isDependentType(),
           T->isDependentType(), T->isDependentType(),
           ContainsUnexpandedParameterPack),
      IntroducerRange(IntroducerRange), CaptureDefaultLoc(CaptureDefaultLoc),
      NumCaptures(Captures.size()), CaptureDefault(CaptureDefault),
      ExplicitParams(ExplicitParams), ExplicitResultType(ExplicitResultType),
      ClosingBrace(ClosingBrace) {
  assert(CaptureInits.size() == Captures.size() && "Wrong number of arguments");
  CXXRecordDecl *Class = getLambdaClass();
  CXXRecordDecl::LambdaDefinitionData &Data = Class->getLambdaData();

  // FIXME: Propagate "has unexpanded parameter pack" bit.

  // Copy captures.
  const ASTContext &Context = Class->getASTContext();
  Data.NumCaptures = NumCaptures;
  Data.NumExplicitCaptures = 0;
  Data.Captures =
      (LambdaCapture *)Context.Allocate(sizeof(LambdaCapture) * NumCaptures);
  LambdaCapture *ToCapture = Data.Captures;
  for (unsigned I = 0, N = Captures.size(); I != N; ++I) {
    if (Captures[I].isExplicit())
      ++Data.NumExplicitCaptures;

    *ToCapture++ = Captures[I];
  }

  // Copy initialization expressions for the non-static data members.
  Stmt **Stored = getStoredStmts();
  for (unsigned I = 0, N = CaptureInits.size(); I != N; ++I)
    *Stored++ = CaptureInits[I];

  // Copy the body of the lambda.
  *Stored++ = getCallOperator()->getBody();
}

LambdaExpr *LambdaExpr::Create(
    const ASTContext &Context, CXXRecordDecl *Class,
    SourceRange IntroducerRange, LambdaCaptureDefault CaptureDefault,
    SourceLocation CaptureDefaultLoc, ArrayRef<LambdaCapture> Captures,
    bool ExplicitParams, bool ExplicitResultType, ArrayRef<Expr *> CaptureInits,
    SourceLocation ClosingBrace, bool ContainsUnexpandedParameterPack) {
  // Determine the type of the expression (i.e., the type of the
  // function object we're creating).
  QualType T = Context.getTypeDeclType(Class);

  unsigned Size = totalSizeToAlloc<Stmt *>(Captures.size() + 1);
  void *Mem = Context.Allocate(Size);
  return new (Mem)
      LambdaExpr(T, IntroducerRange, CaptureDefault, CaptureDefaultLoc,
                 Captures, ExplicitParams, ExplicitResultType, CaptureInits,
                 ClosingBrace, ContainsUnexpandedParameterPack);
}

LambdaExpr *LambdaExpr::CreateDeserialized(const ASTContext &C,
                                           unsigned NumCaptures) {
  unsigned Size = totalSizeToAlloc<Stmt *>(NumCaptures + 1);
  void *Mem = C.Allocate(Size);
  return new (Mem) LambdaExpr(EmptyShell(), NumCaptures);
}

bool LambdaExpr::isInitCapture(const LambdaCapture *C) const {
  return (C->capturesVariable() && C->getCapturedVar()->isInitCapture() &&
          (getCallOperator() == C->getCapturedVar()->getDeclContext()));
}

LambdaExpr::capture_iterator LambdaExpr::capture_begin() const {
  return getLambdaClass()->getLambdaData().Captures;
}

LambdaExpr::capture_iterator LambdaExpr::capture_end() const {
  return capture_begin() + NumCaptures;
}

LambdaExpr::capture_range LambdaExpr::captures() const {
  return capture_range(capture_begin(), capture_end());
}

LambdaExpr::capture_iterator LambdaExpr::explicit_capture_begin() const {
  return capture_begin();
}

LambdaExpr::capture_iterator LambdaExpr::explicit_capture_end() const {
  struct CXXRecordDecl::LambdaDefinitionData &Data
    = getLambdaClass()->getLambdaData();
  return Data.Captures + Data.NumExplicitCaptures;
}

LambdaExpr::capture_range LambdaExpr::explicit_captures() const {
  return capture_range(explicit_capture_begin(), explicit_capture_end());
}

LambdaExpr::capture_iterator LambdaExpr::implicit_capture_begin() const {
  return explicit_capture_end();
}

LambdaExpr::capture_iterator LambdaExpr::implicit_capture_end() const {
  return capture_end();
}

LambdaExpr::capture_range LambdaExpr::implicit_captures() const {
  return capture_range(implicit_capture_begin(), implicit_capture_end());
}

CXXRecordDecl *LambdaExpr::getLambdaClass() const {
  return getType()->getAsCXXRecordDecl();
}

CXXMethodDecl *LambdaExpr::getCallOperator() const {
  CXXRecordDecl *Record = getLambdaClass();
  return Record->getLambdaCallOperator();
}

FunctionTemplateDecl *LambdaExpr::getDependentCallOperator() const {
  CXXRecordDecl *Record = getLambdaClass();
  return Record->getDependentLambdaCallOperator();
}

TemplateParameterList *LambdaExpr::getTemplateParameterList() const {
  CXXRecordDecl *Record = getLambdaClass();
  return Record->getGenericLambdaTemplateParameterList();
}

ArrayRef<NamedDecl *> LambdaExpr::getExplicitTemplateParameters() const {
  const CXXRecordDecl *Record = getLambdaClass();
  return Record->getLambdaExplicitTemplateParameters();
}

CompoundStmt *LambdaExpr::getBody() const {
  // FIXME: this mutation in getBody is bogus. It should be
  // initialized in ASTStmtReader::VisitLambdaExpr, but for reasons I
  // don't understand, that doesn't work.
  if (!getStoredStmts()[NumCaptures])
    *const_cast<Stmt **>(&getStoredStmts()[NumCaptures]) =
        getCallOperator()->getBody();

  return static_cast<CompoundStmt *>(getStoredStmts()[NumCaptures]);
}

bool LambdaExpr::isMutable() const {
  return !getCallOperator()->isConst();
}

ExprWithCleanups::ExprWithCleanups(Expr *subexpr,
                                   bool CleanupsHaveSideEffects,
                                   ArrayRef<CleanupObject> objects)
    : FullExpr(ExprWithCleanupsClass, subexpr) {
  ExprWithCleanupsBits.CleanupsHaveSideEffects = CleanupsHaveSideEffects;
  ExprWithCleanupsBits.NumObjects = objects.size();
  for (unsigned i = 0, e = objects.size(); i != e; ++i)
    getTrailingObjects<CleanupObject>()[i] = objects[i];
}

ExprWithCleanups *ExprWithCleanups::Create(const ASTContext &C, Expr *subexpr,
                                           bool CleanupsHaveSideEffects,
                                           ArrayRef<CleanupObject> objects) {
  void *buffer = C.Allocate(totalSizeToAlloc<CleanupObject>(objects.size()),
                            alignof(ExprWithCleanups));
  return new (buffer)
      ExprWithCleanups(subexpr, CleanupsHaveSideEffects, objects);
}

ExprWithCleanups::ExprWithCleanups(EmptyShell empty, unsigned numObjects)
    : FullExpr(ExprWithCleanupsClass, empty) {
  ExprWithCleanupsBits.NumObjects = numObjects;
}

ExprWithCleanups *ExprWithCleanups::Create(const ASTContext &C,
                                           EmptyShell empty,
                                           unsigned numObjects) {
  void *buffer = C.Allocate(totalSizeToAlloc<CleanupObject>(numObjects),
                            alignof(ExprWithCleanups));
  return new (buffer) ExprWithCleanups(empty, numObjects);
}

CXXUnresolvedConstructExpr::CXXUnresolvedConstructExpr(TypeSourceInfo *TSI,
                                                       SourceLocation LParenLoc,
                                                       ArrayRef<Expr *> Args,
                                                       SourceLocation RParenLoc)
    : Expr(CXXUnresolvedConstructExprClass,
           TSI->getType().getNonReferenceType(),
           (TSI->getType()->isLValueReferenceType()
                ? VK_LValue
                : TSI->getType()->isRValueReferenceType() ? VK_XValue
                                                          : VK_RValue),
           OK_Ordinary,
           TSI->getType()->isDependentType() ||
               TSI->getType()->getContainedDeducedType(),
           true, true, TSI->getType()->containsUnexpandedParameterPack()),
      TSI(TSI), LParenLoc(LParenLoc), RParenLoc(RParenLoc) {
  CXXUnresolvedConstructExprBits.NumArgs = Args.size();
  auto **StoredArgs = getTrailingObjects<Expr *>();
  for (unsigned I = 0; I != Args.size(); ++I) {
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    StoredArgs[I] = Args[I];
  }
}

CXXUnresolvedConstructExpr *CXXUnresolvedConstructExpr::Create(
    const ASTContext &Context, TypeSourceInfo *TSI, SourceLocation LParenLoc,
    ArrayRef<Expr *> Args, SourceLocation RParenLoc) {
  void *Mem = Context.Allocate(totalSizeToAlloc<Expr *>(Args.size()));
  return new (Mem) CXXUnresolvedConstructExpr(TSI, LParenLoc, Args, RParenLoc);
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::CreateEmpty(const ASTContext &Context,
                                        unsigned NumArgs) {
  void *Mem = Context.Allocate(totalSizeToAlloc<Expr *>(NumArgs));
  return new (Mem) CXXUnresolvedConstructExpr(EmptyShell(), NumArgs);
}

SourceLocation CXXUnresolvedConstructExpr::getBeginLoc() const {
  return TSI->getTypeLoc().getBeginLoc();
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(
    const ASTContext &Ctx, Expr *Base, QualType BaseType, bool IsArrow,
    SourceLocation OperatorLoc, NestedNameSpecifierLoc QualifierLoc,
    SourceLocation TemplateKWLoc, NamedDecl *FirstQualifierFoundInScope,
    DeclarationNameInfo MemberNameInfo,
    const TemplateArgumentListInfo *TemplateArgs)
    : Expr(CXXDependentScopeMemberExprClass, Ctx.DependentTy, VK_LValue,
           OK_Ordinary, true, true, true,
           ((Base && Base->containsUnexpandedParameterPack()) ||
            (QualifierLoc && QualifierLoc.getNestedNameSpecifier()
                                 ->containsUnexpandedParameterPack()) ||
            MemberNameInfo.containsUnexpandedParameterPack())),
      Base(Base), BaseType(BaseType), QualifierLoc(QualifierLoc),
      MemberNameInfo(MemberNameInfo) {
  CXXDependentScopeMemberExprBits.IsArrow = IsArrow;
  CXXDependentScopeMemberExprBits.HasTemplateKWAndArgsInfo =
      (TemplateArgs != nullptr) || TemplateKWLoc.isValid();
  CXXDependentScopeMemberExprBits.HasFirstQualifierFoundInScope =
      FirstQualifierFoundInScope != nullptr;
  CXXDependentScopeMemberExprBits.OperatorLoc = OperatorLoc;

  if (TemplateArgs) {
    bool Dependent = true;
    bool InstantiationDependent = true;
    bool ContainsUnexpandedParameterPack = false;
    getTrailingObjects<ASTTemplateKWAndArgsInfo>()->initializeFrom(
        TemplateKWLoc, *TemplateArgs, getTrailingObjects<TemplateArgumentLoc>(),
        Dependent, InstantiationDependent, ContainsUnexpandedParameterPack);
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  } else if (TemplateKWLoc.isValid()) {
    getTrailingObjects<ASTTemplateKWAndArgsInfo>()->initializeFrom(
        TemplateKWLoc);
  }

  if (hasFirstQualifierFoundInScope())
    *getTrailingObjects<NamedDecl *>() = FirstQualifierFoundInScope;
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(
    EmptyShell Empty, bool HasTemplateKWAndArgsInfo,
    bool HasFirstQualifierFoundInScope)
    : Expr(CXXDependentScopeMemberExprClass, Empty) {
  CXXDependentScopeMemberExprBits.HasTemplateKWAndArgsInfo =
      HasTemplateKWAndArgsInfo;
  CXXDependentScopeMemberExprBits.HasFirstQualifierFoundInScope =
      HasFirstQualifierFoundInScope;
}

CXXDependentScopeMemberExpr *CXXDependentScopeMemberExpr::Create(
    const ASTContext &Ctx, Expr *Base, QualType BaseType, bool IsArrow,
    SourceLocation OperatorLoc, NestedNameSpecifierLoc QualifierLoc,
    SourceLocation TemplateKWLoc, NamedDecl *FirstQualifierFoundInScope,
    DeclarationNameInfo MemberNameInfo,
    const TemplateArgumentListInfo *TemplateArgs) {
  bool HasTemplateKWAndArgsInfo =
      (TemplateArgs != nullptr) || TemplateKWLoc.isValid();
  unsigned NumTemplateArgs = TemplateArgs ? TemplateArgs->size() : 0;
  bool HasFirstQualifierFoundInScope = FirstQualifierFoundInScope != nullptr;

  unsigned Size = totalSizeToAlloc<ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc, NamedDecl *>(
      HasTemplateKWAndArgsInfo, NumTemplateArgs, HasFirstQualifierFoundInScope);

  void *Mem = Ctx.Allocate(Size, alignof(CXXDependentScopeMemberExpr));
  return new (Mem) CXXDependentScopeMemberExpr(
      Ctx, Base, BaseType, IsArrow, OperatorLoc, QualifierLoc, TemplateKWLoc,
      FirstQualifierFoundInScope, MemberNameInfo, TemplateArgs);
}

CXXDependentScopeMemberExpr *CXXDependentScopeMemberExpr::CreateEmpty(
    const ASTContext &Ctx, bool HasTemplateKWAndArgsInfo,
    unsigned NumTemplateArgs, bool HasFirstQualifierFoundInScope) {
  assert(NumTemplateArgs == 0 || HasTemplateKWAndArgsInfo);

  unsigned Size = totalSizeToAlloc<ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc, NamedDecl *>(
      HasTemplateKWAndArgsInfo, NumTemplateArgs, HasFirstQualifierFoundInScope);

  void *Mem = Ctx.Allocate(Size, alignof(CXXDependentScopeMemberExpr));
  return new (Mem) CXXDependentScopeMemberExpr(
      EmptyShell(), HasTemplateKWAndArgsInfo, HasFirstQualifierFoundInScope);
}

static bool hasOnlyNonStaticMemberFunctions(UnresolvedSetIterator begin,
                                            UnresolvedSetIterator end) {
  do {
    NamedDecl *decl = *begin;
    if (isa<UnresolvedUsingValueDecl>(decl))
      return false;

    // Unresolved member expressions should only contain methods and
    // method templates.
    if (cast<CXXMethodDecl>(decl->getUnderlyingDecl()->getAsFunction())
            ->isStatic())
      return false;
  } while (++begin != end);

  return true;
}

UnresolvedMemberExpr::UnresolvedMemberExpr(
    const ASTContext &Context, bool HasUnresolvedUsing, Expr *Base,
    QualType BaseType, bool IsArrow, SourceLocation OperatorLoc,
    NestedNameSpecifierLoc QualifierLoc, SourceLocation TemplateKWLoc,
    const DeclarationNameInfo &MemberNameInfo,
    const TemplateArgumentListInfo *TemplateArgs, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End)
    : OverloadExpr(
          UnresolvedMemberExprClass, Context, QualifierLoc, TemplateKWLoc,
          MemberNameInfo, TemplateArgs, Begin, End,
          // Dependent
          ((Base && Base->isTypeDependent()) || BaseType->isDependentType()),
          ((Base && Base->isInstantiationDependent()) ||
           BaseType->isInstantiationDependentType()),
          // Contains unexpanded parameter pack
          ((Base && Base->containsUnexpandedParameterPack()) ||
           BaseType->containsUnexpandedParameterPack())),
      Base(Base), BaseType(BaseType), OperatorLoc(OperatorLoc) {
  UnresolvedMemberExprBits.IsArrow = IsArrow;
  UnresolvedMemberExprBits.HasUnresolvedUsing = HasUnresolvedUsing;

  // Check whether all of the members are non-static member functions,
  // and if so, mark give this bound-member type instead of overload type.
  if (hasOnlyNonStaticMemberFunctions(Begin, End))
    setType(Context.BoundMemberTy);
}

UnresolvedMemberExpr::UnresolvedMemberExpr(EmptyShell Empty,
                                           unsigned NumResults,
                                           bool HasTemplateKWAndArgsInfo)
    : OverloadExpr(UnresolvedMemberExprClass, Empty, NumResults,
                   HasTemplateKWAndArgsInfo) {}

bool UnresolvedMemberExpr::isImplicitAccess() const {
  if (!Base)
    return true;

  return cast<Expr>(Base)->isImplicitCXXThis();
}

UnresolvedMemberExpr *UnresolvedMemberExpr::Create(
    const ASTContext &Context, bool HasUnresolvedUsing, Expr *Base,
    QualType BaseType, bool IsArrow, SourceLocation OperatorLoc,
    NestedNameSpecifierLoc QualifierLoc, SourceLocation TemplateKWLoc,
    const DeclarationNameInfo &MemberNameInfo,
    const TemplateArgumentListInfo *TemplateArgs, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End) {
  unsigned NumResults = End - Begin;
  bool HasTemplateKWAndArgsInfo = TemplateArgs || TemplateKWLoc.isValid();
  unsigned NumTemplateArgs = TemplateArgs ? TemplateArgs->size() : 0;
  unsigned Size = totalSizeToAlloc<DeclAccessPair, ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc>(
      NumResults, HasTemplateKWAndArgsInfo, NumTemplateArgs);
  void *Mem = Context.Allocate(Size, alignof(UnresolvedMemberExpr));
  return new (Mem) UnresolvedMemberExpr(
      Context, HasUnresolvedUsing, Base, BaseType, IsArrow, OperatorLoc,
      QualifierLoc, TemplateKWLoc, MemberNameInfo, TemplateArgs, Begin, End);
}

UnresolvedMemberExpr *UnresolvedMemberExpr::CreateEmpty(
    const ASTContext &Context, unsigned NumResults,
    bool HasTemplateKWAndArgsInfo, unsigned NumTemplateArgs) {
  assert(NumTemplateArgs == 0 || HasTemplateKWAndArgsInfo);
  unsigned Size = totalSizeToAlloc<DeclAccessPair, ASTTemplateKWAndArgsInfo,
                                   TemplateArgumentLoc>(
      NumResults, HasTemplateKWAndArgsInfo, NumTemplateArgs);
  void *Mem = Context.Allocate(Size, alignof(UnresolvedMemberExpr));
  return new (Mem)
      UnresolvedMemberExpr(EmptyShell(), NumResults, HasTemplateKWAndArgsInfo);
}

CXXRecordDecl *UnresolvedMemberExpr::getNamingClass() {
  // Unlike for UnresolvedLookupExpr, it is very easy to re-derive this.

  // If there was a nested name specifier, it names the naming class.
  // It can't be dependent: after all, we were actually able to do the
  // lookup.
  CXXRecordDecl *Record = nullptr;
  auto *NNS = getQualifier();
  if (NNS && NNS->getKind() != NestedNameSpecifier::Super) {
    const Type *T = getQualifier()->getAsType();
    assert(T && "qualifier in member expression does not name type");
    Record = T->getAsCXXRecordDecl();
    assert(Record && "qualifier in member expression does not name record");
  }
  // Otherwise the naming class must have been the base class.
  else {
    QualType BaseType = getBaseType().getNonReferenceType();
    if (isArrow())
      BaseType = BaseType->castAs<PointerType>()->getPointeeType();

    Record = BaseType->getAsCXXRecordDecl();
    assert(Record && "base of member expression does not name record");
  }

  return Record;
}

SizeOfPackExpr *
SizeOfPackExpr::Create(ASTContext &Context, SourceLocation OperatorLoc,
                       NamedDecl *Pack, SourceLocation PackLoc,
                       SourceLocation RParenLoc,
                       Optional<unsigned> Length,
                       ArrayRef<TemplateArgument> PartialArgs) {
  void *Storage =
      Context.Allocate(totalSizeToAlloc<TemplateArgument>(PartialArgs.size()));
  return new (Storage) SizeOfPackExpr(Context.getSizeType(), OperatorLoc, Pack,
                                      PackLoc, RParenLoc, Length, PartialArgs);
}

SizeOfPackExpr *SizeOfPackExpr::CreateDeserialized(ASTContext &Context,
                                                   unsigned NumPartialArgs) {
  void *Storage =
      Context.Allocate(totalSizeToAlloc<TemplateArgument>(NumPartialArgs));
  return new (Storage) SizeOfPackExpr(EmptyShell(), NumPartialArgs);
}

SubstNonTypeTemplateParmPackExpr::
SubstNonTypeTemplateParmPackExpr(QualType T,
                                 ExprValueKind ValueKind,
                                 NonTypeTemplateParmDecl *Param,
                                 SourceLocation NameLoc,
                                 const TemplateArgument &ArgPack)
    : Expr(SubstNonTypeTemplateParmPackExprClass, T, ValueKind, OK_Ordinary,
           true, true, true, true),
      Param(Param), Arguments(ArgPack.pack_begin()),
      NumArguments(ArgPack.pack_size()), NameLoc(NameLoc) {}

TemplateArgument SubstNonTypeTemplateParmPackExpr::getArgumentPack() const {
  return TemplateArgument(llvm::makeArrayRef(Arguments, NumArguments));
}

FunctionParmPackExpr::FunctionParmPackExpr(QualType T, VarDecl *ParamPack,
                                           SourceLocation NameLoc,
                                           unsigned NumParams,
                                           VarDecl *const *Params)
    : Expr(FunctionParmPackExprClass, T, VK_LValue, OK_Ordinary, true, true,
           true, true),
      ParamPack(ParamPack), NameLoc(NameLoc), NumParameters(NumParams) {
  if (Params)
    std::uninitialized_copy(Params, Params + NumParams,
                            getTrailingObjects<VarDecl *>());
}

FunctionParmPackExpr *
FunctionParmPackExpr::Create(const ASTContext &Context, QualType T,
                             VarDecl *ParamPack, SourceLocation NameLoc,
                             ArrayRef<VarDecl *> Params) {
  return new (Context.Allocate(totalSizeToAlloc<VarDecl *>(Params.size())))
      FunctionParmPackExpr(T, ParamPack, NameLoc, Params.size(), Params.data());
}

FunctionParmPackExpr *
FunctionParmPackExpr::CreateEmpty(const ASTContext &Context,
                                  unsigned NumParams) {
  return new (Context.Allocate(totalSizeToAlloc<VarDecl *>(NumParams)))
      FunctionParmPackExpr(QualType(), nullptr, SourceLocation(), 0, nullptr);
}

MaterializeTemporaryExpr::MaterializeTemporaryExpr(
    QualType T, Expr *Temporary, bool BoundToLvalueReference,
    LifetimeExtendedTemporaryDecl *MTD)
    : Expr(MaterializeTemporaryExprClass, T,
           BoundToLvalueReference ? VK_LValue : VK_XValue, OK_Ordinary,
           Temporary->isTypeDependent(), Temporary->isValueDependent(),
           Temporary->isInstantiationDependent(),
           Temporary->containsUnexpandedParameterPack()) {
  if (MTD) {
    State = MTD;
    MTD->ExprWithTemporary = Temporary;
    return;
  }
  State = Temporary;
}

void MaterializeTemporaryExpr::setExtendingDecl(ValueDecl *ExtendedBy,
                                                unsigned ManglingNumber) {
  // We only need extra state if we have to remember more than just the Stmt.
  if (!ExtendedBy)
    return;

  // We may need to allocate extra storage for the mangling number and the
  // extended-by ValueDecl.
  if (!State.is<LifetimeExtendedTemporaryDecl *>())
    State = LifetimeExtendedTemporaryDecl::Create(
        cast<Expr>(State.get<Stmt *>()), ExtendedBy, ManglingNumber);

  auto ES = State.get<LifetimeExtendedTemporaryDecl *>();
  ES->ExtendingDecl = ExtendedBy;
  ES->ManglingNumber = ManglingNumber;
}

TypeTraitExpr::TypeTraitExpr(QualType T, SourceLocation Loc, TypeTrait Kind,
                             ArrayRef<TypeSourceInfo *> Args,
                             SourceLocation RParenLoc,
                             bool Value)
    : Expr(TypeTraitExprClass, T, VK_RValue, OK_Ordinary,
           /*TypeDependent=*/false,
           /*ValueDependent=*/false,
           /*InstantiationDependent=*/false,
           /*ContainsUnexpandedParameterPack=*/false),
      Loc(Loc), RParenLoc(RParenLoc) {
  TypeTraitExprBits.Kind = Kind;
  TypeTraitExprBits.Value = Value;
  TypeTraitExprBits.NumArgs = Args.size();

  auto **ToArgs = getTrailingObjects<TypeSourceInfo *>();

  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    if (Args[I]->getType()->isDependentType())
      setValueDependent(true);
    if (Args[I]->getType()->isInstantiationDependentType())
      setInstantiationDependent(true);
    if (Args[I]->getType()->containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack(true);

    ToArgs[I] = Args[I];
  }
}

TypeTraitExpr *TypeTraitExpr::Create(const ASTContext &C, QualType T,
                                     SourceLocation Loc,
                                     TypeTrait Kind,
                                     ArrayRef<TypeSourceInfo *> Args,
                                     SourceLocation RParenLoc,
                                     bool Value) {
  void *Mem = C.Allocate(totalSizeToAlloc<TypeSourceInfo *>(Args.size()));
  return new (Mem) TypeTraitExpr(T, Loc, Kind, Args, RParenLoc, Value);
}

TypeTraitExpr *TypeTraitExpr::CreateDeserialized(const ASTContext &C,
                                                 unsigned NumArgs) {
  void *Mem = C.Allocate(totalSizeToAlloc<TypeSourceInfo *>(NumArgs));
  return new (Mem) TypeTraitExpr(EmptyShell());
}

CUDAKernelCallExpr::CUDAKernelCallExpr(Expr *Fn, CallExpr *Config,
                                       ArrayRef<Expr *> Args, QualType Ty,
                                       ExprValueKind VK, SourceLocation RP,
                                       unsigned MinNumArgs)
    : CallExpr(CUDAKernelCallExprClass, Fn, /*PreArgs=*/Config, Args, Ty, VK,
               RP, MinNumArgs, NotADL) {}

CUDAKernelCallExpr::CUDAKernelCallExpr(unsigned NumArgs, EmptyShell Empty)
    : CallExpr(CUDAKernelCallExprClass, /*NumPreArgs=*/END_PREARG, NumArgs,
               Empty) {}

CUDAKernelCallExpr *
CUDAKernelCallExpr::Create(const ASTContext &Ctx, Expr *Fn, CallExpr *Config,
                           ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK,
                           SourceLocation RP, unsigned MinNumArgs) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned NumArgs = std::max<unsigned>(Args.size(), MinNumArgs);
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/END_PREARG, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CUDAKernelCallExpr) + SizeOfTrailingObjects,
                           alignof(CUDAKernelCallExpr));
  return new (Mem) CUDAKernelCallExpr(Fn, Config, Args, Ty, VK, RP, MinNumArgs);
}

CUDAKernelCallExpr *CUDAKernelCallExpr::CreateEmpty(const ASTContext &Ctx,
                                                    unsigned NumArgs,
                                                    EmptyShell Empty) {
  // Allocate storage for the trailing objects of CallExpr.
  unsigned SizeOfTrailingObjects =
      CallExpr::sizeOfTrailingObjects(/*NumPreArgs=*/END_PREARG, NumArgs);
  void *Mem = Ctx.Allocate(sizeof(CUDAKernelCallExpr) + SizeOfTrailingObjects,
                           alignof(CUDAKernelCallExpr));
  return new (Mem) CUDAKernelCallExpr(NumArgs, Empty);
}
