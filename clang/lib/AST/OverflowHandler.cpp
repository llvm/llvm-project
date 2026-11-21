//===--- OverflowHandler.cpp - Overflow behavior type handler labels ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OverflowHandler.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/AST/Type.h"

using namespace clang;

IdentifierInfo *clang::getOverflowHandlerName(ASTContext &Ctx) {
  return &Ctx.Idents.get(Ctx.getLangOpts().OverflowBehaviorLabel);
}

static unsigned getCheckedIntWidth(ASTContext &Ctx, QualType T) {
  if (const auto *OBT = T->getAs<OverflowBehaviorType>())
    T = OBT->getUnderlyingType();
  return Ctx.getIntWidth(T);
}

bool clang::isOverflowHandlerCheckedConversion(ASTContext &Ctx, QualType Src,
                                               QualType Dst) {
  if (!Ctx.hasTrapOverflowBehaviorType())
    return false;
  if (!Src->isIntegerType() || !Dst->isIntegerType() || Dst->isBooleanType())
    return false;
  // Mirrors CodeGen: a wrap type on either side suppresses the check, and a
  // trap type on either side forces it.
  if (Src.isWrapType() || Dst.isWrapType())
    return false;
  if (!Src.isTrapType() && !Dst.isTrapType())
    return false;

  unsigned SrcBits = getCheckedIntWidth(Ctx, Src);
  unsigned DstBits = getCheckedIntWidth(Ctx, Dst);
  bool SrcSigned = Src->isSignedIntegerOrEnumerationType();
  bool DstSigned = Dst->isSignedIntegerOrEnumerationType();

  // Truncation check.
  if (SrcBits > DstBits)
    return true;
  // Sign change check.
  if (SrcSigned == DstSigned)
    return false;
  return !(DstBits > SrcBits && DstSigned);
}

bool clang::isOverflowHandlerCheckedOp(ASTContext &Ctx, const Expr *E) {
  if (!Ctx.hasTrapOverflowBehaviorType() || E->isTypeDependent() ||
      E->isValueDependent())
    return false;

  if (const auto *CAO = dyn_cast<CompoundAssignOperator>(E)) {
    // Any compound assignment converts its LHS to the computation type and
    // the result back to the LHS type; either conversion may be checked.
    QualType LHSTy = CAO->getLHS()->getType();
    if (isOverflowHandlerCheckedConversion(Ctx, LHSTy,
                                           CAO->getComputationLHSType()) ||
        isOverflowHandlerCheckedConversion(Ctx, CAO->getComputationResultType(),
                                           LHSTy))
      return true;
  }

  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {
    case BO_Add:
    case BO_Sub:
    case BO_Mul:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_MulAssign:
      break;
    default:
      return false;
    }
    if (BO->hasExcludedOverflowPattern())
      return false;
    QualType Ty = BO->getType();
    if (const auto *CAO = dyn_cast<CompoundAssignOperator>(BO))
      Ty = CAO->getComputationResultType();
    return Ty->isIntegerType() && Ty.isTrapType();
  }

  if (const auto *CE = dyn_cast<CastExpr>(E)) {
    if (CE->getCastKind() != CK_IntegralCast)
      return false;
    return isOverflowHandlerCheckedConversion(Ctx, CE->getSubExpr()->getType(),
                                              CE->getType());
  }

  if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    case UO_PreInc:
    case UO_PostInc:
    case UO_PreDec:
    case UO_PostDec:
    case UO_Minus:
      break;
    default:
      return false;
    }
    QualType Ty = UO->getType();
    if (!Ty->isIntegerType() || !Ty.isTrapType())
      return false;
    return !Ctx.isUnaryOverflowPatternExcluded(UO);
  }

  return false;
}

bool clang::containsOverflowHandlerCheckedOp(ASTContext &Ctx, const Stmt *S,
                                             bool SearchStmtExprs) {
  if (!Ctx.hasTrapOverflowBehaviorType())
    return false;
  SmallVector<const Stmt *, 16> Worklist = {S};
  while (!Worklist.empty()) {
    const Stmt *Cur = Worklist.pop_back_val();
    // Operations within a nested handler branch to it instead.
    if (getOverflowHandlerIntroducedBy(Ctx, Cur) ||
        (!SearchStmtExprs && isa<StmtExpr>(Cur)))
      continue;
    if (const auto *E = dyn_cast<Expr>(Cur);
        E && isOverflowHandlerCheckedOp(Ctx, E))
      return true;
    // A lambda body is a separate function with its own handler; only its
    // capture initializers belong here. Block bodies are not children.
    if (const auto *LE = dyn_cast<LambdaExpr>(Cur)) {
      for (const Expr *Init : LE->capture_inits())
        if (Init)
          Worklist.push_back(Init);
      continue;
    }
    if (const auto *DS = dyn_cast<DeclStmt>(Cur)) {
      for (const Decl *D : DS->decls())
        if (const auto *VD = dyn_cast<VarDecl>(D);
            VD && VD->getInit() && !initializerIgnoresOverflowHandler(VD))
          Worklist.push_back(VD->getInit());
      continue;
    }
    for (const Stmt *Child : Cur->children())
      if (Child && !isOverflowHandlerUnevaluatedChild(Ctx, Cur, Child) &&
          !childIgnoresOverflowHandler(Cur, Child))
        Worklist.push_back(Child);
  }
  return false;
}

LabelDecl *clang::findFunctionOverflowHandler(ASTContext &Ctx,
                                              const Decl *CodeDecl) {
  const auto *DC = dyn_cast_or_null<DeclContext>(CodeDecl);
  if (!DC || !Ctx.hasTrapOverflowBehaviorType())
    return nullptr;
  const IdentifierInfo *Name = getOverflowHandlerName(Ctx);
  // Labels live in the label namespace, which DeclContext::lookup does not
  // search, so scan the declarations directly.
  for (Decl *D : DC->decls()) {
    auto *LD = dyn_cast<LabelDecl>(D);
    if (LD && LD->getIdentifier() == Name && !LD->isGnuLocal() && LD->getStmt())
      return LD;
  }
  return nullptr;
}

static LabelDecl *findLocalOverflowHandler(ASTContext &Ctx,
                                           const CompoundStmt *CS) {
  const IdentifierInfo *Name = getOverflowHandlerName(Ctx);
  // GNU local label declarations must appear at the start of the block.
  for (const Stmt *S : CS->body()) {
    const auto *DS = dyn_cast<DeclStmt>(S);
    if (!DS)
      break;
    for (Decl *D : DS->decls()) {
      auto *LD = dyn_cast<LabelDecl>(D);
      if (LD && LD->getIdentifier() == Name && LD->isGnuLocal())
        return LD;
    }
  }
  return nullptr;
}

std::optional<LabelDecl *>
clang::getOverflowHandlerIntroducedBy(ASTContext &Ctx, const Stmt *S) {
  if (!Ctx.hasTrapOverflowBehaviorType())
    return std::nullopt;
  if (const auto *CS = dyn_cast<CompoundStmt>(S)) {
    if (LabelDecl *LD = findLocalOverflowHandler(Ctx, CS))
      return LD;
    return std::nullopt;
  }
  // Code that cannot be left by a jump, or that is emitted apart from the
  // function body: a captured statement such as an OpenMP region, a __finally
  // block, the body of a defer statement, and an OpenACC compute construct.
  if (isa<CapturedStmt, SEHFinallyStmt, DeferStmt, OpenACCComputeConstruct,
          OpenACCCombinedConstruct>(S))
    return nullptr;
  return std::nullopt;
}

bool clang::initializerIgnoresOverflowHandler(const VarDecl *VD) {
  return VD->isStaticLocal() && !VD->hasConstantInitialization();
}

bool clang::childIgnoresOverflowHandler(const Stmt *Parent, const Stmt *Child) {
  if (const auto *NE = dyn_cast<CXXNewExpr>(Parent))
    return Child == NE->getInitializer();
  if (const auto *ES = dyn_cast<SEHExceptStmt>(Parent))
    return Child == ES->getFilterExpr();
  return isa<CXXThrowExpr>(Parent);
}

bool clang::isOverflowHandlerUnevaluatedChild(ASTContext &Ctx,
                                              const Stmt *Parent,
                                              const Stmt *Child) {
  // Mirrors ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr.
  if (const auto *UE = dyn_cast<UnaryExprOrTypeTraitExpr>(Parent)) {
    UnaryExprOrTypeTrait K = UE->getKind();
    return !((K == UETT_SizeOf || K == UETT_DataSizeOf || K == UETT_CountOf) &&
             Ctx.getAsVariableArrayType(UE->getTypeOfArgument()));
  }
  if (const auto *GSE = dyn_cast<GenericSelectionExpr>(Parent))
    return !GSE->isResultDependent() && Child != GSE->getResultExpr();
  if (const auto *CE = dyn_cast<ChooseExpr>(Parent))
    return !CE->isConditionDependent() && Child != CE->getChosenSubExpr();
  if (const auto *TE = dyn_cast<CXXTypeidExpr>(Parent))
    return !TE->isPotentiallyEvaluated();
  return isa<CXXNoexceptExpr>(Parent);
}

OverflowHandlerScope::OverflowHandlerScope(
    SmallVectorImpl<LabelDecl *> &Handlers, ASTContext &Ctx, const Stmt *S) {
  if (std::optional<LabelDecl *> Handler =
          getOverflowHandlerIntroducedBy(Ctx, S)) {
    Handlers.push_back(*Handler);
    this->Handlers = &Handlers;
  }
}
