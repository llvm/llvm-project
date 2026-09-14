//===--- ExprUtils.cpp - Shared expression emission queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ExprUtils.h"
#include "clang/AST/Attr.h"

namespace clang::CodeGenUtils {

QualType getFixedSizeElementType(const ASTContext &Ctx,
                                 const VariableArrayType *VLA) {
  QualType EltType;
  do {
    EltType = VLA->getElementType();
  } while ((VLA = Ctx.getAsVariableArrayType(EltType)));
  return EltType;
}

bool isBlockVarRef(const Expr *E) {
  // Make sure we look through parens.
  E = E->IgnoreParens();

  // Check for a direct reference to a __block variable.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl());
    return (Var && Var->hasAttr<BlocksAttr>());
  }

  // More complicated stuff.

  // Binary operators.
  if (const BinaryOperator *Op = dyn_cast<BinaryOperator>(E)) {
    // For an assignment or pointer-to-member operation, just care
    // about the LHS.
    if (Op->isAssignmentOp() || Op->isPtrMemOp())
      return isBlockVarRef(Op->getLHS());

    // For a comma, just care about the RHS.
    if (Op->getOpcode() == BO_Comma)
      return isBlockVarRef(Op->getRHS());

    // FIXME: pointer arithmetic?
    return false;

    // Check both sides of a conditional operator.
  } else if (const AbstractConditionalOperator *Op =
                 dyn_cast<AbstractConditionalOperator>(E)) {
    return isBlockVarRef(Op->getTrueExpr()) ||
           isBlockVarRef(Op->getFalseExpr());

    // OVEs are required to support BinaryConditionalOperators.
  } else if (const OpaqueValueExpr *Op = dyn_cast<OpaqueValueExpr>(E)) {
    if (const Expr *Src = Op->getSourceExpr())
      return isBlockVarRef(Src);

    // Casts are necessary to get things like (*(int*)&var) = foo().
    // We don't really care about the kind of cast here, except
    // we don't want to look through l2r casts, because it's okay
    // to get the *value* in a __block variable.
  } else if (const CastExpr *Cast = dyn_cast<CastExpr>(E)) {
    if (Cast->getCastKind() == CK_LValueToRValue)
      return false;
    return isBlockVarRef(Cast->getSubExpr());

    // Handle unary operators.  Again, just aggressively look through
    // it, ignoring the operation.
  } else if (const UnaryOperator *UOp = dyn_cast<UnaryOperator>(E)) {
    return isBlockVarRef(UOp->getSubExpr());

    // Look into the base of a field access.
  } else if (const MemberExpr *Mem = dyn_cast<MemberExpr>(E)) {
    return isBlockVarRef(Mem->getBase());

    // Look into the base of a subscript.
  } else if (const ArraySubscriptExpr *Sub = dyn_cast<ArraySubscriptExpr>(E)) {
    return isBlockVarRef(Sub->getBase());
  }

  return false;
}

bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                            const ASTContext &Ctx) {
  // Anything that is an integer or floating point constant is fine.
  return E->IgnoreParens()->isEvaluatable(Ctx);

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}

bool isTrivialFiller(const Expr *E) {
  if (!E)
    return true;

  if (isa<ImplicitValueInitExpr>(E))
    return true;

  if (const auto *ILE = dyn_cast<InitListExpr>(E)) {
    if (ILE->getNumInits())
      return false;
    return isTrivialFiller(ILE->getArrayFiller());
  }

  if (const auto *Cons = dyn_cast_or_null<CXXConstructExpr>(E))
    return Cons->getConstructor()->isDefaultConstructor() &&
           Cons->getConstructor()->isTrivial();

  // FIXME: Are there other cases where we can avoid emitting an initializer?
  return false;
}

bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD) {
  for (const FunctionDecl *PD = FD; PD; PD = PD->getPreviousDecl())
    if (!PD->isInlineBuiltinDeclaration())
      return false;
  return true;
}

} // namespace clang::CodeGenUtils
