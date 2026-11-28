//===--- ExprUtils.cpp - Shared expression emission queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ExprUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ExprCXX.h"

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

bool castPreservesZero(const CastExpr *CE) {
  switch (CE->getCastKind()) {
    // No-ops.
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
  case CK_BitCast:
  case CK_ToUnion:
  case CK_ToVoid:
    // Conversions between (possibly-complex) integral, (possibly-complex)
    // floating-point, and bool.
  case CK_BooleanToSignedIntegral:
  case CK_FloatingCast:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexToIntegralComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingRealToComplex:
  case CK_FloatingToBoolean:
  case CK_FloatingToIntegral:
  case CK_IntegralCast:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexToFloatingComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralRealToComplex:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
    // Reinterpreting integers as pointers and vice versa.
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
    // Language extensions.
  case CK_VectorSplat:
  case CK_MatrixCast:
  case CK_CoopMatrixCast:
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:
  case CK_HLSLVectorTruncation:
  case CK_HLSLMatrixTruncation:
  case CK_HLSLElementwiseCast:
  case CK_HLSLAggregateSplatCast:
    return true;

  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_NullToMemberPointer:
  case CK_ReinterpretMemberPointer:
    // FIXME: ABI-dependent.
    return false;

  case CK_AnyPointerToBlockPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_CPointerToObjCPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_IntToOCLSampler:
  case CK_ZeroToOCLOpaqueType:
    // FIXME: Check these.
    return false;

  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToFloating:
  case CK_FixedPointToIntegral:
  case CK_FloatingToFixedPoint:
  case CK_IntegralToFixedPoint:
    // FIXME: Do all fixed-point types represent zero as all 0 bits?
    return false;

  case CK_AddressSpaceConversion:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_Dynamic:
  case CK_NullToPointer:
  case CK_PointerToBoolean:
    // FIXME: Preserves zeroes only if zero pointers and null pointers have the
    // same representation in all involved address spaces.
    return false;

  case CK_ARCConsumeObject:
  case CK_ARCExtendBlockObject:
  case CK_ARCProduceObject:
  case CK_ARCReclaimReturnedObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_BuiltinFnToFnPtr:
  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_LValueToRValue:
  case CK_LValueToRValueBitCast:
  case CK_UncheckedDerivedToBase:
  case CK_HLSLArrayRValue:
    return false;
  }
  llvm_unreachable("Unhandled clang::CastKind enum");
}

bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD) {
  for (const FunctionDecl *PD = FD; PD; PD = PD->getPreviousDecl())
    if (!PD->isInlineBuiltinDeclaration())
      return false;
  return true;
}

bool isConstantEmittableObjectType(QualType Type) {
  assert(Type.isCanonical());
  assert(!Type->isReferenceType());

  // Must be const-qualified but non-volatile.
  Qualifiers Qs = Type.getLocalQualifiers();
  if (!Qs.hasConst() || Qs.hasVolatile())
    return false;

  // Otherwise, all object types satisfy this except C++ classes with
  // mutable subobjects or non-trivial copy/destroy behavior.
  if (const auto *RT = dyn_cast<RecordType>(Type))
    if (const auto *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      RD = RD->getDefinitionOrSelf();
      if (RD->hasMutableFields() || !RD->isTrivial())
        return false;
    }

  return true;
}

ConstantEmissionKind checkVarTypeForConstantEmission(QualType Type) {
  Type = Type.getCanonicalType();
  if (const auto *Ref = dyn_cast<ReferenceType>(Type)) {
    if (isConstantEmittableObjectType(Ref->getPointeeType()))
      return CEK_AsValueOrReference;
    return CEK_AsReferenceOnly;
  }
  if (isConstantEmittableObjectType(Type))
    return CEK_AsValueOnly;
  return CEK_None;
}

} // namespace clang::CodeGenUtils
