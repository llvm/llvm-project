#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"

#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace clang::CIRGen;

#ifndef NDEBUG
/// Return the complex type that we are meant to emit.
static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
}
#endif // NDEBUG

namespace {
class ComplexExprEmitter : public StmtVisitor<ComplexExprEmitter, mlir::Value> {
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;

public:
  explicit ComplexExprEmitter(CIRGenFunction &cgf)
      : cgf(cgf), builder(cgf.getBuilder()) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with complex type that represents a value l-value,
  /// this method emits the address of the l-value, then loads and returns the
  /// result.
  mlir::Value emitLoadOfLValue(const Expr *e) {
    return emitLoadOfLValue(cgf.emitLValue(e), e->getExprLoc());
  }

  mlir::Value emitLoadOfLValue(LValue lv, SourceLocation loc);

  /// Store the specified real/imag parts into the
  /// specified value pointer.
  void emitStoreOfComplex(mlir::Location loc, mlir::Value val, LValue lv,
                          bool isInit);

  /// Emit a cast from complex value Val to DestType.
  mlir::Value emitComplexToComplexCast(mlir::Value value, QualType srcType,
                                       QualType destType, SourceLocation loc);

  /// Emit a cast from scalar value Val to DestType.
  mlir::Value emitScalarToComplexCast(mlir::Value value, QualType srcType,
                                      QualType destType, SourceLocation loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *e) {
    return StmtVisitor<ComplexExprEmitter, mlir::Value>::Visit(e);
  }

  mlir::Value VisitStmt(Stmt *s) {
    cgf.cgm.errorNYI(s->getBeginLoc(), "ComplexExprEmitter VisitStmt");
    return {};
  }

  mlir::Value VisitExpr(Expr *e);
  mlir::Value VisitConstantExpr(ConstantExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitConstantExpr");
    return {};
  }

  mlir::Value VisitParenExpr(ParenExpr *pe) { return Visit(pe->getSubExpr()); }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    return Visit(ge->getResultExpr());
  }
  mlir::Value VisitImaginaryLiteral(const ImaginaryLiteral *il);
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe) {
    return Visit(pe->getReplacement());
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *s) {
    cgf.cgm.errorNYI(s->getExprLoc(), "ComplexExprEmitter VisitCoawaitExpr");
    return {};
  }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *s) {
    cgf.cgm.errorNYI(s->getExprLoc(), "ComplexExprEmitter VisitCoyieldExpr");
    return {};
  }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitUnaryCoawait");
    return {};
  }

  mlir::Value emitConstant(const CIRGenFunction::ConstantEmission &constant,
                           Expr *e) {
    assert(constant && "not a constant");
    if (constant.isReference())
      return emitLoadOfLValue(constant.getReferenceLValue(cgf, e),
                              e->getExprLoc());

    mlir::TypedAttr valueAttr = constant.getValue();
    return builder.getConstant(cgf.getLoc(e->getSourceRange()), valueAttr);
  }

  // l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(e))
      return emitConstant(constant, e);
    return emitLoadOfLValue(e);
  }
  mlir::Value VisitObjCIvarRefExpr(ObjCIvarRefExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitObjCIvarRefExpr");
    return {};
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitObjCMessageExpr");
    return {};
  }
  mlir::Value VisitArraySubscriptExpr(Expr *e) { return emitLoadOfLValue(e); }
  mlir::Value VisitMemberExpr(MemberExpr *me) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(me)) {
      cgf.emitIgnoredExpr(me->getBase());
      return emitConstant(constant, me);
    }
    return emitLoadOfLValue(me);
  }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitOpaqueValueExpr");
    return {};
  }

  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitPseudoObjectExpr");
    return {};
  }

  mlir::Value emitCast(CastKind ck, Expr *op, QualType destTy);
  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *e) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    if (e->changesVolatileQualification())
      return emitLoadOfLValue(e);
    return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  mlir::Value VisitCastExpr(CastExpr *e) {
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(e)) {
      // Bind VLAs in the cast type.
      if (ece->getType()->isVariablyModifiedType()) {
        cgf.cgm.errorNYI(e->getExprLoc(),
                         "VisitCastExpr Bind VLAs in the cast type");
        return {};
      }
    }

    if (e->changesVolatileQualification())
      return emitLoadOfLValue(e);

    return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  mlir::Value VisitCallExpr(const CallExpr *e);
  mlir::Value VisitStmtExpr(const StmtExpr *e);

  // Operators.
  mlir::Value VisitPrePostIncDec(const UnaryOperator *e, cir::UnaryOpKind op,
                                 bool isPre) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return cgf.emitComplexPrePostIncDec(e, lv, op, isPre);
  }
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, cir::UnaryOpKind::Dec, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, cir::UnaryOpKind::Inc, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, cir::UnaryOpKind::Dec, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, cir::UnaryOpKind::Inc, true);
  }
  mlir::Value VisitUnaryDeref(const Expr *e) { return emitLoadOfLValue(e); }

  mlir::Value VisitUnaryPlus(const UnaryOperator *e);
  mlir::Value VisitUnaryMinus(const UnaryOperator *e);
  mlir::Value VisitPlusMinus(const UnaryOperator *e, cir::UnaryOpKind kind,
                             QualType promotionType);
  mlir::Value VisitUnaryNot(const UnaryOperator *e);
  // LNot,Real,Imag never return complex.
  mlir::Value VisitUnaryExtension(const UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitUnaryExtension");
    return {};
  }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    cgf.cgm.errorNYI(dae->getExprLoc(),
                     "ComplexExprEmitter VisitCXXDefaultArgExpr");
    return {};
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    cgf.cgm.errorNYI(die->getExprLoc(),
                     "ComplexExprEmitter VisitCXXDefaultInitExpr");
    return {};
  }
  mlir::Value VisitExprWithCleanups(ExprWithCleanups *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitExprWithCleanups");
    return {};
  }
  mlir::Value VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    mlir::Location loc = cgf.getLoc(e->getExprLoc());
    mlir::Type complexTy = cgf.convertType(e->getType());
    return builder.getNullValue(complexTy, loc);
  }
  mlir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitImplicitValueInitExpr");
    return {};
  }

  struct BinOpInfo {
    mlir::Location loc;
    mlir::Value lhs{};
    mlir::Value rhs{};
    QualType ty{}; // Computation Type.
    FPOptions fpFeatures{};
  };

  BinOpInfo emitBinOps(const BinaryOperator *e,
                       QualType promotionTy = QualType());

  mlir::Value emitPromoted(const Expr *e, QualType promotionTy);
  mlir::Value emitPromotedComplexOperand(const Expr *e, QualType promotionTy);
  LValue emitCompoundAssignLValue(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &),
      RValue &value);
  mlir::Value emitCompoundAssign(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &));

  mlir::Value emitBinAdd(const BinOpInfo &op);
  mlir::Value emitBinSub(const BinOpInfo &op);
  mlir::Value emitBinMul(const BinOpInfo &op);
  mlir::Value emitBinDiv(const BinOpInfo &op);

  QualType getPromotionType(QualType ty, bool isDivOpCode = false) {
    if (auto *complexTy = ty->getAs<ComplexType>()) {
      QualType elementTy = complexTy->getElementType();
      if (elementTy.UseExcessPrecision(cgf.getContext()))
        return cgf.getContext().getComplexType(cgf.getContext().FloatTy);
    }

    if (ty.UseExcessPrecision(cgf.getContext()))
      return cgf.getContext().FloatTy;
    return QualType();
  }

#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *e) {                          \
    QualType promotionTy = getPromotionType(                                   \
        e->getType(), e->getOpcode() == BinaryOperatorKind::BO_Div);           \
    mlir::Value result = emitBin##OP(emitBinOps(e, promotionTy));              \
    if (!promotionTy.isNull())                                                 \
      result = cgf.emitUnPromotedValue(result, e->getType());                  \
    return result;                                                             \
  }

  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
#undef HANDLEBINOP

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitCXXRewrittenBinaryOperator");
    return {};
  }

  // Compound assignments.
  mlir::Value VisitBinAddAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinAdd);
  }
  mlir::Value VisitBinSubAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinSub);
  }
  mlir::Value VisitBinMulAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinMul);
  }
  mlir::Value VisitBinDivAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue emitBinAssignLValue(const BinaryOperator *e, mlir::Value &val);
  mlir::Value VisitBinAssign(const BinaryOperator *e);
  mlir::Value VisitBinComma(const BinaryOperator *e);

  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);
  mlir::Value VisitChooseExpr(ChooseExpr *e);

  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitVAArgExpr(VAArgExpr *e);

  mlir::Value VisitAtomicExpr(AtomicExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitAtomicExpr");
    return {};
  }

  mlir::Value VisitPackIndexingExpr(PackIndexingExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitPackIndexingExpr");
    return {};
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfLValue - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
mlir::Value ComplexExprEmitter::emitLoadOfLValue(LValue lv,
                                                 SourceLocation loc) {
  assert(lv.isSimple() && "non-simple complex l-value?");
  if (lv.getType()->isAtomicType())
    cgf.cgm.errorNYI(loc, "emitLoadOfLValue with Atomic LV");

  const Address srcAddr = lv.getAddress();
  return builder.createLoad(cgf.getLoc(loc), srcAddr);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::emitStoreOfComplex(mlir::Location loc, mlir::Value val,
                                            LValue lv, bool isInit) {
  if (lv.getType()->isAtomicType() ||
      (!isInit && cgf.isLValueSuitableForInlineAtomic(lv))) {
    cgf.cgm.errorNYI(loc, "StoreOfComplex with Atomic LV");
    return;
  }

  const Address destAddr = lv.getAddress();
  builder.createStore(loc, val, destAddr);
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

mlir::Value ComplexExprEmitter::VisitExpr(Expr *e) {
  cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitExpr");
  return {};
}

mlir::Value
ComplexExprEmitter::VisitImaginaryLiteral(const ImaginaryLiteral *il) {
  auto ty = mlir::cast<cir::ComplexType>(cgf.convertType(il->getType()));
  mlir::Type elementTy = ty.getElementType();
  mlir::Location loc = cgf.getLoc(il->getExprLoc());

  mlir::TypedAttr realValueAttr;
  mlir::TypedAttr imagValueAttr;

  if (mlir::isa<cir::IntType>(elementTy)) {
    llvm::APInt imagValue = cast<IntegerLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::IntAttr::get(elementTy, 0);
    imagValueAttr = cir::IntAttr::get(elementTy, imagValue);
  } else {
    assert(mlir::isa<cir::FPTypeInterface>(elementTy) &&
           "Expected complex element type to be floating-point");

    llvm::APFloat imagValue =
        cast<FloatingLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::FPAttr::get(
        elementTy, llvm::APFloat::getZero(imagValue.getSemantics()));
    imagValueAttr = cir::FPAttr::get(elementTy, imagValue);
  }

  auto complexAttr = cir::ConstComplexAttr::get(realValueAttr, imagValueAttr);
  return builder.create<cir::ConstantOp>(loc, complexAttr);
}

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);
  return cgf.emitCallExpr(e).getComplexValue();
}

mlir::Value ComplexExprEmitter::VisitStmtExpr(const StmtExpr *e) {
  cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitExpr");
  return {};
}

mlir::Value ComplexExprEmitter::emitComplexToComplexCast(mlir::Value val,
                                                         QualType srcType,
                                                         QualType destType,
                                                         SourceLocation loc) {
  if (srcType == destType)
    return val;

  // Get the src/dest element type.
  QualType srcElemTy = srcType->castAs<ComplexType>()->getElementType();
  QualType destElemTy = destType->castAs<ComplexType>()->getElementType();

  cir::CastKind castOpKind;
  if (srcElemTy->isFloatingType() && destElemTy->isFloatingType())
    castOpKind = cir::CastKind::float_complex;
  else if (srcElemTy->isFloatingType() && destElemTy->isIntegerType())
    castOpKind = cir::CastKind::float_complex_to_int_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isFloatingType())
    castOpKind = cir::CastKind::int_complex_to_float_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isIntegerType())
    castOpKind = cir::CastKind::int_complex;
  else
    llvm_unreachable("unexpected src type or dest type");

  return builder.createCast(cgf.getLoc(loc), castOpKind, val,
                            cgf.convertType(destType));
}

mlir::Value ComplexExprEmitter::emitScalarToComplexCast(mlir::Value val,
                                                        QualType srcType,
                                                        QualType destType,
                                                        SourceLocation loc) {
  cir::CastKind castOpKind;
  if (srcType->isFloatingType())
    castOpKind = cir::CastKind::float_to_complex;
  else if (srcType->isIntegerType())
    castOpKind = cir::CastKind::int_to_complex;
  else
    llvm_unreachable("unexpected src type");

  return builder.createCast(cgf.getLoc(loc), castOpKind, val,
                            cgf.convertType(destType));
}

mlir::Value ComplexExprEmitter::emitCast(CastKind ck, Expr *op,
                                         QualType destTy) {
  switch (ck) {
  case CK_Dependent:
    llvm_unreachable("dependent type must be resolved before the CIR codegen");

  case CK_NoOp:
  case CK_LValueToRValue:
    return Visit(op);

  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_UserDefinedConversion: {
    cgf.cgm.errorNYI(
        "ComplexExprEmitter::emitCast Atmoic & UserDefinedConversion");
    return {};
  }

  case CK_LValueBitCast: {
    LValue origLV = cgf.emitLValue(op);
    Address addr =
        origLV.getAddress().withElementType(builder, cgf.convertType(destTy));
    LValue destLV = cgf.makeAddrLValue(addr, destTy);
    return emitLoadOfLValue(destLV, op->getExprLoc());
  }

  case CK_LValueToRValueBitCast: {
    LValue sourceLVal = cgf.emitLValue(op);
    Address addr = sourceLVal.getAddress().withElementType(
        builder, cgf.convertTypeForMem(destTy));
    LValue destLV = cgf.makeAddrLValue(addr, destTy);
    assert(!cir::MissingFeatures::opTBAA());
    return emitLoadOfLValue(destLV, op->getExprLoc());
  }

  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_BooleanToSignedIntegral:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
  case CK_ZeroToOCLOpaqueType:
  case CK_AddressSpaceConversion:
  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
  case CK_MatrixCast:
  case CK_HLSLVectorTruncation:
  case CK_HLSLArrayRValue:
  case CK_HLSLElementwiseCast:
  case CK_HLSLAggregateSplatCast:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    return emitScalarToComplexCast(cgf.emitScalarExpr(op), op->getType(),
                                   destTy, op->getExprLoc());
  }

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex: {
    assert(!cir::MissingFeatures::cgFPOptionsRAII());
    return emitComplexToComplexCast(Visit(op), op->getType(), destTy,
                                    op->getExprLoc());
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

mlir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  mlir::Value result = VisitPlusMinus(e, cir::UnaryOpKind::Plus, promotionTy);
  if (!promotionTy.isNull())
    return cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  mlir::Value result = VisitPlusMinus(e, cir::UnaryOpKind::Minus, promotionTy);
  if (!promotionTy.isNull())
    return cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::VisitPlusMinus(const UnaryOperator *e,
                                               cir::UnaryOpKind kind,
                                               QualType promotionType) {
  assert(kind == cir::UnaryOpKind::Plus ||
         kind == cir::UnaryOpKind::Minus &&
             "Invalid UnaryOp kind for ComplexType Plus or Minus");

  mlir::Value op;
  if (!promotionType.isNull())
    op = cgf.emitPromotedComplexExpr(e->getSubExpr(), promotionType);
  else
    op = Visit(e->getSubExpr());
  return builder.createUnaryOp(cgf.getLoc(e->getExprLoc()), kind, op);
}

mlir::Value ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *e) {
  mlir::Value op = Visit(e->getSubExpr());
  return builder.createNot(op);
}

mlir::Value ComplexExprEmitter::emitBinAdd(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  assert(!cir::MissingFeatures::cgFPOptionsRAII());

  if (mlir::isa<cir::ComplexType>(op.lhs.getType()) &&
      mlir::isa<cir::ComplexType>(op.rhs.getType()))
    return builder.create<cir::ComplexAddOp>(op.loc, op.lhs, op.rhs);

  if (mlir::isa<cir::ComplexType>(op.lhs.getType())) {
    mlir::Value real = builder.createComplexReal(op.loc, op.lhs);
    mlir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    mlir::Value newReal = builder.createAdd(op.loc, real, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, imag);
  }

  assert(mlir::isa<cir::ComplexType>(op.rhs.getType()));
  mlir::Value real = builder.createComplexReal(op.loc, op.rhs);
  mlir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  mlir::Value newReal = builder.createAdd(op.loc, op.lhs, real);
  return builder.createComplexCreate(op.loc, newReal, imag);
}

mlir::Value ComplexExprEmitter::emitBinSub(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  assert(!cir::MissingFeatures::cgFPOptionsRAII());

  if (mlir::isa<cir::ComplexType>(op.lhs.getType()) &&
      mlir::isa<cir::ComplexType>(op.rhs.getType()))
    return builder.create<cir::ComplexSubOp>(op.loc, op.lhs, op.rhs);

  if (mlir::isa<cir::ComplexType>(op.lhs.getType())) {
    mlir::Value real = builder.createComplexReal(op.loc, op.lhs);
    mlir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    mlir::Value newReal = builder.createSub(op.loc, real, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, imag);
  }

  assert(mlir::isa<cir::ComplexType>(op.rhs.getType()));
  mlir::Value real = builder.createComplexReal(op.loc, op.rhs);
  mlir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  mlir::Value newReal = builder.createSub(op.loc, op.lhs, real);
  return builder.createComplexCreate(op.loc, newReal, imag);
}

static cir::ComplexRangeKind
getComplexRangeAttr(LangOptions::ComplexRangeKind range) {
  switch (range) {
  case LangOptions::CX_Full:
    return cir::ComplexRangeKind::Full;
  case LangOptions::CX_Improved:
    return cir::ComplexRangeKind::Improved;
  case LangOptions::CX_Promoted:
    return cir::ComplexRangeKind::Promoted;
  case LangOptions::CX_Basic:
    return cir::ComplexRangeKind::Basic;
  case LangOptions::CX_None:
    // The default value for ComplexRangeKind is Full if no option is selected
    return cir::ComplexRangeKind::Full;
  }
}

mlir::Value ComplexExprEmitter::emitBinMul(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  assert(!cir::MissingFeatures::cgFPOptionsRAII());

  if (mlir::isa<cir::ComplexType>(op.lhs.getType()) &&
      mlir::isa<cir::ComplexType>(op.rhs.getType())) {
    cir::ComplexRangeKind rangeKind =
        getComplexRangeAttr(op.fpFeatures.getComplexRange());
    return builder.create<cir::ComplexMulOp>(op.loc, op.lhs, op.rhs, rangeKind);
  }

  if (mlir::isa<cir::ComplexType>(op.lhs.getType())) {
    mlir::Value real = builder.createComplexReal(op.loc, op.lhs);
    mlir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    mlir::Value newReal = builder.createMul(op.loc, real, op.rhs);
    mlir::Value newImag = builder.createMul(op.loc, imag, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, newImag);
  }

  assert(mlir::isa<cir::ComplexType>(op.rhs.getType()));
  mlir::Value real = builder.createComplexReal(op.loc, op.rhs);
  mlir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  mlir::Value newReal = builder.createMul(op.loc, op.lhs, real);
  mlir::Value newImag = builder.createMul(op.loc, op.lhs, imag);
  return builder.createComplexCreate(op.loc, newReal, newImag);
}

mlir::Value ComplexExprEmitter::emitBinDiv(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  assert(!cir::MissingFeatures::cgFPOptionsRAII());

  // Handle division between two complex values. In the case of complex integer
  // types mixed with scalar integers, the scalar integer type will always be
  // promoted to a complex integer value with a zero imaginary component when
  // the AST is formed.
  if (mlir::isa<cir::ComplexType>(op.lhs.getType()) &&
      mlir::isa<cir::ComplexType>(op.rhs.getType())) {
    cir::ComplexRangeKind rangeKind =
        getComplexRangeAttr(op.fpFeatures.getComplexRange());
    return cir::ComplexDivOp::create(builder, op.loc, op.lhs, op.rhs,
                                     rangeKind);
  }

  // The C99 standard (G.5.1) defines division of a complex value by a real
  // value in the following simplified form.
  if (mlir::isa<cir::ComplexType>(op.lhs.getType())) {
    assert(mlir::cast<cir::ComplexType>(op.lhs.getType()).getElementType() ==
           op.rhs.getType());
    mlir::Value real = builder.createComplexReal(op.loc, op.lhs);
    mlir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    mlir::Value newReal = builder.createFDiv(op.loc, real, op.rhs);
    mlir::Value newImag = builder.createFDiv(op.loc, imag, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, newImag);
  }

  assert(mlir::isa<cir::ComplexType>(op.rhs.getType()));
  cir::ConstantOp nullValue = builder.getNullValue(op.lhs.getType(), op.loc);
  mlir::Value lhs = builder.createComplexCreate(op.loc, op.lhs, nullValue);
  cir::ComplexRangeKind rangeKind =
      getComplexRangeAttr(op.fpFeatures.getComplexRange());
  return cir::ComplexDivOp::create(builder, op.loc, lhs, op.rhs, rangeKind);
}

mlir::Value CIRGenFunction::emitUnPromotedValue(mlir::Value result,
                                                QualType unPromotionType) {
  assert(!mlir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(unPromotionType));
}

mlir::Value CIRGenFunction::emitPromotedValue(mlir::Value result,
                                              QualType promotionType) {
  assert(!mlir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(promotionType));
}

mlir::Value ComplexExprEmitter::emitPromoted(const Expr *e,
                                             QualType promotionTy) {
  e = e->IgnoreParens();
  if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
    switch (bo->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return emitBin##OP(emitBinOps(bo, promotionTy));
      HANDLE_BINOP(Add)
      HANDLE_BINOP(Sub)
      HANDLE_BINOP(Mul)
      HANDLE_BINOP(Div)
#undef HANDLE_BINOP
    default:
      break;
    }
  } else if (const auto *unaryOp = dyn_cast<UnaryOperator>(e)) {
    switch (unaryOp->getOpcode()) {
    case UO_Minus:
    case UO_Plus: {
      auto kind = unaryOp->getOpcode() == UO_Plus ? cir::UnaryOpKind::Plus
                                                  : cir::UnaryOpKind::Minus;
      return VisitPlusMinus(unaryOp, kind, promotionTy);
    }
    default:
      break;
    }
  }

  mlir::Value result = Visit(const_cast<Expr *>(e));
  if (!promotionTy.isNull())
    return cgf.emitPromotedValue(result, promotionTy);

  return result;
}

mlir::Value CIRGenFunction::emitPromotedComplexExpr(const Expr *e,
                                                    QualType promotionType) {
  return ComplexExprEmitter(*this).emitPromoted(e, promotionType);
}

mlir::Value
ComplexExprEmitter::emitPromotedComplexOperand(const Expr *e,
                                               QualType promotionTy) {
  if (e->getType()->isAnyComplexType()) {
    if (!promotionTy.isNull())
      return cgf.emitPromotedComplexExpr(e, promotionTy);
    return Visit(const_cast<Expr *>(e));
  }

  if (!promotionTy.isNull()) {
    QualType complexElementTy =
        promotionTy->castAs<ComplexType>()->getElementType();
    return cgf.emitPromotedScalarExpr(e, complexElementTy);
  }
  return cgf.emitScalarExpr(e);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::emitBinOps(const BinaryOperator *e, QualType promotionTy) {
  BinOpInfo binOpInfo{cgf.getLoc(e->getExprLoc())};
  binOpInfo.lhs = emitPromotedComplexOperand(e->getLHS(), promotionTy);
  binOpInfo.rhs = emitPromotedComplexOperand(e->getRHS(), promotionTy);
  binOpInfo.ty = promotionTy.isNull() ? e->getType() : promotionTy;
  binOpInfo.fpFeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());
  return binOpInfo;
}

LValue ComplexExprEmitter::emitCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &value) {
  QualType lhsTy = e->getLHS()->getType();
  QualType rhsTy = e->getRHS()->getType();
  SourceLocation exprLoc = e->getExprLoc();
  mlir::Location loc = cgf.getLoc(exprLoc);

  if (lhsTy->getAs<AtomicType>()) {
    cgf.cgm.errorNYI("emitCompoundAssignLValue AtmoicType");
    return {};
  }

  BinOpInfo opInfo{loc};
  opInfo.fpFeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());

  assert(!cir::MissingFeatures::cgFPOptionsRAII());

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  QualType promotionTypeCR = getPromotionType(e->getComputationResultType());
  opInfo.ty = promotionTypeCR.isNull() ? e->getComputationResultType()
                                       : promotionTypeCR;

  QualType complexElementTy =
      opInfo.ty->castAs<ComplexType>()->getElementType();
  QualType promotionTypeRHS = getPromotionType(rhsTy);

  // The RHS should have been converted to the computation type.
  if (e->getRHS()->getType()->isRealFloatingType()) {
    if (!promotionTypeRHS.isNull()) {
      opInfo.rhs = cgf.emitPromotedScalarExpr(e->getRHS(), promotionTypeRHS);
    } else {
      assert(cgf.getContext().hasSameUnqualifiedType(complexElementTy, rhsTy));
      opInfo.rhs = cgf.emitScalarExpr(e->getRHS());
    }
  } else {
    if (!promotionTypeRHS.isNull()) {
      opInfo.rhs = cgf.emitPromotedComplexExpr(e->getRHS(), promotionTypeRHS);
    } else {
      assert(cgf.getContext().hasSameUnqualifiedType(opInfo.ty, rhsTy));
      opInfo.rhs = Visit(e->getRHS());
    }
  }

  LValue lhs = cgf.emitLValue(e->getLHS());

  // Load from the l-value and convert it.
  QualType promotionTypeLHS = getPromotionType(e->getComputationLHSType());
  if (lhsTy->isAnyComplexType()) {
    mlir::Value lhsValue = emitLoadOfLValue(lhs, exprLoc);
    QualType destTy = promotionTypeLHS.isNull() ? opInfo.ty : promotionTypeLHS;
    opInfo.lhs = emitComplexToComplexCast(lhsValue, lhsTy, destTy, exprLoc);
  } else {
    mlir::Value lhsVal = cgf.emitLoadOfScalar(lhs, exprLoc);
    // For floating point real operands we can directly pass the scalar form
    // to the binary operator emission and potentially get more efficient code.
    if (lhsTy->isRealFloatingType()) {
      QualType promotedComplexElementTy;
      if (!promotionTypeLHS.isNull()) {
        promotedComplexElementTy =
            cast<ComplexType>(promotionTypeLHS)->getElementType();
        if (!cgf.getContext().hasSameUnqualifiedType(promotedComplexElementTy,
                                                     promotionTypeLHS))
          lhsVal = cgf.emitScalarConversion(lhsVal, lhsTy,
                                            promotedComplexElementTy, exprLoc);
      } else {
        if (!cgf.getContext().hasSameUnqualifiedType(complexElementTy, lhsTy))
          lhsVal = cgf.emitScalarConversion(lhsVal, lhsTy, complexElementTy,
                                            exprLoc);
      }
      opInfo.lhs = lhsVal;
    } else {
      opInfo.lhs = emitScalarToComplexCast(lhsVal, lhsTy, opInfo.ty, exprLoc);
    }
  }

  // Expand the binary operator.
  mlir::Value result = (this->*func)(opInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (lhsTy->isAnyComplexType()) {
    mlir::Value resultValue =
        emitComplexToComplexCast(result, opInfo.ty, lhsTy, exprLoc);
    emitStoreOfComplex(loc, resultValue, lhs, /*isInit*/ false);
    value = RValue::getComplex(resultValue);
  } else {
    mlir::Value resultValue =
        cgf.emitComplexToScalarConversion(result, opInfo.ty, lhsTy, exprLoc);
    cgf.emitStoreOfScalar(resultValue, lhs, /*isInit*/ false);
    value = RValue::get(resultValue);
  }

  return lhs;
}

mlir::Value ComplexExprEmitter::emitCompoundAssign(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &)) {
  RValue val;
  LValue lv = emitCompoundAssignLValue(e, func, val);

  // The result of an assignment in C is the assigned r-value.
  if (!cgf.getLangOpts().CPlusPlus)
    return val.getComplexValue();

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lv.isVolatileQualified())
    return val.getComplexValue();

  return emitLoadOfLValue(lv, e->getExprLoc());
}

LValue ComplexExprEmitter::emitBinAssignLValue(const BinaryOperator *e,
                                               mlir::Value &value) {
  assert(cgf.getContext().hasSameUnqualifiedType(e->getLHS()->getType(),
                                                 e->getRHS()->getType()) &&
         "Invalid assignment");

  // Emit the RHS.  __block variables need the RHS evaluated first.
  value = Visit(e->getRHS());

  // Compute the address to store into.
  LValue lhs = cgf.emitLValue(e->getLHS());

  // Store the result value into the LHS lvalue.
  emitStoreOfComplex(cgf.getLoc(e->getExprLoc()), value, lhs,
                     /*isInit*/ false);
  return lhs;
}

mlir::Value ComplexExprEmitter::VisitBinAssign(const BinaryOperator *e) {
  mlir::Value value;
  LValue lv = emitBinAssignLValue(e, value);

  // The result of an assignment in C is the assigned r-value.
  if (!cgf.getLangOpts().CPlusPlus)
    return value;

  // If the lvalue is non-volatile, return the computed value of the
  // assignment.
  if (!lv.isVolatile())
    return value;

  return emitLoadOfLValue(lv, e->getExprLoc());
}

mlir::Value ComplexExprEmitter::VisitBinComma(const BinaryOperator *e) {
  cgf.emitIgnoredExpr(e->getLHS());
  return Visit(e->getRHS());
}

mlir::Value ComplexExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  mlir::Value condValue = Visit(e->getCond());
  mlir::Location loc = cgf.getLoc(e->getSourceRange());

  return builder
      .create<cir::TernaryOp>(
          loc, condValue,
          /*thenBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            mlir::Value trueValue = Visit(e->getTrueExpr());
            b.create<cir::YieldOp>(loc, trueValue);
          },
          /*elseBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            mlir::Value falseValue = Visit(e->getFalseExpr());
            b.create<cir::YieldOp>(loc, falseValue);
          })
      .getResult();
}

mlir::Value ComplexExprEmitter::VisitChooseExpr(ChooseExpr *e) {
  return Visit(e->getChosenSubExpr());
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *e) {
  mlir::Location loc = cgf.getLoc(e->getExprLoc());
  if (e->getNumInits() == 2) {
    mlir::Value real = cgf.emitScalarExpr(e->getInit(0));
    mlir::Value imag = cgf.emitScalarExpr(e->getInit(1));
    return builder.createComplexCreate(loc, real, imag);
  }

  if (e->getNumInits() == 1)
    return Visit(e->getInit(0));

  assert(e->getNumInits() == 0 && "Unexpected number of inits");
  mlir::Type complexTy = cgf.convertType(e->getType());
  return builder.getNullValue(complexTy, loc);
}

mlir::Value ComplexExprEmitter::VisitVAArgExpr(VAArgExpr *e) {
  return cgf.emitVAArg(e);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
mlir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitComplexExprIntoLValue(const Expr *e, LValue dest,
                                               bool isInit) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter emitter(*this);
  mlir::Value value = emitter.Visit(const_cast<Expr *>(e));
  emitter.emitStoreOfComplex(getLoc(e->getExprLoc()), value, dest, isInit);
}

/// EmitStoreOfComplex - Store a complex number into the specified l-value.
void CIRGenFunction::emitStoreOfComplex(mlir::Location loc, mlir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}

mlir::Value CIRGenFunction::emitLoadOfComplex(LValue src, SourceLocation loc) {
  return ComplexExprEmitter(*this).emitLoadOfLValue(src, loc);
}

LValue CIRGenFunction::emitComplexAssignmentLValue(const BinaryOperator *e) {
  assert(e->getOpcode() == BO_Assign && "Expected assign op");

  mlir::Value value; // ignored
  LValue lvalue = ComplexExprEmitter(*this).emitBinAssignLValue(e, value);
  if (getLangOpts().OpenMP)
    cgm.errorNYI("emitComplexAssignmentLValue OpenMP");

  return lvalue;
}

using CompoundFunc =
    mlir::Value (ComplexExprEmitter::*)(const ComplexExprEmitter::BinOpInfo &);

static CompoundFunc getComplexOp(BinaryOperatorKind op) {
  switch (op) {
  case BO_MulAssign:
    return &ComplexExprEmitter::emitBinMul;
  case BO_DivAssign:
    return &ComplexExprEmitter::emitBinDiv;
  case BO_SubAssign:
    return &ComplexExprEmitter::emitBinSub;
  case BO_AddAssign:
    return &ComplexExprEmitter::emitBinAdd;
  default:
    llvm_unreachable("unexpected complex compound assignment");
  }
}

LValue CIRGenFunction::emitComplexCompoundAssignmentLValue(
    const CompoundAssignOperator *e) {
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue val;
  return ComplexExprEmitter(*this).emitCompoundAssignLValue(e, op, val);
}

mlir::Value CIRGenFunction::emitComplexPrePostIncDec(const UnaryOperator *e,
                                                     LValue lv,
                                                     cir::UnaryOpKind op,
                                                     bool isPre) {
  assert(op == cir::UnaryOpKind::Inc ||
         op == cir::UnaryOpKind::Dec && "Invalid UnaryOp kind for ComplexType");

  mlir::Value inVal = emitLoadOfComplex(lv, e->getExprLoc());
  mlir::Location loc = getLoc(e->getExprLoc());
  mlir::Value incVal = builder.createUnaryOp(loc, op, inVal);

  // Store the updated result through the lvalue.
  emitStoreOfComplex(loc, incVal, lv, /*isInit=*/false);

  if (getLangOpts().OpenMP)
    cgm.errorNYI(loc, "emitComplexPrePostIncDec OpenMP");

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? incVal : inVal;
}

LValue CIRGenFunction::emitScalarCompoundAssignWithComplex(
    const CompoundAssignOperator *e, mlir::Value &result) {
  // Key Instructions: Don't need to create an atom group here; one will already
  // be active through scalar handling code.
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue value;
  LValue ret = ComplexExprEmitter(*this).emitCompoundAssignLValue(e, op, value);
  result = value.getValue();
  return ret;
}
