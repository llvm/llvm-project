#include "CIRGenBuilder.h"
#include "CIRGenConstantEmitter.h"
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
class ComplexExprEmitter : public StmtVisitor<ComplexExprEmitter, aiir::Value> {
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
  aiir::Value emitLoadOfLValue(const Expr *e) {
    return emitLoadOfLValue(cgf.emitLValue(e), e->getExprLoc());
  }

  aiir::Value emitLoadOfLValue(LValue lv, SourceLocation loc);

  /// Store the specified real/imag parts into the
  /// specified value pointer.
  void emitStoreOfComplex(aiir::Location loc, aiir::Value val, LValue lv,
                          bool isInit);

  /// Emit a cast from complex value Val to DestType.
  aiir::Value emitComplexToComplexCast(aiir::Value value, QualType srcType,
                                       QualType destType, SourceLocation loc);

  /// Emit a cast from scalar value Val to DestType.
  aiir::Value emitScalarToComplexCast(aiir::Value value, QualType srcType,
                                      QualType destType, SourceLocation loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  aiir::Value Visit(Expr *e) {
    return StmtVisitor<ComplexExprEmitter, aiir::Value>::Visit(e);
  }

  aiir::Value VisitStmt(Stmt *s) {
    s->dump(llvm::errs(), cgf.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  aiir::Value VisitExpr(Expr *e);
  aiir::Value VisitConstantExpr(ConstantExpr *e) {
    if (aiir::Attribute result = ConstantEmitter(cgf).tryEmitConstantExpr(e))
      return builder.getConstant(cgf.getLoc(e->getSourceRange()),
                                 aiir::cast<aiir::TypedAttr>(result));

    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitConstantExpr non constantexpr");
    return {};
  }

  aiir::Value VisitParenExpr(ParenExpr *pe) { return Visit(pe->getSubExpr()); }
  aiir::Value VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    return Visit(ge->getResultExpr());
  }
  aiir::Value VisitImaginaryLiteral(const ImaginaryLiteral *il);
  aiir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe) {
    return Visit(pe->getReplacement());
  }
  aiir::Value VisitCoawaitExpr(CoawaitExpr *s) {
    cgf.cgm.errorNYI(s->getExprLoc(), "ComplexExprEmitter VisitCoawaitExpr");
    return {};
  }
  aiir::Value VisitCoyieldExpr(CoyieldExpr *s) {
    cgf.cgm.errorNYI(s->getExprLoc(), "ComplexExprEmitter VisitCoyieldExpr");
    return {};
  }
  aiir::Value VisitUnaryCoawait(const UnaryOperator *e) {
    cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitUnaryCoawait");
    return {};
  }

  aiir::Value emitConstant(const CIRGenFunction::ConstantEmission &constant,
                           Expr *e) {
    assert(constant && "not a constant");
    if (constant.isReference())
      return emitLoadOfLValue(constant.getReferenceLValue(cgf, e),
                              e->getExprLoc());

    aiir::TypedAttr valueAttr = constant.getValue();
    return builder.getConstant(cgf.getLoc(e->getSourceRange()), valueAttr);
  }

  // l-values.
  aiir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(e))
      return emitConstant(constant, e);
    return emitLoadOfLValue(e);
  }
  aiir::Value VisitObjCIvarRefExpr(ObjCIvarRefExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitObjCIvarRefExpr");
    return {};
  }
  aiir::Value VisitObjCMessageExpr(ObjCMessageExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitObjCMessageExpr");
    return {};
  }
  aiir::Value VisitArraySubscriptExpr(Expr *e) { return emitLoadOfLValue(e); }
  aiir::Value VisitMemberExpr(MemberExpr *me) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(me)) {
      cgf.emitIgnoredExpr(me->getBase());
      return emitConstant(constant, me);
    }
    return emitLoadOfLValue(me);
  }
  aiir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    if (e->isGLValue())
      return emitLoadOfLValue(cgf.getOrCreateOpaqueLValueMapping(e),
                              e->getExprLoc());
    return cgf.getOrCreateOpaqueRValueMapping(e).getComplexValue();
  }

  aiir::Value VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitPseudoObjectExpr");
    return {};
  }

  aiir::Value emitCast(CastKind ck, Expr *op, QualType destTy);
  aiir::Value VisitImplicitCastExpr(ImplicitCastExpr *e) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    if (e->changesVolatileQualification())
      return emitLoadOfLValue(e);
    return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  aiir::Value VisitCastExpr(CastExpr *e) {
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
  aiir::Value VisitCallExpr(const CallExpr *e);
  aiir::Value VisitStmtExpr(const StmtExpr *e);

  // Operators.
  aiir::Value VisitPrePostIncDec(const UnaryOperator *e) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return cgf.emitComplexPrePostIncDec(e, lv);
  }
  aiir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e);
  }
  aiir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e);
  }
  aiir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e);
  }
  aiir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e);
  }
  aiir::Value VisitUnaryDeref(const Expr *e) { return emitLoadOfLValue(e); }

  aiir::Value VisitUnaryPlus(const UnaryOperator *e);
  aiir::Value VisitUnaryPlus(const UnaryOperator *e, QualType promotionType);
  aiir::Value VisitUnaryMinus(const UnaryOperator *e);
  aiir::Value VisitUnaryMinus(const UnaryOperator *e, QualType promotionType);
  aiir::Value VisitUnaryNot(const UnaryOperator *e);
  // LNot,Real,Imag never return complex.
  aiir::Value VisitUnaryExtension(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }
  aiir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    CIRGenFunction::CXXDefaultArgExprScope scope(cgf, dae);
    return Visit(dae->getExpr());
  }
  aiir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    CIRGenFunction::CXXDefaultInitExprScope scope(cgf, die);
    return Visit(die->getExpr());
  }
  aiir::Value VisitExprWithCleanups(ExprWithCleanups *e) {
    CIRGenFunction::RunCleanupsScope scope(cgf);
    aiir::Value complexVal = Visit(e->getSubExpr());
    // Defend against dominance problems caused by jumps out of expression
    // evaluation through the shared cleanup block.
    scope.forceCleanup({&complexVal});
    return complexVal;
  }
  aiir::Value VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    aiir::Location loc = cgf.getLoc(e->getExprLoc());
    aiir::Type complexTy = cgf.convertType(e->getType());
    return builder.getNullValue(complexTy, loc);
  }
  aiir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    aiir::Location loc = cgf.getLoc(e->getExprLoc());
    aiir::Type complexTy = cgf.convertType(e->getType());
    return builder.getNullValue(complexTy, loc);
  }

  struct BinOpInfo {
    aiir::Location loc;
    aiir::Value lhs{};
    aiir::Value rhs{};
    QualType ty{}; // Computation Type.
    FPOptions fpFeatures{};
  };

  BinOpInfo emitBinOps(const BinaryOperator *e,
                       QualType promotionTy = QualType());

  aiir::Value emitPromoted(const Expr *e, QualType promotionTy);
  aiir::Value emitPromotedComplexOperand(const Expr *e, QualType promotionTy);
  LValue emitCompoundAssignLValue(
      const CompoundAssignOperator *e,
      aiir::Value (ComplexExprEmitter::*func)(const BinOpInfo &),
      RValue &value);
  aiir::Value emitCompoundAssign(
      const CompoundAssignOperator *e,
      aiir::Value (ComplexExprEmitter::*func)(const BinOpInfo &));

  aiir::Value emitBinAdd(const BinOpInfo &op);
  aiir::Value emitBinSub(const BinOpInfo &op);
  aiir::Value emitBinMul(const BinOpInfo &op);
  aiir::Value emitBinDiv(const BinOpInfo &op);

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
  aiir::Value VisitBin##OP(const BinaryOperator *e) {                          \
    QualType promotionTy = getPromotionType(                                   \
        e->getType(), e->getOpcode() == BinaryOperatorKind::BO_Div);           \
    aiir::Value result = emitBin##OP(emitBinOps(e, promotionTy));              \
    if (!promotionTy.isNull())                                                 \
      result = cgf.emitUnPromotedValue(result, e->getType());                  \
    return result;                                                             \
  }

  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
#undef HANDLEBINOP

  aiir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    cgf.cgm.errorNYI(e->getExprLoc(),
                     "ComplexExprEmitter VisitCXXRewrittenBinaryOperator");
    return {};
  }

  // Compound assignments.
  aiir::Value VisitBinAddAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinAdd);
  }
  aiir::Value VisitBinSubAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinSub);
  }
  aiir::Value VisitBinMulAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinMul);
  }
  aiir::Value VisitBinDivAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue emitBinAssignLValue(const BinaryOperator *e, aiir::Value &val);
  aiir::Value VisitBinAssign(const BinaryOperator *e);
  aiir::Value VisitBinComma(const BinaryOperator *e);

  aiir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);
  aiir::Value VisitChooseExpr(ChooseExpr *e);

  aiir::Value VisitInitListExpr(InitListExpr *e);

  aiir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    return emitLoadOfLValue(e);
  }

  aiir::Value VisitVAArgExpr(VAArgExpr *e);

  aiir::Value VisitAtomicExpr(AtomicExpr *e) {
    return cgf.emitAtomicExpr(e).getComplexValue();
  }

  aiir::Value VisitPackIndexingExpr(PackIndexingExpr *e) {
    return Visit(e->getSelectedExpr());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfLValue - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
aiir::Value ComplexExprEmitter::emitLoadOfLValue(LValue lv,
                                                 SourceLocation loc) {
  assert(lv.isSimple() && "non-simple complex l-value?");
  if (lv.getType()->isAtomicType())
    cgf.cgm.errorNYI(loc, "emitLoadOfLValue with Atomic LV");

  const Address srcAddr = lv.getAddress();
  return builder.createLoad(cgf.getLoc(loc), srcAddr, lv.isVolatileQualified());
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::emitStoreOfComplex(aiir::Location loc, aiir::Value val,
                                            LValue lv, bool isInit) {
  if (lv.getType()->isAtomicType() ||
      (!isInit && cgf.isLValueSuitableForInlineAtomic(lv))) {
    cgf.cgm.errorNYI(loc, "StoreOfComplex with Atomic LV");
    return;
  }

  const Address destAddr = lv.getAddress();
  builder.createStore(loc, val, destAddr, lv.isVolatileQualified());
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

aiir::Value ComplexExprEmitter::VisitExpr(Expr *e) {
  cgf.cgm.errorNYI(e->getExprLoc(), "ComplexExprEmitter VisitExpr");
  return {};
}

aiir::Value
ComplexExprEmitter::VisitImaginaryLiteral(const ImaginaryLiteral *il) {
  auto ty = aiir::cast<cir::ComplexType>(cgf.convertType(il->getType()));
  aiir::Type elementTy = ty.getElementType();
  aiir::Location loc = cgf.getLoc(il->getExprLoc());

  aiir::TypedAttr realValueAttr;
  aiir::TypedAttr imagValueAttr;

  if (aiir::isa<cir::IntType>(elementTy)) {
    llvm::APInt imagValue = cast<IntegerLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::IntAttr::get(elementTy, 0);
    imagValueAttr = cir::IntAttr::get(elementTy, imagValue);
  } else {
    assert(aiir::isa<cir::FPTypeInterface>(elementTy) &&
           "Expected complex element type to be floating-point");

    llvm::APFloat imagValue =
        cast<FloatingLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::FPAttr::get(
        elementTy, llvm::APFloat::getZero(imagValue.getSemantics()));
    imagValueAttr = cir::FPAttr::get(elementTy, imagValue);
  }

  auto complexAttr = cir::ConstComplexAttr::get(realValueAttr, imagValueAttr);
  return cir::ConstantOp::create(builder, loc, complexAttr);
}

aiir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);
  return cgf.emitCallExpr(e).getComplexValue();
}

aiir::Value ComplexExprEmitter::VisitStmtExpr(const StmtExpr *e) {
  CIRGenFunction::StmtExprEvaluation eval(cgf);
  Address retAlloca =
      cgf.createMemTemp(e->getType(), cgf.getLoc(e->getSourceRange()));
  (void)cgf.emitCompoundStmt(*e->getSubStmt(), &retAlloca);
  assert(retAlloca.isValid() && "Expected complex return value");
  return emitLoadOfLValue(cgf.makeAddrLValue(retAlloca, e->getType()),
                          e->getExprLoc());
}

aiir::Value ComplexExprEmitter::emitComplexToComplexCast(aiir::Value val,
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

aiir::Value ComplexExprEmitter::emitScalarToComplexCast(aiir::Value val,
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

aiir::Value ComplexExprEmitter::emitCast(CastKind ck, Expr *op,
                                         QualType destTy) {
  switch (ck) {
  case CK_Dependent:
    llvm_unreachable("dependent type must be resolved before the CIR codegen");

  case CK_NoOp:
  case CK_LValueToRValue:
  case CK_UserDefinedConversion:
    return Visit(op);

  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic: {
    cgf.cgm.errorNYI("ComplexExprEmitter::emitCast Atmoic");
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
  case CK_HLSLMatrixTruncation:
  case CK_HLSLArrayRValue:
  case CK_HLSLElementwiseCast:
  case CK_HLSLAggregateSplatCast:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op);
    return emitScalarToComplexCast(cgf.emitScalarExpr(op), op->getType(),
                                   destTy, op->getExprLoc());
  }

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex: {
    CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op);
    return emitComplexToComplexCast(Visit(op), op->getType(), destTy,
                                    op->getExprLoc());
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

aiir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  aiir::Value result = VisitUnaryPlus(e, promotionTy);
  if (!promotionTy.isNull())
    return cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

aiir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *e,
                                               QualType promotionType) {
  if (!promotionType.isNull())
    return cgf.emitPromotedComplexExpr(e->getSubExpr(), promotionType);
  return Visit(e->getSubExpr());
}

aiir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *e) {
  QualType promotionTy = getPromotionType(e->getSubExpr()->getType());
  aiir::Value result = VisitUnaryMinus(e, promotionTy);
  if (!promotionTy.isNull())
    return cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

aiir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *e,
                                                QualType promotionType) {
  aiir::Value op;
  if (!promotionType.isNull())
    op = cgf.emitPromotedComplexExpr(e->getSubExpr(), promotionType);
  else
    op = Visit(e->getSubExpr());
  return builder.createMinus(cgf.getLoc(e->getExprLoc()), op);
}

aiir::Value ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *e) {
  aiir::Value op = Visit(e->getSubExpr());
  return builder.createNot(op);
}

aiir::Value ComplexExprEmitter::emitBinAdd(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op.fpFeatures);

  if (aiir::isa<cir::ComplexType>(op.lhs.getType()) &&
      aiir::isa<cir::ComplexType>(op.rhs.getType()))
    return cir::ComplexAddOp::create(builder, op.loc, op.lhs, op.rhs);

  if (aiir::isa<cir::ComplexType>(op.lhs.getType())) {
    aiir::Value real = builder.createComplexReal(op.loc, op.lhs);
    aiir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    aiir::Value newReal = builder.createAdd(op.loc, real, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, imag);
  }

  assert(aiir::isa<cir::ComplexType>(op.rhs.getType()));
  aiir::Value real = builder.createComplexReal(op.loc, op.rhs);
  aiir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  aiir::Value newReal = builder.createAdd(op.loc, op.lhs, real);
  return builder.createComplexCreate(op.loc, newReal, imag);
}

aiir::Value ComplexExprEmitter::emitBinSub(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op.fpFeatures);

  if (aiir::isa<cir::ComplexType>(op.lhs.getType()) &&
      aiir::isa<cir::ComplexType>(op.rhs.getType()))
    return cir::ComplexSubOp::create(builder, op.loc, op.lhs, op.rhs);

  if (aiir::isa<cir::ComplexType>(op.lhs.getType())) {
    aiir::Value real = builder.createComplexReal(op.loc, op.lhs);
    aiir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    aiir::Value newReal = builder.createSub(op.loc, real, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, imag);
  }

  assert(aiir::isa<cir::ComplexType>(op.rhs.getType()));
  aiir::Value real = builder.createComplexReal(op.loc, op.rhs);
  aiir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  aiir::Value newReal = builder.createSub(op.loc, op.lhs, real);
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

aiir::Value ComplexExprEmitter::emitBinMul(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op.fpFeatures);

  if (aiir::isa<cir::ComplexType>(op.lhs.getType()) &&
      aiir::isa<cir::ComplexType>(op.rhs.getType())) {
    cir::ComplexRangeKind rangeKind =
        getComplexRangeAttr(op.fpFeatures.getComplexRange());
    return cir::ComplexMulOp::create(builder, op.loc, op.lhs, op.rhs,
                                     rangeKind);
  }

  if (aiir::isa<cir::ComplexType>(op.lhs.getType())) {
    aiir::Value real = builder.createComplexReal(op.loc, op.lhs);
    aiir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    aiir::Value newReal = builder.createMul(op.loc, real, op.rhs);
    aiir::Value newImag = builder.createMul(op.loc, imag, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, newImag);
  }

  assert(aiir::isa<cir::ComplexType>(op.rhs.getType()));
  aiir::Value real = builder.createComplexReal(op.loc, op.rhs);
  aiir::Value imag = builder.createComplexImag(op.loc, op.rhs);
  aiir::Value newReal = builder.createMul(op.loc, op.lhs, real);
  aiir::Value newImag = builder.createMul(op.loc, op.lhs, imag);
  return builder.createComplexCreate(op.loc, newReal, newImag);
}

aiir::Value ComplexExprEmitter::emitBinDiv(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::fastMathFlags());
  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, op.fpFeatures);

  // Handle division between two complex values. In the case of complex integer
  // types mixed with scalar integers, the scalar integer type will always be
  // promoted to a complex integer value with a zero imaginary component when
  // the AST is formed.
  if (aiir::isa<cir::ComplexType>(op.lhs.getType()) &&
      aiir::isa<cir::ComplexType>(op.rhs.getType())) {
    cir::ComplexRangeKind rangeKind =
        getComplexRangeAttr(op.fpFeatures.getComplexRange());
    return cir::ComplexDivOp::create(builder, op.loc, op.lhs, op.rhs,
                                     rangeKind);
  }

  // The C99 standard (G.5.1) defines division of a complex value by a real
  // value in the following simplified form.
  if (aiir::isa<cir::ComplexType>(op.lhs.getType())) {
    assert(aiir::cast<cir::ComplexType>(op.lhs.getType()).getElementType() ==
           op.rhs.getType());
    aiir::Value real = builder.createComplexReal(op.loc, op.lhs);
    aiir::Value imag = builder.createComplexImag(op.loc, op.lhs);
    aiir::Value newReal = builder.createFDiv(op.loc, real, op.rhs);
    aiir::Value newImag = builder.createFDiv(op.loc, imag, op.rhs);
    return builder.createComplexCreate(op.loc, newReal, newImag);
  }

  assert(aiir::isa<cir::ComplexType>(op.rhs.getType()));
  cir::ConstantOp nullValue = builder.getNullValue(op.lhs.getType(), op.loc);
  aiir::Value lhs = builder.createComplexCreate(op.loc, op.lhs, nullValue);
  cir::ComplexRangeKind rangeKind =
      getComplexRangeAttr(op.fpFeatures.getComplexRange());
  return cir::ComplexDivOp::create(builder, op.loc, lhs, op.rhs, rangeKind);
}

aiir::Value CIRGenFunction::emitUnPromotedValue(aiir::Value result,
                                                QualType unPromotionType) {
  assert(!aiir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(unPromotionType));
}

aiir::Value CIRGenFunction::emitPromotedValue(aiir::Value result,
                                              QualType promotionType) {
  assert(!aiir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(promotionType));
}

aiir::Value ComplexExprEmitter::emitPromoted(const Expr *e,
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
    case UO_Plus:
      return VisitUnaryPlus(unaryOp, promotionTy);
    case UO_Minus:
      return VisitUnaryMinus(unaryOp, promotionTy);
    default:
      break;
    }
  }

  aiir::Value result = Visit(const_cast<Expr *>(e));
  if (!promotionTy.isNull())
    return cgf.emitPromotedValue(result, promotionTy);

  return result;
}

aiir::Value CIRGenFunction::emitPromotedComplexExpr(const Expr *e,
                                                    QualType promotionType) {
  return ComplexExprEmitter(*this).emitPromoted(e, promotionType);
}

aiir::Value
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
    aiir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &value) {
  QualType lhsTy = e->getLHS()->getType();
  QualType rhsTy = e->getRHS()->getType();
  SourceLocation exprLoc = e->getExprLoc();
  aiir::Location loc = cgf.getLoc(exprLoc);

  if (lhsTy->getAs<AtomicType>()) {
    cgf.cgm.errorNYI("emitCompoundAssignLValue AtmoicType");
    return {};
  }

  BinOpInfo opInfo{loc};
  opInfo.fpFeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());

  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(cgf, opInfo.fpFeatures);

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
    aiir::Value lhsValue = emitLoadOfLValue(lhs, exprLoc);
    QualType destTy = promotionTypeLHS.isNull() ? opInfo.ty : promotionTypeLHS;
    opInfo.lhs = emitComplexToComplexCast(lhsValue, lhsTy, destTy, exprLoc);
  } else {
    aiir::Value lhsVal = cgf.emitLoadOfScalar(lhs, exprLoc);
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
  aiir::Value result = (this->*func)(opInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (lhsTy->isAnyComplexType()) {
    aiir::Value resultValue =
        emitComplexToComplexCast(result, opInfo.ty, lhsTy, exprLoc);
    emitStoreOfComplex(loc, resultValue, lhs, /*isInit*/ false);
    value = RValue::getComplex(resultValue);
  } else {
    aiir::Value resultValue =
        cgf.emitComplexToScalarConversion(result, opInfo.ty, lhsTy, exprLoc);
    cgf.emitStoreOfScalar(resultValue, lhs, /*isInit*/ false);
    value = RValue::get(resultValue);
  }

  return lhs;
}

aiir::Value ComplexExprEmitter::emitCompoundAssign(
    const CompoundAssignOperator *e,
    aiir::Value (ComplexExprEmitter::*func)(const BinOpInfo &)) {
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
                                               aiir::Value &value) {
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

aiir::Value ComplexExprEmitter::VisitBinAssign(const BinaryOperator *e) {
  aiir::Value value;
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

aiir::Value ComplexExprEmitter::VisitBinComma(const BinaryOperator *e) {
  cgf.emitIgnoredExpr(e->getLHS());
  return Visit(e->getRHS());
}

aiir::Value ComplexExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  aiir::Location loc = cgf.getLoc(e->getSourceRange());

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(cgf, e);

  CIRGenFunction::ConditionalEvaluation eval(cgf);

  Expr *cond = e->getCond()->IgnoreParens();
  aiir::Value condValue = cgf.evaluateExprAsBool(cond);

  return cir::TernaryOp::create(
             builder, loc, condValue,
             /*thenBuilder=*/
             [&](aiir::OpBuilder &b, aiir::Location loc) {
               eval.beginEvaluation();
               aiir::Value trueValue = Visit(e->getTrueExpr());
               cir::YieldOp::create(b, loc, trueValue);
               eval.endEvaluation();
             },
             /*elseBuilder=*/
             [&](aiir::OpBuilder &b, aiir::Location loc) {
               eval.beginEvaluation();
               aiir::Value falseValue = Visit(e->getFalseExpr());
               cir::YieldOp::create(b, loc, falseValue);
               eval.endEvaluation();
             })
      .getResult();
}

aiir::Value ComplexExprEmitter::VisitChooseExpr(ChooseExpr *e) {
  return Visit(e->getChosenSubExpr());
}

aiir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *e) {
  aiir::Location loc = cgf.getLoc(e->getExprLoc());
  if (e->getNumInits() == 2) {
    aiir::Value real = cgf.emitScalarExpr(e->getInit(0));
    aiir::Value imag = cgf.emitScalarExpr(e->getInit(1));
    return builder.createComplexCreate(loc, real, imag);
  }

  if (e->getNumInits() == 1)
    return Visit(e->getInit(0));

  assert(e->getNumInits() == 0 && "Unexpected number of inits");
  aiir::Type complexTy = cgf.convertType(e->getType());
  return builder.getNullValue(complexTy, loc);
}

aiir::Value ComplexExprEmitter::VisitVAArgExpr(VAArgExpr *e) {
  return cgf.emitVAArg(e);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
aiir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitComplexExprIntoLValue(const Expr *e, LValue dest,
                                               bool isInit) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter emitter(*this);
  aiir::Value value = emitter.Visit(const_cast<Expr *>(e));
  emitter.emitStoreOfComplex(getLoc(e->getExprLoc()), value, dest, isInit);
}

/// EmitStoreOfComplex - Store a complex number into the specified l-value.
void CIRGenFunction::emitStoreOfComplex(aiir::Location loc, aiir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}

aiir::Value CIRGenFunction::emitLoadOfComplex(LValue src, SourceLocation loc) {
  return ComplexExprEmitter(*this).emitLoadOfLValue(src, loc);
}

LValue CIRGenFunction::emitComplexAssignmentLValue(const BinaryOperator *e) {
  assert(e->getOpcode() == BO_Assign && "Expected assign op");

  aiir::Value value; // ignored
  LValue lvalue = ComplexExprEmitter(*this).emitBinAssignLValue(e, value);
  if (getLangOpts().OpenMP)
    cgm.errorNYI("emitComplexAssignmentLValue OpenMP");

  return lvalue;
}

using CompoundFunc =
    aiir::Value (ComplexExprEmitter::*)(const ComplexExprEmitter::BinOpInfo &);

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

aiir::Value CIRGenFunction::emitComplexPrePostIncDec(const UnaryOperator *e,
                                                     LValue lv) {
  aiir::Value inVal = emitLoadOfComplex(lv, e->getExprLoc());
  aiir::Location loc = getLoc(e->getExprLoc());
  aiir::Value incVal = e->isIncrementOp() ? builder.createInc(loc, inVal)
                                          : builder.createDec(loc, inVal);

  // Store the updated result through the lvalue.
  emitStoreOfComplex(loc, incVal, lv, /*isInit=*/false);

  if (getLangOpts().OpenMP)
    cgm.errorNYI(loc, "emitComplexPrePostIncDec OpenMP");

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return e->isPrefix() ? incVal : inVal;
}

LValue CIRGenFunction::emitScalarCompoundAssignWithComplex(
    const CompoundAssignOperator *e, aiir::Value &result) {
  // Key Instructions: Don't need to create an atom group here; one will already
  // be active through scalar handling code.
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue value;
  LValue ret = ComplexExprEmitter(*this).emitCompoundAssignLValue(e, op, value);
  result = value.getValue();
  return ret;
}
