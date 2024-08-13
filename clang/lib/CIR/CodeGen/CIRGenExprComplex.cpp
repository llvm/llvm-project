#include "CIRGenBuilder.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"
#include "clang/CIR/MissingFeatures.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;

namespace {

class ComplexExprEmitter : public StmtVisitor<ComplexExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  CIRGenBuilderTy &Builder;
  bool FPHasBeenPromoted;

public:
  explicit ComplexExprEmitter(CIRGenFunction &cgf)
      : CGF(cgf), Builder(cgf.getBuilder()), FPHasBeenPromoted(false) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with complex type that represents a value l-value,
  /// this method emits the address of the l-value, then loads and returns the
  /// result.
  mlir::Value buildLoadOfLValue(const Expr *E) {
    return buildLoadOfLValue(CGF.buildLValue(E), E->getExprLoc());
  }

  mlir::Value buildLoadOfLValue(LValue LV, SourceLocation Loc);

  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void buildStoreOfComplex(mlir::Location Loc, mlir::Value Val, LValue LV,
                           bool isInit);

  /// Emit a cast from complex value Val to DestType.
  mlir::Value buildComplexToComplexCast(mlir::Value Val, QualType SrcType,
                                        QualType DestType, SourceLocation Loc);
  /// Emit a cast from scalar value Val to DestType.
  mlir::Value buildScalarToComplexCast(mlir::Value Val, QualType SrcType,
                                       QualType DestType, SourceLocation Loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *E) {
    assert(!MissingFeatures::generateDebugInfo());
    return StmtVisitor<ComplexExprEmitter, mlir::Value>::Visit(E);
  }

  mlir::Value VisitStmt(Stmt *S) {
    S->dump(llvm::errs(), CGF.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *S) { llvm_unreachable("not supported"); }
  mlir::Value VisitConstantExpr(ConstantExpr *E) {
    if (auto Result = ConstantEmitter(CGF).tryEmitConstantExpr(E))
      return Builder.getConstant(CGF.getLoc(E->getSourceRange()),
                                 mlir::cast<mlir::TypedAttr>(Result));
    return Visit(E->getSubExpr());
  }
  mlir::Value VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    return Visit(GE->getResultExpr());
  }
  mlir::Value VisitImaginaryLiteral(const ImaginaryLiteral *IL);
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE) {
    return Visit(PE->getReplacement());
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *S) { llvm_unreachable("NYI"); }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *S) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }

  mlir::Value emitConstant(const CIRGenFunction::ConstantEmission &Constant,
                           Expr *E) {
    assert(Constant && "not a constant");
    if (Constant.isReference())
      return buildLoadOfLValue(Constant.getReferenceLValue(CGF, E),
                               E->getExprLoc());

    auto valueAttr = Constant.getValue();
    return Builder.getConstant(CGF.getLoc(E->getSourceRange()), valueAttr);
  }

  // l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
    if (CIRGenFunction::ConstantEmission Constant = CGF.tryEmitAsConstant(E))
      return emitConstant(Constant, E);
    return buildLoadOfLValue(E);
  }
  mlir::Value VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArraySubscriptExpr(Expr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitMemberExpr(MemberExpr *ME) { llvm_unreachable("NYI"); }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *E) {
    llvm_unreachable("NYI");
  }

  // FIXME: CompoundLiteralExpr

  mlir::Value buildCast(CastKind CK, Expr *Op, QualType DestTy);
  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *E) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    if (E->changesVolatileQualification())
      return buildLoadOfLValue(E);
    return buildCast(E->getCastKind(), E->getSubExpr(), E->getType());
  }
  mlir::Value VisitCastExpr(CastExpr *E);
  mlir::Value VisitCallExpr(const CallExpr *E);
  mlir::Value VisitStmtExpr(const StmtExpr *E) { llvm_unreachable("NYI"); }

  // Operators.
  mlir::Value VisitPrePostIncDec(const UnaryOperator *E, bool isInc,
                                 bool isPre) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryPostDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, true);
  }
  mlir::Value VisitUnaryDeref(const Expr *E) { llvm_unreachable("NYI"); }

  mlir::Value VisitUnaryPlus(const UnaryOperator *E,
                             QualType PromotionType = QualType());
  mlir::Value VisitPlus(const UnaryOperator *E, QualType PromotionType);
  mlir::Value VisitUnaryMinus(const UnaryOperator *E,
                              QualType PromotionType = QualType());
  mlir::Value VisitMinus(const UnaryOperator *E, QualType PromotionType);
  mlir::Value VisitUnaryNot(const UnaryOperator *E);
  // LNot,Real,Imag never return complex.
  mlir::Value VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExprWithCleanups(ExprWithCleanups *E) {
    CIRGenFunction::RunCleanupsScope Scope(CGF);
    mlir::Value V = Visit(E->getSubExpr());
    // Defend against dominance problems caused by jumps out of expression
    // evaluation through the shared cleanup block.
    Scope.ForceCleanup({&V});
    return V;
  }
  mlir::Value VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    llvm_unreachable("NYI");
  }

  struct BinOpInfo {
    mlir::Location Loc;
    mlir::Value LHS{};
    mlir::Value RHS{};
    QualType Ty{}; // Computation Type.
    FPOptions FPFeatures{};
  };

  BinOpInfo buildBinOps(const BinaryOperator *E,
                        QualType PromotionTy = QualType());
  mlir::Value buildPromoted(const Expr *E, QualType PromotionTy);
  mlir::Value buildPromotedComplexOperand(const Expr *E, QualType PromotionTy);

  LValue buildCompoundAssignLValue(
      const CompoundAssignOperator *E,
      mlir::Value (ComplexExprEmitter::*Func)(const BinOpInfo &), RValue &Val);
  mlir::Value buildCompoundAssign(
      const CompoundAssignOperator *E,
      mlir::Value (ComplexExprEmitter::*Func)(const BinOpInfo &));

  mlir::Value buildBinAdd(const BinOpInfo &Op);
  mlir::Value buildBinSub(const BinOpInfo &Op);
  mlir::Value buildBinMul(const BinOpInfo &Op);
  mlir::Value buildBinDiv(const BinOpInfo &Op);

  QualType GetHigherPrecisionFPType(QualType ElementType) {
    const auto *CurrentBT = cast<BuiltinType>(ElementType);
    switch (CurrentBT->getKind()) {
    case BuiltinType::Kind::Float16:
      return CGF.getContext().FloatTy;
    case BuiltinType::Kind::Float:
    case BuiltinType::Kind::BFloat16:
      return CGF.getContext().DoubleTy;
    case BuiltinType::Kind::Double:
      return CGF.getContext().LongDoubleTy;
    default:
      return ElementType;
    }
  }

  QualType HigherPrecisionTypeForComplexArithmetic(QualType ElementType,
                                                   bool IsDivOpCode) {
    QualType HigherElementType = GetHigherPrecisionFPType(ElementType);
    const llvm::fltSemantics &ElementTypeSemantics =
        CGF.getContext().getFloatTypeSemantics(ElementType);
    const llvm::fltSemantics &HigherElementTypeSemantics =
        CGF.getContext().getFloatTypeSemantics(HigherElementType);
    // Check that the promoted type can handle the intermediate values without
    // overflowing. This can be interpreted as:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal) * 2 <=
    // LargerType.LargestFiniteVal.
    // In terms of exponent it gives this formula:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal
    // doubles the exponent of SmallerType.LargestFiniteVal)
    if (llvm::APFloat::semanticsMaxExponent(ElementTypeSemantics) * 2 + 1 <=
        llvm::APFloat::semanticsMaxExponent(HigherElementTypeSemantics)) {
      FPHasBeenPromoted = true;
      return CGF.getContext().getComplexType(HigherElementType);
    }

    DiagnosticsEngine &Diags = CGF.CGM.getDiags();
    Diags.Report(diag::warn_next_larger_fp_type_same_size_than_fp);
    return QualType();
  }

  QualType getPromotionType(QualType Ty, bool IsDivOpCode = false) {
    if (auto *CT = Ty->getAs<ComplexType>()) {
      QualType ElementType = CT->getElementType();
      if (IsDivOpCode && ElementType->isFloatingType() &&
          CGF.getLangOpts().getComplexRange() ==
              LangOptions::ComplexRangeKind::CX_Promoted)
        return HigherPrecisionTypeForComplexArithmetic(ElementType,
                                                       IsDivOpCode);
      if (ElementType.UseExcessPrecision(CGF.getContext()))
        return CGF.getContext().getComplexType(CGF.getContext().FloatTy);
    }
    if (Ty.UseExcessPrecision(CGF.getContext()))
      return CGF.getContext().FloatTy;
    return QualType();
  }

#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    QualType promotionTy = getPromotionType(                                   \
        E->getType(),                                                          \
        (E->getOpcode() == BinaryOperatorKind::BO_Div) ? true : false);        \
    mlir::Value result = buildBin##OP(buildBinOps(E, promotionTy));            \
    if (!promotionTy.isNull())                                                 \
      result = CGF.buildUnPromotedValue(result, E->getType());                 \
    return result;                                                             \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
#undef HANDLEBINOP

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  // Compound assignments.
  mlir::Value VisitBinAddAssign(const CompoundAssignOperator *E) {
    return buildCompoundAssign(E, &ComplexExprEmitter::buildBinAdd);
  }
  mlir::Value VisitBinSubAssign(const CompoundAssignOperator *E) {
    return buildCompoundAssign(E, &ComplexExprEmitter::buildBinSub);
  }
  mlir::Value VisitBinMulAssign(const CompoundAssignOperator *E) {
    return buildCompoundAssign(E, &ComplexExprEmitter::buildBinMul);
  }
  mlir::Value VisitBinDivAssign(const CompoundAssignOperator *E) {
    return buildCompoundAssign(E, &ComplexExprEmitter::buildBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue buildBinAssignLValue(const BinaryOperator *E, mlir::Value &Val);
  mlir::Value VisitBinAssign(const BinaryOperator *E) {
    mlir::Value Val;
    LValue LV = buildBinAssignLValue(E, Val);

    // The result of an assignment in C is the assigned r-value.
    if (!CGF.getLangOpts().CPlusPlus)
      return Val;

    // If the lvalue is non-volatile, return the computed value of the
    // assignment.
    if (!LV.isVolatileQualified())
      return Val;

    return buildLoadOfLValue(LV, E->getExprLoc());
  };
  mlir::Value VisitBinComma(const BinaryOperator *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *CO) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitChooseExpr(ChooseExpr *CE) { llvm_unreachable("NYI"); }

  mlir::Value VisitInitListExpr(InitListExpr *E);

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitVAArgExpr(VAArgExpr *E) { llvm_unreachable("NYI"); }

  mlir::Value VisitAtomicExpr(AtomicExpr *E) { llvm_unreachable("NYI"); }

  mlir::Value VisitPackIndexingExpr(PackIndexingExpr *E) {
    llvm_unreachable("NYI");
  }
};

} // namespace

static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
}

static mlir::Value createComplexFromReal(CIRGenBuilderTy &builder,
                                         mlir::Location loc, mlir::Value real) {
  mlir::Value imag = builder.getNullValue(real.getType(), loc);
  return builder.createComplexCreate(loc, real, imag);
}

mlir::Value ComplexExprEmitter::buildLoadOfLValue(LValue LV,
                                                  SourceLocation Loc) {
  assert(LV.isSimple() && "non-simple complex l-value?");
  if (LV.getType()->isAtomicType())
    llvm_unreachable("NYI");

  Address SrcPtr = LV.getAddress();
  return Builder.createLoad(CGF.getLoc(Loc), SrcPtr, LV.isVolatileQualified());
}

void ComplexExprEmitter::buildStoreOfComplex(mlir::Location Loc,
                                             mlir::Value Val, LValue LV,
                                             bool isInit) {
  if (LV.getType()->isAtomicType() ||
      (!isInit && CGF.LValueIsSuitableForInlineAtomic(LV)))
    llvm_unreachable("NYI");

  Address DestAddr = LV.getAddress();
  Builder.createStore(Loc, Val, DestAddr, LV.isVolatileQualified());
}

mlir::Value ComplexExprEmitter::buildComplexToComplexCast(mlir::Value Val,
                                                          QualType SrcType,
                                                          QualType DestType,
                                                          SourceLocation Loc) {
  if (SrcType == DestType)
    return Val;

  // Get the src/dest element type.
  QualType SrcElemTy = SrcType->castAs<ComplexType>()->getElementType();
  QualType DestElemTy = DestType->castAs<ComplexType>()->getElementType();

  mlir::cir::CastKind CastOpKind;
  if (SrcElemTy->isFloatingType() && DestElemTy->isFloatingType())
    CastOpKind = mlir::cir::CastKind::float_complex;
  else if (SrcElemTy->isFloatingType() && DestElemTy->isIntegerType())
    CastOpKind = mlir::cir::CastKind::float_complex_to_int_complex;
  else if (SrcElemTy->isIntegerType() && DestElemTy->isFloatingType())
    CastOpKind = mlir::cir::CastKind::int_complex_to_float_complex;
  else if (SrcElemTy->isIntegerType() && DestElemTy->isIntegerType())
    CastOpKind = mlir::cir::CastKind::int_complex;
  else
    llvm_unreachable("unexpected src type or dest type");

  return Builder.createCast(CGF.getLoc(Loc), CastOpKind, Val,
                            CGF.ConvertType(DestType));
}

mlir::Value ComplexExprEmitter::buildScalarToComplexCast(mlir::Value Val,
                                                         QualType SrcType,
                                                         QualType DestType,
                                                         SourceLocation Loc) {
  mlir::cir::CastKind CastOpKind;
  if (SrcType->isFloatingType())
    CastOpKind = mlir::cir::CastKind::float_to_complex;
  else if (SrcType->isIntegerType())
    CastOpKind = mlir::cir::CastKind::int_to_complex;
  else
    llvm_unreachable("unexpected src type");

  return Builder.createCast(CGF.getLoc(Loc), CastOpKind, Val,
                            CGF.ConvertType(DestType));
}

mlir::Value ComplexExprEmitter::buildCast(CastKind CK, Expr *Op,
                                          QualType DestTy) {
  switch (CK) {
  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");

  // Atomic to non-atomic casts may be more than a no-op for some platforms and
  // for some types.
  case CK_LValueToRValue:
    return Visit(Op);

  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
    llvm_unreachable("NYI");

  case CK_LValueBitCast:
    llvm_unreachable("NYI");

  case CK_LValueToRValueBitCast:
    llvm_unreachable("NYI");

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
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex: {
    assert(!MissingFeatures::CGFPOptionsRAII());
    return buildScalarToComplexCast(CGF.buildScalarExpr(Op), Op->getType(),
                                    DestTy, Op->getExprLoc());
  }

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex: {
    assert(!MissingFeatures::CGFPOptionsRAII());
    return buildComplexToComplexCast(Visit(Op), Op->getType(), DestTy,
                                     Op->getExprLoc());
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

mlir::Value ComplexExprEmitter::VisitCastExpr(CastExpr *E) {
  if (const auto *ECE = dyn_cast<ExplicitCastExpr>(E))
    CGF.CGM.buildExplicitCastExprType(ECE, &CGF);
  if (E->changesVolatileQualification())
    return buildLoadOfLValue(E);
  return buildCast(E->getCastKind(), E->getSubExpr(), E->getType());
}

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType(CGF.getContext())->isReferenceType())
    return buildLoadOfLValue(E);

  return CGF.buildCallExpr(E).getComplexVal();
}

mlir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *E,
                                               QualType PromotionType) {
  QualType promotionTy = PromotionType.isNull()
                             ? getPromotionType(E->getSubExpr()->getType())
                             : PromotionType;
  mlir::Value result = VisitPlus(E, promotionTy);
  if (!promotionTy.isNull())
    return CGF.buildUnPromotedValue(result, E->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::VisitPlus(const UnaryOperator *E,
                                          QualType PromotionType) {
  mlir::Value Op;
  if (!PromotionType.isNull())
    Op = CGF.buildPromotedComplexExpr(E->getSubExpr(), PromotionType);
  else
    Op = Visit(E->getSubExpr());

  return Builder.createUnaryOp(CGF.getLoc(E->getExprLoc()),
                               mlir::cir::UnaryOpKind::Plus, Op);
}

mlir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *E,
                                                QualType PromotionType) {
  QualType promotionTy = PromotionType.isNull()
                             ? getPromotionType(E->getSubExpr()->getType())
                             : PromotionType;
  mlir::Value result = VisitMinus(E, promotionTy);
  if (!promotionTy.isNull())
    return CGF.buildUnPromotedValue(result, E->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::VisitMinus(const UnaryOperator *E,
                                           QualType PromotionType) {
  mlir::Value Op;
  if (!PromotionType.isNull())
    Op = CGF.buildPromotedComplexExpr(E->getSubExpr(), PromotionType);
  else
    Op = Visit(E->getSubExpr());

  return Builder.createUnaryOp(CGF.getLoc(E->getExprLoc()),
                               mlir::cir::UnaryOpKind::Minus, Op);
}

mlir::Value ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  mlir::Value Op = Visit(E->getSubExpr());
  return Builder.createUnaryOp(CGF.getLoc(E->getExprLoc()),
                               mlir::cir::UnaryOpKind::Not, Op);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::buildBinOps(const BinaryOperator *E, QualType PromotionTy) {
  BinOpInfo Ops{CGF.getLoc(E->getExprLoc())};

  Ops.LHS = buildPromotedComplexOperand(E->getLHS(), PromotionTy);
  Ops.RHS = buildPromotedComplexOperand(E->getRHS(), PromotionTy);
  if (!PromotionTy.isNull())
    Ops.Ty = PromotionTy;
  else
    Ops.Ty = E->getType();
  Ops.FPFeatures = E->getFPFeaturesInEffect(CGF.getLangOpts());
  return Ops;
}

mlir::Value ComplexExprEmitter::buildPromoted(const Expr *E,
                                              QualType PromotionTy) {
  E = E->IgnoreParens();
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return buildBin##OP(buildBinOps(BO, PromotionTy));
      HANDLE_BINOP(Add)
      HANDLE_BINOP(Sub)
      HANDLE_BINOP(Mul)
      HANDLE_BINOP(Div)
#undef HANDLE_BINOP
    default:
      break;
    }
  } else if (const auto *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    case UO_Minus:
      return VisitMinus(UO, PromotionTy);
    case UO_Plus:
      return VisitPlus(UO, PromotionTy);
    default:
      break;
    }
  }
  auto result = Visit(const_cast<Expr *>(E));
  if (!PromotionTy.isNull())
    return CGF.buildPromotedValue(result, PromotionTy);
  return result;
}

mlir::Value
ComplexExprEmitter::buildPromotedComplexOperand(const Expr *E,
                                                QualType PromotionTy) {
  if (E->getType()->isAnyComplexType()) {
    if (!PromotionTy.isNull())
      return CGF.buildPromotedComplexExpr(E, PromotionTy);
    return Visit(const_cast<Expr *>(E));
  }

  mlir::Value Real;
  if (!PromotionTy.isNull()) {
    QualType ComplexElementTy =
        PromotionTy->castAs<ComplexType>()->getElementType();
    Real = CGF.buildPromotedScalarExpr(E, ComplexElementTy);
  } else
    Real = CGF.buildScalarExpr(E);

  return createComplexFromReal(CGF.getBuilder(), CGF.getLoc(E->getExprLoc()),
                               Real);
}

LValue ComplexExprEmitter::buildCompoundAssignLValue(
    const CompoundAssignOperator *E,
    mlir::Value (ComplexExprEmitter::*Func)(const BinOpInfo &), RValue &Val) {
  QualType LHSTy = E->getLHS()->getType();
  if (const AtomicType *AT = LHSTy->getAs<AtomicType>())
    LHSTy = AT->getValueType();

  BinOpInfo OpInfo{CGF.getLoc(E->getExprLoc())};
  OpInfo.FPFeatures = E->getFPFeaturesInEffect(CGF.getLangOpts());

  assert(!MissingFeatures::CGFPOptionsRAII());

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  QualType PromotionTypeCR;
  PromotionTypeCR = getPromotionType(E->getComputationResultType());
  if (PromotionTypeCR.isNull())
    PromotionTypeCR = E->getComputationResultType();
  OpInfo.Ty = PromotionTypeCR;
  QualType ComplexElementTy =
      OpInfo.Ty->castAs<ComplexType>()->getElementType();
  QualType PromotionTypeRHS = getPromotionType(E->getRHS()->getType());

  // The RHS should have been converted to the computation type.
  if (E->getRHS()->getType()->isRealFloatingType()) {
    if (!PromotionTypeRHS.isNull())
      OpInfo.RHS = createComplexFromReal(
          CGF.getBuilder(), CGF.getLoc(E->getExprLoc()),
          CGF.buildPromotedScalarExpr(E->getRHS(), PromotionTypeRHS));
    else {
      assert(CGF.getContext().hasSameUnqualifiedType(ComplexElementTy,
                                                     E->getRHS()->getType()));
      OpInfo.RHS =
          createComplexFromReal(CGF.getBuilder(), CGF.getLoc(E->getExprLoc()),
                                CGF.buildScalarExpr(E->getRHS()));
    }
  } else {
    if (!PromotionTypeRHS.isNull()) {
      OpInfo.RHS = createComplexFromReal(
          CGF.getBuilder(), CGF.getLoc(E->getExprLoc()),
          CGF.buildPromotedComplexExpr(E->getRHS(), PromotionTypeRHS));
    } else {
      assert(CGF.getContext().hasSameUnqualifiedType(OpInfo.Ty,
                                                     E->getRHS()->getType()));
      OpInfo.RHS = Visit(E->getRHS());
    }
  }

  LValue LHS = CGF.buildLValue(E->getLHS());

  // Load from the l-value and convert it.
  SourceLocation Loc = E->getExprLoc();
  QualType PromotionTypeLHS = getPromotionType(E->getComputationLHSType());
  if (LHSTy->isAnyComplexType()) {
    mlir::Value LHSVal = buildLoadOfLValue(LHS, Loc);
    if (!PromotionTypeLHS.isNull())
      OpInfo.LHS =
          buildComplexToComplexCast(LHSVal, LHSTy, PromotionTypeLHS, Loc);
    else
      OpInfo.LHS = buildComplexToComplexCast(LHSVal, LHSTy, OpInfo.Ty, Loc);
  } else {
    mlir::Value LHSVal = CGF.buildLoadOfScalar(LHS, Loc);
    // For floating point real operands we can directly pass the scalar form
    // to the binary operator emission and potentially get more efficient code.
    if (LHSTy->isRealFloatingType()) {
      QualType PromotedComplexElementTy;
      if (!PromotionTypeLHS.isNull()) {
        PromotedComplexElementTy =
            cast<ComplexType>(PromotionTypeLHS)->getElementType();
        if (!CGF.getContext().hasSameUnqualifiedType(PromotedComplexElementTy,
                                                     PromotionTypeLHS))
          LHSVal = CGF.buildScalarConversion(LHSVal, LHSTy,
                                             PromotedComplexElementTy, Loc);
      } else {
        if (!CGF.getContext().hasSameUnqualifiedType(ComplexElementTy, LHSTy))
          LHSVal =
              CGF.buildScalarConversion(LHSVal, LHSTy, ComplexElementTy, Loc);
      }
      OpInfo.LHS = createComplexFromReal(CGF.getBuilder(),
                                         CGF.getLoc(E->getExprLoc()), LHSVal);
    } else {
      OpInfo.LHS = buildScalarToComplexCast(LHSVal, LHSTy, OpInfo.Ty, Loc);
    }
  }

  // Expand the binary operator.
  mlir::Value Result = (this->*Func)(OpInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (LHSTy->isAnyComplexType()) {
    mlir::Value ResVal =
        buildComplexToComplexCast(Result, OpInfo.Ty, LHSTy, Loc);
    buildStoreOfComplex(CGF.getLoc(E->getExprLoc()), ResVal, LHS,
                        /*isInit*/ false);
    Val = RValue::getComplex(ResVal);
  } else {
    mlir::Value ResVal =
        CGF.buildComplexToScalarConversion(Result, OpInfo.Ty, LHSTy, Loc);
    CGF.buildStoreOfScalar(ResVal, LHS, /*isInit*/ false);
    Val = RValue::get(ResVal);
  }

  return LHS;
}

mlir::Value ComplexExprEmitter::buildCompoundAssign(
    const CompoundAssignOperator *E,
    mlir::Value (ComplexExprEmitter::*Func)(const BinOpInfo &)) {
  RValue Val;
  LValue LV = buildCompoundAssignLValue(E, Func, Val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return Val.getComplexVal();

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LV.isVolatileQualified())
    return Val.getComplexVal();

  return buildLoadOfLValue(LV, E->getExprLoc());
}

mlir::Value ComplexExprEmitter::buildBinAdd(const BinOpInfo &Op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexAdd(Op.Loc, Op.LHS, Op.RHS);
}

mlir::Value ComplexExprEmitter::buildBinSub(const BinOpInfo &Op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexSub(Op.Loc, Op.LHS, Op.RHS);
}

static mlir::cir::ComplexRangeKind
getComplexRangeAttr(LangOptions::ComplexRangeKind range) {
  switch (range) {
  case LangOptions::CX_Full:
    return mlir::cir::ComplexRangeKind::Full;
  case LangOptions::CX_Improved:
    return mlir::cir::ComplexRangeKind::Improved;
  case LangOptions::CX_Promoted:
    return mlir::cir::ComplexRangeKind::Promoted;
  case LangOptions::CX_Basic:
    return mlir::cir::ComplexRangeKind::Basic;
  case LangOptions::CX_None:
    return mlir::cir::ComplexRangeKind::None;
  }
}

mlir::Value ComplexExprEmitter::buildBinMul(const BinOpInfo &Op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexMul(
      Op.Loc, Op.LHS, Op.RHS,
      getComplexRangeAttr(Op.FPFeatures.getComplexRange()), FPHasBeenPromoted);
}

mlir::Value ComplexExprEmitter::buildBinDiv(const BinOpInfo &Op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexDiv(
      Op.Loc, Op.LHS, Op.RHS,
      getComplexRangeAttr(Op.FPFeatures.getComplexRange()), FPHasBeenPromoted);
}

LValue ComplexExprEmitter::buildBinAssignLValue(const BinaryOperator *E,
                                                mlir::Value &Val) {
  assert(CGF.getContext().hasSameUnqualifiedType(E->getLHS()->getType(),
                                                 E->getRHS()->getType()) &&
         "Invalid assignment");

  // Emit the RHS.  __block variables need the RHS evaluated first.
  Val = Visit(E->getRHS());

  // Compute the address to store into.
  LValue LHS = CGF.buildLValue(E->getLHS());

  // Store the result value into the LHS lvalue.
  buildStoreOfComplex(CGF.getLoc(E->getExprLoc()), Val, LHS, /*isInit*/ false);

  return LHS;
}

mlir::Value
ComplexExprEmitter::VisitImaginaryLiteral(const ImaginaryLiteral *IL) {
  auto Loc = CGF.getLoc(IL->getExprLoc());
  auto Ty = mlir::cast<mlir::cir::ComplexType>(CGF.getCIRType(IL->getType()));
  auto ElementTy = Ty.getElementTy();

  mlir::TypedAttr RealValueAttr;
  mlir::TypedAttr ImagValueAttr;
  if (mlir::isa<mlir::cir::IntType>(ElementTy)) {
    auto ImagValue = cast<IntegerLiteral>(IL->getSubExpr())->getValue();
    RealValueAttr = mlir::cir::IntAttr::get(ElementTy, 0);
    ImagValueAttr = mlir::cir::IntAttr::get(ElementTy, ImagValue);
  } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(ElementTy)) {
    auto ImagValue = cast<FloatingLiteral>(IL->getSubExpr())->getValue();
    RealValueAttr = mlir::cir::FPAttr::get(
        ElementTy, llvm::APFloat::getZero(ImagValue.getSemantics()));
    ImagValueAttr = mlir::cir::FPAttr::get(ElementTy, ImagValue);
  } else
    llvm_unreachable("unexpected complex element type");

  auto RealValue = Builder.getConstant(Loc, RealValueAttr);
  auto ImagValue = Builder.getConstant(Loc, ImagValueAttr);
  return Builder.createComplexCreate(Loc, RealValue, ImagValue);
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *E) {
  if (E->getNumInits() == 2) {
    mlir::Value Real = CGF.buildScalarExpr(E->getInit(0));
    mlir::Value Imag = CGF.buildScalarExpr(E->getInit(1));
    return Builder.createComplexCreate(CGF.getLoc(E->getExprLoc()), Real, Imag);
  }

  if (E->getNumInits() == 1)
    return Visit(E->getInit(0));

  // Empty init list initializes to null
  assert(E->getNumInits() == 0 && "Unexpected number of inits");
  QualType Ty = E->getType()->castAs<ComplexType>()->getElementType();
  return Builder.getZero(CGF.getLoc(E->getExprLoc()), CGF.ConvertType(Ty));
}

mlir::Value CIRGenFunction::buildPromotedComplexExpr(const Expr *E,
                                                     QualType PromotionType) {
  return ComplexExprEmitter(*this).buildPromoted(E, PromotionType);
}

mlir::Value CIRGenFunction::buildPromotedValue(mlir::Value result,
                                               QualType PromotionType) {
  assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(
             mlir::cast<mlir::cir::ComplexType>(result.getType())
                 .getElementTy()) &&
         "integral complex will never be promoted");
  return builder.createCast(mlir::cir::CastKind::float_complex, result,
                            ConvertType(PromotionType));
}

mlir::Value CIRGenFunction::buildUnPromotedValue(mlir::Value result,
                                                 QualType UnPromotionType) {
  assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(
             mlir::cast<mlir::cir::ComplexType>(result.getType())
                 .getElementTy()) &&
         "integral complex will never be promoted");
  return builder.createCast(mlir::cir::CastKind::float_complex, result,
                            ConvertType(UnPromotionType));
}

mlir::Value CIRGenFunction::buildComplexExpr(const Expr *E) {
  assert(E && getComplexType(E->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(E));
}

void CIRGenFunction::buildComplexExprIntoLValue(const Expr *E, LValue dest,
                                                bool isInit) {
  assert(E && getComplexType(E->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter Emitter(*this);
  mlir::Value Val = Emitter.Visit(const_cast<Expr *>(E));
  Emitter.buildStoreOfComplex(getLoc(E->getExprLoc()), Val, dest, isInit);
}

void CIRGenFunction::buildStoreOfComplex(mlir::Location Loc, mlir::Value V,
                                         LValue dest, bool isInit) {
  ComplexExprEmitter(*this).buildStoreOfComplex(Loc, V, dest, isInit);
}

Address CIRGenFunction::buildAddrOfRealComponent(mlir::Location loc,
                                                 Address addr,
                                                 QualType complexType) {
  return builder.createRealPtr(loc, addr);
}

Address CIRGenFunction::buildAddrOfImagComponent(mlir::Location loc,
                                                 Address addr,
                                                 QualType complexType) {
  return builder.createImagPtr(loc, addr);
}

LValue CIRGenFunction::buildComplexAssignmentLValue(const BinaryOperator *E) {
  assert(E->getOpcode() == BO_Assign);
  mlir::Value Val; // ignored
  LValue LVal = ComplexExprEmitter(*this).buildBinAssignLValue(E, Val);
  if (getLangOpts().OpenMP)
    llvm_unreachable("NYI");
  return LVal;
}

using CompoundFunc =
    mlir::Value (ComplexExprEmitter::*)(const ComplexExprEmitter::BinOpInfo &);

static CompoundFunc getComplexOp(BinaryOperatorKind Op) {
  switch (Op) {
  case BO_MulAssign:
    return &ComplexExprEmitter::buildBinMul;
  case BO_DivAssign:
    return &ComplexExprEmitter::buildBinDiv;
  case BO_SubAssign:
    return &ComplexExprEmitter::buildBinSub;
  case BO_AddAssign:
    return &ComplexExprEmitter::buildBinAdd;
  default:
    llvm_unreachable("unexpected complex compound assignment");
  }
}

LValue CIRGenFunction::buildComplexCompoundAssignmentLValue(
    const CompoundAssignOperator *E) {
  CompoundFunc Op = getComplexOp(E->getOpcode());
  RValue Val;
  return ComplexExprEmitter(*this).buildCompoundAssignLValue(E, Op, Val);
}
