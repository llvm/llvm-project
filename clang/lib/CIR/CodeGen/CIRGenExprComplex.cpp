#include "CIRGenBuilder.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "clang/CIR/MissingFeatures.h"

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
  mlir::Value VisitCastExpr(CastExpr *E) { llvm_unreachable("NYI"); }
  mlir::Value VisitCallExpr(const CallExpr *E) { llvm_unreachable("NYI"); }
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
                             QualType PromotionType = QualType()) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitPlus(const UnaryOperator *E, QualType PromotionType) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryMinus(const UnaryOperator *E,
                              QualType PromotionType = QualType()) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitMinus(const UnaryOperator *E, QualType PromotionType) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitUnaryNot(const UnaryOperator *E) { llvm_unreachable("NYI"); }
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

#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) { llvm_unreachable("NYI"); }

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
    llvm_unreachable("NYI");
  }
  mlir::Value VisitBinSubAssign(const CompoundAssignOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitBinMulAssign(const CompoundAssignOperator *E) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitBinDivAssign(const CompoundAssignOperator *E) {
    llvm_unreachable("NYI");
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
  case CK_IntegralRealToComplex:
    llvm_unreachable("NYI");

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
    llvm_unreachable("NYI");
  }

  llvm_unreachable("unknown cast resulting in complex value");
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
