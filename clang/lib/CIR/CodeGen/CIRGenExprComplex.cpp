#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"

#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace clang::CIRGen;

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

  LValue emitBinAssignLValue(const BinaryOperator *e, mlir::Value &val);

  mlir::Value emitCast(CastKind ck, Expr *op, QualType destTy);

  mlir::Value emitConstant(const CIRGenFunction::ConstantEmission &constant,
                           Expr *e);

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

  mlir::Value VisitArraySubscriptExpr(Expr *e);
  mlir::Value VisitBinAssign(const BinaryOperator *e);
  mlir::Value VisitBinComma(const BinaryOperator *e);
  mlir::Value VisitCallExpr(const CallExpr *e);
  mlir::Value VisitChooseExpr(ChooseExpr *e);
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e);
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *e);
  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *e);
  mlir::Value VisitInitListExpr(const InitListExpr *e);
  mlir::Value VisitImaginaryLiteral(const ImaginaryLiteral *il);
  mlir::Value VisitParenExpr(ParenExpr *e);
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *e);
  mlir::Value VisitUnaryDeref(const Expr *e);
};
} // namespace

static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
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
  emitStoreOfComplex(cgf.getLoc(e->getExprLoc()), value, lhs, /*isInit*/ false);
  return lhs;
}

mlir::Value ComplexExprEmitter::emitCast(CastKind ck, Expr *op,
                                         QualType destTy) {
  switch (ck) {
  case CK_LValueToRValue:
    return Visit(op);
  default:
    cgf.cgm.errorNYI("ComplexType Cast");
    break;
  }
  return {};
}

mlir::Value ComplexExprEmitter::emitConstant(
    const CIRGenFunction::ConstantEmission &constant, Expr *e) {
  assert(constant && "not a constant");
  if (constant.isReference())
    return emitLoadOfLValue(constant.getReferenceLValue(cgf, e),
                            e->getExprLoc());

  mlir::TypedAttr valueAttr = constant.getValue();
  return builder.getConstant(cgf.getLoc(e->getSourceRange()), valueAttr);
}

mlir::Value ComplexExprEmitter::emitLoadOfLValue(LValue lv,
                                                 SourceLocation loc) {
  assert(lv.isSimple() && "non-simple complex l-value?");
  if (lv.getType()->isAtomicType())
    cgf.cgm.errorNYI(loc, "emitLoadOfLValue with Atomic LV");

  const Address srcAddr = lv.getAddress();
  return builder.createLoad(cgf.getLoc(loc), srcAddr);
}

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

mlir::Value ComplexExprEmitter::VisitArraySubscriptExpr(Expr *e) {
  return emitLoadOfLValue(e);
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

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);

  return cgf.emitCallExpr(e).getValue();
}

mlir::Value ComplexExprEmitter::VisitChooseExpr(ChooseExpr *e) {
  return Visit(e->getChosenSubExpr());
}

mlir::Value ComplexExprEmitter::VisitDeclRefExpr(DeclRefExpr *e) {
  if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(e))
    return emitConstant(constant, e);
  return emitLoadOfLValue(e);
}

mlir::Value
ComplexExprEmitter::VisitGenericSelectionExpr(GenericSelectionExpr *e) {
  return Visit(e->getResultExpr());
}

mlir::Value ComplexExprEmitter::VisitImplicitCastExpr(ImplicitCastExpr *e) {
  // Unlike for scalars, we don't have to worry about function->ptr demotion
  // here.
  if (e->changesVolatileQualification())
    return emitLoadOfLValue(e);
  return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(const InitListExpr *e) {
  mlir::Location loc = cgf.getLoc(e->getExprLoc());
  if (e->getNumInits() == 2) {
    mlir::Value real = cgf.emitScalarExpr(e->getInit(0));
    mlir::Value imag = cgf.emitScalarExpr(e->getInit(1));
    return builder.createComplexCreate(loc, real, imag);
  }

  if (e->getNumInits() == 1) {
    cgf.cgm.errorNYI("Create Complex with InitList with size 1");
    return {};
  }

  assert(e->getNumInits() == 0 && "Unexpected number of inits");
  QualType complexElemTy =
      e->getType()->castAs<clang::ComplexType>()->getElementType();
  mlir::Type complexElemLLVMTy = cgf.convertType(complexElemTy);
  mlir::TypedAttr defaultValue = builder.getZeroInitAttr(complexElemLLVMTy);
  auto complexAttr = cir::ConstComplexAttr::get(defaultValue, defaultValue);
  return builder.create<cir::ConstantOp>(loc, complexAttr);
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

mlir::Value ComplexExprEmitter::VisitParenExpr(ParenExpr *e) {
  return Visit(e->getSubExpr());
}

mlir::Value ComplexExprEmitter::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *e) {
  return Visit(e->getReplacement());
}

mlir::Value ComplexExprEmitter::VisitUnaryDeref(const Expr *e) {
  return emitLoadOfLValue(e);
}

LValue CIRGenFunction::emitComplexAssignmentLValue(const BinaryOperator *e) {
  assert(e->getOpcode() == BO_Assign && "Expected assign op");

  mlir::Value value; // ignored
  LValue lvalue = ComplexExprEmitter(*this).emitBinAssignLValue(e, value);
  if (getLangOpts().OpenMP)
    cgm.errorNYI("emitComplexAssignmentLValue OpenMP");

  return lvalue;
}

mlir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitStoreOfComplex(mlir::Location loc, mlir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}
