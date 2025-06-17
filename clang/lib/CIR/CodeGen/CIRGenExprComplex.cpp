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

  mlir::Value VisitCallExpr(const CallExpr *e);
  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitImaginaryLiteral(const ImaginaryLiteral *il);
};

} // namespace

static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
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

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);

  return cgf.emitCallExpr(e).getValue();
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *e) {
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
    assert(mlir::isa<cir::CIRFPTypeInterface>(elementTy) &&
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

mlir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitStoreOfComplex(mlir::Location loc, mlir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}
