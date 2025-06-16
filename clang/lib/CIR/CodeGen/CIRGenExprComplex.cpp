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

  /// Store the specified real/imag parts into the
  /// specified value pointer.
  void emitStoreOfComplex(mlir::Location loc, mlir::Value val, LValue lv,
                          bool isInit);

  mlir::Value VisitInitListExpr(InitListExpr *e);
};

} // namespace

static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
}

void ComplexExprEmitter::emitStoreOfComplex(mlir::Location loc, mlir::Value val,
                                            LValue lv, bool isInit) {
  if (lv.getType()->isAtomicType() ||
      (!isInit && cgf.isLValueSuitableForInlineAtomic(lv))) {
    cgf.cgm.errorNYI("StoreOfComplex with Atomic LV");
    return;
  }

  const Address destAddr = lv.getAddress();
  builder.createStore(loc, val, destAddr);
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

mlir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitStoreOfComplex(mlir::Location loc, mlir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}
