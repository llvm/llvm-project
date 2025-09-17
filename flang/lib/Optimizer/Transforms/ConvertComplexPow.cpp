//===- ConvertComplexPow.cpp - Convert complex.pow to library calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/static-multimap-view.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_CONVERTCOMPLEXPOW
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {
class ConvertComplexPowPass
    : public fir::impl::ConvertComplexPowBase<ConvertComplexPowPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<fir::FIROpsDialect, complex::ComplexDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};
} // namespace

// Helper to declare or get a math library function.
static func::FuncOp getOrDeclare(fir::FirOpBuilder &builder, Location loc,
                                 StringRef name, FunctionType type) {
  if (auto func = builder.getNamedFunction(name))
    return func;
  auto func = builder.createFunction(loc, name, type);
  func->setAttr(fir::getSymbolAttrName(), builder.getStringAttr(name));
  func->setAttr(fir::FIROpsDialect::getFirRuntimeAttrName(),
                builder.getUnitAttr());
  return func;
}

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<FloatAttr>(cst.getValue()))
      return attr.getValue().isZero();
  return false;
}

void ConvertComplexPowPass::runOnOperation() {
  ModuleOp mod = getOperation();
  fir::FirOpBuilder builder(mod, fir::getKindMapping(mod));

  mod.walk([&](complex::PowOp op) {
    builder.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto complexTy = cast<ComplexType>(op.getType());
    auto elemTy = complexTy.getElementType();

    Value base = op.getLhs();
    Value rhs = op.getRhs();

    Value intExp;
    if (auto create = rhs.getDefiningOp<complex::CreateOp>()) {
      if (isZero(create.getImaginary())) {
        if (auto conv = create.getReal().getDefiningOp<fir::ConvertOp>()) {
          if (auto intTy = dyn_cast<IntegerType>(conv.getValue().getType()))
            intExp = conv.getValue();
        }
      }
    }

    func::FuncOp callee;
    SmallVector<Value> args;
    if (intExp) {
      unsigned realBits = cast<FloatType>(elemTy).getWidth();
      unsigned intBits = cast<IntegerType>(intExp.getType()).getWidth();
      auto funcTy = builder.getFunctionType(
          {complexTy, builder.getIntegerType(intBits)}, {complexTy});
      if (realBits == 32 && intBits == 32)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(cpowi), funcTy);
      else if (realBits == 32 && intBits == 64)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(cpowk), funcTy);
      else if (realBits == 64 && intBits == 32)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(zpowi), funcTy);
      else if (realBits == 64 && intBits == 64)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(zpowk), funcTy);
      else if (realBits == 128 && intBits == 32)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(cqpowi), funcTy);
      else if (realBits == 128 && intBits == 64)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(cqpowk), funcTy);
      else
        return;
      args = {base, intExp};
    } else {
      unsigned realBits = cast<FloatType>(elemTy).getWidth();
      auto funcTy =
          builder.getFunctionType({complexTy, complexTy}, {complexTy});
      if (realBits == 32)
        callee = getOrDeclare(builder, loc, "cpowf", funcTy);
      else if (realBits == 64)
        callee = getOrDeclare(builder, loc, "cpow", funcTy);
      else if (realBits == 128)
        callee = getOrDeclare(builder, loc, RTNAME_STRING(CPowF128), funcTy);
      else
        return;
      args = {base, rhs};
    }

    auto call = fir::CallOp::create(builder, loc, callee, args);
    if (auto fmf = op.getFastmathAttr())
      call.setFastmathAttr(fmf);
    op.replaceAllUsesWith(call.getResult(0));
    op.erase();
  });
}
