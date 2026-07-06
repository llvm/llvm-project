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

void ConvertComplexPowPass::runOnOperation() {
  ModuleOp mod = getOperation();
  fir::FirOpBuilder builder(mod, fir::getKindMapping(mod));

  mod.walk([&](Operation *op) {
    if (auto powIop = dyn_cast<complex::PowiOp>(op)) {
      builder.setInsertionPoint(powIop);
      Location loc = powIop.getLoc();
      auto complexTy = cast<ComplexType>(powIop.getType());
      auto elemTy = complexTy.getElementType();
      Value base = powIop.getLhs();
      Value intExp = powIop.getRhs();
      func::FuncOp callee;
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
      auto call = fir::CallOp::create(builder, loc, callee, {base, intExp});
      if (auto fmf = powIop.getFastmathAttr())
        call.setFastmathAttr(fmf);
      powIop.replaceAllUsesWith(call.getResult(0));
      powIop.erase();
    } else if (auto powOp = dyn_cast<complex::PowOp>(op)) {
      builder.setInsertionPoint(powOp);
      Location loc = powOp.getLoc();
      auto complexTy = cast<ComplexType>(powOp.getType());
      auto elemTy = complexTy.getElementType();
      unsigned realBits = cast<FloatType>(elemTy).getWidth();
      func::FuncOp callee;
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
      auto call = fir::CallOp::create(builder, loc, callee,
                                      {powOp.getLhs(), powOp.getRhs()});
      if (auto fmf = powOp.getFastmathAttr())
        call.setFastmathAttr(fmf);
      powOp.replaceAllUsesWith(call.getResult(0));
      powOp.erase();
    }
  });
}
