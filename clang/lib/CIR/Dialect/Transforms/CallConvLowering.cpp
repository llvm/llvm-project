//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace cir {

struct CallConvLowering {
  CallConvLowering(mlir::ModuleOp module) : module(module) {}

  void lower(FuncOp op) {
    // TODO(cir): Implement calling convention lowering for function definitions
    // and calls once the upstream LowerModule has the necessary APIs.
  }

private:
  mlir::ModuleOp module;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  llvm::StringRef getArgument() const override {
    return "cir-call-conv-lowering";
  };
};

void CallConvLoweringPass::runOnOperation() {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(getOperation());
  CallConvLowering cc(module);
  module.walk([&](FuncOp op) { cc.lower(op); });
}

} // namespace cir

namespace mlir {

std::unique_ptr<mlir::Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
