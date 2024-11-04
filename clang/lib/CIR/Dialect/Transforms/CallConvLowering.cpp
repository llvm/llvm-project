//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TargetLowering/LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

#include "llvm/Support/TimeProfiler.h"

namespace mlir {
namespace cir {

FuncType getFuncPointerTy(mlir::Type typ) {
  if (auto ptr = dyn_cast<PointerType>(typ))
    return dyn_cast<FuncType>(ptr.getPointee());
  return {};
}

bool isFuncPointerTy(mlir::Type typ) { return (bool)getFuncPointerTy(typ); }

struct CallConvLowering {

  CallConvLowering(ModuleOp module)
      : rewriter(module.getContext()),
        lowerModule(createLowerModule(module, rewriter)) {}

  void lower(FuncOp op) {
    // Fail the pass on unimplemented function users
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    auto calls = op.getSymbolUses(module);
    if (calls.has_value()) {
      for (auto call : calls.value()) {
        if (auto g = dyn_cast<GetGlobalOp>(call.getUser()))
          rewriteGetGlobalOp(g);
        else if (auto c = dyn_cast<CallOp>(call.getUser()))
          lowerDirectCallOp(c, op);
        else {
          cir_cconv_assert_or_abort(!::cir::MissingFeatures::ABIFuncPtr(),
                                    "NYI");
        }
      }
    }

    op.walk([&](CallOp c) {
      if (c.isIndirect())
        lowerIndirectCallOp(c);
    });

    if (lowerModule->rewriteFunctionDefinition(op).failed())
      op.emitError("Unable to rewrite function definition");
  }

private:
  FuncType convert(FuncType t) {
    auto &typs = lowerModule->getTypes();
    return typs.getFunctionType(typs.arrangeFreeFunctionType(t));
  }

  mlir::Type convert(mlir::Type t) {
    if (auto fTy = getFuncPointerTy(t))
      return PointerType::get(rewriter.getContext(), convert(fTy));
    return t;
  }

  void bitcast(Value src, Type newTy) {
    if (src.getType() != newTy) {
      auto cast =
          rewriter.create<CastOp>(src.getLoc(), newTy, CastKind::bitcast, src);
      rewriter.replaceAllUsesExcept(src, cast, cast);
    }
  }

  void rewriteGetGlobalOp(GetGlobalOp op) {
    auto resTy = op.getResult().getType();
    if (isFuncPointerTy(resTy)) {
      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.replaceOpWithNewOp<GetGlobalOp>(op, convert(resTy),
                                                            op.getName());
      rewriter.setInsertionPointAfter(newOp);
      bitcast(newOp, resTy);
    }
  }

  void lowerDirectCallOp(CallOp op, FuncOp callee) {
    if (lowerModule->rewriteFunctionCall(op, callee).failed())
      op.emitError("Unable to rewrite function call");
  }

  void lowerIndirectCallOp(CallOp op) {
    cir_cconv_assert(op.isIndirect());

    rewriter.setInsertionPoint(op);
    auto typ = op.getIndirectCall().getType();
    if (!isFuncPointerTy(typ))
      return;

    if (lowerModule->rewriteFunctionCall(op).failed())
      op.emitError("Unable to rewrite function call");
  }

private:
  mlir::PatternRewriter rewriter;
  std::unique_ptr<LowerModule> lowerModule;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "cir-call-conv-lowering"; };
};

void CallConvLoweringPass::runOnOperation() {
  auto module = dyn_cast<ModuleOp>(getOperation());
  CallConvLowering cc(module);
  module.walk([&](FuncOp op) { cc.lower(op); });
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
