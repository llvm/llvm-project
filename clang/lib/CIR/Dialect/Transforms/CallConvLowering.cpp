//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME(cir): This header file is not exposed to the public API, but can be
// reused by CIR ABI lowering since it holds target-specific information.
#include "../../../Basic/Targets.h"

#include "TargetLowering/LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace mlir {
namespace cir {

namespace {

LowerModule createLowerModule(FuncOp op, PatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();

  // Fetch the LLVM data layout string.
  auto dataLayoutStr =
      module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName())
          .cast<StringAttr>();

  // Fetch target information.
  llvm::Triple triple(
      module->getAttr("cir.triple").cast<StringAttr>().getValue());
  clang::TargetOptions targetOptions;
  targetOptions.Triple = triple.str();
  auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

  // FIXME(cir): This just uses the default language options. We need to account
  // for custom options.
  // Create context.
  assert(!::cir::MissingFeatures::langOpts());
  clang::LangOptions langOpts;
  auto context = CIRLowerContext(module.getContext(), langOpts);
  context.initBuiltinTypes(*targetInfo);

  return LowerModule(context, module, dataLayoutStr, *targetInfo, rewriter);
}

} // namespace

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

struct CallConvLoweringPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const final {
    const auto module = op->getParentOfType<mlir::ModuleOp>();

    if (!op.getAst())
      return op.emitError("function has no AST information");

    LowerModule lowerModule = createLowerModule(op, rewriter);

    // Rewrite function calls before definitions. This should be done before
    // lowering the definition.
    auto calls = op.getSymbolUses(module);
    if (calls.has_value()) {
      for (auto call : calls.value()) {
        auto callOp = cast<CallOp>(call.getUser());
        if (lowerModule.rewriteFunctionCall(callOp, op).failed())
          return failure();
      }
    }

    // Rewrite function definition.
    // FIXME(cir): This is a workaround to avoid an infinite loop in the driver.
    rewriter.replaceOp(op, rewriter.clone(*op));
    return success();
  }
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

void populateCallConvLoweringPassPatterns(RewritePatternSet &patterns) {
  patterns.add<CallConvLoweringPattern>(patterns.getContext());
}

void CallConvLoweringPass::runOnOperation() {

  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCallConvLoweringPassPatterns(patterns);

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](FuncOp op) { ops.push_back(op); });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
