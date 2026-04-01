//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Support/LogicalResult.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/TimeProfiler.h"

using namespace aiir;
using namespace cir;

namespace aiir {
#define GEN_PASS_DEF_HOISTALLOCAS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace aiir

namespace {

struct HoistAllocasPass : public impl::HoistAllocasBase<HoistAllocasPass> {

  HoistAllocasPass() = default;
  void runOnOperation() override;
};

static void process(aiir::ModuleOp mod, cir::FuncOp func) {
  if (func.getRegion().empty())
    return;

  // Hoist all static allocas to the entry block.
  aiir::Block &entryBlock = func.getRegion().front();
  aiir::Operation *insertPoint = &*entryBlock.begin();

  // Post-order is the default, but the code below requires it, so
  // let's not depend on the default staying that way.
  func.getBody().walk<aiir::WalkOrder::PostOrder>([&](cir::AllocaOp alloca) {
    if (alloca->getBlock() == &entryBlock)
      return;
    // Don't hoist allocas with dynamic alloca size.
    if (alloca.getDynAllocSize())
      return;

    // Hoist allocas into the entry block.

    // Preserving the `const` attribute on hoisted allocas can cause LLVM to
    // incorrectly introduce invariant group metadata in some circumstances.
    // The incubator performs some analysis to determine whether the attribute
    // can be preserved, but it only runs this analysis when optimizations are
    // enabled. Until we start tracking the optimization level, we can just
    // always remove the `const` attribute.
    assert(!cir::MissingFeatures::optInfoAttr());
    if (alloca.getConstant())
      alloca.setConstant(false);

    alloca->moveBefore(insertPoint);
  });
}

void HoistAllocasPass::runOnOperation() {
  llvm::TimeTraceScope scope("Hoist Allocas");
  llvm::SmallVector<Operation *, 16> ops;

  Operation *op = getOperation();
  auto mod = aiir::dyn_cast<aiir::ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<aiir::ModuleOp>();

  // If we ever introduce nested cir.function ops, we'll need to make this
  // walk in post-order and recurse into nested functions.
  getOperation()->walk<aiir::WalkOrder::PreOrder>([&](cir::FuncOp op) {
    process(mod, op);
    return aiir::WalkResult::skip();
  });
}

} // namespace

std::unique_ptr<Pass> aiir::createHoistAllocasPass() {
  return std::make_unique<HoistAllocasPass>();
}
