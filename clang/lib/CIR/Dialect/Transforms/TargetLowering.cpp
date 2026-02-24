//===- TargetLowering.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the cir-target-lowering pass.
//
//===----------------------------------------------------------------------===//

#include "TargetLowering/LowerModule.h"

#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_TARGETLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct TargetLoweringPass
    : public impl::TargetLoweringBase<TargetLoweringPass> {
  TargetLoweringPass() = default;
  void runOnOperation() override;
};

} // namespace

static void convertSyncScopeIfPresent(mlir::Operation *op,
                                      cir::LowerModule &lowerModule) {
  auto syncScopeAttr =
      mlir::cast_if_present<cir::SyncScopeKindAttr>(op->getAttr("sync_scope"));
  if (syncScopeAttr) {
    cir::SyncScopeKind convertedSyncScope =
        lowerModule.getTargetLoweringInfo().convertSyncScope(
            syncScopeAttr.getValue());
    op->setAttr("sync_scope", cir::SyncScopeKindAttr::get(op->getContext(),
                                                          convertedSyncScope));
  }
}

void TargetLoweringPass::runOnOperation() {
  auto mod = mlir::cast<mlir::ModuleOp>(getOperation());
  std::unique_ptr<cir::LowerModule> lowerModule = cir::createLowerModule(mod);
  // If lower module is not available, skip the target lowering pass.
  if (!lowerModule) {
    mod.emitWarning("Cannot create a CIR lower module, skipping the ")
        << getName() << " pass";
    return;
  }

  mod->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::LoadOp, cir::StoreOp, cir::AtomicXchgOp,
                  cir::AtomicCmpXchgOp, cir::AtomicFetchOp>(op))
      convertSyncScopeIfPresent(op, *lowerModule);
  });
}

std::unique_ptr<Pass> mlir::createTargetLoweringPass() {
  return std::make_unique<TargetLoweringPass>();
}
