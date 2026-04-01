//===- TestLoopMapping.cpp --- Parametric loop mapping pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to parametrically map scf.for loops to virtual
// processing element dimensions.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/LoopUtils.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/IR/Builders.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;
using namespace aiir::affine;

namespace {
struct TestLoopMappingPass
    : public PassWrapper<TestLoopMappingPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopMappingPass)

  StringRef getArgument() const final {
    return "test-mapping-to-processing-elements";
  }
  StringRef getDescription() const final {
    return "test mapping a single loop on a virtual processor grid";
  }
  explicit TestLoopMappingPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    // SSA values for the transformation are created out of thin air by
    // unregistered "new_processor_id_and_range" operations. This is enough to
    // emulate mapping conditions.
    SmallVector<Value, 8> processorIds, numProcessors;
    getOperation()->walk([&processorIds, &numProcessors](Operation *op) {
      if (op->getName().getStringRef() != "new_processor_id_and_range")
        return;
      processorIds.push_back(op->getResult(0));
      numProcessors.push_back(op->getResult(1));
    });

    getOperation()->walk([&processorIds, &numProcessors](scf::ForOp op) {
      // Ignore nested loops.
      if (op->getParentRegion()->getParentOfType<scf::ForOp>())
        return;
      mapLoopToProcessorIds(op, processorIds, numProcessors);
    });
  }
};
} // namespace

namespace aiir {
namespace test {
void registerTestLoopMappingPass() { PassRegistration<TestLoopMappingPass>(); }
} // namespace test
} // namespace aiir
