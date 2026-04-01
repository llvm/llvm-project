//===- TestGPUMemoryPromotionPass.cpp - Test pass for GPU promotion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass testing the utilities for moving data across
// different levels of the GPU memory hierarchy.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/GPU/Transforms/MemoryPromotion.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {
/// Simple pass for testing the promotion to workgroup memory in GPU functions.
/// Promotes all arguments with "gpu.test_promote_workgroup" attribute. This
/// does not check whether the promotion is legal (e.g., amount of memory used)
/// or beneficial (e.g., makes previously uncoalesced loads coalesced).
struct TestGpuMemoryPromotionPass
    : public PassWrapper<TestGpuMemoryPromotionPass,
                         OperationPass<gpu::GPUFuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGpuMemoryPromotionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }
  StringRef getArgument() const final { return "test-gpu-memory-promotion"; }
  StringRef getDescription() const final {
    return "Promotes the annotated arguments of gpu.func to workgroup memory.";
  }

  void runOnOperation() override {
    gpu::GPUFuncOp op = getOperation();
    for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
      if (op.getArgAttrOfType<UnitAttr>(i, "gpu.test_promote_workgroup"))
        promoteToWorkgroupMemory(op, i);
    }
  }
};
} // namespace

namespace aiir {
void registerTestGpuMemoryPromotionPass() {
  PassRegistration<TestGpuMemoryPromotionPass>();
}
} // namespace aiir
