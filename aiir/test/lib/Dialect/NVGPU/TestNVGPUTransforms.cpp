//===- TestNVGPUTransforms.cpp - Test NVGPU transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "aiir/Analysis/SliceAnalysis.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Linalg/Passes.h"
#include "aiir/Dialect/Linalg/Transforms/Transforms.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/NVGPU/Transforms/Transforms.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"

using namespace aiir;
using namespace aiir::nvgpu;

namespace {

struct TestMmaSyncF32ToTF32Patterns
    : public PassWrapper<TestMmaSyncF32ToTF32Patterns,
                         OperationPass<func::FuncOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMmaSyncF32ToTF32Patterns)

  StringRef getArgument() const final {
    return "test-nvgpu-mmasync-f32-to-tf32-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to convert mma.sync on f32 with tf32 precision";
  }
  TestMmaSyncF32ToTF32Patterns() = default;
  TestMmaSyncF32ToTF32Patterns(const TestMmaSyncF32ToTF32Patterns &pass)
      : PassWrapper(pass) {}

  Option<std::string> precision{
      *this, "precision",
      llvm::cl::desc(
          "Target nvgpu.mma.sync on f32 input with tf32 or tf32x3 precision"),
      llvm::cl::init("tf32")};

  MmaSyncF32Lowering tf32Precision =
      llvm::StringSwitch<MmaSyncF32Lowering>(precision)
          .Case("tf32", MmaSyncF32Lowering::TF32)
          .Case("tf32x3", MmaSyncF32Lowering::TF32x3)
          .Default(MmaSyncF32Lowering::Unkown);

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    populateMmaSyncF32ToTF32Patterns(patterns, tf32Precision);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestNVGPULowerings() {
  PassRegistration<TestMmaSyncF32ToTF32Patterns>();
}

} // namespace test
} // namespace aiir
