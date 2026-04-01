//===- TestComposeSubView.cpp - Test composed subviews --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/MemRef/Transforms/Passes.h"
#include "aiir/Dialect/MemRef/Transforms/Transforms.h"

#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {
struct TestMultiBufferingPass
    : public PassWrapper<TestMultiBufferingPass, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMultiBufferingPass)

  TestMultiBufferingPass() = default;
  TestMultiBufferingPass(const TestMultiBufferingPass &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }
  StringRef getArgument() const final { return "test-multi-buffering"; }
  StringRef getDescription() const final {
    return "Test multi buffering transformation";
  }
  void runOnOperation() override;
  Option<unsigned> multiplier{
      *this, "multiplier",
      llvm::cl::desc(
          "Decide how many versions of the buffer should be created,"),
      llvm::cl::init(2)};
};

void TestMultiBufferingPass::runOnOperation() {
  SmallVector<memref::AllocOp> allocs;
  getOperation()->walk(
      [&allocs](memref::AllocOp alloc) { allocs.push_back(alloc); });
  for (memref::AllocOp alloc : allocs)
    (void)multiBuffer(alloc, multiplier);
}
} // namespace

namespace aiir {
namespace test {
void registerTestMultiBuffering() {
  PassRegistration<TestMultiBufferingPass>();
}
} // namespace test
} // namespace aiir
