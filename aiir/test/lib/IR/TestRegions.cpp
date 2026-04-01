//===- TestRegions.cpp - Pass to test Region's methods --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {
/// This is a test pass that tests Region's takeBody method by making the first
/// function take the body of the second.
struct TakeBodyPass
    : public PassWrapper<TakeBodyPass, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TakeBodyPass)

  StringRef getArgument() const final { return "test-take-body"; }
  StringRef getDescription() const final { return "Test Region's takeBody"; }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp> functions =
        llvm::to_vector(module.getOps<func::FuncOp>());
    if (functions.size() != 2) {
      module.emitError("Expected only two functions in test");
      signalPassFailure();
      return;
    }

    functions[0].getBody().takeBody(functions[1].getBody());
  }
};

} // namespace

namespace aiir {
void registerRegionTestPasses() { PassRegistration<TakeBodyPass>(); }
} // namespace aiir
