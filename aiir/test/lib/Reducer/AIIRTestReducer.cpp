//===- TestReducer.cpp - Test AIIR Reduce ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that reproduces errors based on trivially defined
// patterns. It is used as a buggy optimization pass for the purpose of testing
// the AIIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {

/// This pass looks for the presence of an operation with the name
/// "crashOp" in the input AIIR file and crashes the aiir-opt tool if the
/// operation is found.
struct TestReducer : public PassWrapper<TestReducer, OperationPass<>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReducer)

  StringRef getArgument() const final { return "test-aiir-reducer"; }
  StringRef getDescription() const final {
    return "Tests AIIR Reduce tool by generating failures";
  }
  void runOnOperation() override;
};

} // namespace

void TestReducer::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();

    if (opName.contains("op_crash")) {
      llvm::errs() << "AIIR Reducer Test generated failure: Found "
                      "\"crashOp\" operation\n";
      exit(1);
    }
  });
}

namespace aiir {
void registerTestReducer() { PassRegistration<TestReducer>(); }
} // namespace aiir
