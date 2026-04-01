//===- TestCallGraph.cpp - Test callgraph construction and iteration ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and iterating over a
// callgraph.
//
//===----------------------------------------------------------------------===//

#include "aiir/Analysis/CallGraph.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {
struct TestCallGraphPass
    : public PassWrapper<TestCallGraphPass, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCallGraphPass)

  StringRef getArgument() const final { return "test-print-callgraph"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed callgraph.";
  }
  void runOnOperation() override {
    llvm::errs() << "Testing : "
                 << getOperation()->getDiscardableAttr("test.name") << "\n";
    getAnalysis<CallGraph>().print(llvm::errs());
  }
};
} // namespace

namespace aiir {
namespace test {
void registerTestCallGraphPass() { PassRegistration<TestCallGraphPass>(); }
} // namespace test
} // namespace aiir
