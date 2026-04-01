//===- TestLiveness.cpp - Test liveness construction and information ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving liveness
// information.
//
//===----------------------------------------------------------------------===//

#include "aiir/Analysis/Liveness.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {

struct TestLivenessPass
    : public PassWrapper<TestLivenessPass, InterfacePass<SymbolOpInterface>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLivenessPass)

  StringRef getArgument() const final { return "test-print-liveness"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed liveness information.";
  }
  void runOnOperation() override {
    llvm::errs() << "Testing : " << getOperation().getName() << "\n";
    getAnalysis<Liveness>().print(llvm::errs());
  }
};

} // namespace

namespace aiir {
namespace test {
void registerTestLivenessPass() { PassRegistration<TestLivenessPass>(); }
} // namespace test
} // namespace aiir
