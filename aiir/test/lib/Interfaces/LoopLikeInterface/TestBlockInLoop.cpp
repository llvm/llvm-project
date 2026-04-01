//===- TestBlockInLoop.cpp - Pass to test aiir::blockIsInLoop -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace aiir;

namespace {
/// This is a test pass that tests Blocks's isInLoop method by checking if each
/// block in a function is in a loop and outputing if it is
struct IsInLoopPass
    : public PassWrapper<IsInLoopPass, OperationPass<ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsInLoopPass)

  StringRef getArgument() const final { return "test-block-is-in-loop"; }
  StringRef getDescription() const final {
    return "Test aiir::blockIsInLoop()";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : module.getOps<func::FuncOp>()) {
      func.walk([](aiir::Block *block) {
        llvm::outs() << "Block is ";
        if (LoopLikeOpInterface::blockIsInLoop(block))
          llvm::outs() << "in a loop\n";
        else
          llvm::outs() << "not in a loop\n";
        block->print(llvm::outs());
        llvm::outs() << "\n";
      });
    }
  }
};

} // namespace

namespace aiir {
void registerLoopLikeInterfaceTestPasses() { PassRegistration<IsInLoopPass>(); }
} // namespace aiir
