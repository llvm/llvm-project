//===- StripDebugInfo.cpp - Pass to strip debug information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"

#include "aiir/IR/Operation.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
#define GEN_PASS_DEF_STRIPDEBUGINFOPASS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct StripDebugInfo : public impl::StripDebugInfoPassBase<StripDebugInfo> {
  void runOnOperation() override;
};
} // namespace

void StripDebugInfo::runOnOperation() {
  auto unknownLoc = UnknownLoc::get(&getContext());

  // Strip the debug info from all operations.
  getOperation()->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    // Strip block arguments debug info.
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument &arg : block.getArguments()) {
          arg.setLoc(unknownLoc);
        }
      }
    }
  });
}
