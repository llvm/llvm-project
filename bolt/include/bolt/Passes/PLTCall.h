//===- bolt/Passes/PLTCall.h - PLT call optimization ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_PLTCALL_H
#define BOLT_PASSES_PLTCALL_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class PLTCall : public BinaryFunctionPass {
public:
  /// PLT optimization type
  enum OptType : char {
    OT_NONE = 0, /// Do not optimize
    OT_HOT = 1,  /// Optimize hot PLT calls
    OT_ALL = 2   /// Optimize all PLT calls
  };

  explicit PLTCall(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "PLT call optimization"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF);
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
