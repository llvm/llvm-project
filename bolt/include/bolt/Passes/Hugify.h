//===- bolt/Passes/Hugify.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_HUGIFY_H
#define BOLT_PASSES_HUGIFY_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class HugePage : public BinaryFunctionPass {
public:
  HugePage(const cl::opt<bool> &PrintPass) : BinaryFunctionPass(PrintPass) {}

  void runOnFunctions(BinaryContext &BC) override;

  const char *getName() const override { return "HugePage"; }
};

} // namespace bolt
} // namespace llvm

#endif
