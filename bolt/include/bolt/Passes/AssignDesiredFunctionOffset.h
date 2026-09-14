//===- bolt/Passes/AssignDesiredFunctionOffset.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to assign the desired output offsets from --function-layout-file to the
// corresponding functions.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_ASSIGNDESIREDFUNCTIONOFFSET_H
#define BOLT_PASSES_ASSIGNDESIREDFUNCTIONOFFSET_H

#include "bolt/Passes/BinaryPasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace opts {
extern cl::opt<std::string> FunctionLayoutFile;
} // namespace opts

namespace llvm {
namespace bolt {

class AssignDesiredFunctionOffset : public BinaryFunctionPass {
public:
  explicit AssignDesiredFunctionOffset() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "apply-function-layout"; }

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
