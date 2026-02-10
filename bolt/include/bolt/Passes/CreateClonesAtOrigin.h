//===- bolt/Passes/CreateClonesAtOrigin.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass for creating clones at origin of functions.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CREATE_CLONES_AT_ORIGIN_H
#define BOLT_PASSES_CREATE_CLONES_AT_ORIGIN_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Pass for creating clones at origin of functions.
///
/// This pass creates clones at the original function addresses and runs
/// scanExternalRefs() to update references from clone code to relocated
/// functions. Unlike patching (which redirects execution from original to
/// optimized code), cloning keeps the original code executable.
class CreateClonesAtOrigin : public BinaryFunctionPass {
public:
  explicit CreateClonesAtOrigin() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "create-clones-at-origin"; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
