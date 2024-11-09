//===- bolt/Passes/Discover.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_DISCOVERNORETURN_H
#define BOLT_PASSES_DISCOVERNORETURN_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"
#include <unordered_map>

namespace llvm {

namespace bolt {

class DiscoverNoReturnPass : public BinaryFunctionPass {
public:
  explicit DiscoverNoReturnPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "discover-no-return"; }

  Error runOnFunctions(BinaryContext &BC) override;

private:
  std::unordered_map<BinaryFunction *, bool> Visited;
  bool traverseFromFunction(BinaryFunction *Func, BinaryContext &BC);
};
} // namespace bolt
} // namespace llvm

#endif
