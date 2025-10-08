//===- bolt/Passes/MarkRAStates.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MarkRAStates class.
//
//===----------------------------------------------------------------------===//
#ifndef BOLT_PASSES_MARK_RA_STATES
#define BOLT_PASSES_MARK_RA_STATES

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class MarkRAStates : public BinaryFunctionPass {
public:
  explicit MarkRAStates() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "mark-ra-states"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  bool runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm
#endif
