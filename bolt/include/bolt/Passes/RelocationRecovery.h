//===- bolt/Passes/RelocationRecovery.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_RELOCATIONRECOVERY_H
#define BOLT_PASSES_RELOCATIONRECOVERY_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Reconstruct code and data address references that are missing from the
/// input static relocation records. If the complete target of an AArch64
/// ADRP/ADD reference cannot be determined unambiguously, rewrite the ADRP to
/// reproduce its original absolute page and leave the ADD immediate unchanged.
class RelocationRecovery : public BinaryFunctionPass {
public:
  explicit RelocationRecovery(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "recover-relocations"; }
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_RELOCATIONRECOVERY_H
