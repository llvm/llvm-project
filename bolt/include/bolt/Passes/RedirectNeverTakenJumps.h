//===- bolt/Passes/RedirectNeverTakenJumps.h - Code size reduction --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass reduces code size in X86 by redirecting never-taken jumps that take
// 5 or 6 bytes to nearby jumps with the same jump target and compatible
// condition codes. Doing each such redirection will save 3 or 4 bytes depending
// on if the redirected jump is unconditional or conditional, since a short jump
// takes only 2 bytes. The pass can be turned on with BOLT option
// -redirect-never-taken-jumps.
//
// There are two modes for classifying "never-taken" jumps: aggressive and
// conservative. The aggressive mode classifies any jump with zero execution
// count as never-taken, and can be turned on with BOLT option
// -aggressive-never-taken. The conservative mode is used by default and
// accounts for potential errors in the input profile. It infers if a jump with
// zero execution count is actually never-taken by checking the gap between the
// inflow (resp. outflow) and block execution count for each basic block.
// The conservativeness is controlled by BOLT option
// -conservative-never-taken-threshold. The smaller the threshold, the more
// conservative the classification is. In most realistic settings, the value
// should exceed 1.0. The current default is 1.25.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REDIRECT_NEVER_TAKEN_JUMPS_H
#define BOLT_PASSES_REDIRECT_NEVER_TAKEN_JUMPS_H

#include "bolt/Passes/BinaryPasses.h"
#include <atomic>

namespace llvm {
namespace bolt {

class RedirectNeverTakenJumps : public BinaryFunctionPass {
private:
  std::atomic<uint64_t> TotalHotSizeSavings{0ull};
  std::atomic<uint64_t> TotalSizeSavings{0ull};

public:
  explicit RedirectNeverTakenJumps(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "redirect-never-taken-jumps"; }

  Error runOnFunctions(BinaryContext &BC) override;

  void performRedirections(BinaryFunction &Function);
};

} // namespace bolt
} // namespace llvm

#endif
