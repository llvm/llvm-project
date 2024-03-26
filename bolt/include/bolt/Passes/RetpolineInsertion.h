//===- bolt/Passes/RetpolineInsertion.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_RETPOLINE_INSERTION_H
#define BOLT_PASSES_RETPOLINE_INSERTION_H

#include "bolt/Passes/BinaryPasses.h"
#include <string>
#include <unordered_map>

namespace llvm {
namespace bolt {

struct IndirectBranchInfo {
private:
  bool IsMem = false;
  bool IsCall = false;
  bool IsTailCall = false;

public:
  IndirectBranchInfo(MCInst &Inst, MCPlusBuilder &MIB);
  bool isMem() const { return IsMem; }
  bool isReg() const { return !IsMem; }
  bool isCall() const { return IsCall; }
  bool isJump() const { return !IsCall; }
  bool isTailCall() const { return IsTailCall; }

  using MemOpInfo = MCPlusBuilder::X86MemOperand;

  union {
    // Register branch information
    MCPhysReg BranchReg;

    // Memory branch information
    MemOpInfo Memory;
  };
};

class RetpolineInsertion : public BinaryFunctionPass {
private:
  std::unordered_map<std::string, BinaryFunction *> CreatedRetpolines;

  BinaryFunction *getOrCreateRetpoline(BinaryContext &BC,
                                       const IndirectBranchInfo &BrInfo,
                                       bool R11Available);

public:
  /// Register r11 availability options
  enum AvailabilityOptions : char {
    ALWAYS = 0, ///  r11 available before calls and jumps
    ABI = 1,    ///  r11 available before calls
    NEVER = 2   ///  r11 not available
  };

  explicit RetpolineInsertion(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "retpoline-insertion"; }

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_RETPOLINE_INSERTION_H
