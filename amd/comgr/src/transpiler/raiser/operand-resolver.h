//===- operand-resolver.h - Transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_OPERAND_RESOLVER_H
#define TRANSPILER_OPERAND_RESOLVER_H

#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/raise-context.h"

#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace COMGR::transpiler {

// Width of a vector register, and of the halves a 16-bit operation addresses.
constexpr unsigned RegisterWidthInBits = 32;
constexpr unsigned HalfWidthInBits = 16;

// Destination and source values of a binary instruction.
struct BinaryOperands {
  ParsedReg Dst;
  llvm::Value *Src0;
  llvm::Value *Src1;
};

/// Destination and source values of a ternary 32-bit instruction.
struct TernaryOperands {
  ParsedReg Dst;
  llvm::Value *Src0;
  llvm::Value *Src1;
  llvm::Value *Src2;
};

// Operand access for one decoded instruction, handed to a handler: source
// reads through the decoded srcMap at 32-bit, 64-bit or EXEC width, register
// names for sources and destinations, and immediates. Float source reads apply
// the neg/abs modifiers.
struct OperandResolver {
  // Context the operands are read through.
  RaiseContext &Ctx;
  // Instruction whose operands are being read.
  const DecodedInst &Di;

  // MC operand index of the I-th source.
  unsigned srcIdx(unsigned I) const {
    assert(I < Di.SrcMap.size() && "source index out of range");
    return Di.SrcMap[I];
  }
  // Number of sources the instruction takes.
  unsigned nSrcs() const { return static_cast<unsigned>(Di.SrcMap.size()); }

  // Modifier bits attached to the I-th source, 0 when it carries none. Bit 0
  // negates and bit 1 takes the absolute value.
  unsigned srcMod(unsigned I) const;

  // Apply the I-th source's modifiers to V, which the caller has already read.
  // An integer-typed V round-trips through float, since the modifiers are
  // defined on the float interpretation of the bits.
  llvm::Value *applyMods(unsigned I, llvm::Value *V);

  // Read the I-th source as a 32-bit value.
  llvm::Expected<llvm::Value *> src(unsigned I) {
    return Ctx.registers().readOp32(Di, srcIdx(I));
  }
  /// Read the I-th source as an f64 value with its modifiers applied.
  llvm::Expected<llvm::Value *> srcF64(unsigned I);
  // Read the I-th source as an f32 value with its modifiers applied.
  llvm::Expected<llvm::Value *> srcF32(unsigned I);
  // Read the selected half of the I-th source as an f16 value with its
  // modifiers applied.
  llvm::Expected<llvm::Value *> srcF16(unsigned I);
  // Read the selected half of the I-th source as an i16 value.
  llvm::Expected<llvm::Value *> src16(unsigned I);
  // Whether the I-th source names the high half of its 32-bit register. The
  // true16 encodings spell the half as a register suffix and the others as an
  // op_sel modifier bit; the two agree wherever both are present.
  bool srcIsHighHalf(unsigned I) const;
  // Whether the destination names the high half of its 32-bit register, which
  // the VOP3 encodings carry as the op_sel bit in src0's modifiers.
  bool dstIsHighHalf() const;
  // Whether a 16-bit result leaves the other half of the destination register
  // alone. The encodings that cannot name a destination half zero it instead.
  bool dstKeepsOtherHalf() const;
  // Read the I-th source as a 64-bit value.
  llvm::Expected<llvm::Value *> src64(unsigned I) {
    return Ctx.registers().readOp64(Di, srcIdx(I));
  }
  // Read the I-th source at the width selected by Is64.
  llvm::Expected<llvm::Value *> src(unsigned I, bool Is64) {
    return Is64 ? src64(I) : src(I);
  }
  // Read the I-th source as a wave mask at target EXEC width.
  llvm::Expected<llvm::Value *> srcExecWidth(unsigned I) {
    return Ctx.registers().readOpExecWidth(Di, srcIdx(I));
  }
  // Read the I-th source as the current source wave's slice of a wave mask.
  llvm::Expected<llvm::Value *> srcSourceWaveMask32(unsigned I) {
    return Ctx.registers().readOpSourceWaveMask32(Di, srcIdx(I));
  }
  // Read the I-th source's per-lane wave-mask value, or null when unavailable.
  llvm::Expected<llvm::Value *> srcWaveMaskI1(unsigned I) {
    return Ctx.registers().readOpWaveMaskI1(Di, srcIdx(I));
  }
  // Return the source constant, including constants represented as
  // expressions.
  int64_t srcImm(unsigned I) {
    std::optional<int64_t> Constant = evalOperandAsConst(Di.Inst, srcIdx(I));
    assert(Constant && "source operand must be a constant");
    return *Constant;
  }

  // Register the I-th destination names.
  llvm::Expected<ParsedReg> dst(unsigned I = 0) {
    assert(Di.isReg(I) && "destination operand must be a register");
    return Ctx.registers().parseReg(Di, I);
  }
  // Value the I-th destination holds before the instruction runs, for an
  // opcode that leaves it alone on one side of a condition.
  llvm::Expected<llvm::Value *> dstValue(unsigned I = 0) {
    return Ctx.registers().readOp32(Di, I);
  }
  llvm::Expected<llvm::Value *> dstValue64(unsigned I = 0) {
    return Ctx.registers().readOp64(Di, I);
  }
  // Whether the I-th source is a register rather than an immediate.
  bool isSrcReg(unsigned I) { return Di.isReg(srcIdx(I)); }
  // Register the I-th source names, or no value when it is an immediate.
  llvm::Expected<std::optional<ParsedReg>> srcReg(unsigned I);

  // Read the destination and two 32-bit sources.
  llvm::Expected<BinaryOperands> readBinary32();
  // Read the destination and two 64-bit sources.
  llvm::Expected<BinaryOperands> readBinary64();
  /// Read the destination and three 32-bit sources.
  llvm::Expected<TernaryOperands> readTernary32();
};

} // namespace COMGR::transpiler

#endif
