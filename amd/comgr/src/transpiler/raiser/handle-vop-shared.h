//===- handle-vop-shared.h - Shared VOP lowering helpers --------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPILER_HANDLE_VOP_SHARED_H
#define TRANSPILER_HANDLE_VOP_SHARED_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

namespace COMGR::transpiler {

struct OperandResolver;
struct DecodedInst;
class RaiseContext;

/// Builds the result of a two-source instruction from its already-read sources.
using BinaryBuilder = llvm::function_ref<llvm::Value *(
    llvm::IRBuilder<> &, llvm::Value *, llvm::Value *)>;

/// Raise a 32-bit move shared by VOP1 and VOP3 encodings.
llvm::Error raiseMove32(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op);

/// Raise a 64-bit move shared by VOP1 and VOP3 encodings.
llvm::Error raiseMove64(RaiseContext &Ctx, const DecodedInst &Di,
                        OperandResolver &Op);

/// Raise a unary 32-bit bit operation shared by VOP1 and VOP3 encodings.
llvm::Error raiseUnaryBit32(RaiseContext &Ctx, const DecodedInst &Di,
                            OperandResolver &Op);

/// Raise a unary F32 operation shared by VOP1 and VOP3 encodings.
llvm::Error raiseUnaryFloat32(RaiseContext &Ctx, const DecodedInst &Di,
                              OperandResolver &Op);
llvm::Error raiseUnaryFloat64(RaiseContext &Ctx, const DecodedInst &Di,
                              OperandResolver &Op);

/// Raise floating-point addition, subtraction, or multiplication.
llvm::Error raiseFloatBinary(RaiseContext &Ctx, const DecodedInst &Di,
                             OperandResolver &Op);

/// Raise F32 minimumNumber or maximumNumber, quieting source 0 if both are NaN.
llvm::Error raiseNumericExtremum32(RaiseContext &Ctx, const DecodedInst &Di,
                                   OperandResolver &Op);

llvm::Error raiseFloatMac(RaiseContext &Ctx, const DecodedInst &Di,
                          OperandResolver &Op);

/// Raise a 32-bit floating-point conversion shared by VOP1 and VOP3 encodings.
llvm::Error raiseFloatConversion32(RaiseContext &Ctx, const DecodedInst &Di,
                                   OperandResolver &Op);
llvm::Error raiseFloatConversion64(RaiseContext &Ctx, const DecodedInst &Di,
                                   OperandResolver &Op);

/// Raise V_CNDMASK_B32 with an implicit or explicit wave-mask condition.
llvm::Error raiseCndMask32(RaiseContext &Ctx, const DecodedInst &Di,
                           OperandResolver &Op);

/// Raise a binary 32-bit operation and write its result.
llvm::Error raiseBinary32(RaiseContext &Ctx, OperandResolver &Op,
                          BinaryBuilder Build);

/// Raise a binary 64-bit operation and write its result.
llvm::Error raiseBinary64(RaiseContext &Ctx, OperandResolver &Op,
                          BinaryBuilder Build);

/// Read the clamp operand, which selects a saturating result on the opcodes
/// that define it. Opcodes whose encoding reserves the field have no named
/// operand and are necessarily unclamped.
llvm::Expected<bool> readClamp(RaiseContext &Ctx, const DecodedInst &Di);

/// Read the lane condition a V_CNDMASK selects with: VCC for the two-source
/// encodings, the named wave mask for the three-source ones.
llvm::Expected<llvm::Value *> readCndMaskCondition(RaiseContext &Ctx,
                                                   OperandResolver &Op);

/// Combine three sources the way V_BITOP3 does: bit i of the result is
/// `TruthTable[{Src0[i], Src1[i], Src2[i]}]`, Src0 contributing the high index
/// bit. The sources share one integer type, which the result also has.
llvm::Value *emitBitOp3(llvm::IRBuilder<> &B, llvm::Value *Src0,
                        llvm::Value *Src1, llvm::Value *Src2,
                        uint8_t TruthTable);

/// Write a 16-bit result to the destination half the instruction selects. The
/// other half of the destination register keeps its value on the encodings
/// that name a half and is zeroed on those that cannot.
llvm::Error writeDestination16(RaiseContext &Ctx, OperandResolver &Op,
                               llvm::Value *Result);

/// Apply the f16 sign modifiers in `Mods` to the raw bits of a 16-bit value.
/// Both modifiers act on the sign bit alone, so an opcode that only moves bits
/// around can honor them without interpreting the rest of the value. Hardware
/// negates the absolute value when both are set.
llvm::Value *applyFloat16SignModifiers(llvm::IRBuilder<> &B, llvm::Value *Bits,
                                       unsigned Mods);

/// Raise a binary operation over the selected 16-bit halves of both sources.
llvm::Error raiseBinary16(RaiseContext &Ctx, OperandResolver &Op,
                          BinaryBuilder Build);

/// Mask a shift amount to the low bits read by hardware.
llvm::Value *maskShiftAmount(llvm::IRBuilder<> &B, llvm::Value *Amount,
                             unsigned Width);

/// Raise V_BFM_B32 using its five-bit width and offset operands.
llvm::Error raiseBitMask(RaiseContext &Ctx, OperandResolver &Op);

/// Raise V_BCNT_U32_B32 by adding popcount(src0) to src1.
llvm::Error raiseBitCount(RaiseContext &Ctx, OperandResolver &Op);

/// Raise V_LSHLREV_B64 with a 32-bit shift amount and 64-bit value.
llvm::Error raiseShiftLeft64(RaiseContext &Ctx, OperandResolver &Op);

} // namespace COMGR::transpiler

#endif // TRANSPILER_HANDLE_VOP_SHARED_H
