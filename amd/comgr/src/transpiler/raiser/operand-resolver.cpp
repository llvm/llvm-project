//===- operand-resolver.cpp - Transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/operand-resolver.h"

#include "transpiler/raiser/raise_failure.h"

#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "SIDefines.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

#include <climits>

using namespace llvm;

namespace COMGR::transpiler {

unsigned OperandResolver::srcMod(unsigned I) const {
  assert(I < Di.ModMap.size() && "source modifier index out of range");
  unsigned ModIdx = Di.ModMap[I];
  if (ModIdx == UINT_MAX)
    return 0;
  assert(Di.isImm(ModIdx) && "source modifier must be an immediate");
  return static_cast<unsigned>(Di.getImm(ModIdx) & 0xF);
}

Value *OperandResolver::applyMods(unsigned I, Value *V) {
  unsigned Mods = srcMod(I);
  if (Mods == 0)
    return V;
  bool IsI32 = (V->getType() == Ctx.B.getInt32Ty());
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.B.getFloatTy());
  if (Mods & 2)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr, "abs");
  if (Mods & 1)
    V = Ctx.B.CreateFNeg(V, "neg");
  if (IsI32)
    V = Ctx.B.CreateBitCast(V, Ctx.B.getInt32Ty());
  return V;
}

Expected<Value *> OperandResolver::srcF32(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();
  unsigned AllowedModifiers = SISrcMods::NEG | SISrcMods::ABS;
  if (I == 0 && dstKeepsOtherHalf())
    AllowedModifiers |= SISrcMods::DST_OP_SEL;
  if (srcMod(I) & ~AllowedModifiers)
    return unsupportedInstruction(Ctx, Di, "unsupported f32 source modifier");
  Value *Float = Ctx.B.CreateBitCast(*V, Ctx.B.getFloatTy());
  return applyMods(I, Float);
}

Expected<Value *> OperandResolver::srcF64(unsigned I) {
  Expected<Value *> V = src64(I);
  if (!V)
    return V.takeError();
  if (srcMod(I) & ~(SISrcMods::NEG | SISrcMods::ABS))
    return unsupportedInstruction(Ctx, Di, "unsupported f64 source modifier");
  const MCOperand &Operand = Di.Inst.getOperand(srcIdx(I));
  if (Operand.isExpr()) {
    const auto *Literal = dyn_cast<AMDGPUMCExpr>(Operand.getExpr());
    // A forced 32-bit f64 literal supplies the high word.
    if (Literal && Literal->getKind() == AMDGPUMCExpr::AGVK_Lit)
      *V = Ctx.B.CreateShl(*V, 32);
  }
  Value *Float = Ctx.B.CreateBitCast(*V, Ctx.B.getDoubleTy());
  return applyMods(I, Float);
}

// Whether the register named by operand `OpIdx` is a `_HI16` subregister.
static bool namesHighHalfRegister(const DecodedInst &Di,
                                  const MCRegisterInfo &MRI, unsigned OpIdx) {
  return Di.isReg(OpIdx) &&
         (MRI.getEncodingValue(Di.getReg(OpIdx)) & AMDGPU::HWEncoding::IS_HI16);
}

bool OperandResolver::srcIsHighHalf(unsigned I) const {
  return (srcMod(I) & SISrcMods::OP_SEL_0) != 0 ||
         namesHighHalfRegister(Di, *Ctx.MC.RegInfo, srcIdx(I));
}

bool OperandResolver::dstIsHighHalf() const {
  return (srcMod(0) & SISrcMods::DST_OP_SEL) != 0 ||
         namesHighHalfRegister(Di, *Ctx.MC.RegInfo, 0);
}

bool OperandResolver::dstKeepsOtherHalf() const {
  // A VOP3 op_sel field names the destination half, and a true16 encoding
  // names it as a 16-bit register; either way the half that is not named is
  // left alone. The older encodings have no way to name a half and write the
  // result zero-extended over the whole register.
  const MCInstrDesc &Desc = Ctx.MC.InstrInfo->get(Di.Inst.getOpcode());
  if (SIInstrFlags::hasVOP3OpSel(Desc))
    return true;
  int16_t DstClass = Desc.operands()[0].RegClass;
  return DstClass >= 0 &&
         Ctx.MC.RegInfo->getRegClass(DstClass).getSizeInBits() ==
             HalfWidthInBits;
}

// Narrow the 32-bit register value `Bits` to the selected 16-bit half.
static Value *selectHalf(IRBuilder<> &B, Value *Bits, bool HighHalf) {
  if (HighHalf)
    Bits = B.CreateLShr(Bits, HalfWidthInBits, "src.hi");
  return B.CreateTrunc(Bits, B.getInt16Ty(), "src.i16");
}

Expected<Value *> OperandResolver::srcF16(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();

  constexpr unsigned AllowedModifiers =
      SISrcMods::NEG | SISrcMods::ABS | SISrcMods::OP_SEL_0;
  if (srcMod(I) & ~AllowedModifiers)
    return unsupportedInstruction(Ctx, Di, "unsupported f16 source modifier");

  Value *LowBits = selectHalf(Ctx.B, *V, srcIsHighHalf(I));
  Value *Half = Ctx.B.CreateBitCast(LowBits, Ctx.B.getHalfTy());
  return applyMods(I, Half);
}

Expected<Value *> OperandResolver::src16(unsigned I) {
  Expected<Value *> V = Ctx.registers().readOp32(Di, srcIdx(I));
  if (!V)
    return V.takeError();
  return selectHalf(Ctx.B, *V, srcIsHighHalf(I));
}

Expected<std::optional<ParsedReg>> OperandResolver::srcReg(unsigned I) {
  unsigned Index = srcIdx(I);
  if (!Di.isReg(Index))
    return std::optional<ParsedReg>();
  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  return std::optional<ParsedReg>(*Reg);
}

Expected<BinaryOperands> OperandResolver::readBinary32() {
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

Expected<BinaryOperands> OperandResolver::readBinary64() {
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src64(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src64(1);
  if (!Src1)
    return Src1.takeError();
  return BinaryOperands{*Dst, *Src0, *Src1};
}

Expected<TernaryOperands> OperandResolver::readTernary32() {
  assert(nSrcs() >= 3 && "ternary instruction must have three sources");
  Expected<ParsedReg> Dst = dst();
  if (!Dst)
    return Dst.takeError();
  Expected<Value *> Src0 = src(0);
  if (!Src0)
    return Src0.takeError();
  Expected<Value *> Src1 = src(1);
  if (!Src1)
    return Src1.takeError();
  Expected<Value *> Src2 = src(2);
  if (!Src2)
    return Src2.takeError();
  return TernaryOperands{*Dst, *Src0, *Src1, *Src2};
}

} // namespace COMGR::transpiler
