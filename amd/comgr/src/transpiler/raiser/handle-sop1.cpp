//===- handle-sop1.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/decode.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/raiser/source-image.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/FormatVariadic.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <variant>

using namespace llvm;

namespace COMGR::transpiler {

// Whether a whole-wave mask of Bits bits is the mask the raiser holds: both
// the source wave and the EXEC storage are that wide, so a mask combined with
// EXEC is the same mask that is written back.
static bool waveMaskWidthIs(const RaiseContext &Ctx, unsigned Bits) {
  return Ctx.Projection.sourceWaveMaskTy()->getIntegerBitWidth() == Bits &&
         Ctx.Projection.execStorageTy()->getIntegerBitWidth() == Bits;
}

// The two widths waveMaskWidthIs compares against, for naming in a refusal.
static std::string heldWaveMaskWidths(const RaiseContext &Ctx) {
  return (Twine("the source wave is ") +
          Twine(Ctx.Projection.sourceWaveMaskTy()->getIntegerBitWidth()) +
          " bits wide and EXEC holds " +
          Twine(Ctx.Projection.execStorageTy()->getIntegerBitWidth()) + " bits")
      .str();
}

// Refuse a register-indirect control transfer that the analysis could not
// resolve, phrasing each reason to follow the mnemonic in the diagnostic.
static Error refuseSetPc(RaiseContext &Ctx, const DecodedInst &Di,
                         const SetPcUnresolvable &Site) {
  std::string Pair =
      (Twine("s[") + Twine(Site.Subject) + ":" + Twine(Site.Subject + 1) + "]")
          .str();
  switch (Site.Why) {
  case SetPcRefusal::NotARegisterPair:
    return unsupported(Ctx, Di,
                       "reads its target from something other than a scalar "
                       "register pair");
  case SetPcRefusal::BlockUnreachable:
    return unsupported(Ctx, Di,
                       (Twine("reads ") + Pair +
                        ", and no path the decode recovered reaches its block")
                           .str());
  case SetPcRefusal::PairWrittenWithoutOffset:
    return unsupported(Ctx, Di,
                       (Twine("reads ") + Pair +
                        ", which some path writes without computing a source "
                        "offset in it")
                           .str());
  case SetPcRefusal::PairNeverHeldOffset:
    return unsupported(
        Ctx, Di,
        (Twine("reads ") + Pair + ", which nothing gives a source offset")
            .str());
  case SetPcRefusal::TooManyTargets:
    return unsupported(Ctx, Di,
                       (Twine("reads ") + Pair +
                        ", which more source offsets reach than the analysis "
                        "enumerates")
                           .str());
  case SetPcRefusal::TargetNotAnInstruction:
    return unsupported(Ctx, Di,
                       (Twine("reaches source offset 0x") +
                        Twine::utohexstr(Site.Subject) +
                        ", which no decoded instruction starts at")
                           .str());
  }
  llvm_unreachable("unhandled set-pc refusal");
}

// Write V to Dst at the width the opcode operates on.
static void writeDst(RegisterState &Registers, ParsedReg Dst, Value *V,
                     bool Is64) {
  if (Is64)
    Registers.writeReg64(Dst, V);
  else
    Registers.writeReg32(Dst, V);
}

// Whether CanonOp is one of the scalar float opcodes, each of which takes one
// 32-bit source SGPR and writes one 32-bit destination SGPR.
static bool isScalarFloat(CanonicalOp CanonOp) {
  switch (CanonOp) {
  case CanonicalOp::S_CEIL_F32:
  case CanonicalOp::S_FLOOR_F32:
  case CanonicalOp::S_TRUNC_F32:
  case CanonicalOp::S_RNDNE_F32:
  case CanonicalOp::S_CVT_F32_I32:
  case CanonicalOp::S_CVT_F32_U32:
  case CanonicalOp::S_CVT_I32_F32:
  case CanonicalOp::S_CVT_U32_F32:
  case CanonicalOp::S_CVT_F16_F32:
  case CanonicalOp::S_CVT_F32_F16:
  case CanonicalOp::S_CVT_HI_F32_F16:
  case CanonicalOp::S_CEIL_F16:
  case CanonicalOp::S_FLOOR_F16:
  case CanonicalOp::S_TRUNC_F16:
  case CanonicalOp::S_RNDNE_F16:
    return true;
  default:
    return false;
  }
}

// Round the f32 in Src with intrinsic ID. The result stays in floating-point
// form, so it goes back as an f32 bit pattern rather than as an integer.
static Value *roundF32(IRBuilder<> &B, Value *Src, Intrinsic::ID ID) {
  Value *Rounded =
      B.CreateUnaryIntrinsic(ID, B.CreateBitCast(Src, B.getFloatTy()));
  return B.CreateBitCast(Rounded, B.getInt32Ty());
}

// Round the f16 in the low half of Src with intrinsic ID. A scalar f16 opcode
// reads the low half of its source SGPR and zeroes the high half of its
// destination.
static Value *roundF16(IRBuilder<> &B, Value *Src, Intrinsic::ID ID) {
  Value *Half =
      B.CreateBitCast(B.CreateTrunc(Src, B.getInt16Ty()), B.getHalfTy());
  Value *Rounded = B.CreateUnaryIntrinsic(ID, Half);
  return B.CreateZExt(B.CreateBitCast(Rounded, B.getInt16Ty()), B.getInt32Ty());
}

// The 32 bits the scalar float opcode CanonOp writes to its destination SGPR,
// given the 32 bits Src of its source. CanonOp must satisfy isScalarFloat.
static Value *emitScalarFloat(IRBuilder<> &B, CanonicalOp CanonOp, Value *Src) {
  switch (CanonOp) {
  case CanonicalOp::S_CEIL_F32:
    return roundF32(B, Src, Intrinsic::ceil);
  case CanonicalOp::S_FLOOR_F32:
    return roundF32(B, Src, Intrinsic::floor);
  case CanonicalOp::S_TRUNC_F32:
    return roundF32(B, Src, Intrinsic::trunc);
  case CanonicalOp::S_RNDNE_F32:
    return roundF32(B, Src, Intrinsic::roundeven);
  case CanonicalOp::S_CEIL_F16:
    return roundF16(B, Src, Intrinsic::ceil);
  case CanonicalOp::S_FLOOR_F16:
    return roundF16(B, Src, Intrinsic::floor);
  case CanonicalOp::S_TRUNC_F16:
    return roundF16(B, Src, Intrinsic::trunc);
  case CanonicalOp::S_RNDNE_F16:
    return roundF16(B, Src, Intrinsic::roundeven);
  case CanonicalOp::S_CVT_F32_I32:
    return B.CreateBitCast(B.CreateSIToFP(Src, B.getFloatTy()), B.getInt32Ty());
  case CanonicalOp::S_CVT_F32_U32:
    return B.CreateBitCast(B.CreateUIToFP(Src, B.getFloatTy()), B.getInt32Ty());
  // The hardware saturates an out-of-range input and converts NaN to zero,
  // which is what the saturating intrinsics are defined to do. Plain fptosi
  // and fptoui make both of those poison.
  case CanonicalOp::S_CVT_I32_F32:
    return B.CreateIntrinsic(Intrinsic::fptosi_sat,
                             {B.getInt32Ty(), B.getFloatTy()},
                             {B.CreateBitCast(Src, B.getFloatTy())});
  case CanonicalOp::S_CVT_U32_F32:
    return B.CreateIntrinsic(Intrinsic::fptoui_sat,
                             {B.getInt32Ty(), B.getFloatTy()},
                             {B.CreateBitCast(Src, B.getFloatTy())});
  case CanonicalOp::S_CVT_F16_F32: {
    Value *Half =
        B.CreateFPTrunc(B.CreateBitCast(Src, B.getFloatTy()), B.getHalfTy());
    return B.CreateZExt(B.CreateBitCast(Half, B.getInt16Ty()), B.getInt32Ty());
  }
  case CanonicalOp::S_CVT_F32_F16:
  case CanonicalOp::S_CVT_HI_F32_F16: {
    Value *Bits =
        CanonOp == CanonicalOp::S_CVT_HI_F32_F16 ? B.CreateLShr(Src, 16) : Src;
    Value *Half =
        B.CreateBitCast(B.CreateTrunc(Bits, B.getInt16Ty()), B.getHalfTy());
    return B.CreateBitCast(B.CreateFPExt(Half, B.getFloatTy()), B.getInt32Ty());
  }
  default:
    llvm_unreachable("not a scalar float opcode");
  }
}

// Refuse Di as a relative access that does not resolve statically.
static Error refuseMovrel(RaiseContext &Ctx, const DecodedInst &Di,
                          const Twine &Detail) {
  return unsupported(Ctx, Di, Twine("movrel: ") + Detail);
}

// The M0 value a relative register index is displaced by. The register file is
// one alloca per register rather than addressable memory, so only a constant M0
// resolves to a named register.
static Expected<uint64_t> constantM0(RaiseContext &Ctx, const DecodedInst &Di) {
  std::optional<uint64_t> M0 = Ctx.registers().getM0Const();
  if (!M0)
    return refuseMovrel(Ctx, Di, "M0 does not hold a constant here");
  return *M0;
}

// The SGPR the register operand at OpIdx names once displaced by Displacement.
// Refuses an operand that is not an SGPR, an odd index for a 64-bit access,
// and a displaced index outside the scalar register file.
static Expected<unsigned> displacedSgpr(RaiseContext &Ctx,
                                        const DecodedInst &Di, unsigned OpIdx,
                                        uint64_t Displacement,
                                        unsigned WidthInDwords) {
  if (!Di.isReg(OpIdx))
    return refuseMovrel(Ctx, Di, "relative operand is not a register");
  Expected<ParsedReg> Base = Ctx.registers().parseReg(Di, OpIdx);
  if (!Base)
    return Base.takeError();
  if (Base->RegKind != ParsedReg::SGPR)
    return refuseMovrel(Ctx, Di, "relative operand is not an SGPR");
  assert(Base->BaseIdx && "SGPR must have a base register index");

  uint64_t Idx = *Base->BaseIdx + Displacement;
  if (WidthInDwords == 2 && Idx % 2 != 0)
    return refuseMovrel(
        Ctx, Di, "64-bit access resolves to odd SGPR index " + Twine(Idx));
  if (Idx + WidthInDwords > Ctx.registers().numSgprs())
    return refuseMovrel(
        Ctx, Di, "resolved SGPR index " + Twine(Idx) + " is out of range");
  return static_cast<unsigned>(Idx);
}

// How a bit-search opcode looks for the bit whose position it reports: the
// counting intrinsic, whether the source is a register pair, and whether the
// bit sought is the first one differing from the sign rather than the first
// set one.
struct BitSearch {
  Intrinsic::ID Count;
  bool Is64;
  bool AgainstSign;
};

// The search CanonOp performs, or no value when it is not a bit-search opcode.
static std::optional<BitSearch> bitSearch(CanonicalOp CanonOp) {
  switch (CanonOp) {
  case CanonicalOp::S_FF1_I32_B32:
    return BitSearch{Intrinsic::cttz, false, false};
  case CanonicalOp::S_FF1_I32_B64:
    return BitSearch{Intrinsic::cttz, true, false};
  case CanonicalOp::S_FLBIT_I32_B32:
    return BitSearch{Intrinsic::ctlz, false, false};
  case CanonicalOp::S_FLBIT_I32_B64:
    return BitSearch{Intrinsic::ctlz, true, false};
  case CanonicalOp::S_FLBIT_I32:
    return BitSearch{Intrinsic::ctlz, false, true};
  case CanonicalOp::S_FLBIT_I32_I64:
    return BitSearch{Intrinsic::ctlz, true, true};
  default:
    return std::nullopt;
  }
}

// The position Search reports for Src, as the single dword every bit-search
// opcode writes whatever width it searched. Searching against the sign is
// searching the source exclusive-ored with its own sign, which leaves a
// non-negative source alone and complements a negative one, so that in all
// three cases a zero operand is exactly the input with no bit to find. That is
// the input the hardware answers with -1 and the count intrinsics answer with
// the operand width, so it is selected out here.
static Value *emitBitSearch(IRBuilder<> &B, BitSearch Search, Value *Src) {
  Value *Searched = Src;
  if (Search.AgainstSign) {
    unsigned Width = Src->getType()->getIntegerBitWidth();
    Value *Sign = B.CreateAShr(Src, Width - 1);
    Searched = B.CreateXor(Src, Sign, "s_cls_unsigned");
  }
  Value *Count = B.CreateBinaryIntrinsic(Search.Count, Searched, B.getFalse(),
                                         {}, "s_pos");
  Value *Position = B.CreateZExtOrTrunc(Count, B.getInt32Ty());
  Value *Found = B.CreateIsNotNull(Searched);
  return B.CreateSelect(Found, Position, B.getInt32(-1), "s_pos_found");
}

// Src with each of its 32 bits doubled into a 64-bit value. Every step spreads
// the bits half as far apart as the one before it, leaving them on the even
// positions, and the last one copies each into the odd position above it.
static Value *emitBitReplicate(IRBuilder<> &B, Value *Src) {
  static constexpr struct {
    unsigned Shift;
    uint64_t Keep;
  } KSpread[] = {{16, 0x0000ffff0000ffffULL},
                 {8, 0x00ff00ff00ff00ffULL},
                 {4, 0x0f0f0f0f0f0f0f0fULL},
                 {2, 0x3333333333333333ULL},
                 {1, 0x5555555555555555ULL}};
  Value *Spread = B.CreateZExt(Src, B.getInt64Ty());
  for (auto [Shift, Keep] : KSpread) {
    Value *Shifted = B.CreateShl(Spread, Shift);
    Value *Widened = B.CreateOr(Spread, Shifted);
    Spread = B.CreateAnd(Widened, Keep);
  }
  Value *Doubled = B.CreateShl(Spread, 1);
  return B.CreateOr(Spread, Doubled, "s_bitreplicate");
}

// Src with the lowest bit of each group of four set iff any bit of that group
// is, and every other bit clear. Both quad-mask opcodes start from this.
static Value *emitQuadAnyBit(IRBuilder<> &B, Value *Src) {
  IntegerType *Ty = cast<IntegerType>(Src->getType());
  unsigned Width = Ty->getBitWidth();
  assert((Width == 32 || Width == 64) && "a wave mask is one or two dwords");
  Value *Any = B.CreateOr(Src, B.CreateLShr(Src, 2));
  Any = B.CreateOr(Any, B.CreateLShr(Any, 1));
  return B.CreateAnd(Any,
                     ConstantInt::get(Ty, APInt::getSplat(Width, APInt(4, 1))));
}

// Src reduced to one bit per group of four, packed into the low quarter of the
// result.
static Value *emitQuadMask(IRBuilder<> &B, Value *Src) {
  IntegerType *Ty = cast<IntegerType>(Src->getType());
  unsigned Width = Ty->getBitWidth();
  Value *Packed = emitQuadAnyBit(B, Src);
  for (unsigned Gathered = 2; Gathered * 4 <= Width; Gathered *= 2) {
    // Each step merges a field with the one above it, doubling how many bits
    // sit gathered at the bottom of a field. On entry Gathered / 2 bits sit at
    // the bottom of every 2 * Gathered bits, so the upper field's bits start at
    // bit 2 * Gathered and belong at bit Gathered / 2, a right shift of
    // 3 * Gathered / 2. The first step shifts by 3, which lands the bit of
    // nibble 1 at bit 1, next to the bit of nibble 0.
    Value *Halved = B.CreateOr(Packed, B.CreateLShr(Packed, Gathered * 3 / 2));
    // Keep the Gathered bits now adjacent in each field of Gathered * 4 bits.
    Constant *Keep = ConstantInt::get(
        Ty,
        APInt::getSplat(Width, APInt::getLowBitsSet(Gathered * 4, Gathered)));
    // The last step leaves the gathered bits adjacent, which is the result.
    bool LastStep = Gathered * 8 > Width;
    Packed = B.CreateAnd(Halved, Keep, LastStep ? "s_quadmask" : "");
  }
  return Packed;
}

// Src with each group of four bits set whole iff any bit of the group is set.
static Value *emitWholeQuadMask(IRBuilder<> &B, Value *Src) {
  Value *Pair = emitQuadAnyBit(B, Src);
  Pair = B.CreateOr(Pair, B.CreateShl(Pair, 1));
  return B.CreateOr(Pair, B.CreateShl(Pair, 2), "s_wqm");
}

// The bitwise operation an EXEC-combining opcode applies to its source and
// EXEC.
enum class MaskOperation { And, Or, Xor };

// Which of the source, EXEC, or the combined result the opcode complements on
// the way. No opcode in the family complements more than one of the three.
enum class MaskNegate { None, Source, Exec, Result };

// Which mask the opcode leaves in its scalar destination: the EXEC it
// replaced, or the one it just computed.
enum class MaskDestination { OldExec, NewExec };

// One row of kExecCombines.
struct ExecCombine {
  CanonicalOp Opcode;
  MaskOperation Operation;
  MaskNegate Negate;
  MaskDestination Destination;
  unsigned MaskBits;
};

// clang-format off
static const ExecCombine kExecCombines[] = {
    {CanonicalOp::S_AND_SAVEEXEC_B32,   MaskOperation::And, MaskNegate::None,   MaskDestination::OldExec, 32},
    {CanonicalOp::S_AND_SAVEEXEC_B64,   MaskOperation::And, MaskNegate::None,   MaskDestination::OldExec, 64},
    {CanonicalOp::S_OR_SAVEEXEC_B32,    MaskOperation::Or,  MaskNegate::None,   MaskDestination::OldExec, 32},
    {CanonicalOp::S_OR_SAVEEXEC_B64,    MaskOperation::Or,  MaskNegate::None,   MaskDestination::OldExec, 64},
    {CanonicalOp::S_XOR_SAVEEXEC_B32,   MaskOperation::Xor, MaskNegate::None,   MaskDestination::OldExec, 32},
    {CanonicalOp::S_XOR_SAVEEXEC_B64,   MaskOperation::Xor, MaskNegate::None,   MaskDestination::OldExec, 64},
    {CanonicalOp::S_NAND_SAVEEXEC_B32,  MaskOperation::And, MaskNegate::Result, MaskDestination::OldExec, 32},
    {CanonicalOp::S_NAND_SAVEEXEC_B64,  MaskOperation::And, MaskNegate::Result, MaskDestination::OldExec, 64},
    {CanonicalOp::S_NOR_SAVEEXEC_B32,   MaskOperation::Or,  MaskNegate::Result, MaskDestination::OldExec, 32},
    {CanonicalOp::S_NOR_SAVEEXEC_B64,   MaskOperation::Or,  MaskNegate::Result, MaskDestination::OldExec, 64},
    {CanonicalOp::S_XNOR_SAVEEXEC_B32,  MaskOperation::Xor, MaskNegate::Result, MaskDestination::OldExec, 32},
    {CanonicalOp::S_XNOR_SAVEEXEC_B64,  MaskOperation::Xor, MaskNegate::Result, MaskDestination::OldExec, 64},
    {CanonicalOp::S_ANDN1_SAVEEXEC_B32, MaskOperation::And, MaskNegate::Source, MaskDestination::OldExec, 32},
    {CanonicalOp::S_ANDN1_SAVEEXEC_B64, MaskOperation::And, MaskNegate::Source, MaskDestination::OldExec, 64},
    {CanonicalOp::S_ORN1_SAVEEXEC_B32,  MaskOperation::Or,  MaskNegate::Source, MaskDestination::OldExec, 32},
    {CanonicalOp::S_ORN1_SAVEEXEC_B64,  MaskOperation::Or,  MaskNegate::Source, MaskDestination::OldExec, 64},
    {CanonicalOp::S_ANDN2_SAVEEXEC_B32, MaskOperation::And, MaskNegate::Exec,   MaskDestination::OldExec, 32},
    {CanonicalOp::S_ANDN2_SAVEEXEC_B64, MaskOperation::And, MaskNegate::Exec,   MaskDestination::OldExec, 64},
    {CanonicalOp::S_ORN2_SAVEEXEC_B32,  MaskOperation::Or,  MaskNegate::Exec,   MaskDestination::OldExec, 32},
    {CanonicalOp::S_ORN2_SAVEEXEC_B64,  MaskOperation::Or,  MaskNegate::Exec,   MaskDestination::OldExec, 64},
    {CanonicalOp::S_ANDN1_WREXEC_B32,   MaskOperation::And, MaskNegate::Source, MaskDestination::NewExec, 32},
    {CanonicalOp::S_ANDN1_WREXEC_B64,   MaskOperation::And, MaskNegate::Source, MaskDestination::NewExec, 64},
    {CanonicalOp::S_ANDN2_WREXEC_B32,   MaskOperation::And, MaskNegate::Exec,   MaskDestination::NewExec, 32},
    {CanonicalOp::S_ANDN2_WREXEC_B64,   MaskOperation::And, MaskNegate::Exec,   MaskDestination::NewExec, 64},
};
// clang-format on

// The kExecCombines row for CanonOp, or null if CanonOp does not combine a
// mask with EXEC. A scan over a table this size costs less than the dynamic
// initialization a map keyed on the opcode would need.
static const ExecCombine *execCombine(CanonicalOp CanonOp) {
  const ExecCombine *Row = llvm::find_if(
      kExecCombines, [&](const ExecCombine &C) { return C.Opcode == CanonOp; });
  return Row == std::end(kExecCombines) ? nullptr : Row;
}

// The mask Combine computes from source mask Src and the EXEC it replaces.
static Value *emitMaskCombine(IRBuilder<> &B, const ExecCombine &Combine,
                              Value *Src, Value *Exec) {
  bool NegatesResult = Combine.Negate == MaskNegate::Result;
  Value *Lhs = Combine.Negate == MaskNegate::Source ? B.CreateNot(Src) : Src;
  Value *Rhs = Combine.Negate == MaskNegate::Exec ? B.CreateNot(Exec) : Exec;
  StringRef Name = NegatesResult ? "" : "new_exec";
  Value *Combined = nullptr;
  switch (Combine.Operation) {
  case MaskOperation::And:
    Combined = B.CreateAnd(Lhs, Rhs, Name);
    break;
  case MaskOperation::Or:
    Combined = B.CreateOr(Lhs, Rhs, Name);
    break;
  case MaskOperation::Xor:
    Combined = B.CreateXor(Lhs, Rhs, Name);
    break;
  }
  return NegatesResult ? B.CreateNot(Combined, "new_exec") : Combined;
}

Error handleSOP1(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  if (Di.CanonOp == CanonicalOp::S_MOV_B32 ||
      Di.CanonOp == CanonicalOp::S_MOV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_MOV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    // A copy of a pair holding a source code-object address holds the same
    // address, and loads off the copy still have to read it from the image.
    // Take the address from the pair rather than reading the registers, which
    // refuse to answer a read that would let the address escape.
    std::optional<uint64_t> SourceAddress;
    if (Is64) {
      Expected<std::optional<uint64_t>> Read =
          sourceImageOperandAddr(Ctx, Di, Op.srcIdx(0));
      if (!Read)
        return Read.takeError();
      SourceAddress = *Read;
    }
    if (SourceAddress && (Dst->RegKind != ParsedReg::SGPR || !Dst->BaseIdx))
      return unsupported(Ctx, Di,
                         "puts a source address outside an SGPR pair, which "
                         "is not tracked as one");
    Value *SrcValue = nullptr;
    if (SourceAddress) {
      SrcValue = ConstantInt::get(Ctx.B.getInt64Ty(), *SourceAddress);
    } else {
      Expected<Value *> Src = Op.src(0, Is64);
      if (!Src)
        return Src.takeError();
      SrcValue = *Src;
    }
    writeDst(Ctx.registers(), *Dst, SrcValue, Is64);
    if (SourceAddress)
      Ctx.registers().recordSourceImageSgprPairAddr(*Dst->BaseIdx,
                                                    *SourceAddress);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_BREV_B32 ||
      Di.CanonOp == CanonicalOp::S_BREV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_BREV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Value *Reversed = Ctx.B.CreateUnaryIntrinsic(Intrinsic::bitreverse, *Src,
                                                 /*FMFSource=*/{}, "s_brev");
    writeDst(Ctx.registers(), *Dst, Reversed, Is64);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_NOT_B32 ||
      Di.CanonOp == CanonicalOp::S_NOT_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_NOT_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Value *Result = Ctx.B.CreateNot(*Src, "s_not");
    writeDst(Ctx.registers(), *Dst, Result, Is64);
    // SCC is the result compared against zero, which storeSCC does for a
    // value wider than the single bit SCC holds.
    Ctx.registers().storeSCC(Ctx.B, Result);
    return Error::success();
  }

  // A clear SCC leaves the destination alone. The MC form carries no tied
  // operand for that preserved value, so it is read back off the destination.
  if (Di.CanonOp == CanonicalOp::S_CMOV_B32 ||
      Di.CanonOp == CanonicalOp::S_CMOV_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_CMOV_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Expected<Value *> Old = Is64 ? Op.dstValue64() : Op.dstValue();
    if (!Old)
      return Old.takeError();
    Expected<Value *> Scc = Ctx.registers().readSCC(Di);
    if (!Scc)
      return Scc.takeError();
    Value *Moved = Ctx.B.CreateSelect(*Scc, *Src, *Old, "s_cmov");
    writeDst(Ctx.registers(), *Dst, Moved, Is64);
    return Error::success();
  }

  if (isScalarFloat(Di.CanonOp)) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg32(*Dst, emitScalarFloat(Ctx.B, Di.CanonOp, *Src));
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_MOVRELS_B32 ||
      Di.CanonOp == CanonicalOp::S_MOVRELS_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_MOVRELS_B64;
    Expected<uint64_t> M0 = constantM0(Ctx, Di);
    if (!M0)
      return M0.takeError();
    Expected<unsigned> Src =
        displacedSgpr(Ctx, Di, Op.srcIdx(0), *M0, Is64 ? 2 : 1);
    if (!Src)
      return Src.takeError();
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Moved = Is64 ? Ctx.registers().readSgpr64(Di, *Src)
                                   : Ctx.registers().readSgpr32(Di, *Src);
    if (!Moved)
      return Moved.takeError();
    if (Is64)
      Ctx.registers().writeReg64(*Dst, *Moved);
    else
      Ctx.registers().writeReg32(*Dst, *Moved);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_MOVRELD_B32 ||
      Di.CanonOp == CanonicalOp::S_MOVRELD_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_MOVRELD_B64;
    Expected<uint64_t> M0 = constantM0(Ctx, Di);
    if (!M0)
      return M0.takeError();
    // These opcodes only read their destination register for its index, so it
    // is an input operand and heads the source map.
    Expected<unsigned> DstIdx =
        displacedSgpr(Ctx, Di, Op.srcIdx(0), *M0, Is64 ? 2 : 1);
    if (!DstIdx)
      return DstIdx.takeError();
    Expected<Value *> Src = Is64 ? Op.src64(1) : Op.src(1);
    if (!Src)
      return Src.takeError();
    ParsedReg Dst{ParsedReg::SGPR, *DstIdx, static_cast<uint8_t>(Is64 ? 2 : 1)};
    if (Is64)
      Ctx.registers().writeReg64(Dst, *Src);
    else
      Ctx.registers().writeReg32(Dst, *Src);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_MOVRELSD_2_B32) {
    Expected<uint64_t> M0 = constantM0(Ctx, Di);
    if (!M0)
      return M0.takeError();
    // The source index is displaced by M0[9:0] and the destination index by
    // M0[25:16].
    constexpr uint64_t FieldMask = (1u << 10) - 1;
    Expected<unsigned> Src =
        displacedSgpr(Ctx, Di, Op.srcIdx(0), *M0 & FieldMask, 1);
    if (!Src)
      return Src.takeError();
    Expected<unsigned> DstIdx =
        displacedSgpr(Ctx, Di, /*OpIdx=*/0, (*M0 >> 16) & FieldMask, 1);
    if (!DstIdx)
      return DstIdx.takeError();
    Expected<Value *> Moved = Ctx.registers().readSgpr32(Di, *Src);
    if (!Moved)
      return Moved.takeError();
    Ctx.registers().writeReg32(ParsedReg{ParsedReg::SGPR, *DstIdx, 1}, *Moved);
    return Error::success();
  }

  if (std::optional<BitSearch> Search = bitSearch(Di.CanonOp)) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Search->Is64);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg32(*Dst, emitBitSearch(Ctx.B, *Search, *Src));
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_SEXT_I32_I8 ||
      Di.CanonOp == CanonicalOp::S_SEXT_I32_I16) {
    unsigned Width = Di.CanonOp == CanonicalOp::S_SEXT_I32_I8 ? 8 : 16;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Value *Narrow = Ctx.B.CreateTrunc(*Src, Ctx.B.getIntNTy(Width));
    Ctx.registers().writeReg32(
        *Dst, Ctx.B.CreateSExt(Narrow, Ctx.B.getInt32Ty(), "s_sext"));
    return Error::success();
  }

  // The bit index comes from the low bits of the source and everything else in
  // the destination is preserved, so the destination is an input too. The MC
  // form drops the tied operand that carries it, so it is read back off the
  // destination register.
  if (Di.CanonOp == CanonicalOp::S_BITSET0_B32 ||
      Di.CanonOp == CanonicalOp::S_BITSET0_B64 ||
      Di.CanonOp == CanonicalOp::S_BITSET1_B32 ||
      Di.CanonOp == CanonicalOp::S_BITSET1_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_BITSET0_B64 ||
                Di.CanonOp == CanonicalOp::S_BITSET1_B64;
    bool Sets = Di.CanonOp == CanonicalOp::S_BITSET1_B32 ||
                Di.CanonOp == CanonicalOp::S_BITSET1_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    // The source is a dword whichever width the destination is, and only as
    // many of its low bits as index a bit of the destination are read.
    unsigned DestinationWidth = Is64 ? 64 : 32;
    Expected<Value *> Index = Op.src(0);
    if (!Index)
      return Index.takeError();
    Value *Shift =
        Ctx.B.CreateAnd(*Index, DestinationWidth - 1, "s_bitset_index");
    if (Is64)
      Shift = Ctx.B.CreateZExt(Shift, Ctx.B.getInt64Ty());
    Value *Bit = Ctx.B.CreateShl(ConstantInt::get(Shift->getType(), 1), Shift,
                                 "s_bitset_bit");
    Expected<Value *> Old = Is64 ? Op.dstValue64() : Op.dstValue();
    if (!Old)
      return Old.takeError();
    Value *Result;
    if (Sets) {
      Result = Ctx.B.CreateOr(*Old, Bit, "s_bitset1");
    } else {
      Value *Mask = Ctx.B.CreateNot(Bit);
      Result = Ctx.B.CreateAnd(*Old, Mask, "s_bitset0");
    }
    writeDst(Ctx.registers(), *Dst, Result, Is64);
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_BITREPLICATE_B64_B32) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Ctx.registers().writeReg64(*Dst, emitBitReplicate(Ctx.B, *Src));
    return Error::success();
  }

  if (Di.CanonOp == CanonicalOp::S_QUADMASK_B32 ||
      Di.CanonOp == CanonicalOp::S_QUADMASK_B64 ||
      Di.CanonOp == CanonicalOp::S_WQM_B32 ||
      Di.CanonOp == CanonicalOp::S_WQM_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_QUADMASK_B64 ||
                Di.CanonOp == CanonicalOp::S_WQM_B64;
    bool Reduces = Di.CanonOp == CanonicalOp::S_QUADMASK_B32 ||
                   Di.CanonOp == CanonicalOp::S_QUADMASK_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Value *Result =
        Reduces ? emitQuadMask(Ctx.B, *Src) : emitWholeQuadMask(Ctx.B, *Src);
    writeDst(Ctx.registers(), *Dst, Result, Is64);
    Ctx.registers().storeSCC(Ctx.B, Result);
    return Error::success();
  }

  // EXEC is written before the destination, so a destination naming EXEC ends
  // up holding what the destination rule says rather than the combined mask.
  // SCC reports the combined mask either way.
  if (const ExecCombine *Combine = execCombine(Di.CanonOp)) {
    if (!waveMaskWidthIs(Ctx, Combine->MaskBits))
      return unsupported(Ctx, Di,
                         "combines a " + Twine(Combine->MaskBits) +
                             "-bit mask with EXEC, but " +
                             heldWaveMaskWidths(Ctx));
    bool Is64 = Combine->MaskBits == 64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Value *OldExec = Ctx.registers().readExec();
    Value *NewExec = emitMaskCombine(Ctx.B, *Combine, *Src, OldExec);
    Ctx.registers().storeExec(NewExec);
    Ctx.registers().storeSCC(Ctx.B, NewExec);
    writeDst(Ctx.registers(), *Dst,
             Combine->Destination == MaskDestination::NewExec ? NewExec
                                                              : OldExec,
             Is64);
    return Error::success();
  }

  // The most negative input has no positive counterpart and the hardware keeps
  // it as it is, which is what llvm.abs does when overflow is not poison.
  if (Di.CanonOp == CanonicalOp::S_ABS_I32) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0);
    if (!Src)
      return Src.takeError();
    Value *Result = Ctx.B.CreateBinaryIntrinsic(Intrinsic::abs, *Src,
                                                Ctx.B.getFalse(), {}, "s_abs");
    Ctx.registers().writeReg32(*Dst, Result);
    Ctx.registers().storeSCC(Ctx.B, Result);
    return Error::success();
  }

  // Counting zeros is counting the ones of the complement. Either way the
  // result is one dword, so the 64-bit forms narrow their count, which cannot
  // lose anything below 65.
  if (Di.CanonOp == CanonicalOp::S_BCNT0_I32_B32 ||
      Di.CanonOp == CanonicalOp::S_BCNT0_I32_B64 ||
      Di.CanonOp == CanonicalOp::S_BCNT1_I32_B32 ||
      Di.CanonOp == CanonicalOp::S_BCNT1_I32_B64) {
    bool Is64 = Di.CanonOp == CanonicalOp::S_BCNT0_I32_B64 ||
                Di.CanonOp == CanonicalOp::S_BCNT1_I32_B64;
    bool CountsZeros = Di.CanonOp == CanonicalOp::S_BCNT0_I32_B32 ||
                       Di.CanonOp == CanonicalOp::S_BCNT0_I32_B64;
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    Expected<Value *> Src = Op.src(0, Is64);
    if (!Src)
      return Src.takeError();
    Value *Counted = CountsZeros ? Ctx.B.CreateNot(*Src, "s_bcnt0_bits") : *Src;
    Value *Count =
        Ctx.B.CreateUnaryIntrinsic(Intrinsic::ctpop, Counted, {}, "s_bcnt");
    Value *Result = Ctx.B.CreateZExtOrTrunc(Count, Ctx.B.getInt32Ty());
    Ctx.registers().writeReg32(*Dst, Result);
    Ctx.registers().storeSCC(Ctx.B, Result);
    return Error::success();
  }

  // A constant displacement names a source offset rather than an address, and
  // a source offset leads a block, so the jump is a branch between blocks. The
  // decode already worked out where it goes; recomputing it here would be a
  // second chance to disagree about the target.
  if (Di.CanonOp == CanonicalOp::S_ADD_PC_I64) {
    if (!hasStaticBranchTarget(Di))
      return unsupported(Ctx, Di,
                         "displacement is not a constant, so the offset it "
                         "jumps to is not known");
    Expected<uint64_t> Target = staticBranchTarget(Di);
    if (!Target)
      return Target.takeError();
    Ctx.B.CreateBr(Ctx.lookupBB(*Target));
    return Error::success();
  }

  // The captured address is where the next instruction sits in the source code
  // object, which the decode already knows as a constant. Nothing in the raised
  // kernel runs at that address, so the value only stands for what the source
  // image holds there. Track the pair as a source address so that a
  // PC-relative scalar load off it reads the literal the source meant instead
  // of reaching into target memory.
  if (Di.CanonOp == CanonicalOp::S_GETPC_B64) {
    Expected<ParsedReg> Dst = Op.dst();
    if (!Dst)
      return Dst.takeError();
    if (Dst->RegKind != ParsedReg::SGPR || !Dst->BaseIdx)
      return unsupported(Ctx, Di,
                         "captures a source address outside an SGPR pair, "
                         "which is not tracked as one");
    uint64_t NextInstAddress =
        Ctx.sourceTextBaseAddress() + Di.Offset + Di.sizeInBytes();
    Ctx.registers().writeReg64(
        *Dst, ConstantInt::get(Ctx.B.getInt64Ty(), NextInstAddress));
    Ctx.registers().recordSourceImageSgprPairAddr(*Dst->BaseIdx,
                                                  NextInstAddress);
    return Error::success();
  }

  // A jump through a register names no offset of its own, so it goes wherever
  // the analysis found in the pair that it reads. Those offsets lead blocks,
  // reported by the analysis before any block was made. One offset is a branch
  // and several are a switch. A call additionally leaves behind the offset it
  // returns to, which is the source address of the instruction after it, the
  // same value that a program-counter capture there would have produced.
  if (Di.CanonOp == CanonicalOp::S_SETPC_B64 ||
      Di.CanonOp == CanonicalOp::S_SWAPPC_B64) {
    const SetPcSite *Site = Ctx.setPcSite(Di.Offset);
    assert(Site && "every register-indirect transfer is classified");
    if (const SetPcUnresolvable *Why = std::get_if<SetPcUnresolvable>(Site))
      return refuseSetPc(Ctx, Di, *Why);

    const SetPcResolved &Resolved = std::get<SetPcResolved>(*Site);
    assert(Resolved.Targets.size() <= std::numeric_limits<unsigned>::max() &&
           "the analysis caps how many offsets one transfer enumerates");

    // Read the target before the call writes its destination, which may name
    // the very pair the target comes from. The pair holds a source address,
    // which an operand read refuses because nothing in the raised kernel can
    // address one. Here the address only picks which of the offsets the
    // analysis enumerated this jump takes, so the read goes to the register
    // file directly, past the refusal.
    Value *Target = nullptr;
    if (Resolved.Targets.size() > 1) {
      Expected<std::optional<ParsedReg>> Src = Op.srcReg(0);
      if (!Src)
        return Src.takeError();
      assert(*Src && (*Src)->RegKind == ParsedReg::SGPR && (*Src)->BaseIdx &&
             "a transfer reaching several offsets reads a scalar pair");
      Target = Ctx.registers().regFile().loadSGPR64(Ctx.B, *(*Src)->BaseIdx);
    }

    if (Di.CanonOp == CanonicalOp::S_SWAPPC_B64) {
      Expected<ParsedReg> Dst = Op.dst();
      if (!Dst)
        return Dst.takeError();
      if (Dst->RegKind != ParsedReg::SGPR || !Dst->BaseIdx)
        return unsupported(Ctx, Di,
                           "leaves its return address outside an SGPR pair, "
                           "which is not tracked as one");
      uint64_t ReturnAddress =
          Ctx.sourceTextBaseAddress() + Di.Offset + Di.sizeInBytes();
      Ctx.registers().writeReg64(
          *Dst, ConstantInt::get(Ctx.B.getInt64Ty(), ReturnAddress));
      Ctx.registers().recordSourceImageSgprPairAddr(*Dst->BaseIdx,
                                                    ReturnAddress);
    }

    if (!Target) {
      Ctx.B.CreateBr(Ctx.lookupBB(Resolved.Targets.front()));
      return Error::success();
    }

    // The offsets the analysis enumerated are the only ones the pair can hold,
    // so the fall-out-of-the-table case cannot be reached and traps rather
    // than picking one of them.
    BasicBlock *Trap =
        BasicBlock::Create(Ctx.B.getContext(), formatv("trap_{0:x}", Di.Offset),
                           Ctx.B.GetInsertBlock()->getParent());
    SwitchInst *Dispatch = Ctx.B.CreateSwitch(
        Target, Trap, static_cast<unsigned>(Resolved.Targets.size()));
    for (uint64_t Offset : Resolved.Targets)
      Dispatch->addCase(ConstantInt::get(Ctx.B.getInt64Ty(),
                                         Ctx.sourceTextBaseAddress() + Offset),
                        Ctx.lookupBB(Offset));

    IRBuilderBase::InsertPointGuard Guard(Ctx.B);
    Ctx.B.SetInsertPoint(Trap);
    Ctx.B.CreateIntrinsic(Ctx.B.getVoidTy(), Intrinsic::trap, {});
    Ctx.B.CreateUnreachable();
    return Error::success();
  }

  switch (Di.CanonOp) {
  // The source splits a barrier in two: this arrival, which does not block,
  // and a release in SOPP, which does. The raise has one barrier, and it
  // arrives and waits at once, so standing it in for the arrival alone holds
  // the wave where the source let it run on. A program that made progress only
  // because the arrival did not block deadlocks under that reading, so the
  // arrival is refused rather than widened into a whole barrier.
  //
  // An arrival the matching release always follows is the exception: the pair
  // is raised as the one barrier it amounts to, at the release, leaving the
  // arrival nothing of its own to state.
  case CanonicalOp::S_BARRIER_SIGNAL_IMM:
    if (Ctx.isPairedSplitBarrier(Di.Offset))
      return Error::success();
    return unsupported(Ctx, Di,
                       "arrives at a barrier without waiting there, and the "
                       "raise has only a barrier that also waits");

  // The barrier this arrives at is whichever one m0 names at run time, so
  // which barrier it is cannot be read off the instruction and matched to a
  // release.
  case CanonicalOp::S_BARRIER_SIGNAL_M0:
    return unsupported(Ctx, Di,
                       "arrives at the barrier m0 names, which the raise "
                       "cannot match to the wait that releases it");

  // The rest of the family speaks about a named barrier: a barrier a subset of
  // the workgroup joins, leaves, sizes and polls. The raise carries no barrier
  // membership and no arrival count, so lifting these would invent the
  // synchronization they describe.
  case CanonicalOp::S_BARRIER_SIGNAL_ISFIRST_IMM:
  case CanonicalOp::S_BARRIER_SIGNAL_ISFIRST_M0:
    return unsupported(Ctx, Di,
                       "reports whether this wave arrived at the barrier "
                       "first, which the raise does not track");
  case CanonicalOp::S_GET_BARRIER_STATE_IMM:
  case CanonicalOp::S_GET_BARRIER_STATE_M0:
    return unsupported(Ctx, Di,
                       "reads the arrival and membership counts of a barrier "
                       "the raise keeps none for");
  case CanonicalOp::S_BARRIER_INIT_IMM:
  case CanonicalOp::S_BARRIER_INIT_M0:
    return unsupported(Ctx, Di, "sizes the membership of a named barrier");
  case CanonicalOp::S_BARRIER_JOIN_IMM:
  case CanonicalOp::S_BARRIER_JOIN_M0:
    return unsupported(Ctx, Di, "joins this wave to a named barrier");
  case CanonicalOp::S_WAKEUP_BARRIER_IMM:
  case CanonicalOp::S_WAKEUP_BARRIER_M0:
    return unsupported(Ctx, Di, "wakes the waves waiting on a named barrier");

  case CanonicalOp::S_RFE_B64:
    return unsupported(Ctx, Di, "returns from an exception handler");

  // Free-running cycle counts are a property of the shader clock the source
  // wave was issued against. The raised kernel runs on another one, at another
  // rate and from another epoch, so any difference the kernel takes between two
  // reads means something else there.
  case CanonicalOp::S_GET_SHADER_CYCLES_U64:
    return unsupported(Ctx, Di,
                       "reads the source shader clock, whose rate and epoch "
                       "the raised kernel does not share");

  // Every return message answers with where the source wave sits in the
  // hardware and in its queue: its doorbell, its dispatch, its trap handler
  // base, its shader engine. None of that describes the wave that runs the
  // raised kernel.
  case CanonicalOp::S_SENDMSG_RTN_B32:
  case CanonicalOp::S_SENDMSG_RTN_B64:
    return unsupported(Ctx, Di,
                       "reads back message " + Twine(Op.srcImm(0)) +
                           ", which reports on the source wave's queue and "
                           "hardware placement");

  // The register file backs every VGPR index a source encoding can name, so
  // the allocation this asks for is already in hand and cannot fail. Answering
  // otherwise is not the safe direction: a source that checks SCC retries until
  // it succeeds, so a false answer spins forever.
  //
  // Allocating also waits for the outstanding memory the shrinking form could
  // otherwise strand in registers it releases, so the source may rely on this
  // in place of a wait it does not spell out. Waiting for the stores too is
  // wider than the hardware does and costs a raised kernel nothing here.
  case CanonicalOp::S_ALLOC_VGPR:
    emitMemoryWaitAll(Ctx);
    Ctx.registers().storeSCC(Ctx.B, Ctx.B.getTrue());
    return Error::success();

  // Spends issue slots and leaves nothing behind that a later instruction can
  // read, in either operand form.
  case CanonicalOp::S_SLEEP_VAR:
    return Error::success();

  default:
    break;
  }

  return unsupported(Ctx, Di);
}

} // namespace COMGR::transpiler
