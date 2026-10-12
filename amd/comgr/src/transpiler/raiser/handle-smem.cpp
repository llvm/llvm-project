//===- handle-smem.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "SIDefines.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/source-image.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <climits>
#include <cstdint>
#include <optional>
#include <string>

using namespace llvm;

namespace COMGR::transpiler {

// Every address component is aligned down to the data size, byte loads use
// the full address, 16-bit loads ignore bit 0, and wider loads ignore bits
// [1:0].
static constexpr Align MaxSmemAddressAlignment = Align::Constant<4>();

// Report decoded operands that contradict the generated instruction metadata.
[[noreturn]] static void invalidOperandLayout(const MCState &MC,
                                              const DecodedInst &Di,
                                              StringRef Detail) {
  std::string Message =
      formatv("transpiler: instruction '{0}' (MC opcode {1}, format {2}, "
              "offset 0x{3:x}) has invalid operand layout: {4}",
              strippedMnemonic(MC, Di.Inst), Di.Inst.getOpcode(),
              formatName(Di.TargetSpecificFlags), Di.Offset, Detail)
          .str();
  report_fatal_error(StringRef(Message));
}

// Return the index assigned to a TableGen-named operand, if present.
static std::optional<unsigned> namedOperandIndex(const MCState &MC,
                                                 const DecodedInst &Di,
                                                 AMDGPU::OpName Name,
                                                 StringRef OperandName) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Index < 0)
    return std::nullopt;
  if (static_cast<unsigned>(Index) >= Di.numOperands())
    invalidOperandLayout(
        MC, Di,
        formatv("operand '{0}' has index {1}, but the instruction has {2} "
                "operands",
                OperandName, Index, Di.numOperands())
            .str());
  return static_cast<unsigned>(Index);
}

// Return the index of an operand every mapped scalar load must carry.
static unsigned requiredNamedOperandIndex(const MCState &MC,
                                          const DecodedInst &Di,
                                          AMDGPU::OpName Name,
                                          StringRef OperandName) {
  std::optional<unsigned> Index = namedOperandIndex(MC, Di, Name, OperandName);
  if (!Index)
    invalidOperandLayout(MC, Di,
                         formatv("missing required operand '{0}' (OpName {1})",
                                 OperandName, static_cast<unsigned>(Name))
                             .str());
  return *Index;
}

// The data size in bytes of a supported non-buffer scalar load, plus how a
// sub-dword result reaches its destination. The size drives the address
// alignment, the SCALE_OFFSET factor and the load type.
struct ScalarLoadInfo {
  unsigned SizeInBytes;
  bool SignExtends;
};

static std::optional<ScalarLoadInfo> scalarLoadInfo(CanonicalOp Operation) {
  switch (Operation) {
  case CanonicalOp::S_LOAD_I8:
    return ScalarLoadInfo{1, true};
  case CanonicalOp::S_LOAD_U8:
    return ScalarLoadInfo{1, false};
  case CanonicalOp::S_LOAD_I16:
    return ScalarLoadInfo{2, true};
  case CanonicalOp::S_LOAD_U16:
    return ScalarLoadInfo{2, false};
  case CanonicalOp::S_LOAD_B32:
    return ScalarLoadInfo{4, false};
  case CanonicalOp::S_LOAD_B64:
    return ScalarLoadInfo{8, false};
  case CanonicalOp::S_LOAD_B96:
    return ScalarLoadInfo{12, false};
  case CanonicalOp::S_LOAD_B128:
    return ScalarLoadInfo{16, false};
  case CanonicalOp::S_LOAD_B256:
    return ScalarLoadInfo{32, false};
  case CanonicalOp::S_LOAD_B512:
    return ScalarLoadInfo{64, false};
  default:
    return std::nullopt;
  }
}

// Tell whether the scalar-offset operand adds nothing to the address. An
// encoding that carries the slot but does not use it spells the operand NOREG.
static Expected<bool>
scalarOffsetIsZero(RaiseContext &Ctx, const DecodedInst &Di, unsigned Index) {
  // Resolve expressions before parsing the operand as a register.
  if (std::optional<int64_t> Constant = evalOperandAsConst(Di.Inst, Index))
    return *Constant == 0;
  if (!Di.isReg(Index))
    return unsupported(Ctx, Di,
                       "scalar offset is neither a constant nor a register");
  Expected<ParsedReg> Register = Ctx.registers().parseReg(Di, Index);
  if (!Register)
    return Register.takeError();
  return Register->RegKind == ParsedReg::NOREG;
}

// Answer a scalar load whose base addresses the source code object rather than
// the memory the raised kernel runs against. Every dword it names is a literal
// the source compiled in, so write each one to its destination register as a
// constant.
static Error loadFromSourceImage(RaiseContext &Ctx, const DecodedInst &Di,
                                 ParsedReg Destination, uint64_t BaseAddress,
                                 int64_t ImmediateOffset,
                                 unsigned LoadWidthInDwords) {
  // The hardware drops the low bits of the base and of the offset one by one
  // rather than of their sum, so read what the source read instead of failing
  // on a misaligned offset.
  uint64_t DwordBytes = MaxSmemAddressAlignment.value();
  Expected<uint64_t> FirstDword = moveSourceImageAddress(
      Ctx, Di, alignDown(BaseAddress, DwordBytes),
      alignDown(static_cast<uint64_t>(ImmediateOffset), DwordBytes));
  if (!FirstDword)
    return FirstDword.takeError();
  SmallVector<uint32_t> Dwords;
  Dwords.reserve(LoadWidthInDwords);
  for (unsigned I = 0; I != LoadWidthInDwords; ++I) {
    Expected<uint64_t> DwordAddress =
        moveSourceImageAddress(Ctx, Di, *FirstDword, I * DwordBytes);
    if (!DwordAddress)
      return DwordAddress.takeError();
    std::optional<uint32_t> Dword = readSourceImageDword(Ctx, *DwordAddress);
    if (!Dword)
      return unsupported(Ctx, Di,
                         "reads a source address that no section of the "
                         "source code object covers");
    Dwords.push_back(*Dword);
  }

  if (LoadWidthInDwords == 1) {
    Ctx.registers().writeReg32(Destination,
                               ConstantInt::get(Ctx.B.getInt32Ty(), Dwords[0]));
  } else if (LoadWidthInDwords == 2) {
    uint64_t Pair = static_cast<uint64_t>(Dwords[1]) << 32 | Dwords[0];
    Ctx.registers().writeReg64(Destination,
                               ConstantInt::get(Ctx.B.getInt64Ty(), Pair));
  } else {
    Ctx.registers().writeRegVec(
        Destination, ConstantDataVector::get(Ctx.B.getContext(), Dwords));
  }
  return Error::success();
}

// Build the byte address a scalar load reads from:
//
//   Address = (Base & Mask) + alignDown(ImmediateOffset, AddressAlignment) +
//             ((zext(ScalarOffset) [* LoadSizeInBytes]) & Mask)
//
// Mask clears the low alignment bits; SCALE_OFFSET applies the multiply before
// masking. ScalarOffset is null when no SGPR offset is encoded.
static Value *
emitScalarLoadAddress(IRBuilder<> &B, Value *Base, int64_t ImmediateOffset,
                      Value *ScalarOffset, bool ScalesScalarOffset,
                      unsigned LoadSizeInBytes, Align AddressAlignment) {
  Type *I64Ty = B.getInt64Ty();
  // A byte load aligns to one, so it masks nothing and the address components
  // reach the pointer unchanged.
  bool MasksAddress = AddressAlignment > Align(1);
  Constant *AddressMask = ConstantInt::get(
      I64Ty, maskTrailingZeros<uint64_t>(Log2(AddressAlignment)));

  Value *Address = Base;
  if (MasksAddress)
    Address = B.CreateAnd(Address, AddressMask, "smem_base");
  uint64_t Offset = alignDown(static_cast<uint64_t>(ImmediateOffset),
                              AddressAlignment.value());
  Address = B.CreateAdd(Address, ConstantInt::get(I64Ty, Offset), "smem_addr");
  if (!ScalarOffset)
    return Address;

  // SCALE_OFFSET treats the 32-bit SGPR value as an element index. Widen it
  // first, then scale it by the load size.
  Value *WideScalarOffset = B.CreateZExt(ScalarOffset, I64Ty, "smem_soff");
  if (ScalesScalarOffset && LoadSizeInBytes > 1)
    WideScalarOffset =
        B.CreateMul(WideScalarOffset, ConstantInt::get(I64Ty, LoadSizeInBytes),
                    "smem_soff_scaled");
  if (MasksAddress)
    WideScalarOffset =
        B.CreateAnd(WideScalarOffset, AddressMask, "smem_soff_dword");
  return B.CreateAdd(Address, WideScalarOffset, "smem_addr_soff");
}

Error handleSMEM(RaiseContext &Ctx, const DecodedInst &Di, OperandResolver &) {
  std::optional<ScalarLoadInfo> Info = scalarLoadInfo(Di.CanonOp);
  if (!Info)
    return unsupported(Ctx, Di, "unsupported scalar memory operation");
  unsigned LoadSizeInBytes = Info->SizeInBytes;
  // Narrow loads transfer less than one dword: i8/u8/i16/u16.
  bool IsNarrowLoad = LoadSizeInBytes < MaxSmemAddressAlignment.value();
  // A narrow load extends into a single dword; wider loads fill a tuple.
  unsigned DestinationWidthInDwords =
      IsNarrowLoad ? 1 : LoadSizeInBytes / MaxSmemAddressAlignment.value();
  Align AddressAlignment =
      IsNarrowLoad ? Align(LoadSizeInBytes) : MaxSmemAddressAlignment;

  unsigned DestinationIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sdst, "sdst");
  unsigned BaseIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::sbase, "sbase");
  unsigned CachePolicyIndex =
      requiredNamedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::cpol, "cpol");
  std::optional<unsigned> OffsetIndex =
      namedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::offset, "offset");
  std::optional<unsigned> ScalarOffsetIndex =
      namedOperandIndex(Ctx.MC, Di, AMDGPU::OpName::soffset, "soffset");
  if (!Di.isReg(DestinationIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'sdst' is not a register");
  if (!Di.isReg(BaseIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'sbase' is not a register");
  if (!Di.isImm(CachePolicyIndex))
    invalidOperandLayout(Ctx.MC, Di, "operand 'cpol' is not an immediate");
  // SCALE_OFFSET is encoded in the cache-policy field but changes the address:
  // it makes the SGPR offset an element index scaled by the load size. The
  // scale is handled below.
  //
  // NV only affects whether fine-grained cache write-back or invalidation
  // includes the line; it does not affect the value loaded. The raised IR
  // cannot represent this hint, so accept it only for gfx12 sources and drop
  // it, using the target's default NV=0 behavior.
  //
  // Other cache-policy modifiers remain unsupported and are rejected below.
  int64_t CachePolicy = Di.getImm(CachePolicyIndex);
  bool ScaleScalarOffset = (CachePolicy & AMDGPU::CPol::SCAL) != 0;
  int64_t ModeledCachePolicy = AMDGPU::CPol::SCAL;
  if (Ctx.Projection.SourceSTI.hasFeature(AMDGPU::FeatureGFX12Insts))
    ModeledCachePolicy |= AMDGPU::CPol::NV;
  if (CachePolicy & ~ModeledCachePolicy)
    return unsupported(Ctx, Di,
                       "scalar load carries a cache-policy modifier the raise "
                       "does not model");

  // The immediate and SGPR offsets are separate operands that add together.
  // An encoding may carry either or both, but not neither.
  if (!OffsetIndex && !ScalarOffsetIndex)
    invalidOperandLayout(Ctx.MC, Di, "scalar load has no offset operand");
  int64_t ImmediateOffset = 0;
  if (OffsetIndex) {
    if (!Di.isImm(*OffsetIndex))
      invalidOperandLayout(Ctx.MC, Di, "operand 'offset' is not an immediate");
    ImmediateOffset = Di.getImm(*OffsetIndex);
  }

  Expected<ParsedReg> Destination =
      Ctx.registers().parseReg(Di, DestinationIndex);
  if (!Destination)
    return Destination.takeError();
  if (Destination->RegKind != ParsedReg::SGPR)
    return unsupported(Ctx, Di, "scalar load requires an SGPR destination");
  if (!Destination->BaseIdx)
    invalidOperandLayout(Ctx.MC, Di,
                         "SGPR destination has no base register index");
  if (Destination->WidthInDwords != DestinationWidthInDwords)
    invalidOperandLayout(Ctx.MC, Di,
                         "destination width does not match the load opcode");

  Expected<ParsedReg> Base = Ctx.registers().parseReg(Di, BaseIndex);
  if (!Base)
    return Base.takeError();
  if (Base->RegKind != ParsedReg::SGPR)
    return unsupported(Ctx, Di, "scalar load requires an SGPR-pair base");
  if (!Base->BaseIdx)
    invalidOperandLayout(Ctx.MC, Di, "SGPR base has no base register index");
  if (Base->WidthInDwords != 2)
    invalidOperandLayout(Ctx.MC, Di, "scalar load base is not two dwords");

  // A base the source computed from its own program counter addresses the
  // source code object. Nothing of the source image is mapped where the raised
  // kernel runs, so answer the load from the captured image instead of emitting
  // it, which would read target memory at a source address.
  Expected<std::optional<uint64_t>> SourceImageBase =
      sourceImageSgprPairAddr(Ctx, Di, *Base->BaseIdx);
  if (!SourceImageBase)
    return SourceImageBase.takeError();
  if (*SourceImageBase) {
    // The source image is read a dword at a time, which a sub-dword load does
    // not divide into: its address needs no alignment and its result needs
    // extending.
    if (IsNarrowLoad)
      return unsupported(Ctx, Di,
                         "sub-dword scalar loads from the source code object "
                         "are not supported");
    if (ScalarOffsetIndex) {
      Expected<bool> IsZero = scalarOffsetIsZero(Ctx, Di, *ScalarOffsetIndex);
      if (!IsZero)
        return IsZero.takeError();
      if (!*IsZero)
        return unsupported(Ctx, Di,
                           "reads the source code object at an offset only "
                           "the running kernel knows");
    }
    return loadFromSourceImage(Ctx, Di, *Destination, **SourceImageBase,
                               ImmediateOffset, DestinationWidthInDwords);
  }

  // The immediate is a signed byte offset, and a load off a raw address may
  // reach backwards with it. Only the source-image path above follows it that
  // way; the emitted load aligns the base and the offset apart, which a
  // negative offset does not survive.
  if (ImmediateOffset < 0)
    return unsupported(Ctx, Di,
                       "negative scalar load offsets are not supported");

  Expected<Value *> BaseValue = Ctx.registers().readOp64(Di, BaseIndex);
  if (!BaseValue)
    return BaseValue.takeError();

  Value *ScalarOffsetValue = nullptr;
  if (ScalarOffsetIndex) {
    Expected<Value *> Read = Ctx.registers().readOp32(Di, *ScalarOffsetIndex);
    if (!Read)
      return Read.takeError();
    ScalarOffsetValue = *Read;
  }

  Value *Address = emitScalarLoadAddress(Ctx.B, *BaseValue, ImmediateOffset,
                                         ScalarOffsetValue, ScaleScalarOffset,
                                         LoadSizeInBytes, AddressAlignment);
  PointerType *PointerTy =
      PointerType::get(Ctx.B.getContext(), AMDGPUAS::GLOBAL_ADDRESS);
  Value *Pointer = Ctx.B.CreateIntToPtr(Address, PointerTy, "smem_ptr");

  Type *LoadType = Ctx.B.getInt32Ty();
  if (IsNarrowLoad)
    LoadType = Ctx.B.getIntNTy(LoadSizeInBytes * CHAR_BIT);
  else if (DestinationWidthInDwords == 2)
    LoadType = Ctx.B.getInt64Ty();
  else if (DestinationWidthInDwords > 2)
    LoadType =
        FixedVectorType::get(Ctx.B.getInt32Ty(), DestinationWidthInDwords);
  if (!Ctx.Projection.allowsDivergentScalarControlFlow())
    Ctx.requireWaveUniform(
        Pointer, Di, "projection requires uniform scalar memory addresses");
  Value *Loaded = Ctx.registers().emitMemoryValue(
      [&] {
        return Ctx.B.CreateAlignedLoad(LoadType, Pointer, AddressAlignment,
                                       "smem_load");
      },
      /*IsScalar=*/true);
  if (IsNarrowLoad) {
    Value *Extended =
        Info->SignExtends
            ? Ctx.B.CreateSExt(Loaded, Ctx.B.getInt32Ty(), "smem_load_sext")
            : Ctx.B.CreateZExt(Loaded, Ctx.B.getInt32Ty(), "smem_load_zext");
    Ctx.registers().writeReg32(*Destination, Extended);
  } else if (DestinationWidthInDwords == 1) {
    Ctx.registers().writeReg32(*Destination, Loaded);
  } else if (DestinationWidthInDwords == 2) {
    Ctx.registers().writeReg64(*Destination, Loaded);
  } else {
    // Every wider load is a vector of i32 written across an SGPR tuple.
    Ctx.registers().writeRegVec(*Destination, Loaded);
  }
  return Error::success();
}

} // namespace COMGR::transpiler
