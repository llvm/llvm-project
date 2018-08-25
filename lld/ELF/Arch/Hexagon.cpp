//===-- Hexagon.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class Hexagon final : public TargetInfo {
public:
  uint32_t calcEFlags() const override;
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
};
} // namespace

// Support V60 only at the moment.
uint32_t Hexagon::calcEFlags() const { return 0x60; }

static uint32_t applyMask(uint32_t Mask, uint32_t Data) {
  uint32_t Result = 0;
  size_t Off = 0;

  for (size_t Bit = 0; Bit != 32; ++Bit) {
    uint32_t ValBit = (Data >> Off) & 1;
    uint32_t MaskBit = (Mask >> Bit) & 1;
    if (MaskBit) {
      Result |= (ValBit << Bit);
      ++Off;
    }
  }
  return Result;
}

RelExpr Hexagon::getRelExpr(RelType Type, const Symbol &S,
                            const uint8_t *Loc) const {
  switch (Type) {
  case R_HEX_B15_PCREL:
  case R_HEX_B15_PCREL_X:
  case R_HEX_B22_PCREL:
  case R_HEX_B22_PCREL_X:
  case R_HEX_B32_PCREL_X:
  case R_HEX_6_PCREL_X:
    return R_PC;
  default:
    return R_ABS;
  }
}

static uint32_t findMaskR6(uint32_t Insn) {
  // There are (arguably too) many relocation masks for the DSP's
  // R_HEX_6_X type.  The table below is used to select the correct mask
  // for the given instruction.
  struct InstructionMask {
    uint32_t CmpMask;
    uint32_t RelocMask;
  };

  static const InstructionMask R6[] = {
      {0x38000000, 0x0000201f}, {0x39000000, 0x0000201f},
      {0x3e000000, 0x00001f80}, {0x3f000000, 0x00001f80},
      {0x40000000, 0x000020f8}, {0x41000000, 0x000007e0},
      {0x42000000, 0x000020f8}, {0x43000000, 0x000007e0},
      {0x44000000, 0x000020f8}, {0x45000000, 0x000007e0},
      {0x46000000, 0x000020f8}, {0x47000000, 0x000007e0},
      {0x6a000000, 0x00001f80}, {0x7c000000, 0x001f2000},
      {0x9a000000, 0x00000f60}, {0x9b000000, 0x00000f60},
      {0x9c000000, 0x00000f60}, {0x9d000000, 0x00000f60},
      {0x9f000000, 0x001f0100}, {0xab000000, 0x0000003f},
      {0xad000000, 0x0000003f}, {0xaf000000, 0x00030078},
      {0xd7000000, 0x006020e0}, {0xd8000000, 0x006020e0},
      {0xdb000000, 0x006020e0}, {0xdf000000, 0x006020e0}};

  // Duplex forms have a fixed mask and parse bits 15:14 are always
  // zero.  Non-duplex insns will always have at least one bit set in the
  // parse field.
  if ((0xC000 & Insn) == 0x0)
    return 0x03f00000;

  for (InstructionMask I : R6)
    if ((0xff000000 & Insn) == I.CmpMask)
      return I.RelocMask;

  error("unrecognized instruction for R_HEX_6 relocation: 0x" +
        utohexstr(Insn));
  return 0;
}

static uint32_t findMaskR8(uint32_t Insn) {
  if ((0xff000000 & Insn) == 0xde000000)
    return 0x00e020e8;
  if ((0xff000000 & Insn) == 0x3c000000)
    return 0x0000207f;
  return 0x00001fe0;
}

static uint32_t findMaskR16(uint32_t Insn) {
  if ((0xff000000 & Insn) == 0x48000000)
    return 0x061f20ff;
  if ((0xff000000 & Insn) == 0x49000000)
    return 0x061f3fe0;
  if ((0xff000000 & Insn) == 0x78000000)
    return 0x00df3fe0;
  if ((0xff000000 & Insn) == 0xb0000000)
    return 0x0fe03fe0;

  error("unrecognized instruction for R_HEX_16_X relocation: 0x" +
        utohexstr(Insn));
  return 0;
}

static void or32le(uint8_t *P, int32_t V) { write32le(P, read32le(P) | V); }

void Hexagon::relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_HEX_NONE:
    break;
  case R_HEX_6_PCREL_X:
  case R_HEX_6_X:
    or32le(Loc, applyMask(findMaskR6(read32le(Loc)), Val));
    break;
  case R_HEX_8_X:
    or32le(Loc, applyMask(findMaskR8(read32le(Loc)), Val));
    break;
  case R_HEX_12_X:
    or32le(Loc, applyMask(0x000007e0, Val));
    break;
  case R_HEX_16_X:  // This reloc only has 6 effective bits.
    or32le(Loc, applyMask(findMaskR16(read32le(Loc)), Val & 0x3f));
    break;
  case R_HEX_32:
    or32le(Loc, Val);
    break;
  case R_HEX_32_6_X:
    or32le(Loc, applyMask(0x0fff3fff, Val >> 6));
    break;
  case R_HEX_B15_PCREL:
    or32le(Loc, applyMask(0x00df20fe, Val >> 2));
    break;
  case R_HEX_B15_PCREL_X:
    or32le(Loc, applyMask(0x00df20fe, Val & 0x3f));
    break;
  case R_HEX_B22_PCREL:
    or32le(Loc, applyMask(0x1ff3ffe, Val >> 2));
    break;
  case R_HEX_B22_PCREL_X:
    or32le(Loc, applyMask(0x1ff3ffe, Val & 0x3f));
    break;
  case R_HEX_B32_PCREL_X:
    or32le(Loc, applyMask(0x0fff3fff, Val >> 6));
    break;
  case R_HEX_HI16:
    or32le(Loc, applyMask(0x00c03fff, Val >> 16));
    break;
  case R_HEX_LO16:
    or32le(Loc, applyMask(0x00c03fff, Val));
    break;
  default:
    error(getErrorLocation(Loc) + "unrecognized reloc " + toString(Type));
    break;
  }
}

TargetInfo *elf::getHexagonTargetInfo() {
  static Hexagon Target;
  return &Target;
}
