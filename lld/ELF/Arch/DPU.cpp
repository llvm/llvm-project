//===- DPU.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Relocations.h"
#include "Target.h"
#include <Symbols.h>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

#define DEBUG_TYPE "dpu-ld"

namespace {

class DPU final : public TargetInfo {
public:
  DPU();
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const override;
  uint64_t fixupTargetVA(uint64_t TargetVA) const override;
  uint32_t calcEFlags() const override;
};

} // end anonymous namespace

DPU::DPU() {}

RelExpr DPU::getRelExpr(const RelType Type, const Symbol &S,
                        const uint8_t *Loc) const {
  switch (Type) {
  default:
    error(getErrorLocation(Loc) + "unrecognized reloc " + Twine(Type));
    return R_NONE;
  case R_DPU_NONE:
    return R_NONE;
  case R_DPU_8:
  case R_DPU_16:
  case R_DPU_32:
  case R_DPU_64:
  case R_DPU_PC:
  case R_DPU_IMM5:
  case R_DPU_IMM8_DMA:
  case R_DPU_IMM24_PC:
  case R_DPU_IMM27_PC:
  case R_DPU_IMM28_PC_OPC8:
  case R_DPU_IMM8_STR:
  case R_DPU_IMM12_STR:
  case R_DPU_IMM16_STR:
  case R_DPU_IMM16_ATM:
  case R_DPU_IMM24:
  case R_DPU_IMM24_RB:
  case R_DPU_IMM27:
  case R_DPU_IMM28:
  case R_DPU_IMM32:
  case R_DPU_IMM32_ZERO_RB:
  case R_DPU_IMM17_24:
  case R_DPU_IMM32_DUS_RB:
    return R_ABS;
  }
}

#define ATOMIC_SECTION_OFFSET (0xF0000000ULL)
#define IRAM_SECTION_OFFSET (0x80000000ULL)
#define MRAM_SECTION_OFFSET (0x08000000ULL)
uint64_t DPU::fixupTargetVA(uint64_t TargetVA) const {
  // BE CAREFUL: this code is based on the assertions defined by the DPU linker
  // script:
  //  - IRAM: memory mapped at virtual address 0x8XXXXXXX
  //  - MRAM: memory mapped at virtual address 0x08XXXXXX
  //  - WRAM: no address remap
  // The LSBits may be the addition of an instruction offset to the address,
  // which must be converted to an absolute address.
  // TODO: fix pointer length everywhere. For example:
  //  __sys_bootstrap:
  //  release zero, 0x5, nz, . + 1
  // In obj:
  //  Ltmp0 + 1
  // Ltmp0 becomes 0x8xxxxxx1 at this level.

  // Make sure MSBs are clean.
  TargetVA &= 0xFFFFFFFF;

  if ((TargetVA & ATOMIC_SECTION_OFFSET) == ATOMIC_SECTION_OFFSET) {
    return TargetVA & ~ATOMIC_SECTION_OFFSET;
  } else if (TargetVA & IRAM_SECTION_OFFSET) {
    const uint32_t shift = 3;
    TargetVA &= ~IRAM_SECTION_OFFSET;
    return (TargetVA >> shift) + (TargetVA & ((1ULL << shift) - 1ULL));
  } else if (TargetVA & MRAM_SECTION_OFFSET) {
    return (TargetVA & ~MRAM_SECTION_OFFSET);
  } else {
    return TargetVA;
  }
}

void DPU::relocate(uint8_t *loc, const Relocation &rel, const uint64_t val) const {
  const endianness E = support::little;
  uint64_t Data = read64<E>(loc);
  RelType Type = rel.type;

  switch (Type) {
  case R_DPU_8:
    *loc = (uint8_t)val;
    return;
  case R_DPU_32:
    write32<E>(loc, (uint32_t)val);
    return;
  case R_DPU_64:
    write64<E>(loc, val);
    return;
  case R_DPU_16:
  case R_DPU_PC:
    write16<E>(loc, (uint16_t)val);
    return;
  case R_DPU_IMM8_DMA:
    *(loc + 3) = (uint8_t)val;
    return;
  case R_DPU_IMM5:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0x1l) << 28);
    break;
  case R_DPU_IMM24_PC:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16);
    break;
  case R_DPU_IMM27_PC:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x7l) << 39);
    break;
  case R_DPU_IMM28_PC_OPC8:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x7l) << 39) | (((val >> 11) & 0x1l) << 44);
    break;
  case R_DPU_IMM8_STR:
    Data |= (((val >> 0) & 0xfl) << 16) | (((val >> 4) & 0xfl) << 0);
    break;
  case R_DPU_IMM12_STR:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0x7l) << 39) |
            (((val >> 7) & 0x1l) << 24) | (((val >> 8) & 0x1l) << 15) |
            (((val >> 9) & 0x1l) << 14) | (((val >> 10) & 0x1l) << 13) |
            (((val >> 11) & 0x1l) << 12);
    break;
  case R_DPU_IMM16_STR:
    Data |= (((val >> 0) & 0xfl) << 16) | (((val >> 4) & 0xfffl) << 0);
    break;
  case R_DPU_IMM16_ATM:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1fl) << 26) | (((val >> 13) & 0x7l) << 39);
    break;
  case R_DPU_IMM24:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1l) << 15) | (((val >> 9) & 0x1l) << 14) |
            (((val >> 10) & 0x1l) << 13) | (((val >> 11) & 0x1l) << 12) |
            (((val >> 12) & 0xfffl) << 0);
    break;
  case R_DPU_IMM24_RB:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0x7l) << 39) |
            (((val >> 7) & 0x1l) << 24) | (((val >> 8) & 0x1l) << 15) |
            (((val >> 9) & 0x1l) << 14) | (((val >> 10) & 0x1l) << 13) |
            (((val >> 11) & 0x1l) << 12) | (((val >> 12) & 0xfffl) << 0);
    break;
  case R_DPU_IMM27:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1l) << 15) | (((val >> 9) & 0x1l) << 14) |
            (((val >> 10) & 0x1l) << 13) | (((val >> 11) & 0x1l) << 12) |
            (((val >> 12) & 0xfffl) << 0) | (((val >> 24) & 0x7l) << 39);
    break;
  case R_DPU_IMM28:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1l) << 15) | (((val >> 9) & 0x1l) << 14) |
            (((val >> 10) & 0x1l) << 13) | (((val >> 11) & 0x1l) << 12) |
            (((val >> 12) & 0xfffl) << 0) | (((val >> 24) & 0x7l) << 39) |
            (((val >> 27) & 0x1l) << 44);
    break;
  case R_DPU_IMM32:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1l) << 15) | (((val >> 9) & 0x1l) << 14) |
            (((val >> 10) & 0x1l) << 13) | (((val >> 11) & 0x1l) << 12) |
            (((val >> 12) & 0xfffl) << 0) | (((val >> 24) & 0xffl) << 24);
    break;
  case R_DPU_IMM32_ZERO_RB:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0x7l) << 34) |
            (((val >> 7) & 0x1l) << 39) | (((val >> 8) & 0x1l) << 15) |
            (((val >> 9) & 0x1l) << 14) | (((val >> 10) & 0x1l) << 13) |
            (((val >> 11) & 0x1l) << 12) | (((val >> 12) & 0xfffl) << 0) |
            (((val >> 24) & 0xffl) << 24);
    break;
  case R_DPU_IMM17_24:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0xfl) << 16) |
            (((val >> 8) & 0x1l) << 15) | (((val >> 9) & 0x1l) << 14) |
            (((val >> 10) & 0x1l) << 13) | (((val >> 11) & 0x1l) << 12) |
            (((val >> 12) & 0x1fl) << 0) | (((val >> 16) & 0x1l) << 5) |
            (((val >> 16) & 0x1l) << 6) | (((val >> 16) & 0x1l) << 7) |
            (((val >> 16) & 0x1l) << 8) | (((val >> 16) & 0x1l) << 9) |
            (((val >> 16) & 0x1l) << 10) | (((val >> 16) & 0x1l) << 11);
    break;
  case R_DPU_IMM32_DUS_RB:
    Data |= (((val >> 0) & 0xfl) << 20) | (((val >> 4) & 0x7l) << 34) |
            (((val >> 7) & 0x1l) << 44) | (((val >> 8) & 0x1l) << 15) |
            (((val >> 9) & 0x1l) << 14) | (((val >> 10) & 0x1l) << 13) |
            (((val >> 11) & 0x1l) << 12) | (((val >> 12) & 0xfffl) << 0) |
            (((val >> 24) & 0xffl) << 24);
    break;
  default:
    error(getErrorLocation(loc) + "unrecognized reloc " + Twine(Type));
  }
  write64<E>(loc, Data);
}

#define UNKNOWN_E_FLAGS (0xffffffff)
static uint32_t getEFlags(InputFile *File) {
  return cast<ObjFile<ELF32LE>>(File)->getObj().getHeader().e_flags;
}

static uint32_t getEABI(uint32_t e_flags) {
  if (!(e_flags & llvm::ELF::EF_DPU_EABI_SET))
    return ~llvm::ELF::EF_DPU_EABI_SET;
  else
    return EF_EABI_DPU_GET(e_flags);
}

static uint32_t getEABI(InputFile *File) { return getEABI(getEFlags(File)); }

uint32_t DPU::calcEFlags() const {
  uint32_t e_flags = UNKNOWN_E_FLAGS;
  InputFile *first_file = NULL;
  for (InputFile *F : objectFiles) {
    if (e_flags == UNKNOWN_E_FLAGS) {
      e_flags = getEFlags(F);
      first_file = F;
    } else if (getEABI(e_flags) != getEABI(F)) {
      error("uncompatible abi between '" + toString(first_file) + "' and '" +
            toString(F) + "' (" + std::to_string(getEABI(e_flags)) +
            " != " + std::to_string(getEABI(F)) + ")");
    }
  }
  return e_flags;
}

TargetInfo *elf::getDPUTargetInfo() {
  static DPU Target;
  return &Target;
}
