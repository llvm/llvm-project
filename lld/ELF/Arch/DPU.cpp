//===- DPU.cpp ----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
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
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
  uint64_t fixupTargetVA(uint64_t TargetVA) const override;
};

} // end anonymous namespace

DPU::DPU() {}

RelExpr DPU::getRelExpr(const RelType Type, const Symbol &S,
                        const uint8_t *Loc) const {
  switch (Type) {
  default:
    error(getErrorLocation(Loc) + "unrecognized reloc " + Twine(Type));
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
  if (TargetVA & 0x80000000) {
    uint64_t optional_add = TargetVA & (8 - 1);
    // Mask with ~0xF0000000 because the shift may have introduced ones.
    return ((TargetVA >> 3) & ~0xF0000000) + optional_add;
  } else if (TargetVA & 0x08000000) {
    return (TargetVA & ~0x08000000);
  } else {
    return TargetVA;
  }
}

void DPU::relocateOne(uint8_t *Loc, const RelType Type,
                      const uint64_t Val) const {
  const endianness E = support::little;

  uint64_t Data = read64<E>(Loc);
  switch (Type) {
  case R_DPU_8:
    *Loc = (uint8_t)Val;
    return;
  case R_DPU_32:
    write32<E>(Loc, (uint32_t)Val);
    return;
  case R_DPU_64:
    write64<E>(Loc, Val);
    return;
  case R_DPU_16:
  case R_DPU_PC:
    write16<E>(Loc, (uint16_t)Val);
    return;
  case R_DPU_IMM8_DMA:
    *(Loc + 3) = (uint8_t)Val;
    return;
  case R_DPU_IMM5:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0x1l) << 28);
    break;
  case R_DPU_IMM24_PC:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16);
    break;
  case R_DPU_IMM27_PC:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x7l) << 39);
    break;
  case R_DPU_IMM28_PC_OPC8:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x7l) << 39) | (((Val >> 11) & 0x1l) << 44);
    break;
  case R_DPU_IMM8_STR:
    Data |= (((Val >> 0) & 0xfl) << 16) | (((Val >> 4) & 0xfl) << 0);
    break;
  case R_DPU_IMM12_STR:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0x7l) << 39) |
            (((Val >> 7) & 0x1l) << 24) | (((Val >> 8) & 0x1l) << 15) |
            (((Val >> 9) & 0x1l) << 14) | (((Val >> 10) & 0x1l) << 13) |
            (((Val >> 11) & 0x1l) << 12);
    break;
  case R_DPU_IMM16_STR:
    Data |= (((Val >> 0) & 0xfl) << 16) | (((Val >> 4) & 0xfffl) << 0);
    break;
  case R_DPU_IMM16_ATM:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1fl) << 26) | (((Val >> 13) & 0x7l) << 39);
    break;
  case R_DPU_IMM24:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1l) << 15) | (((Val >> 9) & 0x1l) << 14) |
            (((Val >> 10) & 0x1l) << 13) | (((Val >> 11) & 0x1l) << 12) |
            (((Val >> 12) & 0xfffl) << 0);
    break;
  case R_DPU_IMM24_RB:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0x7l) << 39) |
            (((Val >> 7) & 0x1l) << 24) | (((Val >> 8) & 0x1l) << 15) |
            (((Val >> 9) & 0x1l) << 14) | (((Val >> 10) & 0x1l) << 13) |
            (((Val >> 11) & 0x1l) << 12) | (((Val >> 12) & 0xfffl) << 0);
    break;
  case R_DPU_IMM27:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1l) << 15) | (((Val >> 9) & 0x1l) << 14) |
            (((Val >> 10) & 0x1l) << 13) | (((Val >> 11) & 0x1l) << 12) |
            (((Val >> 12) & 0xfffl) << 0) | (((Val >> 24) & 0x7l) << 39);
    break;
  case R_DPU_IMM28:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1l) << 15) | (((Val >> 9) & 0x1l) << 14) |
            (((Val >> 10) & 0x1l) << 13) | (((Val >> 11) & 0x1l) << 12) |
            (((Val >> 12) & 0xfffl) << 0) | (((Val >> 24) & 0x7l) << 39) |
            (((Val >> 27) & 0x1l) << 44);
    break;
  case R_DPU_IMM32:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1l) << 15) | (((Val >> 9) & 0x1l) << 14) |
            (((Val >> 10) & 0x1l) << 13) | (((Val >> 11) & 0x1l) << 12) |
            (((Val >> 12) & 0xfffl) << 0) | (((Val >> 24) & 0xffl) << 24);
    break;
  case R_DPU_IMM32_ZERO_RB:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0x7l) << 34) |
            (((Val >> 7) & 0x1l) << 39) | (((Val >> 8) & 0x1l) << 15) |
            (((Val >> 9) & 0x1l) << 14) | (((Val >> 10) & 0x1l) << 13) |
            (((Val >> 11) & 0x1l) << 12) | (((Val >> 12) & 0xfffl) << 0) |
            (((Val >> 24) & 0xffl) << 24);
    break;
  case R_DPU_IMM17_24:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0xfl) << 16) |
            (((Val >> 8) & 0x1l) << 15) | (((Val >> 9) & 0x1l) << 14) |
            (((Val >> 10) & 0x1l) << 13) | (((Val >> 11) & 0x1l) << 12) |
            (((Val >> 12) & 0x1fl) << 0) | (((Val >> 16) & 0x1l) << 5) |
            (((Val >> 16) & 0x1l) << 6) | (((Val >> 16) & 0x1l) << 7) |
            (((Val >> 16) & 0x1l) << 8) | (((Val >> 16) & 0x1l) << 9) |
            (((Val >> 16) & 0x1l) << 10) | (((Val >> 16) & 0x1l) << 11);
    break;
  case R_DPU_IMM32_DUS_RB:
    Data |= (((Val >> 0) & 0xfl) << 20) | (((Val >> 4) & 0x7l) << 34) |
            (((Val >> 7) & 0x1l) << 44) | (((Val >> 8) & 0x1l) << 15) |
            (((Val >> 9) & 0x1l) << 14) | (((Val >> 10) & 0x1l) << 13) |
            (((Val >> 11) & 0x1l) << 12) | (((Val >> 12) & 0xfffl) << 0) |
            (((Val >> 24) & 0xffl) << 24);
    break;
  default:
    error(getErrorLocation(Loc) + "unrecognized reloc " + Twine(Type));
  }
  write64<E>(Loc, Data);
}

TargetInfo *elf::getDPUTargetInfo() {
  static DPU Target;
  return &Target;
}
