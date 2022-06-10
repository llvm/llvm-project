//===- bolt/Core/Relocation.cpp - Object file relocations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Relocation class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Relocation.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ELF.h"

using namespace llvm;
using namespace bolt;

Triple::ArchType Relocation::Arch;

namespace {

bool isSupportedX86(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_X86_64_8:
  case ELF::R_X86_64_16:
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S:
  case ELF::R_X86_64_64:
  case ELF::R_X86_64_PC8:
  case ELF::R_X86_64_PC32:
  case ELF::R_X86_64_PC64:
  case ELF::R_X86_64_PLT32:
  case ELF::R_X86_64_GOTPCREL:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_TPOFF32:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
    return true;
  }
}

bool isSupportedAArch64(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_CALL26:
  case ELF::R_AARCH64_JUMP26:
  case ELF::R_AARCH64_TSTBR14:
  case ELF::R_AARCH64_CONDBR19:
  case ELF::R_AARCH64_ADR_PREL_LO21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC:
  case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
  case ELF::R_AARCH64_ADD_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_TLSDESC_CALL:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_PREL16:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_PREL64:
  case ELF::R_AARCH64_ABS16:
  case ELF::R_AARCH64_ABS32:
  case ELF::R_AARCH64_ABS64:
  case ELF::R_AARCH64_MOVW_UABS_G0:
  case ELF::R_AARCH64_MOVW_UABS_G0_NC:
  case ELF::R_AARCH64_MOVW_UABS_G1:
  case ELF::R_AARCH64_MOVW_UABS_G1_NC:
  case ELF::R_AARCH64_MOVW_UABS_G2:
  case ELF::R_AARCH64_MOVW_UABS_G2_NC:
  case ELF::R_AARCH64_MOVW_UABS_G3:
    return true;
  }
}

size_t getSizeForTypeX86(uint64_t Type) {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_X86_64, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_X86_64_8:
  case ELF::R_X86_64_PC8:
    return 1;
  case ELF::R_X86_64_16:
    return 2;
  case ELF::R_X86_64_PLT32:
  case ELF::R_X86_64_PC32:
  case ELF::R_X86_64_32S:
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_GOTPCREL:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_TPOFF32:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
    return 4;
  case ELF::R_X86_64_PC64:
  case ELF::R_X86_64_64:
    return 8;
  }
}

size_t getSizeForTypeAArch64(uint64_t Type) {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_AARCH64, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_AARCH64_ABS16:
  case ELF::R_AARCH64_PREL16:
    return 2;
  case ELF::R_AARCH64_CALL26:
  case ELF::R_AARCH64_JUMP26:
  case ELF::R_AARCH64_TSTBR14:
  case ELF::R_AARCH64_CONDBR19:
  case ELF::R_AARCH64_ADR_PREL_LO21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC:
  case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
  case ELF::R_AARCH64_ADD_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_TLSDESC_CALL:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_MOVW_UABS_G0:
  case ELF::R_AARCH64_MOVW_UABS_G0_NC:
  case ELF::R_AARCH64_MOVW_UABS_G1:
  case ELF::R_AARCH64_MOVW_UABS_G1_NC:
  case ELF::R_AARCH64_MOVW_UABS_G2:
  case ELF::R_AARCH64_MOVW_UABS_G2_NC:
  case ELF::R_AARCH64_MOVW_UABS_G3:
  case ELF::R_AARCH64_ABS32:
    return 4;
  case ELF::R_AARCH64_ABS64:
  case ELF::R_AARCH64_PREL64:
    return 8;
  }
}

bool skipRelocationProcessX86(uint64_t Type, uint64_t Contents) {
  return false;
}

bool skipRelocationProcessAArch64(uint64_t Type, uint64_t Contents) {
  auto IsMov = [](uint64_t Contents) -> bool {
    // The bits 28-23 are 0b100101
    return (Contents & 0x1f800000) == 0x12800000;
  };

  auto IsB = [](uint64_t Contents) -> bool {
    // The bits 31-26 are 0b000101
    return (Contents & 0xfc000000) == 0x14000000;
  };

  auto IsAdr = [](uint64_t Contents) -> bool {
    // The bits 31-24 are 0b0xx10000
    return (Contents & 0x9f000000) == 0x10000000;
  };

  auto IsNop = [](uint64_t Contents) -> bool { return Contents == 0xd503201f; };

  // The linker might eliminate the instruction and replace it with NOP, ignore
  if (IsNop(Contents))
    return true;

  // The linker might perform TLS relocations relaxations, such as
  // changed TLS access model (e.g. changed global dynamic model
  // to initial exec), thus changing the instructions. The static
  // relocations might be invalid at this point and we might no
  // need to proccess these relocations anymore.
  // More information could be found by searching
  // elfNN_aarch64_tls_relax in bfd
  switch (Type) {
  default:
    break;
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21: {
    if (IsMov(Contents))
      return true;
  }
  }

  // The linker might replace load/store instruction with jump and
  // veneer due to errata 843419
  // https://documentation-service.arm.com/static/5fa29fddb209f547eebd361d
  // Thus load/store relocations for these instructions must be ignored
  // NOTE: We only process GOT and TLS relocations this way since the
  // addend used in load/store instructions won't change after bolt
  // (it is important since the instruction in veneer won't have relocation)
  switch (Type) {
  default:
    break;
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12: {
    if (IsB(Contents))
      return true;
  }
  }

  // The linker might relax ADRP+ADD or ADRP+LDR sequences to the ADR+NOP
  switch (Type) {
  default:
    break;
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADD_ABS_LO12_NC:
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
    if (IsAdr(Contents))
      return true;
  }

  return false;
}

uint64_t adjustValueX86(uint64_t Type, uint64_t Value, uint64_t PC) {
  switch (Type) {
  default:
    llvm_unreachable("not supported relocation");
  case ELF::R_X86_64_32:
    break;
  case ELF::R_X86_64_PC32:
    Value -= PC;
    break;
  }
  return Value;
}

uint64_t adjustValueAArch64(uint64_t Type, uint64_t Value, uint64_t PC) {
  switch (Type) {
  default:
    llvm_unreachable("not supported relocation");
  case ELF::R_AARCH64_ABS32:
    break;
  case ELF::R_AARCH64_PREL16:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_PREL64:
    Value -= PC;
    break;
  }
  return Value;
}

uint64_t extractValueX86(uint64_t Type, uint64_t Contents, uint64_t PC) {
  if (Type == ELF::R_X86_64_32S)
    return SignExtend64<32>(Contents);
  if (Relocation::isPCRelative(Type))
    return SignExtend64(Contents, 8 * Relocation::getSizeForType(Type));
  return Contents;
}

uint64_t extractValueAArch64(uint64_t Type, uint64_t Contents, uint64_t PC) {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_AARCH64, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_AARCH64_ABS16:
  case ELF::R_AARCH64_ABS32:
  case ELF::R_AARCH64_ABS64:
    return Contents;
  case ELF::R_AARCH64_PREL16:
    return static_cast<int64_t>(PC) + SignExtend64<16>(Contents & 0xffff);
  case ELF::R_AARCH64_PREL32:
    return static_cast<int64_t>(PC) + SignExtend64<32>(Contents & 0xffffffff);
  case ELF::R_AARCH64_PREL64:
    return static_cast<int64_t>(PC) + Contents;
  case ELF::R_AARCH64_TLSDESC_CALL:
  case ELF::R_AARCH64_JUMP26:
  case ELF::R_AARCH64_CALL26:
    // Immediate goes in bits 25:0 of B and BL.
    Contents &= ~0xfffffffffc000000ULL;
    return static_cast<int64_t>(PC) + SignExtend64<28>(Contents << 2);
  case ELF::R_AARCH64_TSTBR14:
    // Immediate:15:2 goes in bits 18:5 of TBZ, TBNZ
    Contents &= ~0xfffffffffff8001fULL;
    return static_cast<int64_t>(PC) + SignExtend64<16>(Contents >> 3);
  case ELF::R_AARCH64_CONDBR19:
    // Immediate:20:2 goes in bits 23:5 of Bcc, CBZ, CBNZ
    Contents &= ~0xffffffffff00001fULL;
    return static_cast<int64_t>(PC) + SignExtend64<21>(Contents >> 3);
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_ADR_PREL_LO21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC: {
    // Bits 32:12 of Symbol address goes in bits 30:29 + 23:5 of ADRP
    // and ADR instructions
    bool IsAdr = !!(((Contents >> 31) & 0x1) == 0);
    Contents &= ~0xffffffff9f00001fUll;
    uint64_t LowBits = (Contents >> 29) & 0x3;
    uint64_t HighBits = (Contents >> 5) & 0x7ffff;
    Contents = LowBits | (HighBits << 2);
    if (IsAdr)
      return static_cast<int64_t>(PC) + SignExtend64<21>(Contents);

    // ADRP instruction
    Contents = static_cast<int64_t>(PC) + SignExtend64<33>(Contents << 12);
    Contents &= ~0xfffUll;
    return Contents;
  }
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_LDST64_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of LD/ST instruction, taken
    // from bits 11:3 of Symbol address
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 3);
  }
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_ADD_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of ADD instruction
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 0);
  }
  case ELF::R_AARCH64_LDST128_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of ADD instruction, taken
    // from bits 11:4 of Symbol address
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 4);
  }
  case ELF::R_AARCH64_LDST32_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of ADD instruction, taken
    // from bits 11:2 of Symbol address
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 2);
  }
  case ELF::R_AARCH64_LDST16_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of ADD instruction, taken
    // from bits 11:1 of Symbol address
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 1);
  }
  case ELF::R_AARCH64_LDST8_ABS_LO12_NC: {
    // Immediate goes in bits 21:10 of ADD instruction, taken
    // from bits 11:0 of Symbol address
    Contents &= ~0xffffffffffc003ffU;
    return Contents >> (10 - 0);
  }
  case ELF::R_AARCH64_MOVW_UABS_G3:
  case ELF::R_AARCH64_MOVW_UABS_G2_NC:
  case ELF::R_AARCH64_MOVW_UABS_G2:
  case ELF::R_AARCH64_MOVW_UABS_G1_NC:
  case ELF::R_AARCH64_MOVW_UABS_G1:
  case ELF::R_AARCH64_MOVW_UABS_G0_NC:
  case ELF::R_AARCH64_MOVW_UABS_G0:
    // The shift goest in bits 22:21 of MOV* instructions
    uint8_t Shift = (Contents >> 21) & 0x3;
    // Immediate goes in bits 20:5
    Contents = (Contents >> 5) & 0xffff;
    return Contents << (16 * Shift);
  }
}

bool isGOTX86(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_X86_64_GOT32:
  case ELF::R_X86_64_GOTPCREL:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_GOTOFF64:
  case ELF::R_X86_64_GOTPC32:
  case ELF::R_X86_64_GOT64:
  case ELF::R_X86_64_GOTPCREL64:
  case ELF::R_X86_64_GOTPC64:
  case ELF::R_X86_64_GOTPLT64:
  case ELF::R_X86_64_GOTPC32_TLSDESC:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
    return true;
  }
}

bool isGOTAArch64(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_TLSDESC_CALL:
    return true;
  }
}

bool isTLSX86(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_X86_64_TPOFF32:
  case ELF::R_X86_64_TPOFF64:
  case ELF::R_X86_64_GOTTPOFF:
    return true;
  }
}

bool isTLSAArch64(uint64_t Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_TLSDESC_CALL:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    return true;
  }
}

bool isPCRelativeX86(uint64_t Type) {
  switch (Type) {
  default:
    llvm_unreachable("Unknown relocation type");
  case ELF::R_X86_64_64:
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S:
  case ELF::R_X86_64_16:
  case ELF::R_X86_64_8:
  case ELF::R_X86_64_TPOFF32:
    return false;
  case ELF::R_X86_64_PC8:
  case ELF::R_X86_64_PC32:
  case ELF::R_X86_64_PC64:
  case ELF::R_X86_64_GOTPCREL:
  case ELF::R_X86_64_PLT32:
  case ELF::R_X86_64_GOTOFF64:
  case ELF::R_X86_64_GOTPC32:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
    return true;
  }
}

bool isPCRelativeAArch64(uint64_t Type) {
  switch (Type) {
  default:
    llvm_unreachable("Unknown relocation type");
  case ELF::R_AARCH64_ABS16:
  case ELF::R_AARCH64_ABS32:
  case ELF::R_AARCH64_ABS64:
  case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
  case ELF::R_AARCH64_ADD_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
  case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_MOVW_UABS_G0:
  case ELF::R_AARCH64_MOVW_UABS_G0_NC:
  case ELF::R_AARCH64_MOVW_UABS_G1:
  case ELF::R_AARCH64_MOVW_UABS_G1_NC:
  case ELF::R_AARCH64_MOVW_UABS_G2:
  case ELF::R_AARCH64_MOVW_UABS_G2_NC:
  case ELF::R_AARCH64_MOVW_UABS_G3:
    return false;
  case ELF::R_AARCH64_TLSDESC_CALL:
  case ELF::R_AARCH64_CALL26:
  case ELF::R_AARCH64_JUMP26:
  case ELF::R_AARCH64_TSTBR14:
  case ELF::R_AARCH64_CONDBR19:
  case ELF::R_AARCH64_ADR_PREL_LO21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC:
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_PREL16:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_PREL64:
    return true;
  }
}

} // end anonymous namespace

bool Relocation::isSupported(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return isSupportedAArch64(Type);
  return isSupportedX86(Type);
}

size_t Relocation::getSizeForType(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return getSizeForTypeAArch64(Type);
  return getSizeForTypeX86(Type);
}

bool Relocation::skipRelocationProcess(uint64_t Type, uint64_t Contents) {
  if (Arch == Triple::aarch64)
    return skipRelocationProcessAArch64(Type, Contents);
  return skipRelocationProcessX86(Type, Contents);
}

uint64_t Relocation::adjustValue(uint64_t Type, uint64_t Value,
                                 uint64_t PC) {
  if (Arch == Triple::aarch64)
    return adjustValueAArch64(Type, Value, PC);
  return adjustValueX86(Type, Value, PC);
}

uint64_t Relocation::extractValue(uint64_t Type, uint64_t Contents,
                                  uint64_t PC) {
  if (Arch == Triple::aarch64)
    return extractValueAArch64(Type, Contents, PC);
  return extractValueX86(Type, Contents, PC);
}

bool Relocation::isGOT(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return isGOTAArch64(Type);
  return isGOTX86(Type);
}

bool Relocation::isX86GOTPCRELX(uint64_t Type) {
  if (Arch != Triple::x86_64)
    return false;
  return Type == ELF::R_X86_64_GOTPCRELX || Type == ELF::R_X86_64_REX_GOTPCRELX;
}

bool Relocation::isNone(uint64_t Type) { return Type == getNone(); }

bool Relocation::isRelative(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return Type == ELF::R_AARCH64_RELATIVE;
  return Type == ELF::R_X86_64_RELATIVE;
}

bool Relocation::isIRelative(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return Type == ELF::R_AARCH64_IRELATIVE;
  return Type == ELF::R_X86_64_IRELATIVE;
}

bool Relocation::isTLS(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return isTLSAArch64(Type);
  return isTLSX86(Type);
}

uint64_t Relocation::getNone() {
  if (Arch == Triple::aarch64)
    return ELF::R_AARCH64_NONE;
  return ELF::R_X86_64_NONE;
}

uint64_t Relocation::getPC32() {
  if (Arch == Triple::aarch64)
    return ELF::R_AARCH64_PREL32;
  return ELF::R_X86_64_PC32;
}

uint64_t Relocation::getPC64() {
  if (Arch == Triple::aarch64)
    return ELF::R_AARCH64_PREL64;
  return ELF::R_X86_64_PC64;
}

bool Relocation::isPCRelative(uint64_t Type) {
  if (Arch == Triple::aarch64)
    return isPCRelativeAArch64(Type);
  return isPCRelativeX86(Type);
}

size_t Relocation::emit(MCStreamer *Streamer) const {
  const size_t Size = getSizeForType(Type);
  MCContext &Ctx = Streamer->getContext();
  if (isPCRelative(Type)) {
    MCSymbol *TempLabel = Ctx.createNamedTempSymbol();
    Streamer->emitLabel(TempLabel);
    const MCExpr *Value = nullptr;
    if (Symbol) {
      Value = MCSymbolRefExpr::create(Symbol, Ctx);
      if (Addend) {
        Value = MCBinaryExpr::createAdd(
            Value, MCConstantExpr::create(Addend, Ctx), Ctx);
      }
    } else {
      Value = MCConstantExpr::create(Addend, Ctx);
    }
    Value = MCBinaryExpr::createSub(
        Value, MCSymbolRefExpr::create(TempLabel, Ctx), Ctx);
    Streamer->emitValue(Value, Size);

    return Size;
  }

  if (Symbol && Addend) {
    auto Value =
        MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Symbol, Ctx),
                                MCConstantExpr::create(Addend, Ctx), Ctx);
    Streamer->emitValue(Value, Size);
  } else if (Symbol) {
    Streamer->emitSymbolValue(Symbol, Size);
  } else {
    Streamer->emitIntValue(Addend, Size);
  }

  return Size;
}

#define ELF_RELOC(name, value) #name,

void Relocation::print(raw_ostream &OS) const {
  static const char *X86RelocNames[] = {
#include "llvm/BinaryFormat/ELFRelocs/x86_64.def"
  };
  static const char *AArch64RelocNames[] = {
#include "llvm/BinaryFormat/ELFRelocs/AArch64.def"
  };
  if (Arch == Triple::aarch64)
    OS << AArch64RelocNames[Type];
  else
    OS << X86RelocNames[Type];
  OS << ", 0x" << Twine::utohexstr(Offset);
  if (Symbol) {
    OS << ", " << Symbol->getName();
  }
  if (int64_t(Addend) < 0)
    OS << ", -0x" << Twine::utohexstr(-int64_t(Addend));
  else
    OS << ", 0x" << Twine::utohexstr(Addend);
  OS << ", 0x" << Twine::utohexstr(Value);
}
