//===- AArch64RelocationHandler.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 relocation handler.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Relocation.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace bolt;

namespace {

class AArch64RelocationHandler final : public RelocationHandler {
public:
  bool isSupported(uint32_t Type) const override;
  size_t getSizeForType(uint32_t Type) const override;
  bool skipRelocationType(uint32_t Type) const override;

  uint64_t encodeValue(uint32_t Type, uint64_t Value,
                       uint64_t PC) const override;
  bool canEncodeValue(uint32_t Type, uint64_t Value,
                      uint64_t PC) const override;
  uint64_t extractValue(uint32_t Type, uint64_t Contents,
                        uint64_t PC) const override;

  bool isGOT(uint32_t Type) const override;
  bool isRelative(uint32_t Type) const override;
  bool isIRelative(uint32_t Type) const override;
  bool isTLS(uint32_t Type) const override;
  bool isPCRelative(uint32_t Type) const override;

  uint32_t getNone() const override;
  uint32_t getPC32() const override;
  uint32_t getPC64() const override;
  uint32_t getAbs64() const override;
  uint32_t getRelative() const override;

  void printType(raw_ostream &OS, uint32_t Type) const override;
};

} // namespace

bool AArch64RelocationHandler::isRelative(uint32_t Type) const {
  return Type == ELF::R_AARCH64_RELATIVE;
}

bool AArch64RelocationHandler::isIRelative(uint32_t Type) const {
  return Type == ELF::R_AARCH64_IRELATIVE;
}

uint32_t AArch64RelocationHandler::getNone() const {
  return ELF::R_AARCH64_NONE;
}

uint32_t AArch64RelocationHandler::getPC32() const {
  return ELF::R_AARCH64_PREL32;
}

uint32_t AArch64RelocationHandler::getPC64() const {
  return ELF::R_AARCH64_PREL64;
}

uint32_t AArch64RelocationHandler::getAbs64() const {
  return ELF::R_AARCH64_ABS64;
}

uint32_t AArch64RelocationHandler::getRelative() const {
  return ELF::R_AARCH64_RELATIVE;
}

void AArch64RelocationHandler::printType(raw_ostream &OS, uint32_t Type) const {
  OS << object::getELFRelocationTypeName(ELF::EM_AARCH64, Type);
}

bool AArch64RelocationHandler::isSupported(uint32_t Type) const {
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
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
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
  case ELF::R_AARCH64_PLT32:
    return true;
  }
}

size_t AArch64RelocationHandler::getSizeForType(uint32_t Type) const {
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
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
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
  case ELF::R_AARCH64_PLT32:
    return 4;
  case ELF::R_AARCH64_ABS64:
  case ELF::R_AARCH64_PREL64:
    return 8;
  }
}

bool AArch64RelocationHandler::skipRelocationType(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_NONE:
  case ELF::R_AARCH64_LD_PREL_LO19:
  case ELF::R_AARCH64_TLSDESC_CALL:
    return true;
  }
}

bool AArch64RelocationHandler::canEncodeValue(uint32_t Type, uint64_t Value,
                                              uint64_t PC) const {
  switch (Type) {
  default:
    llvm_unreachable("unsupported relocation");
  case ELF::R_AARCH64_CALL26:
  case ELF::R_AARCH64_JUMP26:
    return isInt<28>(Value - PC);
  }
}

uint64_t AArch64RelocationHandler::encodeValue(uint32_t Type, uint64_t Value,
                                               uint64_t PC) const {
  switch (Type) {
  default:
    llvm_unreachable("unsupported relocation");
  case ELF::R_AARCH64_ABS16:
  case ELF::R_AARCH64_ABS32:
  case ELF::R_AARCH64_ABS64:
    break;
  case ELF::R_AARCH64_PREL16:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_PREL64:
    Value -= PC;
    break;
  case ELF::R_AARCH64_CALL26:
    Value -= PC;
    assert(isInt<28>(Value) && "only PC +/- 128MB is allowed for direct call");
    // Immediate goes in bits 25:0 of BL.
    // OP 1001_01 goes in bits 31:26 of BL.
    Value = ((Value >> 2) & 0x3ffffff) | 0x94000000ULL;
    break;
  case ELF::R_AARCH64_JUMP26:
    Value -= PC;
    assert(isInt<28>(Value) &&
           "only PC +/- 128MB is allowed for direct branch");
    // Immediate goes in bits 25:0 of B.
    // OP 0001_01 goes in bits 31:26 of B.
    Value = ((Value >> 2) & 0x3ffffff) | 0x14000000ULL;
    break;
  }
  return Value;
}

uint64_t AArch64RelocationHandler::extractValue(uint32_t Type,
                                                uint64_t Contents,
                                                uint64_t PC) const {
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
  case ELF::R_AARCH64_PLT32:
    return static_cast<int64_t>(PC) + SignExtend64<32>(Contents & 0xffffffff);
  case ELF::R_AARCH64_PREL64:
    return static_cast<int64_t>(PC) + Contents;
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
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
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
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
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
    // The shift goes in bits 22:21 of MOV* instructions
    uint8_t Shift = (Contents >> 21) & 0x3;
    // Immediate goes in bits 20:5
    Contents = (Contents >> 5) & 0xffff;
    return Contents << (16 * Shift);
  }
}

bool AArch64RelocationHandler::isGOT(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_LD64_GOT_LO12_NC:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
    return true;
  }
}

bool AArch64RelocationHandler::isTLS(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC:
  case ELF::R_AARCH64_TLSDESC_LD64_LO12:
  case ELF::R_AARCH64_TLSDESC_ADD_LO12:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    return true;
  }
}

bool AArch64RelocationHandler::isPCRelative(uint32_t Type) const {
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
  case ELF::R_AARCH64_TLSGD_ADD_LO12_NC:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0:
  case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC:
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
  case ELF::R_AARCH64_CALL26:
  case ELF::R_AARCH64_JUMP26:
  case ELF::R_AARCH64_TSTBR14:
  case ELF::R_AARCH64_CONDBR19:
  case ELF::R_AARCH64_ADR_PREL_LO21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21:
  case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC:
  case ELF::R_AARCH64_ADR_GOT_PAGE:
  case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
  case ELF::R_AARCH64_TLSGD_ADR_PAGE21:
  case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
  case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
  case ELF::R_AARCH64_PREL16:
  case ELF::R_AARCH64_PREL32:
  case ELF::R_AARCH64_PREL64:
  case ELF::R_AARCH64_PLT32:
    return true;
  }
}

namespace llvm::bolt {

std::unique_ptr<RelocationHandler> createAArch64RelocationHandler() {
  return std::make_unique<AArch64RelocationHandler>();
}

} // namespace llvm::bolt
