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
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace bolt;

namespace ELFReserved {
enum {
  R_RISCV_TPREL_I = 49,
  R_RISCV_TPREL_S = 50,
};
} // namespace ELFReserved

namespace {

class X86RelocationHandler final : public RelocationHandler {
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

class RISCVRelocationHandler final : public RelocationHandler {
  bool Is64Bit;

public:
  explicit RISCVRelocationHandler(bool Is64Bit) : Is64Bit(Is64Bit) {}

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
  bool isInstructionReference(uint32_t Type) const override;
  bool isPCRelative(uint32_t Type) const override;

  uint32_t getNone() const override;
  uint32_t getPC32() const override;
  uint32_t getPC64() const override;
  uint32_t getAbs64() const override;
  uint32_t getRelative() const override;

  MCBinaryExpr::Opcode getComposeOpcodeFor(uint32_t Type) const override;
  void printType(raw_ostream &OS, uint32_t Type) const override;
};

} // namespace

bool X86RelocationHandler::canEncodeValue(uint32_t, uint64_t, uint64_t) const {
  return true;
}

bool X86RelocationHandler::isRelative(uint32_t Type) const {
  return Type == ELF::R_X86_64_RELATIVE;
}

bool X86RelocationHandler::isIRelative(uint32_t Type) const {
  return Type == ELF::R_X86_64_IRELATIVE;
}

uint32_t X86RelocationHandler::getNone() const { return ELF::R_X86_64_NONE; }

uint32_t X86RelocationHandler::getPC32() const { return ELF::R_X86_64_PC32; }

uint32_t X86RelocationHandler::getPC64() const { return ELF::R_X86_64_PC64; }

uint32_t X86RelocationHandler::getAbs64() const { return ELF::R_X86_64_64; }

uint32_t X86RelocationHandler::getRelative() const {
  return ELF::R_X86_64_RELATIVE;
}

void X86RelocationHandler::printType(raw_ostream &OS, uint32_t Type) const {
  OS << object::getELFRelocationTypeName(ELF::EM_X86_64, Type);
}

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

bool RISCVRelocationHandler::isRelative(uint32_t Type) const {
  return Type == ELF::R_RISCV_RELATIVE;
}

bool RISCVRelocationHandler::isIRelative(uint32_t) const {
  llvm_unreachable("not implemented");
}

bool RISCVRelocationHandler::isInstructionReference(uint32_t Type) const {
  if (!Is64Bit)
    return false;
  return Type == ELF::R_RISCV_PCREL_LO12_I || Type == ELF::R_RISCV_PCREL_LO12_S;
}

uint32_t RISCVRelocationHandler::getNone() const { return ELF::R_RISCV_NONE; }

uint32_t RISCVRelocationHandler::getPC32() const {
  return ELF::R_RISCV_32_PCREL;
}

uint32_t RISCVRelocationHandler::getPC64() const {
  llvm_unreachable("not implemented");
}

uint32_t RISCVRelocationHandler::getAbs64() const { return ELF::R_RISCV_64; }

uint32_t RISCVRelocationHandler::getRelative() const {
  llvm_unreachable("not implemented");
}

MCBinaryExpr::Opcode
RISCVRelocationHandler::getComposeOpcodeFor(uint32_t Type) const {
  switch (Type) {
  default:
    llvm_unreachable("not implemented");
  case ELF::R_RISCV_ADD32:
    return MCBinaryExpr::Add;
  case ELF::R_RISCV_SUB32:
    return MCBinaryExpr::Sub;
  }
}

void RISCVRelocationHandler::printType(raw_ostream &OS, uint32_t Type) const {
  OS << object::getELFRelocationTypeName(ELF::EM_RISCV, Type);
}

bool X86RelocationHandler::isSupported(uint32_t Type) const {
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
  case ELF::R_X86_64_GOTPC64:
  case ELF::R_X86_64_GOTPCREL:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_TPOFF32:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
    return true;
  }
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

bool RISCVRelocationHandler::isSupported(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_RISCV_JAL:
  case ELF::R_RISCV_CALL:
  case ELF::R_RISCV_CALL_PLT:
  case ELF::R_RISCV_BRANCH:
  case ELF::R_RISCV_RELAX:
  case ELF::R_RISCV_GOT_HI20:
  case ELF::R_RISCV_PCREL_HI20:
  case ELF::R_RISCV_PCREL_LO12_I:
  case ELF::R_RISCV_PCREL_LO12_S:
  case ELF::R_RISCV_RVC_JUMP:
  case ELF::R_RISCV_RVC_BRANCH:
  case ELF::R_RISCV_ADD32:
  case ELF::R_RISCV_SUB32:
  case ELF::R_RISCV_HI20:
  case ELF::R_RISCV_LO12_I:
  case ELF::R_RISCV_LO12_S:
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
  case ELF::R_RISCV_TPREL_HI20:
  case ELF::R_RISCV_TPREL_ADD:
  case ELF::R_RISCV_TPREL_LO12_I:
  case ELF::R_RISCV_TPREL_LO12_S:
  case ELFReserved::R_RISCV_TPREL_I:
  case ELFReserved::R_RISCV_TPREL_S:
    return true;
  }
}

size_t X86RelocationHandler::getSizeForType(uint32_t Type) const {
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
  case ELF::R_X86_64_GOTPC64:
    return 8;
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

size_t RISCVRelocationHandler::getSizeForType(uint32_t Type) const {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_RISCV, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_RISCV_RVC_JUMP:
  case ELF::R_RISCV_RVC_BRANCH:
    return 2;
  case ELF::R_RISCV_JAL:
  case ELF::R_RISCV_BRANCH:
  case ELF::R_RISCV_PCREL_HI20:
  case ELF::R_RISCV_PCREL_LO12_I:
  case ELF::R_RISCV_PCREL_LO12_S:
  case ELF::R_RISCV_32_PCREL:
  case ELF::R_RISCV_CALL:
  case ELF::R_RISCV_CALL_PLT:
  case ELF::R_RISCV_ADD32:
  case ELF::R_RISCV_SUB32:
  case ELF::R_RISCV_HI20:
  case ELF::R_RISCV_LO12_I:
  case ELF::R_RISCV_LO12_S:
  case ELF::R_RISCV_32:
    return 4;
  case ELF::R_RISCV_64:
  case ELF::R_RISCV_GOT_HI20:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
    // See extractValue for why this is necessary.
    return 8;
  }
}

bool X86RelocationHandler::skipRelocationType(uint32_t Type) const {
  return Type == ELF::R_X86_64_NONE;
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

bool RISCVRelocationHandler::skipRelocationType(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_RISCV_NONE:
  case ELF::R_RISCV_RELAX:
    return true;
  }
}

uint64_t X86RelocationHandler::encodeValue(uint32_t Type, uint64_t Value,
                                           uint64_t PC) const {
  switch (Type) {
  default:
    llvm_unreachable("unsupported relocation");
  case ELF::R_X86_64_64:
  case ELF::R_X86_64_32:
    break;
  case ELF::R_X86_64_PC32:
    Value -= PC;
    break;
  }
  return Value;
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

bool RISCVRelocationHandler::canEncodeValue(uint32_t Type, uint64_t Value,
                                            uint64_t PC) const {
  switch (Type) {
  default:
    llvm_unreachable("unsupported relocation");
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
    return true;
  }
}

uint64_t RISCVRelocationHandler::encodeValue(uint32_t Type, uint64_t Value,
                                             uint64_t PC) const {
  switch (Type) {
  default:
    llvm_unreachable("unsupported relocation");
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
    break;
  }
  return Value;
}

uint64_t X86RelocationHandler::extractValue(uint32_t Type, uint64_t Contents,
                                            uint64_t PC) const {
  if (Type == ELF::R_X86_64_32S)
    return SignExtend64<32>(Contents);
  if (isPCRelative(Type))
    return SignExtend64(Contents, 8 * getSizeForType(Type));
  return Contents;
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

static uint64_t extractUImmRISCV(uint32_t Contents) {
  return SignExtend64<32>(Contents & 0xfffff000);
}

static uint64_t extractIImmRISCV(uint32_t Contents) {
  return SignExtend64<12>(Contents >> 20);
}

static uint64_t extractSImmRISCV(uint32_t Contents) {
  return SignExtend64<12>(((Contents >> 7) & 0x1f) | ((Contents >> 25) << 5));
}

static uint64_t extractJImmRISCV(uint32_t Contents) {
  return SignExtend64<21>(
      (((Contents >> 21) & 0x3ff) << 1) | (((Contents >> 20) & 0x1) << 11) |
      (((Contents >> 12) & 0xff) << 12) | (((Contents >> 31) & 0x1) << 20));
}

static uint64_t extractBImmRISCV(uint32_t Contents) {
  return SignExtend64<13>(
      (((Contents >> 8) & 0xf) << 1) | (((Contents >> 25) & 0x3f) << 5) |
      (((Contents >> 7) & 0x1) << 11) | (((Contents >> 31) & 0x1) << 12));
}

uint64_t RISCVRelocationHandler::extractValue(uint32_t Type, uint64_t Contents,
                                              uint64_t PC) const {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_RISCV, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_RISCV_JAL:
    return extractJImmRISCV(Contents);
  case ELF::R_RISCV_CALL:
  case ELF::R_RISCV_CALL_PLT:
    return extractUImmRISCV(Contents);
  case ELF::R_RISCV_BRANCH:
    return extractBImmRISCV(Contents);
  case ELF::R_RISCV_GOT_HI20:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
    // We need to know the exact address of the GOT entry so we extract the
    // value from both the AUIPC and L[D|W]. We cannot rely on the symbol in the
    // relocation for this since it simply refers to the object that is stored
    // in the GOT entry, not to the entry itself.
    return extractUImmRISCV(Contents & 0xffffffff) +
           extractIImmRISCV(Contents >> 32);
  case ELF::R_RISCV_PCREL_HI20:
  case ELF::R_RISCV_HI20:
    return extractUImmRISCV(Contents);
  case ELF::R_RISCV_PCREL_LO12_I:
  case ELF::R_RISCV_LO12_I:
    return extractIImmRISCV(Contents);
  case ELF::R_RISCV_PCREL_LO12_S:
  case ELF::R_RISCV_LO12_S:
    return extractSImmRISCV(Contents);
  case ELF::R_RISCV_RVC_JUMP:
    return SignExtend64<11>(Contents >> 2);
  case ELF::R_RISCV_RVC_BRANCH:
    return SignExtend64<8>(((Contents >> 2) & 0x1f) | ((Contents >> 5) & 0xe0));
  case ELF::R_RISCV_ADD32:
  case ELF::R_RISCV_SUB32:
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
    return Contents;
  }
}

bool X86RelocationHandler::isGOT(uint32_t Type) const {
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

bool RISCVRelocationHandler::isGOT(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_RISCV_GOT_HI20:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
    return true;
  }
}

bool X86RelocationHandler::isTLS(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_X86_64_TPOFF32:
  case ELF::R_X86_64_TPOFF64:
  case ELF::R_X86_64_GOTTPOFF:
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

bool RISCVRelocationHandler::isTLS(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
  case ELF::R_RISCV_TPREL_HI20:
  case ELF::R_RISCV_TPREL_ADD:
  case ELF::R_RISCV_TPREL_LO12_I:
  case ELF::R_RISCV_TPREL_LO12_S:
  case ELFReserved::R_RISCV_TPREL_I:
  case ELFReserved::R_RISCV_TPREL_S:
    return true;
  }
}

bool X86RelocationHandler::isPCRelative(uint32_t Type) const {
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
  case ELF::R_X86_64_GOTPC64:
  case ELF::R_X86_64_GOTTPOFF:
  case ELF::R_X86_64_GOTPCRELX:
  case ELF::R_X86_64_REX_GOTPCRELX:
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

bool RISCVRelocationHandler::isPCRelative(uint32_t Type) const {
  switch (Type) {
  default:
    llvm_unreachable("Unknown relocation type");
  case ELF::R_RISCV_ADD32:
  case ELF::R_RISCV_SUB32:
  case ELF::R_RISCV_HI20:
  case ELF::R_RISCV_LO12_I:
  case ELF::R_RISCV_LO12_S:
  case ELF::R_RISCV_32:
  case ELF::R_RISCV_64:
    return false;
  case ELF::R_RISCV_JAL:
  case ELF::R_RISCV_CALL:
  case ELF::R_RISCV_CALL_PLT:
  case ELF::R_RISCV_BRANCH:
  case ELF::R_RISCV_GOT_HI20:
  case ELF::R_RISCV_PCREL_HI20:
  case ELF::R_RISCV_PCREL_LO12_I:
  case ELF::R_RISCV_PCREL_LO12_S:
  case ELF::R_RISCV_RVC_JUMP:
  case ELF::R_RISCV_RVC_BRANCH:
  case ELF::R_RISCV_32_PCREL:
  case ELF::R_RISCV_TLS_GOT_HI20:
  case ELF::R_RISCV_TLS_GD_HI20:
    return true;
  }
}

MCBinaryExpr::Opcode
RelocationHandler::getComposeOpcodeFor(uint32_t Type) const {
  llvm_unreachable("composed relocations are unsupported for this target");
}

std::unique_ptr<RelocationHandler>
llvm::bolt::createRelocationHandler(Triple::ArchType Arch) {
  switch (Arch) {
  default:
    llvm_unreachable("Unsupported architecture");
  case Triple::aarch64:
    return std::make_unique<AArch64RelocationHandler>();
  case Triple::riscv32:
    return std::make_unique<RISCVRelocationHandler>(false);
  case Triple::riscv64:
    return std::make_unique<RISCVRelocationHandler>(true);
  case Triple::x86_64:
    return std::make_unique<X86RelocationHandler>();
  }
}

uint32_t Relocation::getType(const object::RelocationRef &Rel) {
  uint64_t RelType = Rel.getType();
  assert(isUInt<32>(RelType) && "BOLT relocation types are 32 bits");
  return static_cast<uint32_t>(RelType);
}

size_t Relocation::emit(MCStreamer *Streamer,
                        const RelocationHandler &RH) const {
  const size_t Size = RH.getSizeForType(Type);
  const auto *Value = createExpr(Streamer, RH);
  Streamer->emitValue(Value, Size);
  return Size;
}

const MCExpr *Relocation::createExpr(MCStreamer *Streamer,
                                     const RelocationHandler &RH) const {
  MCContext &Ctx = Streamer->getContext();
  const MCExpr *Value = nullptr;

  if (Symbol && Addend) {
    Value = MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Symbol, Ctx),
                                    MCConstantExpr::create(Addend, Ctx), Ctx);
  } else if (Symbol) {
    Value = MCSymbolRefExpr::create(Symbol, Ctx);
  } else {
    Value = MCConstantExpr::create(Addend, Ctx);
  }

  if (RH.isPCRelative(Type)) {
    MCSymbol *TempLabel = Ctx.createNamedTempSymbol();
    Streamer->emitLabel(TempLabel);
    Value = MCBinaryExpr::createSub(
        Value, MCSymbolRefExpr::create(TempLabel, Ctx), Ctx);
  }

  return Value;
}

const MCExpr *Relocation::createExpr(MCStreamer *Streamer,
                                     const MCExpr *RetainedValue,
                                     const RelocationHandler &RH) const {
  const auto *Value = createExpr(Streamer, RH);

  if (RetainedValue) {
    Value = MCBinaryExpr::create(RH.getComposeOpcodeFor(Type), RetainedValue,
                                 Value, Streamer->getContext());
  }

  return Value;
}

void Relocation::print(raw_ostream &OS, const RelocationHandler &RH) const {
  RH.printType(OS, Type);
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
