//===- bolt/Target/PowerPC/PPCRelocationHandler.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PowerPC64 relocation handler.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Relocation.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bolt-reloc"

using namespace llvm;
using namespace bolt;

namespace {

class PPC64RelocationHandler final : public RelocationHandler {
public:
  bool isSupported(uint32_t Type) const override;
  size_t getSizeForType(uint32_t Type) const override;
  bool skipRelocationType(uint32_t Type) const override;

  uint64_t encodeValue(uint32_t Type, uint64_t Value, uint64_t PC,
                       uint32_t OriginalInst) const override;
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

  bool emitsBareSymbol(uint32_t Type) const override;
  bool composesRelocations() const override;

  void printType(raw_ostream &OS, uint32_t Type) const override;
};

} // namespace

bool PPC64RelocationHandler::isSupported(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_PPC64_ADDR16:
  case ELF::R_PPC64_ADDR16_LO:
  case ELF::R_PPC64_TOC16_DS:
  case ELF::R_PPC64_TOC16_LO_DS:
  case ELF::R_PPC64_ADDR16_HI:
  case ELF::R_PPC64_ADDR16_HA:
  case ELF::R_PPC64_ADDR32:
  case ELF::R_PPC64_ADDR64:
  case ELF::R_PPC64_REL64:
  case ELF::R_PPC64_ADDR16_DS:
  case ELF::R_PPC64_ADDR16_LO_DS:
  case ELF::R_PPC64_REL14:
  case ELF::R_PPC64_REL14_BRTAKEN:
  case ELF::R_PPC64_REL14_BRNTAKEN:
  case ELF::R_PPC64_REL24:
  case ELF::R_PPC64_REL32:
  case ELF::R_PPC64_TOC16:
  case ELF::R_PPC64_TOC16_LO:
  case ELF::R_PPC64_TOC16_HI:
  case ELF::R_PPC64_TOC16_HA:
  case ELF::R_PPC64_TOC:
  case ELF::R_PPC64_DTPREL16:
  case ELF::R_PPC64_DTPREL16_LO:
  case ELF::R_PPC64_DTPREL16_HI:
  case ELF::R_PPC64_DTPREL16_HA:
  case ELF::R_PPC64_DTPREL64:
  case ELF::R_PPC64_GOT16:
  case ELF::R_PPC64_GOT16_LO:
  case ELF::R_PPC64_GOT16_HI:
  case ELF::R_PPC64_GOT16_HA:
    return true;
  }
}

size_t PPC64RelocationHandler::getSizeForType(uint32_t Type) const {
  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_PPC64, Type) << '\n';
    llvm_unreachable("unsupported relocation type");
  case ELF::R_PPC64_ADDR16:
  case ELF::R_PPC64_ADDR16_LO:
  case ELF::R_PPC64_ADDR16_HI:
  case ELF::R_PPC64_ADDR16_HA:
  case ELF::R_PPC64_TOC16:
  case ELF::R_PPC64_TOC16_LO:
  case ELF::R_PPC64_TOC16_HI:
  case ELF::R_PPC64_TOC16_HA:
  case ELF::R_PPC64_DTPREL16:
  case ELF::R_PPC64_DTPREL16_LO:
  case ELF::R_PPC64_TOC16_DS:
  case ELF::R_PPC64_TOC16_LO_DS:
  case ELF::R_PPC64_DTPREL16_HI:
  case ELF::R_PPC64_DTPREL16_HA:
  case ELF::R_PPC64_GOT16:
  case ELF::R_PPC64_GOT16_LO:
  case ELF::R_PPC64_GOT16_HI:
  case ELF::R_PPC64_GOT16_HA:
    return 2;
  case ELF::R_PPC64_ADDR32:
  case ELF::R_PPC64_REL24:
  case ELF::R_PPC64_REL14:
  case ELF::R_PPC64_REL14_BRTAKEN:
  case ELF::R_PPC64_REL14_BRNTAKEN:
    return 4;
  case ELF::R_PPC64_ADDR64:
  case ELF::R_PPC64_REL32:
  case ELF::R_PPC64_REL64:
  case ELF::R_PPC64_TOC:
    return 8;
  case ELF::R_PPC64_NONE:
    return 0;
  }
}

bool PPC64RelocationHandler::skipRelocationType(uint32_t Type) const {
  return Type == ELF::R_PPC64_NONE;
}

// The PowerPC backend encodes the value into the instruction itself through
// its own fixups, so the value is passed through unchanged.
uint64_t PPC64RelocationHandler::encodeValue(uint32_t, uint64_t Value, uint64_t,
                                             uint32_t) const {
  return Value;
}

bool PPC64RelocationHandler::canEncodeValue(uint32_t, uint64_t,
                                            uint64_t) const {
  return true;
}

uint64_t PPC64RelocationHandler::extractValue(uint32_t Type, uint64_t Contents,
                                              uint64_t /*PC*/) const {
  LLVM_DEBUG(dbgs() << "[extractValuePPC64] Type=" << Type << " Contents=0x"
                    << llvm::format_hex(Contents, 10) << "\n");

  switch (Type) {
  default:
    errs() << object::getELFRelocationTypeName(ELF::EM_PPC64, Type) << '\n';
    llvm_unreachable("unsupported relocation type");

  // Data / address / TOC / GOT / TLS classes → return the RELA addend (often 0)
  case ELF::R_PPC64_ADDR16:
  case ELF::R_PPC64_ADDR16_LO:
  case ELF::R_PPC64_ADDR16_HI:
  case ELF::R_PPC64_ADDR16_HA:
  case ELF::R_PPC64_ADDR32:
  case ELF::R_PPC64_ADDR64:
  case ELF::R_PPC64_REL64:
  case ELF::R_PPC64_TOC16_DS:
  case ELF::R_PPC64_TOC16_LO_DS:
  case ELF::R_PPC64_ADDR16_DS:
  case ELF::R_PPC64_ADDR16_LO_DS:
  case ELF::R_PPC64_TOC:
  case ELF::R_PPC64_TOC16:
  case ELF::R_PPC64_TOC16_LO:
  case ELF::R_PPC64_TOC16_HI:
  case ELF::R_PPC64_TOC16_HA:
  case ELF::R_PPC64_DTPREL16:
  case ELF::R_PPC64_DTPREL16_LO:
  case ELF::R_PPC64_DTPREL16_HI:
  case ELF::R_PPC64_DTPREL16_HA:
  case ELF::R_PPC64_DTPREL64:
  case ELF::R_PPC64_GOT16:
  case ELF::R_PPC64_GOT16_LO:
  case ELF::R_PPC64_GOT16_HI:
  case ELF::R_PPC64_GOT16_HA:
  case ELF::R_PPC64_REL32:
    return 0;

  case ELF::R_PPC64_REL24: {
    uint32_t li = (Contents >> 2) & 0x00FFFFFF; // bits [6..29]
    int32_t off = (int32_t)(li << 8) >> 8;      // sign-extend 24 bits
    LLVM_DEBUG(dbgs() << "[extractValuePPC64] REL24 offset=" << off
                      << " bytes\n");

    return off << 2; // bytes
  }
  case ELF::R_PPC64_REL14:
  case ELF::R_PPC64_REL14_BRTAKEN:
  case ELF::R_PPC64_REL14_BRNTAKEN: {
    uint32_t bd = (Contents >> 2) & 0x00003FFF; // bits [16..29]
    int32_t off = (int32_t)(bd << 18) >> 18;    // sign-extend 14 bits
    LLVM_DEBUG(dbgs() << "[extractValuePPC64] REL14 extracted offset=" << off
                      << "\n");
    return off << 2;
  }

  case ELF::R_PPC64_NONE:
    LLVM_DEBUG(dbgs() << "[extractValuePPC64] R_PPC64_NONE\n");
    return 0;
  }
}

bool PPC64RelocationHandler::isGOT(uint32_t) const { return false; }

bool PPC64RelocationHandler::isRelative(uint32_t Type) const {
  return Type == ELF::R_PPC64_RELATIVE;
}

bool PPC64RelocationHandler::isIRelative(uint32_t) const {
  llvm_unreachable("Unsupported architecture");
}

bool PPC64RelocationHandler::isTLS(uint32_t) const { return false; }

bool PPC64RelocationHandler::isPCRelative(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_PPC64_REL32:
  case ELF::R_PPC64_REL24:
  case ELF::R_PPC64_REL14:
  case ELF::R_PPC64_REL64:
  case ELF::R_PPC64_REL14_BRTAKEN:
  case ELF::R_PPC64_REL14_BRNTAKEN:
    return true;
  }
}

uint32_t PPC64RelocationHandler::getNone() const { return ELF::R_PPC64_NONE; }

uint32_t PPC64RelocationHandler::getPC32() const { return ELF::R_PPC64_REL32; }

uint32_t PPC64RelocationHandler::getPC64() const { return ELF::R_PPC64_REL64; }

uint32_t PPC64RelocationHandler::getAbs64() const {
  return ELF::R_PPC64_ADDR64;
}

uint32_t PPC64RelocationHandler::getRelative() const {
  return ELF::R_PPC64_RELATIVE;
}

// For these types the PowerPC backend attaches the @ha, @lo and @ds fixups to
// the instructions itself, so BOLT emits only the symbol. Adding the addend as
// well would encode the immediate twice.
bool PPC64RelocationHandler::emitsBareSymbol(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case ELF::R_PPC64_ADDR16:
  case ELF::R_PPC64_ADDR16_HI:
  case ELF::R_PPC64_ADDR16_HA:
  case ELF::R_PPC64_ADDR16_LO:
  case ELF::R_PPC64_ADDR16_DS:
  case ELF::R_PPC64_ADDR16_LO_DS:
  case ELF::R_PPC64_TOC16:
  case ELF::R_PPC64_TOC16_HI:
  case ELF::R_PPC64_TOC16_HA:
  case ELF::R_PPC64_TOC16_LO:
  case ELF::R_PPC64_TOC16_DS:
  case ELF::R_PPC64_TOC16_LO_DS:
  case ELF::R_PPC64_REL14:
  case ELF::R_PPC64_REL14_BRTAKEN:
  case ELF::R_PPC64_REL14_BRNTAKEN:
  case ELF::R_PPC64_REL24:
  case ELF::R_PPC64_REL32:
  case ELF::R_PPC64_ADDR64:
    return true;
  }
}

// PowerPC never composes relocations that share an offset; the MC layer
// handles @ha, @lo and the _ds forms itself.
bool PPC64RelocationHandler::composesRelocations() const { return false; }

void PPC64RelocationHandler::printType(raw_ostream &OS, uint32_t Type) const {
  OS << "RType:" << Twine::utohexstr(Type);
}

std::unique_ptr<RelocationHandler> llvm::bolt::createPPC64RelocationHandler() {
  return std::make_unique<PPC64RelocationHandler>();
}
