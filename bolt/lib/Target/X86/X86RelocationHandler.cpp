//===- X86RelocationHandler.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 relocation handler.
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

bool X86RelocationHandler::skipRelocationType(uint32_t Type) const {
  return Type == ELF::R_X86_64_NONE;
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

uint64_t X86RelocationHandler::extractValue(uint32_t Type, uint64_t Contents,
                                            uint64_t PC) const {
  if (Type == ELF::R_X86_64_32S)
    return SignExtend64<32>(Contents);
  if (isPCRelative(Type))
    return SignExtend64(Contents, 8 * getSizeForType(Type));
  return Contents;
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

namespace llvm::bolt {

std::unique_ptr<RelocationHandler> createX86RelocationHandler() {
  return std::make_unique<X86RelocationHandler>();
}

} // namespace llvm::bolt
