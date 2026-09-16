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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace bolt;

namespace llvm::bolt {
std::unique_ptr<RelocationHandler> createAArch64RelocationHandler();
std::unique_ptr<RelocationHandler> createRISCVRelocationHandler(bool Is64Bit);
std::unique_ptr<RelocationHandler> createX86RelocationHandler();
} // namespace llvm::bolt

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
    return createAArch64RelocationHandler();
  case Triple::riscv32:
    return createRISCVRelocationHandler(false);
  case Triple::riscv64:
    return createRISCVRelocationHandler(true);
  case Triple::x86_64:
    return createX86RelocationHandler();
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
