//===- bolt/Target/AArch64/AArch64MCSymbolizer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64MCSymbolizer.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/Relocation.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bolt-symbolizer"

namespace llvm {
namespace bolt {

AArch64MCSymbolizer::~AArch64MCSymbolizer() {}

bool AArch64MCSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream &CStream, int64_t Value, uint64_t InstAddress,
    bool IsBranch, uint64_t ImmOffset, uint64_t ImmSize, uint64_t InstSize) {
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();

  // NOTE: the callee may incorrectly set IsBranch.
  if (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst))
    return false;

  const uint64_t InstOffset = InstAddress - Function.getAddress();
  const Relocation *Relocation = Function.getRelocationAt(InstOffset);

  /// Add symbolic operand to the instruction with an optional addend.
  auto addOperand = [&](const MCSymbol *Symbol, uint64_t Addend,
                        uint64_t RelType) {
    const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);
    if (Addend)
      Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Addend, *Ctx),
                                     *Ctx);
    Inst.addOperand(MCOperand::createExpr(
        BC.MIB->getTargetExprFor(Inst, Expr, *Ctx, RelType)));
  };

  // The linker can convert ADRP+ADD and ADRP+LDR instruction sequences into
  // NOP+ADR. After the conversion, the linker might keep the relocations and
  // if we try to symbolize ADR's operand using outdated relocations, we might
  // get unexpected results. Hence, we check for the conversion/relaxation, and
  // ignore the relocation. The symbolization is done based on the PC-relative
  // value of the operand instead.
  if (Relocation && BC.MIB->isADR(Inst)) {
    if (Relocation->Type == ELF::R_AARCH64_ADD_ABS_LO12_NC ||
        Relocation->Type == ELF::R_AARCH64_LD64_GOT_LO12_NC) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation at 0x"
                        << Twine::utohexstr(InstAddress) << '\n');
      Relocation = nullptr;
    }
  }

  if (Relocation) {
    addOperand(Relocation->Symbol, Relocation->Addend, Relocation->Type);
    return true;
  }

  if (!BC.MIB->hasPCRelOperand(Inst))
    return false;

  Value += InstAddress;
  const MCSymbol *TargetSymbol;
  uint64_t TargetOffset;
  if (!CreateNewSymbols) {
    if (BinaryData *BD = BC.getBinaryDataContainingAddress(Value)) {
      TargetSymbol = BD->getSymbol();
      TargetOffset = Value - BD->getAddress();
    } else {
      return false;
    }
  } else {
    std::tie(TargetSymbol, TargetOffset) =
        BC.handleAddressRef(Value, Function, /*IsPCRel*/ true);
  }

  addOperand(TargetSymbol, TargetOffset, 0);

  return true;
}

void AArch64MCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                          int64_t Value,
                                                          uint64_t Address) {}

} // namespace bolt
} // namespace llvm
