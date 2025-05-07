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

  if (Relocation) {
    auto AdjustedRel = adjustRelocation(*Relocation, Inst);
    if (AdjustedRel) {
      addOperand(AdjustedRel->Symbol, AdjustedRel->Addend, AdjustedRel->Type);
      return true;
    }

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation at 0x"
                      << Twine::utohexstr(InstAddress) << '\n');
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

std::optional<Relocation>
AArch64MCSymbolizer::adjustRelocation(const Relocation &Rel,
                                      const MCInst &Inst) const {
  BinaryContext &BC = Function.getBinaryContext();

  // The linker can convert ADRP+ADD and ADRP+LDR instruction sequences into
  // NOP+ADR. After the conversion, the linker might keep the relocations and
  // if we try to symbolize ADR's operand using outdated relocations, we might
  // get unexpected results. Hence, we check for the conversion/relaxation, and
  // ignore the relocation. The symbolization is done based on the PC-relative
  // value of the operand instead.
  if (BC.MIB->isADR(Inst) && (Rel.Type == ELF::R_AARCH64_ADD_ABS_LO12_NC ||
                              Rel.Type == ELF::R_AARCH64_LD64_GOT_LO12_NC))
    return std::nullopt;

  // The linker might perform TLS relocations relaxations, such as changed TLS
  // access model (e.g. changed global dynamic model to initial exec), thus
  // changing the instructions. The static relocations might be invalid at this
  // point and we don't have to process these relocations anymore. More
  // information could be found by searching elfNN_aarch64_tls_relax in bfd.
  if (BC.MIB->isMOVW(Inst)) {
    switch (Rel.Type) {
    default:
      break;
    case ELF::R_AARCH64_TLSDESC_LD64_LO12:
    case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
    case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
    case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
      return std::nullopt;
    }
  }

  if (!Relocation::isGOT(Rel.Type))
    return Rel;

  Relocation AdjustedRel = Rel;
  if (Rel.Type == ELF::R_AARCH64_LD64_GOT_LO12_NC && BC.MIB->isAddXri(Inst)) {
    // The ADRP+LDR sequence was converted into ADRP+ADD. We are looking at the
    // second instruction and have to use the relocation type for ADD.
    AdjustedRel.Type = ELF::R_AARCH64_ADD_ABS_LO12_NC;
    return AdjustedRel;
  }

  // ADRP is a special case since the linker can leave the instruction opcode
  // intact and modify only the operand. We are doing our best to detect when
  // such conversion has happened without looking at the next instruction.
  //
  // If we detect that a page referenced by the ADRP cannot belong to GOT, and
  // that it matches the symbol from the relocation, then we can be certain
  // that the linker converted the GOT reference into the local one. Otherwise,
  // we leave the disambiguation resolution to FixRelaxationPass.
  //
  // Note that ADRP relaxation described above cannot happen for TLS relocation.
  // Since TLS relocations may not even have a valid symbol (not supported by
  // BOLT), we explicitly exclude them from the check.
  if (BC.MIB->isADRP(Inst) && Rel.Addend == 0 && !Relocation::isTLS(Rel.Type)) {
    ErrorOr<uint64_t> SymbolValue = BC.getSymbolValue(*Rel.Symbol);
    assert(SymbolValue && "Symbol value should be set");
    const uint64_t SymbolPageAddr = *SymbolValue & ~0xfffULL;

    if (SymbolPageAddr == Rel.Value &&
        !isPageAddressValidForGOT(SymbolPageAddr)) {
      AdjustedRel.Type = ELF::R_AARCH64_ADR_PREL_PG_HI21;
      return AdjustedRel;
    }
  }

  // For instructions that reference GOT, ignore the referenced symbol and
  // use value at the relocation site. FixRelaxationPass will look at
  // instruction pairs and will perform necessary adjustments.
  AdjustedRel.Symbol = BC.registerNameAtAddress("__BOLT_got_zero", 0, 0, 0);
  AdjustedRel.Addend = Rel.Value;

  return AdjustedRel;
}

bool AArch64MCSymbolizer::isPageAddressValidForGOT(uint64_t PageAddress) const {
  assert(!(PageAddress & 0xfffULL) && "Page address not aligned at 4KB");

  ErrorOr<BinarySection &> GOT =
      Function.getBinaryContext().getUniqueSectionByName(".got");
  if (!GOT || !GOT->getSize())
    return false;

  const uint64_t GOTFirstPageAddress = GOT->getAddress() & ~0xfffULL;
  const uint64_t GOTLastPageAddress =
      (GOT->getAddress() + GOT->getSize() - 1) & ~0xfffULL;

  return PageAddress >= GOTFirstPageAddress &&
         PageAddress <= GOTLastPageAddress;
}

void AArch64MCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                          int64_t Value,
                                                          uint64_t Address) {}

} // namespace bolt
} // namespace llvm
