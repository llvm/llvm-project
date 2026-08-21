//===- bolt/Target/RISCV/RISCVMCSymbolizer.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMCSymbolizer.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/Relocation.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCInst.h"

#define DEBUG_TYPE "bolt-symbolizer"

namespace llvm {
namespace bolt {

RISCVMCSymbolizer::RISCVMCSymbolizer(BinaryFunction &Function,
                                     bool CreateNewSymbols)
    : MCSymbolizer(*Function.getBinaryContext().Ctx, nullptr),
      Function(Function), CreateNewSymbols(CreateNewSymbols) {
  // Discover instruction references before decoding starts. This lets us
  // attach a label while decoding the referenced %pcrel_hi instruction even
  // though its %pcrel_lo user is normally decoded later.
  for (uint64_t SearchOffset = 0; SearchOffset < Function.getSize();) {
    const Relocation *Rel =
        Function.getRelocationInRange(SearchOffset, Function.getSize());
    if (!Rel)
      break;

    if (Relocation::isInstructionReference(Rel->Type)) {
      assert(Rel->Value >= Function.getAddress() &&
             Rel->Value < Function.getAddress() + Function.getSize() &&
             "RISC-V instruction reference outside of function");
      const uint64_t ReferencedOffset = Rel->Value - Function.getAddress();
      InstructionReferences.try_emplace(ReferencedOffset, Rel);
      if (CreateNewSymbols)
        InstructionLabels.try_emplace(ReferencedOffset, nullptr);
    }

    SearchOffset = Rel->Offset + 1;
  }
}

RISCVMCSymbolizer::~RISCVMCSymbolizer() {}

MCSymbol *RISCVMCSymbolizer::getOrCreateInstructionLabel(uint64_t Offset) {
  auto [It, Inserted] = InstructionLabels.try_emplace(Offset, nullptr);
  (void)Inserted;
  if (!It->second)
    It->second = Ctx.createNamedTempSymbol();
  return It->second;
}

uint64_t RISCVMCSymbolizer::getGOTValue(const Relocation &Rel) const {
  BinaryContext &BC = Function.getBinaryContext();
  const uint64_t HiAddress = Function.getAddress() + Rel.Offset;

  // A GOT high relocation records a combined high/low value. Locate the low
  // relocation by its reference back to this AUIPC instead of assuming that
  // the low instruction is adjacent.
  auto It = InstructionReferences.find(Rel.Offset);
  if (It != InstructionReferences.end()) {
    const Relocation *LoRel = It->second;
    ErrorOr<uint64_t> HiContents = BC.getUnsignedValueAtAddress(HiAddress, 4);
    ErrorOr<uint64_t> LoContents =
        BC.getUnsignedValueAtAddress(Function.getAddress() + LoRel->Offset,
                                     Relocation::getSizeForType(LoRel->Type));
    assert(HiContents && LoContents &&
           "cannot read RISC-V GOT relocation pair");

    return Relocation::extractValue(ELF::R_RISCV_PCREL_HI20, *HiContents,
                                    HiAddress) +
           Relocation::extractValue(LoRel->Type, *LoContents,
                                    Function.getAddress() + LoRel->Offset);
  }

  return Rel.Value;
}

bool RISCVMCSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream &CStream, int64_t Value, uint64_t InstAddress,
    bool IsBranch, uint64_t ImmOffset, uint64_t ImmSize, uint64_t InstSize) {
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();
  const uint64_t InstOffset = InstAddress - Function.getAddress();

  // Branches and calls are resolved by BinaryFunction's target-independent
  // control-flow handling.
  if (BC.MIB->isBranch(Inst) || BC.MIB->isCall(Inst))
    return false;

  // Linker processing of R_RISCV_ALIGN can leave emitted relocations at an
  // offset inside the instruction they apply to. Match the whole instruction
  // range, as BinaryFunction::disassemble() did before this target-specific
  // handling moved into the symbolizer.
  const Relocation *Rel =
      Function.getRelocationInRange(InstOffset, InstOffset + InstSize);
  if (!Rel)
    return false;

  MCSymbol *Symbol = Rel->Symbol;
  uint64_t Addend = Rel->Addend;

  if (Relocation::isInstructionReference(Rel->Type)) {
    if (!CreateNewSymbols)
      return false;
    Symbol = getOrCreateInstructionLabel(Rel->Value - Function.getAddress());
    // The input addend reflects the original AUIPC location. The label now
    // follows the instruction, so the assembler must derive the low bits from
    // its new location.
    Addend = 0;
  }

  // GOT high relocations name the object stored in the GOT, not the GOT entry
  // addressed by AUIPC. Preserve the actual entry address using a zero-based
  // symbol, as the RISC-V emitter reuses the input GOT.
  if (Relocation::isGOT(Rel->Type)) {
    assert(Relocation::isPCRelative(Rel->Type) &&
           "GOT relocation must be PC-relative on RISC-V");
    Symbol = BC.registerNameAtAddress("__BOLT_got_zero", 0, 0, 0);
    Addend = getGOTValue(*Rel) + InstAddress;
  }

  assert(Symbol && "RISC-V relocation without a symbol");
  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);
  if (Addend)
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Addend, *Ctx),
                                   *Ctx);
  Inst.addOperand(MCOperand::createExpr(
      BC.MIB->getTargetExprFor(Inst, Expr, *Ctx, Rel->Type)));

  // MC annotations must follow every real operand. Attach the instruction
  // label only after the symbolized immediate has been appended.
  if (InstructionLabels.find(InstOffset) != InstructionLabels.end())
    BC.MIB->setInstLabel(Inst, getOrCreateInstructionLabel(InstOffset));

  return true;
}

void RISCVMCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                        int64_t Value,
                                                        uint64_t Address) {}

} // namespace bolt
} // namespace llvm
