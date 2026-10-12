//===- bolt/Target/RISCV/RISCVMCSymbolizer.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMCSymbolizer.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/Relocation.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"

#define DEBUG_TYPE "bolt-symbolizer"

namespace llvm {
namespace bolt {

static bool isValidControlTransferAUIPC(const MCInst &Inst) {
  if (Inst.getOpcode() != RISCV::AUIPC ||
      MCPlus::getNumPrimeOperands(Inst) != 2)
    return false;

  const MCOperand &Base = Inst.getOperand(0);
  return Base.isReg() && Base.getReg() != RISCV::X0 &&
         Inst.getOperand(1).isImm();
}

static bool isValidControlTransferJALR(const MCInst &Inst) {
  if (Inst.getOpcode() != RISCV::JALR || MCPlus::getNumPrimeOperands(Inst) != 3)
    return false;

  return Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg() &&
         Inst.getOperand(2).isImm();
}

static bool isLinkerResolvedControlTransfer(const MCInst &AUIPC,
                                            const MCInst &JALR) {
  if (!isValidControlTransferAUIPC(AUIPC) || !isValidControlTransferJALR(JALR))
    return false;

  const MCRegister Base = AUIPC.getOperand(0).getReg();
  if (JALR.getOperand(1).getReg() != Base)
    return false;

  const MCRegister Link = JALR.getOperand(0).getReg();
  return Link == RISCV::X0 || Link == Base;
}

static int64_t getLinkerResolvedControlTransferOffset(const MCInst &AUIPC,
                                                      const MCInst &JALR) {
  // The symbolic AUIPC decoder passes the sign-extended 20-bit U-immediate
  // already shifted left by 12. JALR adds its sign-extended 12-bit immediate
  // and clears target bit zero.
  return (AUIPC.getOperand(1).getImm() + JALR.getOperand(2).getImm()) & ~1LL;
}

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
      InstructionReferences.try_emplace(ReferencedOffset,
                                        InstructionReferenceInfo{Rel, nullptr});
    }

    SearchOffset = Rel->Offset + 1;
  }
}

RISCVMCSymbolizer::~RISCVMCSymbolizer() {}

MCSymbol *RISCVMCSymbolizer::getOrCreateInstructionLabel(
    InstructionReferenceInfo &ReferenceInfo) {
  if (!ReferenceInfo.Label)
    ReferenceInfo.Label = Ctx.createNamedTempSymbol();
  return ReferenceInfo.Label;
}

uint64_t RISCVMCSymbolizer::getGOTValue(const Relocation &Rel) const {
  BinaryContext &BC = Function.getBinaryContext();
  const uint64_t HiAddress = Function.getAddress() + Rel.Offset;

  // A GOT high relocation records a combined high/low value. Locate the low
  // relocation by its reference back to this AUIPC instead of assuming that
  // the low instruction is adjacent.
  auto It = InstructionReferences.find(Rel.Offset);
  if (It != InstructionReferences.end() && It->second.LowRelocation) {
    const Relocation *LoRel = It->second.LowRelocation;
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

bool RISCVMCSymbolizer::trySymbolizeLinkerResolvedControlTransfer(
    MCInst &Inst, int64_t Value, uint64_t InstAddress) {
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();
  const uint64_t InstOffset = InstAddress - Function.getAddress();

  // Recover RV64 linker-resolved intra-section calls and tail calls without
  // relocations. Decode the following JALR and attach a call expression to the
  // AUIPC before function reordering can move the caller relative to the
  // callee.
  if (Inst.getOpcode() != RISCV::AUIPC || !CreateNewSymbols ||
      !BC.TheTriple->isRISCV64() || InstOffset + 8 > Function.getSize() ||
      Function.getRelocationInRange(InstOffset, InstOffset + 8))
    return false;

  ErrorOr<ArrayRef<uint8_t>> FunctionData = Function.getData();
  if (!FunctionData)
    return false;

  MCInst JALR;
  uint64_t JALRSize = 0;
  if (!BC.DisAsm->getInstruction(JALR, JALRSize,
                                 FunctionData->slice(InstOffset + 4),
                                 InstAddress + 4, nulls()) ||
      JALRSize != 4)
    return false;

  MCInst AUIPC = Inst;
  AUIPC.addOperand(MCOperand::createImm(Value));
  if (!isLinkerResolvedControlTransfer(AUIPC, JALR))
    return false;

  const uint64_t Target =
      InstAddress + getLinkerResolvedControlTransferOffset(AUIPC, JALR);
  BinaryFunction *TargetBF = BC.getBinaryFunctionContainingAddress(Target);
  if (!TargetBF)
    return false;

  // JALR with rd=x0 is a no-link jump. It is a tail call only when it
  // transfers control to another function; a target in the current function
  // is an intraprocedural long jump and must not be represented as a call.
  if (JALR.getOperand(0).getReg() == RISCV::X0 && TargetBF == &Function)
    return false;

  BC.addInterproceduralReference(&Function, Target);
  MCSymbol *TargetSymbol =
      BC.handleExternalBranchTarget(Target, Function, *TargetBF);
  if (!TargetSymbol)
    return false;

  const MCExpr *Expr = MCSymbolRefExpr::create(TargetSymbol, *Ctx);
  Inst.addOperand(MCOperand::createExpr(
      BC.MIB->getTargetExprFor(Inst, Expr, *Ctx, ELF::R_RISCV_CALL_PLT)));
  return true;
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
    return trySymbolizeLinkerResolvedControlTransfer(Inst, Value, InstAddress);

  MCSymbol *Symbol = Rel->Symbol;
  uint64_t Addend = Rel->Addend;

  if (Relocation::isInstructionReference(Rel->Type)) {
    if (!CreateNewSymbols)
      return false;
    auto [It, _] =
        InstructionReferences.try_emplace(Rel->Value - Function.getAddress());
    Symbol = getOrCreateInstructionLabel(It->second);
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
  if (CreateNewSymbols) {
    auto It = InstructionReferences.find(InstOffset);
    if (It != InstructionReferences.end())
      BC.MIB->setInstLabel(Inst, getOrCreateInstructionLabel(It->second));
  }

  return true;
}

void RISCVMCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                        int64_t Value,
                                                        uint64_t Address) {}

} // namespace bolt
} // namespace llvm
