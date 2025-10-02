//===- bolt/Target/AArch64/AArch64MCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MCSymbolizer.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64FixupKinds.h"
#include "MCTargetDesc/AArch64MCAsmInfo.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCInstUtils.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltInstrCategory;
static cl::opt<bool> NoLSEAtomics(
    "no-lse-atomics",
    cl::desc("generate instrumentation code sequence without using LSE atomic "
             "instruction"),
    cl::init(false), cl::Optional, cl::cat(BoltInstrCategory));
} // namespace opts

namespace {

static void getSystemFlag(MCInst &Inst, MCPhysReg RegName) {
  Inst.setOpcode(AArch64::MRS);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(RegName));
  Inst.addOperand(MCOperand::createImm(AArch64SysReg::NZCV));
}

static void setSystemFlag(MCInst &Inst, MCPhysReg RegName) {
  Inst.setOpcode(AArch64::MSR);
  Inst.clear();
  Inst.addOperand(MCOperand::createImm(AArch64SysReg::NZCV));
  Inst.addOperand(MCOperand::createReg(RegName));
}

static void createPushRegisters(MCInst &Inst, MCPhysReg Reg1, MCPhysReg Reg2) {
  Inst.clear();
  unsigned NewOpcode = AArch64::STPXpre;
  Inst.setOpcode(NewOpcode);
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createReg(Reg1));
  Inst.addOperand(MCOperand::createReg(Reg2));
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createImm(-2));
}

static void createPopRegisters(MCInst &Inst, MCPhysReg Reg1, MCPhysReg Reg2) {
  Inst.clear();
  unsigned NewOpcode = AArch64::LDPXpost;
  Inst.setOpcode(NewOpcode);
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createReg(Reg1));
  Inst.addOperand(MCOperand::createReg(Reg2));
  Inst.addOperand(MCOperand::createReg(AArch64::SP));
  Inst.addOperand(MCOperand::createImm(2));
}

static void loadReg(MCInst &Inst, MCPhysReg To, MCPhysReg From) {
  Inst.setOpcode(AArch64::LDRXui);
  Inst.clear();
  if (From == AArch64::SP) {
    Inst.setOpcode(AArch64::LDRXpost);
    Inst.addOperand(MCOperand::createReg(From));
    Inst.addOperand(MCOperand::createReg(To));
    Inst.addOperand(MCOperand::createReg(From));
    Inst.addOperand(MCOperand::createImm(16));
  } else {
    Inst.addOperand(MCOperand::createReg(To));
    Inst.addOperand(MCOperand::createReg(From));
    Inst.addOperand(MCOperand::createImm(0));
  }
}

static void storeReg(MCInst &Inst, MCPhysReg From, MCPhysReg To) {
  Inst.setOpcode(AArch64::STRXui);
  Inst.clear();
  if (To == AArch64::SP) {
    Inst.setOpcode(AArch64::STRXpre);
    Inst.addOperand(MCOperand::createReg(To));
    Inst.addOperand(MCOperand::createReg(From));
    Inst.addOperand(MCOperand::createReg(To));
    Inst.addOperand(MCOperand::createImm(-16));
  } else {
    Inst.addOperand(MCOperand::createReg(From));
    Inst.addOperand(MCOperand::createReg(To));
    Inst.addOperand(MCOperand::createImm(0));
  }
}

static void atomicAdd(MCInst &Inst, MCPhysReg RegTo, MCPhysReg RegCnt) {
  assert(!opts::NoLSEAtomics && "Supports only ARM with LSE extension");
  Inst.setOpcode(AArch64::LDADDX);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(AArch64::XZR));
  Inst.addOperand(MCOperand::createReg(RegCnt));
  Inst.addOperand(MCOperand::createReg(RegTo));
}

static void createMovz(MCInst &Inst, MCPhysReg Reg, uint64_t Imm) {
  assert(Imm <= UINT16_MAX && "Invalid Imm size");
  Inst.clear();
  Inst.setOpcode(AArch64::MOVZXi);
  Inst.addOperand(MCOperand::createReg(Reg));
  Inst.addOperand(MCOperand::createImm(Imm & 0xFFFF));
  Inst.addOperand(MCOperand::createImm(0));
}

static InstructionListType createIncMemory(MCPhysReg RegTo, MCPhysReg RegTmp) {
  InstructionListType Insts;
  Insts.emplace_back();
  createMovz(Insts.back(), RegTmp, 1);
  Insts.emplace_back();
  atomicAdd(Insts.back(), RegTo, RegTmp);
  return Insts;
}
class AArch64MCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  BinaryFunction *InstrCounterIncrFunc{nullptr};

  std::unique_ptr<MCSymbolizer>
  createTargetSymbolizer(BinaryFunction &Function,
                         bool CreateNewSymbols) const override {
    return std::make_unique<AArch64MCSymbolizer>(Function, CreateNewSymbols);
  }

  MCPhysReg getStackPointer() const override { return AArch64::SP; }
  MCPhysReg getFramePointer() const override { return AArch64::FP; }

  bool isPush(const MCInst &Inst) const override {
    return isStoreToStack(Inst);
  };

  bool isPop(const MCInst &Inst) const override {
    return isLoadFromStack(Inst);
  };

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    createDirectCall(Inst, Target, Ctx, false);
  }

  bool convertTailCallToCall(MCInst &Inst) override {
    int NewOpcode;
    switch (Inst.getOpcode()) {
    default:
      return false;
    case AArch64::B:
      NewOpcode = AArch64::BL;
      break;
    case AArch64::BR:
      NewOpcode = AArch64::BLR;
      break;
    }

    Inst.setOpcode(NewOpcode);
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    clearOffset(Inst);
    return true;
  }

  bool equals(const MCSpecifierExpr &A, const MCSpecifierExpr &B,
              CompFuncTy Comp) const override {
    if (A.getSpecifier() != B.getSpecifier())
      return false;

    return MCPlusBuilder::equals(*A.getSubExpr(), *B.getSubExpr(), Comp);
  }

  bool shortenInstruction(MCInst &, const MCSubtargetInfo &) const override {
    return false;
  }

  SmallVector<MCPhysReg> getTrustedLiveInRegs() const override {
    return {AArch64::LR};
  }

  std::optional<MCPhysReg>
  getWrittenAuthenticatedReg(const MCInst &Inst,
                             bool &IsChecked) const override {
    IsChecked = false;
    switch (Inst.getOpcode()) {
    case AArch64::AUTIAZ:
    case AArch64::AUTIBZ:
    case AArch64::AUTIASP:
    case AArch64::AUTIBSP:
    case AArch64::AUTIASPPCi:
    case AArch64::AUTIBSPPCi:
    case AArch64::AUTIASPPCr:
    case AArch64::AUTIBSPPCr:
      return AArch64::LR;
    case AArch64::AUTIA1716:
    case AArch64::AUTIB1716:
    case AArch64::AUTIA171615:
    case AArch64::AUTIB171615:
      return AArch64::X17;
    case AArch64::AUTIA:
    case AArch64::AUTIB:
    case AArch64::AUTDA:
    case AArch64::AUTDB:
    case AArch64::AUTIZA:
    case AArch64::AUTIZB:
    case AArch64::AUTDZA:
    case AArch64::AUTDZB:
      return Inst.getOperand(0).getReg();
    case AArch64::LDRAAwriteback:
    case AArch64::LDRABwriteback:
      // Note that LDRA(A|B)indexed are not listed here, as they do not write
      // an authenticated pointer back to the register.
      IsChecked = true;
      return Inst.getOperand(2).getReg();
    default:
      return std::nullopt;
    }
  }

  bool isPSignOnLR(const MCInst &Inst) const override {
    std::optional<MCPhysReg> SignReg = getSignedReg(Inst);
    return SignReg && *SignReg == AArch64::LR;
  }

  bool isPAuthOnLR(const MCInst &Inst) const override {
    // LDR(A|B) should not be covered.
    bool IsChecked;
    std::optional<MCPhysReg> AuthReg =
        getWrittenAuthenticatedReg(Inst, IsChecked);
    return !IsChecked && AuthReg && *AuthReg == AArch64::LR;
  }

  bool isPAuthAndRet(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::RETAA ||
           Inst.getOpcode() == AArch64::RETAB ||
           Inst.getOpcode() == AArch64::RETAASPPCi ||
           Inst.getOpcode() == AArch64::RETABSPPCi ||
           Inst.getOpcode() == AArch64::RETAASPPCr ||
           Inst.getOpcode() == AArch64::RETABSPPCr;
  }

  std::optional<MCPhysReg> getSignedReg(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case AArch64::PACIA:
    case AArch64::PACIB:
    case AArch64::PACDA:
    case AArch64::PACDB:
    case AArch64::PACIZA:
    case AArch64::PACIZB:
    case AArch64::PACDZA:
    case AArch64::PACDZB:
      return Inst.getOperand(0).getReg();
    case AArch64::PACIAZ:
    case AArch64::PACIBZ:
    case AArch64::PACIASP:
    case AArch64::PACIBSP:
    case AArch64::PACIASPPC:
    case AArch64::PACIBSPPC:
    case AArch64::PACNBIASPPC:
    case AArch64::PACNBIBSPPC:
      return AArch64::LR;
    case AArch64::PACIA1716:
    case AArch64::PACIB1716:
    case AArch64::PACIA171615:
    case AArch64::PACIB171615:
      return AArch64::X17;
    default:
      return std::nullopt;
    }
  }

  std::optional<MCPhysReg>
  getRegUsedAsRetDest(const MCInst &Inst,
                      bool &IsAuthenticatedInternally) const override {
    assert(isReturn(Inst));
    switch (Inst.getOpcode()) {
    case AArch64::RET:
      IsAuthenticatedInternally = false;
      return Inst.getOperand(0).getReg();

    case AArch64::RETAA:
    case AArch64::RETAB:
    case AArch64::RETAASPPCi:
    case AArch64::RETABSPPCi:
    case AArch64::RETAASPPCr:
    case AArch64::RETABSPPCr:
      IsAuthenticatedInternally = true;
      return AArch64::LR;
    case AArch64::ERET:
    case AArch64::ERETAA:
    case AArch64::ERETAB:
      // The ERET* instructions use either register ELR_EL1, ELR_EL2 or
      // ELR_EL3, depending on the current Exception Level at run-time.
      //
      // Furthermore, these registers are not modelled by LLVM as a regular
      // MCPhysReg, so there is no way to indicate that through the current API.
      return std::nullopt;
    default:
      llvm_unreachable("Unhandled return instruction");
    }
  }

  MCPhysReg getRegUsedAsIndirectBranchDest(
      const MCInst &Inst, bool &IsAuthenticatedInternally) const override {
    assert(isIndirectCall(Inst) || isIndirectBranch(Inst));

    switch (Inst.getOpcode()) {
    case AArch64::BR:
    case AArch64::BLR:
      IsAuthenticatedInternally = false;
      return Inst.getOperand(0).getReg();
    case AArch64::BRAA:
    case AArch64::BRAB:
    case AArch64::BRAAZ:
    case AArch64::BRABZ:
    case AArch64::BLRAA:
    case AArch64::BLRAB:
    case AArch64::BLRAAZ:
    case AArch64::BLRABZ:
      IsAuthenticatedInternally = true;
      return Inst.getOperand(0).getReg();
    default:
      llvm_unreachable("Unhandled indirect branch or call");
    }
  }

  std::optional<MCPhysReg>
  getMaterializedAddressRegForPtrAuth(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case AArch64::ADR:
    case AArch64::ADRP:
      // These instructions produce an address value based on the information
      // encoded into the instruction itself (which should reside in a read-only
      // code memory) and the value of PC register (that is, the location of
      // this instruction), so the produced value is not attacker-controlled.
      return Inst.getOperand(0).getReg();
    default:
      return std::nullopt;
    }
  }

  std::optional<std::pair<MCPhysReg, MCPhysReg>>
  analyzeAddressArithmeticsForPtrAuth(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return std::nullopt;
    case AArch64::ADDXri:
    case AArch64::SUBXri:
      // The immediate addend is encoded into the instruction itself, so it is
      // not attacker-controlled under Pointer Authentication threat model.
      return std::make_pair(Inst.getOperand(0).getReg(),
                            Inst.getOperand(1).getReg());
    case AArch64::ORRXrs:
      // "mov Xd, Xm" is equivalent to "orr Xd, XZR, Xm, lsl #0"
      if (Inst.getOperand(1).getReg() != AArch64::XZR ||
          Inst.getOperand(3).getImm() != 0)
        return std::nullopt;

      return std::make_pair(Inst.getOperand(0).getReg(),
                            Inst.getOperand(2).getReg());
    }
  }

  std::optional<std::pair<MCPhysReg, MCInst *>>
  getAuthCheckedReg(BinaryBasicBlock &BB) const override {
    // Match several possible hard-coded sequences of instructions which can be
    // emitted by LLVM backend to check that the authenticated pointer is
    // correct (see AArch64AsmPrinter::emitPtrauthCheckAuthenticatedValue).
    //
    // This function only matches sequences involving branch instructions.
    // All these sequences have the form:
    //
    // (0) ... regular code that authenticates a pointer in Xn ...
    // (1) analyze Xn
    // (2) branch to .Lon_success if the pointer is correct
    // (3) BRK #imm (fall-through basic block)
    //
    // In the above pseudocode, (1) + (2) is one of the following sequences:
    //
    // - eor Xtmp, Xn, Xn, lsl #1
    //   tbz Xtmp, #62, .Lon_success
    //
    // - mov Xtmp, Xn
    //   xpac(i|d) Xn (or xpaclri if Xn is LR)
    //   cmp Xtmp, Xn
    //   b.eq .Lon_success
    //
    // Note that any branch destination operand is accepted as .Lon_success -
    // it is the responsibility of the caller of getAuthCheckedReg to inspect
    // the list of successors of this basic block as appropriate.

    // Any of the above code sequences assume the fall-through basic block
    // is a dead-end trap instruction.
    const BinaryBasicBlock *BreakBB = BB.getFallthrough();
    if (!BreakBB || BreakBB->empty() || !isTrap(BreakBB->front()))
      return std::nullopt;

    // Iterate over the instructions of BB in reverse order, matching opcodes
    // and operands.

    auto It = BB.end();
    auto StepBack = [&]() {
      while (It != BB.begin()) {
        --It;
        // Skip any CFI instructions, but no other pseudos are expected here.
        if (!isCFI(*It))
          return true;
      }
      return false;
    };
    // Step to the last non-CFI instruction.
    if (!StepBack())
      return std::nullopt;

    using namespace llvm::bolt::LowLevelInstMatcherDSL;
    Reg TestedReg;
    Reg ScratchReg;

    if (matchInst(*It, AArch64::Bcc, Imm(AArch64CC::EQ) /*, .Lon_success*/)) {
      if (!StepBack() || !matchInst(*It, AArch64::SUBSXrs, Reg(AArch64::XZR),
                                    TestedReg, ScratchReg, Imm(0)))
        return std::nullopt;

      // Either XPAC(I|D) ScratchReg, ScratchReg
      // or     XPACLRI
      if (!StepBack())
        return std::nullopt;
      if (matchInst(*It, AArch64::XPACLRI)) {
        // No operands to check, but using XPACLRI forces TestedReg to be X30.
        if (TestedReg.get() != AArch64::LR)
          return std::nullopt;
      } else if (!matchInst(*It, AArch64::XPACI, ScratchReg, ScratchReg) &&
                 !matchInst(*It, AArch64::XPACD, ScratchReg, ScratchReg)) {
        return std::nullopt;
      }

      if (!StepBack() || !matchInst(*It, AArch64::ORRXrs, ScratchReg,
                                    Reg(AArch64::XZR), TestedReg, Imm(0)))
        return std::nullopt;

      return std::make_pair(TestedReg.get(), &*It);
    }

    if (matchInst(*It, AArch64::TBZX, ScratchReg, Imm(62) /*, .Lon_success*/)) {
      if (!StepBack() || !matchInst(*It, AArch64::EORXrs, ScratchReg, TestedReg,
                                    TestedReg, Imm(1)))
        return std::nullopt;

      return std::make_pair(TestedReg.get(), &*It);
    }

    return std::nullopt;
  }

  std::optional<MCPhysReg> getAuthCheckedReg(const MCInst &Inst,
                                             bool MayOverwrite) const override {
    // Cannot trivially reuse AArch64InstrInfo::getMemOperandWithOffsetWidth()
    // method as it accepts an instance of MachineInstr, not MCInst.
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());

    // If signing oracles are considered, the particular value left in the base
    // register after this instruction is important. This function checks that
    // if the base register was overwritten, it is due to address write-back:
    //
    //     ; good:
    //     autdza  x1           ; x1 is authenticated (may fail)
    //     ldr     x0, [x1, #8] ; x1 is checked and not changed
    //     pacdzb  x1
    //
    //     ; also good:
    //     autdza  x1
    //     ldr     x0, [x1, #8]! ; x1 is checked and incremented by 8
    //     pacdzb  x1
    //
    //     ; bad (the value being signed is not the authenticated one):
    //     autdza  x1
    //     ldr     x1, [x1, #8]  ; x1 is overwritten with an unrelated value
    //     pacdzb  x1
    //
    //     ; also bad:
    //     autdza  x1
    //     pacdzb  x1  ; possibly signing the result of failed authentication
    //
    // Note that this function is not needed for authentication oracles, as the
    // particular value left in the register after a successful memory access
    // is not important.
    auto ClobbersBaseRegExceptWriteback = [&](unsigned BaseRegUseIndex) {
      // FIXME: Compute the indices of address operands (base reg and written-
      //        back result) in AArch64InstrInfo instead of this ad-hoc code.
      MCPhysReg BaseReg = Inst.getOperand(BaseRegUseIndex).getReg();
      unsigned WrittenBackDefIndex = Desc.getOperandConstraint(
          BaseRegUseIndex, MCOI::OperandConstraint::TIED_TO);

      for (unsigned DefIndex = 0; DefIndex < Desc.getNumDefs(); ++DefIndex) {
        // Address write-back is permitted:
        //
        //    autda x0, x2
        //    ; x0 is authenticated
        //    ldr   x1, [x0, #8]!
        //    ; x0 is trusted (as authenticated and checked)
        if (DefIndex == WrittenBackDefIndex)
          continue;

        // Any other overwriting is not permitted:
        //
        //    autda x0, x2
        //    ; x0 is authenticated
        //    ldr   w0, [x0]
        //    ; x0 is not authenticated anymore
        if (RegInfo->regsOverlap(Inst.getOperand(DefIndex).getReg(), BaseReg))
          return true;
      }

      return false;
    };

    // FIXME: Not all load instructions are handled by this->mayLoad(Inst).
    //        On the other hand, MCInstrDesc::mayLoad() is permitted to return
    //        true for non-load instructions (such as AArch64::HINT) which
    //        would result in false negatives.
    if (mayLoad(Inst)) {
      // The first Use operand is the base address register.
      unsigned BaseRegIndex = Desc.getNumDefs();

      // Reject non-immediate offsets, as adding a 64-bit register can change
      // the resulting address arbitrarily.
      for (unsigned I = BaseRegIndex + 1, E = Desc.getNumOperands(); I < E; ++I)
        if (Inst.getOperand(I).isReg())
          return std::nullopt;

      if (!MayOverwrite && ClobbersBaseRegExceptWriteback(BaseRegIndex))
        return std::nullopt;

      return Inst.getOperand(BaseRegIndex).getReg();
    }

    // Store instructions are not handled yet, as they are not important for
    // pauthtest ABI. Though, they could be handled similar to loads, if needed.

    return std::nullopt;
  }

  bool isADRP(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADRP;
  }

  bool isADR(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADR;
  }

  bool isAddXri(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::ADDXri;
  }

  MCPhysReg getADRReg(const MCInst &Inst) const {
    assert((isADR(Inst) || isADRP(Inst)) && "Not an ADR instruction");
    assert(MCPlus::getNumPrimeOperands(Inst) != 0 &&
           "No operands for ADR instruction");
    assert(Inst.getOperand(0).isReg() &&
           "Unexpected operand in ADR instruction");
    return Inst.getOperand(0).getReg();
  }

  InstructionListType undoAdrpAddRelaxation(const MCInst &ADRInst,
                                            MCContext *Ctx) const override {
    assert(isADR(ADRInst) && "ADR instruction expected");

    const MCPhysReg Reg = getADRReg(ADRInst);
    const MCSymbol *Target = getTargetSymbol(ADRInst);
    const uint64_t Addend = getTargetAddend(ADRInst);
    return materializeAddress(Target, Ctx, Reg, Addend);
  }

  bool isTB(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::TBNZW ||
            Inst.getOpcode() == AArch64::TBNZX ||
            Inst.getOpcode() == AArch64::TBZW ||
            Inst.getOpcode() == AArch64::TBZX);
  }

  bool isCB(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::CBNZW ||
            Inst.getOpcode() == AArch64::CBNZX ||
            Inst.getOpcode() == AArch64::CBZW ||
            Inst.getOpcode() == AArch64::CBZX);
  }

  bool isMOVW(const MCInst &Inst) const override {
    return (Inst.getOpcode() == AArch64::MOVKWi ||
            Inst.getOpcode() == AArch64::MOVKXi ||
            Inst.getOpcode() == AArch64::MOVNWi ||
            Inst.getOpcode() == AArch64::MOVNXi ||
            Inst.getOpcode() == AArch64::MOVZXi ||
            Inst.getOpcode() == AArch64::MOVZWi);
  }

  bool isADD(const MCInst &Inst) const {
    return (Inst.getOpcode() == AArch64::ADDSWri ||
            Inst.getOpcode() == AArch64::ADDSWrr ||
            Inst.getOpcode() == AArch64::ADDSWrs ||
            Inst.getOpcode() == AArch64::ADDSWrx ||
            Inst.getOpcode() == AArch64::ADDSXri ||
            Inst.getOpcode() == AArch64::ADDSXrr ||
            Inst.getOpcode() == AArch64::ADDSXrs ||
            Inst.getOpcode() == AArch64::ADDSXrx ||
            Inst.getOpcode() == AArch64::ADDSXrx64 ||
            Inst.getOpcode() == AArch64::ADDWri ||
            Inst.getOpcode() == AArch64::ADDWrr ||
            Inst.getOpcode() == AArch64::ADDWrs ||
            Inst.getOpcode() == AArch64::ADDWrx ||
            Inst.getOpcode() == AArch64::ADDXri ||
            Inst.getOpcode() == AArch64::ADDXrr ||
            Inst.getOpcode() == AArch64::ADDXrs ||
            Inst.getOpcode() == AArch64::ADDXrx ||
            Inst.getOpcode() == AArch64::ADDXrx64);
  }

  bool isLDRB(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRBpost:
    case AArch64::LDRBBpost:
    case AArch64::LDRBBpre:
    case AArch64::LDRBBroW:
    case AArch64::LDRBroW:
    case AArch64::LDRBroX:
    case AArch64::LDRBBroX:
    case AArch64::LDRBBui:
    case AArch64::LDRBui:
    case AArch64::LDRBpre:
    case AArch64::LDRSBWpost:
    case AArch64::LDRSBWpre:
    case AArch64::LDRSBWroW:
    case AArch64::LDRSBWroX:
    case AArch64::LDRSBWui:
    case AArch64::LDRSBXpost:
    case AArch64::LDRSBXpre:
    case AArch64::LDRSBXroW:
    case AArch64::LDRSBXroX:
    case AArch64::LDRSBXui:
    case AArch64::LDURBi:
    case AArch64::LDURBBi:
    case AArch64::LDURSBWi:
    case AArch64::LDURSBXi:
    case AArch64::LDTRBi:
    case AArch64::LDTRSBWi:
    case AArch64::LDTRSBXi:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRH(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRHpost:
    case AArch64::LDRHHpost:
    case AArch64::LDRHHpre:
    case AArch64::LDRHroW:
    case AArch64::LDRHHroW:
    case AArch64::LDRHroX:
    case AArch64::LDRHHroX:
    case AArch64::LDRHHui:
    case AArch64::LDRHui:
    case AArch64::LDRHpre:
    case AArch64::LDRSHWpost:
    case AArch64::LDRSHWpre:
    case AArch64::LDRSHWroW:
    case AArch64::LDRSHWroX:
    case AArch64::LDRSHWui:
    case AArch64::LDRSHXpost:
    case AArch64::LDRSHXpre:
    case AArch64::LDRSHXroW:
    case AArch64::LDRSHXroX:
    case AArch64::LDRSHXui:
    case AArch64::LDURHi:
    case AArch64::LDURHHi:
    case AArch64::LDURSHWi:
    case AArch64::LDURSHXi:
    case AArch64::LDTRHi:
    case AArch64::LDTRSHWi:
    case AArch64::LDTRSHXi:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRW(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRWpost:
    case AArch64::LDRWpre:
    case AArch64::LDRWroW:
    case AArch64::LDRWroX:
    case AArch64::LDRWui:
    case AArch64::LDRWl:
    case AArch64::LDRSWl:
    case AArch64::LDURWi:
    case AArch64::LDRSWpost:
    case AArch64::LDRSWpre:
    case AArch64::LDRSWroW:
    case AArch64::LDRSWroX:
    case AArch64::LDRSWui:
    case AArch64::LDURSWi:
    case AArch64::LDTRWi:
    case AArch64::LDTRSWi:
    case AArch64::LDPWi:
    case AArch64::LDPWpost:
    case AArch64::LDPWpre:
    case AArch64::LDPSWi:
    case AArch64::LDPSWpost:
    case AArch64::LDPSWpre:
    case AArch64::LDNPWi:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRX(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRXpost:
    case AArch64::LDRXpre:
    case AArch64::LDRXroW:
    case AArch64::LDRXroX:
    case AArch64::LDRXui:
    case AArch64::LDRXl:
    case AArch64::LDURXi:
    case AArch64::LDTRXi:
    case AArch64::LDNPXi:
    case AArch64::LDPXi:
    case AArch64::LDPXpost:
    case AArch64::LDPXpre:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRS(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRSl:
    case AArch64::LDRSui:
    case AArch64::LDRSroW:
    case AArch64::LDRSroX:
    case AArch64::LDURSi:
    case AArch64::LDPSi:
    case AArch64::LDNPSi:
    case AArch64::LDRSpre:
    case AArch64::LDRSpost:
    case AArch64::LDPSpost:
    case AArch64::LDPSpre:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRD(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRDl:
    case AArch64::LDRDui:
    case AArch64::LDRDpre:
    case AArch64::LDRDpost:
    case AArch64::LDRDroW:
    case AArch64::LDRDroX:
    case AArch64::LDURDi:
    case AArch64::LDPDi:
    case AArch64::LDNPDi:
    case AArch64::LDPDpost:
    case AArch64::LDPDpre:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isLDRQ(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();
    switch (opcode) {
    case AArch64::LDRQui:
    case AArch64::LDRQl:
    case AArch64::LDRQpre:
    case AArch64::LDRQpost:
    case AArch64::LDRQroW:
    case AArch64::LDRQroX:
    case AArch64::LDURQi:
    case AArch64::LDPQi:
    case AArch64::LDNPQi:
    case AArch64::LDPQpost:
    case AArch64::LDPQpre:
      return true;
    default:
      break;
    }

    return false;
  }

  bool isBRA(const MCInst &Inst) const {
    switch (Inst.getOpcode()) {
    case AArch64::BRAA:
    case AArch64::BRAB:
    case AArch64::BRAAZ:
    case AArch64::BRABZ:
      return true;
    default:
      return false;
    }
  }

  bool mayLoad(const MCInst &Inst) const override {
    // FIXME: Probably this could be tablegen-erated not to miss any existing
    //        or future opcodes.
    return isLDRB(Inst) || isLDRH(Inst) || isLDRW(Inst) || isLDRX(Inst) ||
           isLDRQ(Inst) || isLDRD(Inst) || isLDRS(Inst);
  }

  bool isAArch64ExclusiveLoad(const MCInst &Inst) const override {
    return (Inst.getOpcode() == AArch64::LDXPX ||
            Inst.getOpcode() == AArch64::LDXPW ||
            Inst.getOpcode() == AArch64::LDXRX ||
            Inst.getOpcode() == AArch64::LDXRW ||
            Inst.getOpcode() == AArch64::LDXRH ||
            Inst.getOpcode() == AArch64::LDXRB ||
            Inst.getOpcode() == AArch64::LDAXPX ||
            Inst.getOpcode() == AArch64::LDAXPW ||
            Inst.getOpcode() == AArch64::LDAXRX ||
            Inst.getOpcode() == AArch64::LDAXRW ||
            Inst.getOpcode() == AArch64::LDAXRH ||
            Inst.getOpcode() == AArch64::LDAXRB);
  }

  bool isAArch64ExclusiveStore(const MCInst &Inst) const override {
    return (Inst.getOpcode() == AArch64::STXPX ||
            Inst.getOpcode() == AArch64::STXPW ||
            Inst.getOpcode() == AArch64::STXRX ||
            Inst.getOpcode() == AArch64::STXRW ||
            Inst.getOpcode() == AArch64::STXRH ||
            Inst.getOpcode() == AArch64::STXRB ||
            Inst.getOpcode() == AArch64::STLXPX ||
            Inst.getOpcode() == AArch64::STLXPW ||
            Inst.getOpcode() == AArch64::STLXRX ||
            Inst.getOpcode() == AArch64::STLXRW ||
            Inst.getOpcode() == AArch64::STLXRH ||
            Inst.getOpcode() == AArch64::STLXRB);
  }

  bool isAArch64ExclusiveClear(const MCInst &Inst) const override {
    return (Inst.getOpcode() == AArch64::CLREX);
  }

  bool isLoadFromStack(const MCInst &Inst) const {
    if (!mayLoad(Inst))
      return false;
    for (const MCOperand &Operand : useOperands(Inst)) {
      if (!Operand.isReg())
        continue;
      unsigned Reg = Operand.getReg();
      if (Reg == AArch64::SP || Reg == AArch64::WSP)
        return true;
    }
    return false;
  }

  bool isRegToRegMove(const MCInst &Inst, MCPhysReg &From,
                      MCPhysReg &To) const override {
    if (Inst.getOpcode() == AArch64::FMOVDXr) {
      From = Inst.getOperand(1).getReg();
      To = Inst.getOperand(0).getReg();
      return true;
    }

    if (Inst.getOpcode() != AArch64::ORRXrs)
      return false;
    if (Inst.getOperand(1).getReg() != AArch64::XZR)
      return false;
    if (Inst.getOperand(3).getImm() != 0)
      return false;
    From = Inst.getOperand(2).getReg();
    To = Inst.getOperand(0).getReg();
    return true;
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    return isIndirectCallOpcode(Inst.getOpcode());
  }

  MCPhysReg getSpRegister(int Size) const {
    switch (Size) {
    case 4:
      return AArch64::WSP;
    case 8:
      return AArch64::SP;
    default:
      llvm_unreachable("Unexpected size");
    }
  }

  MCPhysReg getIntArgRegister(unsigned ArgNo) const override {
    switch (ArgNo) {
    case 0:
      return AArch64::X0;
    case 1:
      return AArch64::X1;
    case 2:
      return AArch64::X2;
    case 3:
      return AArch64::X3;
    case 4:
      return AArch64::X4;
    case 5:
      return AArch64::X5;
    case 6:
      return AArch64::X6;
    case 7:
      return AArch64::X7;
    default:
      return getNoRegister();
    }
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    // ADRP is blacklisted and is an exception. Even though it has a
    // PC-relative operand, this operand is not a complete symbol reference
    // and BOLT shouldn't try to process it in isolation.
    if (isADRP(Inst))
      return false;

    if (isADR(Inst))
      return true;

    // Look for literal addressing mode (see C1-143 ARM DDI 0487B.a)
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I)
      if (MCII.operands()[I].OperandType == MCOI::OPERAND_PCREL)
        return true;

    return false;
  }

  bool evaluateADR(const MCInst &Inst, int64_t &Imm,
                   const MCExpr **DispExpr) const {
    assert((isADR(Inst) || isADRP(Inst)) && "Not an ADR instruction");

    const MCOperand &Label = Inst.getOperand(1);
    if (!Label.isImm()) {
      assert(Label.isExpr() && "Unexpected ADR operand");
      assert(DispExpr && "DispExpr must be set");
      *DispExpr = Label.getExpr();
      return false;
    }

    if (Inst.getOpcode() == AArch64::ADR) {
      Imm = Label.getImm();
      return true;
    }
    Imm = Label.getImm() << 12;
    return true;
  }

  bool evaluateAArch64MemoryOperand(const MCInst &Inst, int64_t &DispImm,
                                    const MCExpr **DispExpr = nullptr) const {
    if (isADR(Inst) || isADRP(Inst))
      return evaluateADR(Inst, DispImm, DispExpr);

    // Literal addressing mode
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
      if (MCII.operands()[I].OperandType != MCOI::OPERAND_PCREL)
        continue;

      if (!Inst.getOperand(I).isImm()) {
        assert(Inst.getOperand(I).isExpr() && "Unexpected PCREL operand");
        assert(DispExpr && "DispExpr must be set");
        *DispExpr = Inst.getOperand(I).getExpr();
        return true;
      }

      DispImm = Inst.getOperand(I).getImm() * 4;
      return true;
    }
    return false;
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    int64_t DispValue;
    const MCExpr *DispExpr = nullptr;
    if (!evaluateAArch64MemoryOperand(Inst, DispValue, &DispExpr))
      return false;

    // Make sure it's a well-formed addressing we can statically evaluate.
    if (DispExpr)
      return false;

    Target = DispValue;
    if (Inst.getOpcode() == AArch64::ADRP)
      Target += Address & ~0xFFFULL;
    else
      Target += Address;
    return true;
  }

  MCInst::iterator getMemOperandDisp(MCInst &Inst) const override {
    MCInst::iterator OI = Inst.begin();
    if (isADR(Inst) || isADRP(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 2 &&
             "Unexpected number of operands");
      return ++OI;
    }
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = MCII.getNumOperands(); I != E; ++I) {
      if (MCII.operands()[I].OperandType == MCOI::OPERAND_PCREL)
        break;
      ++OI;
    }
    assert(OI != Inst.end() && "Literal operand not found");
    return OI;
  }

  bool replaceMemOperandDisp(MCInst &Inst, MCOperand Operand) const override {
    MCInst::iterator OI = getMemOperandDisp(Inst);
    *OI = Operand;
    return true;
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    Regs |= getAliases(AArch64::X18);
    Regs |= getAliases(AArch64::X19);
    Regs |= getAliases(AArch64::X20);
    Regs |= getAliases(AArch64::X21);
    Regs |= getAliases(AArch64::X22);
    Regs |= getAliases(AArch64::X23);
    Regs |= getAliases(AArch64::X24);
    Regs |= getAliases(AArch64::X25);
    Regs |= getAliases(AArch64::X26);
    Regs |= getAliases(AArch64::X27);
    Regs |= getAliases(AArch64::X28);
    Regs |= getAliases(AArch64::LR);
    Regs |= getAliases(AArch64::FP);
  }

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint32_t RelType) const override {

    if (isADR(Inst) || RelType == ELF::R_AARCH64_ADR_PREL_LO21 ||
        RelType == ELF::R_AARCH64_TLSDESC_ADR_PREL21) {
      return MCSpecifierExpr::create(Expr, AArch64::S_ABS, Ctx);
    } else if (isADRP(Inst) || RelType == ELF::R_AARCH64_ADR_PREL_PG_HI21 ||
               RelType == ELF::R_AARCH64_ADR_PREL_PG_HI21_NC ||
               RelType == ELF::R_AARCH64_TLSDESC_ADR_PAGE21 ||
               RelType == ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 ||
               RelType == ELF::R_AARCH64_ADR_GOT_PAGE) {
      // Never emit a GOT reloc, we handled this in
      // RewriteInstance::readRelocations().
      return MCSpecifierExpr::create(Expr, AArch64::S_ABS_PAGE, Ctx);
    } else {
      switch (RelType) {
      case ELF::R_AARCH64_ADD_ABS_LO12_NC:
      case ELF::R_AARCH64_LD64_GOT_LO12_NC:
      case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
      case ELF::R_AARCH64_TLSDESC_ADD_LO12:
      case ELF::R_AARCH64_TLSDESC_LD64_LO12:
      case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
      case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
        return MCSpecifierExpr::create(Expr, AArch64::S_LO12, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G3:
        return MCSpecifierExpr::create(Expr, AArch64::S_ABS_G3, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G2:
      case ELF::R_AARCH64_MOVW_UABS_G2_NC:
        return MCSpecifierExpr::create(Expr, AArch64::S_ABS_G2_NC, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G1:
      case ELF::R_AARCH64_MOVW_UABS_G1_NC:
        return MCSpecifierExpr::create(Expr, AArch64::S_ABS_G1_NC, Ctx);
      case ELF::R_AARCH64_MOVW_UABS_G0:
      case ELF::R_AARCH64_MOVW_UABS_G0_NC:
        return MCSpecifierExpr::create(Expr, AArch64::S_ABS_G0_NC, Ctx);
      default:
        break;
      }
    }
    return Expr;
  }

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    if (OpNum >= MCPlus::getNumPrimeOperands(Inst))
      return false;

    // Auto-select correct operand number
    if (OpNum == 0) {
      if (isConditionalBranch(Inst) || isADR(Inst) || isADRP(Inst) ||
          isMOVW(Inst))
        OpNum = 1;
      if (isTB(Inst) || isAddXri(Inst))
        OpNum = 2;
    }

    return true;
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *AArchExpr = dyn_cast<MCSpecifierExpr>(Expr);
    if (AArchExpr && AArchExpr->getSubExpr())
      return getTargetSymbol(AArchExpr->getSubExpr());

    return MCPlusBuilder::getTargetSymbol(Expr);
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (!OpNum && !getSymbolRefOperandNum(Inst, OpNum))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
  }

  int64_t getTargetAddend(const MCExpr *Expr) const override {
    auto *AArchExpr = dyn_cast<MCSpecifierExpr>(Expr);
    if (AArchExpr && AArchExpr->getSubExpr())
      return getTargetAddend(AArchExpr->getSubExpr());

    auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr);
    if (BinExpr && BinExpr->getOpcode() == MCBinaryExpr::Add)
      return getTargetAddend(BinExpr->getRHS());

    auto *ConstExpr = dyn_cast<MCConstantExpr>(Expr);
    if (ConstExpr)
      return ConstExpr->getValue();

    return 0;
  }

  int64_t getTargetAddend(const MCInst &Inst,
                          unsigned OpNum = 0) const override {
    if (!getSymbolRefOperandNum(Inst, OpNum))
      return 0;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return 0;

    return getTargetAddend(Op.getExpr());
  }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");
    assert(MCPlus::getNumPrimeOperands(Inst) >= 1 &&
           "Invalid number of operands");
    MCInst::iterator OI = Inst.begin();

    if (isConditionalBranch(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 2 &&
             "Invalid number of operands");
      ++OI;
    }

    if (isTB(Inst)) {
      assert(MCPlus::getNumPrimeOperands(Inst) >= 3 &&
             "Invalid number of operands");
      OI = Inst.begin() + 2;
    }

    *OI = MCOperand::createExpr(MCSymbolRefExpr::create(TBB, *Ctx));
  }

  /// Matches indirect branch patterns in AArch64 related to a jump table (JT),
  /// helping us to build the complete CFG. A typical indirect branch to
  /// a jump table entry in AArch64 looks like the following:
  ///
  ///   adrp    x1, #-7585792           # Get JT Page location
  ///   add     x1, x1, #692            # Complement with JT Page offset
  ///   ldrh    w0, [x1, w0, uxtw #1]   # Loads JT entry
  ///   adr     x1, #12                 # Get PC + 12 (end of this BB) used next
  ///   add     x0, x1, w0, sxth #2     # Finish building branch target
  ///                                   # (entries in JT are relative to the end
  ///                                   #  of this BB)
  ///   br      x0                      # Indirect jump instruction
  ///
  /// Return true on successful jump table instruction sequence match, false
  /// otherwise.
  bool analyzeIndirectBranchFragment(
      const MCInst &Inst,
      DenseMap<const MCInst *, SmallVector<MCInst *, 4>> &UDChain,
      const MCExpr *&JumpTable, int64_t &Offset, int64_t &ScaleValue,
      MCInst *&PCRelBase) const {
    // The only kind of indirect branches we match is jump table, thus ignore
    // authenticating branch instructions early.
    if (isBRA(Inst))
      return false;

    // Expect AArch64 BR
    assert(Inst.getOpcode() == AArch64::BR && "Unexpected opcode");

    JumpTable = nullptr;

    // Match the indirect branch pattern for aarch64
    SmallVector<MCInst *, 4> &UsesRoot = UDChain[&Inst];
    if (UsesRoot.size() == 0 || UsesRoot[0] == nullptr)
      return false;

    const MCInst *DefAdd = UsesRoot[0];

    // Now we match an ADD
    if (!isADD(*DefAdd)) {
      // If the address is not broken up in two parts, this is not branching
      // according to a jump table entry. Fail.
      return false;
    }
    if (DefAdd->getOpcode() == AArch64::ADDXri) {
      // This can happen when there is no offset, but a direct jump that was
      // transformed into an indirect one  (indirect tail call) :
      //   ADRP   x2, Perl_re_compiler
      //   ADD    x2, x2, :lo12:Perl_re_compiler
      //   BR     x2
      return false;
    }
    if (DefAdd->getOpcode() == AArch64::ADDXrs) {
      // Covers the less common pattern where JT entries are relative to
      // the JT itself (like x86). Seems less efficient since we can't
      // assume the JT is aligned at 4B boundary and thus drop 2 bits from
      // JT values.
      // cde264:
      //    adrp    x12, #21544960  ; 216a000
      //    add     x12, x12, #1696 ; 216a6a0  (JT object in .rodata)
      //    ldrsw   x8, [x12, x8, lsl #2]   --> loads e.g. 0xfeb73bd8
      //  * add     x8, x8, x12   --> = cde278, next block
      //    br      x8
      // cde278:
      //
      // Parsed as ADDXrs reg:x8 reg:x8 reg:x12 imm:0
      return false;
    }
    if (DefAdd->getOpcode() != AArch64::ADDXrx)
      return false;

    // Validate ADD operands
    int64_t OperandExtension = DefAdd->getOperand(3).getImm();
    unsigned ShiftVal = AArch64_AM::getArithShiftValue(OperandExtension);
    AArch64_AM::ShiftExtendType ExtendType =
        AArch64_AM::getArithExtendType(OperandExtension);
    if (ShiftVal != 2) {
      // TODO: Handle the patten where ShiftVal != 2.
      // The following code sequence below has no shift amount,
      // the range could be 0 to 4.
      // The pattern comes from libc, it occurs when the binary is static.
      //   adr     x6, 0x219fb0 <sigall_set+0x88>
      //   add     x6, x6, x14, lsl #2
      //   ldr     w7, [x6]
      //   add     x6, x6, w7, sxtw => no shift amount
      //   br      x6
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: "
                           "failed to match indirect branch: ShiftVAL != 2\n");
      return false;
    }

    if (ExtendType == AArch64_AM::SXTB)
      ScaleValue = 1LL;
    else if (ExtendType == AArch64_AM::SXTH)
      ScaleValue = 2LL;
    else if (ExtendType == AArch64_AM::SXTW)
      ScaleValue = 4LL;
    else
      return false;

    // Match an ADR to load base address to be used when addressing JT targets
    SmallVector<MCInst *, 4> &UsesAdd = UDChain[DefAdd];
    if (UsesAdd.size() <= 1 || UsesAdd[1] == nullptr || UsesAdd[2] == nullptr) {
      // This happens when we don't have enough context about this jump table
      // because the jumping code sequence was split in multiple basic blocks.
      // This was observed in the wild in HHVM code (dispatchImpl).
      return false;
    }
    MCInst *DefBaseAddr = UsesAdd[1];
    if (DefBaseAddr->getOpcode() != AArch64::ADR)
      return false;

    PCRelBase = DefBaseAddr;
    // Match LOAD to load the jump table (relative) target
    const MCInst *DefLoad = UsesAdd[2];
    if (!mayLoad(*DefLoad) || (ScaleValue == 1LL && !isLDRB(*DefLoad)) ||
        (ScaleValue == 2LL && !isLDRH(*DefLoad)))
      return false;

    // Match ADD that calculates the JumpTable Base Address (not the offset)
    SmallVector<MCInst *, 4> &UsesLoad = UDChain[DefLoad];
    const MCInst *DefJTBaseAdd = UsesLoad[1];
    MCPhysReg From, To;
    if (DefJTBaseAdd == nullptr || isLoadFromStack(*DefJTBaseAdd) ||
        isRegToRegMove(*DefJTBaseAdd, From, To)) {
      // Sometimes base address may have been defined in another basic block
      // (hoisted). Return with no jump table info.
      return true;
    }

    if (DefJTBaseAdd->getOpcode() == AArch64::ADR) {
      // TODO: Handle the pattern where there is no adrp/add pair.
      // It also occurs when the binary is static.
      //  adr     x13, 0x215a18 <_nl_value_type_LC_COLLATE+0x50>
      //  ldrh    w13, [x13, w12, uxtw #1]
      //  adr     x12, 0x247b30 <__gettextparse+0x5b0>
      //  add     x13, x12, w13, sxth #2
      //  br      x13
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: failed to match indirect branch: "
                           "nop/adr instead of adrp/add\n");
      return false;
    }

    if (DefJTBaseAdd->getOpcode() != AArch64::ADDXri) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: failed to match jump table base "
                           "address pattern! (1)\n");
      return false;
    }

    if (DefJTBaseAdd->getOperand(2).isImm())
      Offset = DefJTBaseAdd->getOperand(2).getImm();
    SmallVector<MCInst *, 4> &UsesJTBaseAdd = UDChain[DefJTBaseAdd];
    const MCInst *DefJTBasePage = UsesJTBaseAdd[1];
    if (DefJTBasePage == nullptr || isLoadFromStack(*DefJTBasePage)) {
      return true;
    }
    if (DefJTBasePage->getOpcode() != AArch64::ADRP)
      return false;

    if (DefJTBasePage->getOperand(1).isExpr())
      JumpTable = DefJTBasePage->getOperand(1).getExpr();
    return true;
  }

  DenseMap<const MCInst *, SmallVector<MCInst *, 4>>
  computeLocalUDChain(const MCInst *CurInstr, InstructionIterator Begin,
                      InstructionIterator End) const {
    DenseMap<int, MCInst *> RegAliasTable;
    DenseMap<const MCInst *, SmallVector<MCInst *, 4>> Uses;

    auto addInstrOperands = [&](const MCInst &Instr) {
      // Update Uses table
      for (const MCOperand &Operand : MCPlus::primeOperands(Instr)) {
        if (!Operand.isReg())
          continue;
        unsigned Reg = Operand.getReg();
        MCInst *AliasInst = RegAliasTable[Reg];
        Uses[&Instr].push_back(AliasInst);
        LLVM_DEBUG({
          dbgs() << "Adding reg operand " << Reg << " refs ";
          if (AliasInst != nullptr)
            AliasInst->dump();
          else
            dbgs() << "\n";
        });
      }
    };

    LLVM_DEBUG(dbgs() << "computeLocalUDChain\n");
    bool TerminatorSeen = false;
    for (auto II = Begin; II != End; ++II) {
      MCInst &Instr = *II;
      // Ignore nops and CFIs
      if (isPseudo(Instr) || isNoop(Instr))
        continue;
      if (TerminatorSeen) {
        RegAliasTable.clear();
        Uses.clear();
      }

      LLVM_DEBUG(dbgs() << "Now updating for:\n ");
      LLVM_DEBUG(Instr.dump());
      addInstrOperands(Instr);

      BitVector Regs = BitVector(RegInfo->getNumRegs(), false);
      getWrittenRegs(Instr, Regs);

      // Update register definitions after this point
      for (int Idx : Regs.set_bits()) {
        RegAliasTable[Idx] = &Instr;
        LLVM_DEBUG(dbgs() << "Setting reg " << Idx
                          << " def to current instr.\n");
      }

      TerminatorSeen = isTerminator(Instr);
    }

    // Process the last instruction, which is not currently added into the
    // instruction stream
    if (CurInstr)
      addInstrOperands(*CurInstr);

    return Uses;
  }

  IndirectBranchType
  analyzeIndirectBranch(MCInst &Instruction, InstructionIterator Begin,
                        InstructionIterator End, const unsigned PtrSize,
                        MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
                        unsigned &IndexRegNumOut, int64_t &DispValueOut,
                        const MCExpr *&DispExprOut, MCInst *&PCRelBaseOut,
                        MCInst *&FixedEntryLoadInstr) const override {
    MemLocInstrOut = nullptr;
    BaseRegNumOut = AArch64::NoRegister;
    IndexRegNumOut = AArch64::NoRegister;
    DispValueOut = 0;
    DispExprOut = nullptr;
    FixedEntryLoadInstr = nullptr;

    // An instruction referencing memory used by jump instruction (directly or
    // via register). This location could be an array of function pointers
    // in case of indirect tail call, or a jump table.
    MCInst *MemLocInstr = nullptr;

    // Analyze the memory location.
    int64_t ScaleValue, DispValue;
    const MCExpr *DispExpr;

    DenseMap<const MCInst *, SmallVector<llvm::MCInst *, 4>> UDChain =
        computeLocalUDChain(&Instruction, Begin, End);
    MCInst *PCRelBase;
    if (!analyzeIndirectBranchFragment(Instruction, UDChain, DispExpr,
                                       DispValue, ScaleValue, PCRelBase))
      return IndirectBranchType::UNKNOWN;

    MemLocInstrOut = MemLocInstr;
    DispValueOut = DispValue;
    DispExprOut = DispExpr;
    PCRelBaseOut = PCRelBase;
    return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
  }

  ///  Matches PLT entry pattern and returns the associated GOT entry address.
  ///  Typical PLT entry looks like the following:
  ///
  ///    adrp    x16, 230000
  ///    ldr     x17, [x16, #3040]
  ///    add     x16, x16, #0xbe0
  ///    br      x17
  ///
  ///  The other type of trampolines are located in .plt.got, that are used for
  ///  non-lazy bindings so doesn't use x16 arg to transfer .got entry address:
  ///
  ///    adrp    x16, 230000
  ///    ldr     x17, [x16, #3040]
  ///    br      x17
  ///    nop
  ///
  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override {
    // Check branch instruction
    MCInst *Branch = &Instruction;
    assert(Branch->getOpcode() == AArch64::BR && "Unexpected opcode");

    DenseMap<const MCInst *, SmallVector<llvm::MCInst *, 4>> UDChain =
        computeLocalUDChain(Branch, Begin, End);

    // Match ldr instruction
    SmallVector<MCInst *, 4> &BranchUses = UDChain[Branch];
    if (BranchUses.size() < 1 || BranchUses[0] == nullptr)
      return 0;

    // Check ldr instruction
    const MCInst *Ldr = BranchUses[0];
    if (Ldr->getOpcode() != AArch64::LDRXui)
      return 0;

    // Get ldr value
    const unsigned ScaleLdr = 8; // LDRX operates on 8 bytes segments
    assert(Ldr->getOperand(2).isImm() && "Unexpected ldr operand");
    const uint64_t Offset = Ldr->getOperand(2).getImm() * ScaleLdr;

    // Match adrp instruction
    SmallVector<MCInst *, 4> &LdrUses = UDChain[Ldr];
    if (LdrUses.size() < 2 || LdrUses[1] == nullptr)
      return 0;

    // Check adrp instruction
    MCInst *Adrp = LdrUses[1];
    if (Adrp->getOpcode() != AArch64::ADRP)
      return 0;

    // Get adrp instruction PC
    const unsigned InstSize = 4;
    uint64_t AdrpPC = BeginPC;
    for (InstructionIterator It = Begin; It != End; ++It) {
      if (&(*It) == Adrp)
        break;
      AdrpPC += InstSize;
    }

    // Get adrp value
    uint64_t Base;
    assert(Adrp->getOperand(1).isImm() && "Unexpected adrp operand");
    bool Ret = evaluateMemOperandTarget(*Adrp, Base, AdrpPC, InstSize);
    assert(Ret && "Failed to evaluate adrp");
    (void)Ret;

    return Base + Offset;
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case AArch64::TBZW:     return AArch64::TBNZW;
    case AArch64::TBZX:     return AArch64::TBNZX;
    case AArch64::TBNZW:    return AArch64::TBZW;
    case AArch64::TBNZX:    return AArch64::TBZX;
    case AArch64::CBZW:     return AArch64::CBNZW;
    case AArch64::CBZX:     return AArch64::CBNZX;
    case AArch64::CBNZW:    return AArch64::CBZW;
    case AArch64::CBNZX:    return AArch64::CBZX;
    }
  }

  unsigned getCondCode(const MCInst &Inst) const override {
    // AArch64 does not use conditional codes, so we just return the opcode
    // of the conditional branch here.
    return Inst.getOpcode();
  }

  unsigned getCanonicalBranchCondCode(unsigned Opcode) const override {
    switch (Opcode) {
    default:
      return Opcode;
    case AArch64::TBNZW:    return AArch64::TBZW;
    case AArch64::TBNZX:    return AArch64::TBZX;
    case AArch64::CBNZW:    return AArch64::CBZW;
    case AArch64::CBNZX:    return AArch64::CBZX;
    }
  }

  void reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    if (isTB(Inst) || isCB(Inst)) {
      Inst.setOpcode(getInvertedBranchOpcode(Inst.getOpcode()));
      assert(Inst.getOpcode() != 0 && "Invalid branch instruction");
    } else if (Inst.getOpcode() == AArch64::Bcc) {
      Inst.getOperand(0).setImm(AArch64CC::getInvertedCondCode(
          static_cast<AArch64CC::CondCode>(Inst.getOperand(0).getImm())));
      assert(Inst.getOperand(0).getImm() != AArch64CC::AL &&
             Inst.getOperand(0).getImm() != AArch64CC::NV &&
             "Can't reverse ALWAYS cond code");
    } else {
      LLVM_DEBUG(Inst.dump());
      llvm_unreachable("Unrecognized branch instruction");
    }
    replaceBranchTarget(Inst, TBB, Ctx);
  }

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("Failed to get pcrel encoding size");
      return 0;
    case AArch64::TBZW:     return 16;
    case AArch64::TBZX:     return 16;
    case AArch64::TBNZW:    return 16;
    case AArch64::TBNZX:    return 16;
    case AArch64::CBZW:     return 21;
    case AArch64::CBZX:     return 21;
    case AArch64::CBNZW:    return 21;
    case AArch64::CBNZX:    return 21;
    case AArch64::B:        return 28;
    case AArch64::BL:       return 28;
    case AArch64::Bcc:      return 21;
    }
  }

  int getShortJmpEncodingSize() const override { return 33; }

  int getUncondBranchEncodingSize() const override { return 28; }

  // This helper function creates the snippet of code that compares a register
  // RegNo with an immedaite Imm, and jumps to Target if they are equal.
  // cmp RegNo, #Imm
  // b.eq Target
  // where cmp is an alias for subs, which results in the code below:
  // subs xzr, RegNo, #Imm
  // b.eq Target.
  InstructionListType createCmpJE(MCPhysReg RegNo, int64_t Imm,
                                  const MCSymbol *Target,
                                  MCContext *Ctx) const override {
    InstructionListType Code;
    Code.emplace_back(MCInstBuilder(AArch64::SUBSXri)
                          .addReg(AArch64::XZR)
                          .addReg(RegNo)
                          .addImm(Imm)
                          .addImm(0));
    Code.emplace_back(MCInstBuilder(AArch64::Bcc)
                          .addImm(AArch64CC::EQ)
                          .addExpr(MCSymbolRefExpr::create(Target, *Ctx)));
    return Code;
  }

  // This helper function creates the snippet of code that compares a register
  // RegNo with an immedaite Imm, and jumps to Target if they are not equal.
  // cmp RegNo, #Imm
  // b.ne Target
  // where cmp is an alias for subs, which results in the code below:
  // subs xzr, RegNo, #Imm
  // b.ne Target.
  InstructionListType createCmpJNE(MCPhysReg RegNo, int64_t Imm,
                                   const MCSymbol *Target,
                                   MCContext *Ctx) const override {
    InstructionListType Code;
    Code.emplace_back(MCInstBuilder(AArch64::SUBSXri)
                          .addReg(AArch64::XZR)
                          .addReg(RegNo)
                          .addImm(Imm)
                          .addImm(0));
    Code.emplace_back(MCInstBuilder(AArch64::Bcc)
                          .addImm(AArch64CC::NE)
                          .addExpr(MCSymbolRefExpr::create(Target, *Ctx)));
    return Code;
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createDirectCall(Inst, Target, Ctx, /*IsTailCall*/ true);
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    createShortJmp(Seq, Target, Ctx, /*IsTailCall*/ true);
  }

  void createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(AArch64::BRK);
    Inst.addOperand(MCOperand::createImm(1));
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    setTailCall(Inst);
    return true;
  }

  bool convertTailCallToJmp(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    clearOffset(Inst);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  InstructionListType createIndirectPLTCall(MCInst &&DirectCall,
                                            const MCSymbol *TargetLocation,
                                            MCContext *Ctx) override {
    const bool IsTailCall = isTailCall(DirectCall);
    assert((DirectCall.getOpcode() == AArch64::BL ||
            (DirectCall.getOpcode() == AArch64::B && IsTailCall)) &&
           "64-bit direct (tail) call instruction expected");

    InstructionListType Code;
    // Code sequence for indirect plt call:
    // adrp	x16 <symbol>
    // ldr	x17, [x16, #<offset>]
    // blr	x17  ; or 'br' for tail calls

    MCInst InstAdrp;
    InstAdrp.setOpcode(AArch64::ADRP);
    InstAdrp.addOperand(MCOperand::createReg(AArch64::X16));
    InstAdrp.addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(InstAdrp, /* OpNum */ 1, TargetLocation,
                          /* Addend */ 0, Ctx, ELF::R_AARCH64_ADR_GOT_PAGE);
    Code.emplace_back(InstAdrp);

    MCInst InstLoad;
    InstLoad.setOpcode(AArch64::LDRXui);
    InstLoad.addOperand(MCOperand::createReg(AArch64::X17));
    InstLoad.addOperand(MCOperand::createReg(AArch64::X16));
    InstLoad.addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(InstLoad, /* OpNum */ 2, TargetLocation,
                          /* Addend */ 0, Ctx, ELF::R_AARCH64_LD64_GOT_LO12_NC);
    Code.emplace_back(InstLoad);

    MCInst InstCall;
    InstCall.setOpcode(IsTailCall ? AArch64::BR : AArch64::BLR);
    InstCall.addOperand(MCOperand::createReg(AArch64::X17));
    moveAnnotations(std::move(DirectCall), InstCall);
    Code.emplace_back(InstCall);

    return Code;
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  bool isNoop(const MCInst &Inst) const override {
    return Inst.getOpcode() == AArch64::HINT &&
           Inst.getOperand(0).getImm() == 0;
  }

  void createNoop(MCInst &Inst) const override {
    Inst.setOpcode(AArch64::HINT);
    Inst.clear();
    Inst.addOperand(MCOperand::createImm(0));
  }

  bool isTrap(const MCInst &Inst) const override {
    if (Inst.getOpcode() != AArch64::BRK)
      return false;
    // Only match the immediate values that are likely to indicate this BRK
    // instruction is emitted to terminate the program immediately and not to
    // be handled by a SIGTRAP handler, for example.
    switch (Inst.getOperand(0).getImm()) {
    case 0xc470:
    case 0xc471:
    case 0xc472:
    case 0xc473:
      // Explicit Pointer Authentication check failed, see
      // AArch64AsmPrinter::emitPtrauthCheckAuthenticatedValue().
      return true;
    case 0x1:
      // __builtin_trap(), as emitted by Clang.
      return true;
    case 0x3e8: // decimal 1000
      // __builtin_trap(), as emitted by GCC.
      return true;
    default:
      // Some constants may indicate intentionally recoverable break-points.
      // This is the case at least for 0xf000, which is used by
      // __builtin_debugtrap() supported by Clang.
      return false;
    }
  }

  bool isStorePair(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();

    auto isStorePairImmOffset = [&]() {
      switch (opcode) {
      case AArch64::STPWi:
      case AArch64::STPXi:
      case AArch64::STPSi:
      case AArch64::STPDi:
      case AArch64::STPQi:
      case AArch64::STNPWi:
      case AArch64::STNPXi:
      case AArch64::STNPSi:
      case AArch64::STNPDi:
      case AArch64::STNPQi:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStorePairPostIndex = [&]() {
      switch (opcode) {
      case AArch64::STPWpost:
      case AArch64::STPXpost:
      case AArch64::STPSpost:
      case AArch64::STPDpost:
      case AArch64::STPQpost:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStorePairPreIndex = [&]() {
      switch (opcode) {
      case AArch64::STPWpre:
      case AArch64::STPXpre:
      case AArch64::STPSpre:
      case AArch64::STPDpre:
      case AArch64::STPQpre:
        return true;
      default:
        break;
      }

      return false;
    };

    return isStorePairImmOffset() || isStorePairPostIndex() ||
           isStorePairPreIndex();
  }

  bool isStoreReg(const MCInst &Inst) const {
    const unsigned opcode = Inst.getOpcode();

    auto isStoreRegUnscaleImm = [&]() {
      switch (opcode) {
      case AArch64::STURBi:
      case AArch64::STURBBi:
      case AArch64::STURHi:
      case AArch64::STURHHi:
      case AArch64::STURWi:
      case AArch64::STURXi:
      case AArch64::STURSi:
      case AArch64::STURDi:
      case AArch64::STURQi:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStoreRegScaledImm = [&]() {
      switch (opcode) {
      case AArch64::STRBui:
      case AArch64::STRBBui:
      case AArch64::STRHui:
      case AArch64::STRHHui:
      case AArch64::STRWui:
      case AArch64::STRXui:
      case AArch64::STRSui:
      case AArch64::STRDui:
      case AArch64::STRQui:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStoreRegImmPostIndexed = [&]() {
      switch (opcode) {
      case AArch64::STRBpost:
      case AArch64::STRBBpost:
      case AArch64::STRHpost:
      case AArch64::STRHHpost:
      case AArch64::STRWpost:
      case AArch64::STRXpost:
      case AArch64::STRSpost:
      case AArch64::STRDpost:
      case AArch64::STRQpost:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStoreRegImmPreIndexed = [&]() {
      switch (opcode) {
      case AArch64::STRBpre:
      case AArch64::STRBBpre:
      case AArch64::STRHpre:
      case AArch64::STRHHpre:
      case AArch64::STRWpre:
      case AArch64::STRXpre:
      case AArch64::STRSpre:
      case AArch64::STRDpre:
      case AArch64::STRQpre:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStoreRegUnscaleUnpriv = [&]() {
      switch (opcode) {
      case AArch64::STTRBi:
      case AArch64::STTRHi:
      case AArch64::STTRWi:
      case AArch64::STTRXi:
        return true;
      default:
        break;
      }

      return false;
    };

    auto isStoreRegTrunc = [&]() {
      switch (opcode) {
      case AArch64::STRBBroW:
      case AArch64::STRBBroX:
      case AArch64::STRBroW:
      case AArch64::STRBroX:
      case AArch64::STRDroW:
      case AArch64::STRDroX:
      case AArch64::STRHHroW:
      case AArch64::STRHHroX:
      case AArch64::STRHroW:
      case AArch64::STRHroX:
      case AArch64::STRQroW:
      case AArch64::STRQroX:
      case AArch64::STRSroW:
      case AArch64::STRSroX:
      case AArch64::STRWroW:
      case AArch64::STRWroX:
      case AArch64::STRXroW:
      case AArch64::STRXroX:
        return true;
      default:
        break;
      }

      return false;
    };

    return isStoreRegUnscaleImm() || isStoreRegScaledImm() ||
           isStoreRegImmPreIndexed() || isStoreRegImmPostIndexed() ||
           isStoreRegUnscaleUnpriv() || isStoreRegTrunc();
  }

  bool mayStore(const MCInst &Inst) const override {
    return isStorePair(Inst) || isStoreReg(Inst) ||
           isAArch64ExclusiveStore(Inst);
  }

  bool isStoreToStack(const MCInst &Inst) const {
    if (!mayStore(Inst))
      return false;

    for (const MCOperand &Operand : useOperands(Inst)) {
      if (!Operand.isReg())
        continue;

      unsigned Reg = Operand.getReg();
      if (Reg == AArch64::SP || Reg == AArch64::WSP)
        return true;
    }

    return false;
  }

  void createDirectCall(MCInst &Inst, const MCSymbol *Target, MCContext *Ctx,
                        bool IsTailCall) override {
    Inst.setOpcode(IsTailCall ? AArch64::B : AArch64::BL);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
        Inst, MCSymbolRefExpr::create(Target, *Ctx), *Ctx, 0)));
    if (IsTailCall)
      convertJmpToTailCall(Inst);
  }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I) || isNoop(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I) || isTailCall(*I) || !isBranch(*I))
        break;

      // Handle unconditional branches.
      if (isUnconditionalBranch(*I)) {
        // If any code was seen after this unconditional branch, we've seen
        // unreachable code. Ignore them.
        CondBranch = nullptr;
        UncondBranch = &*I;
        const MCSymbol *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I))
        return false;

      if (CondBranch == nullptr) {
        const MCSymbol *TargetBB = getTargetSymbol(*I);
        if (TargetBB == nullptr) {
          // Unrecognized branch target
          return false;
        }
        FBB = TBB;
        TBB = TargetBB;
        CondBranch = &*I;
        continue;
      }

      llvm_unreachable("multiple conditional branches in one BB");
    }
    return true;
  }

  void createLongJmp(InstructionListType &Seq, const MCSymbol *Target,
                     MCContext *Ctx, bool IsTailCall) override {
    // ip0 (r16) is reserved to the linker (refer to 5.3.1.1 of "Procedure Call
    //   Standard for the ARM 64-bit Architecture (AArch64)".
    // The sequence of instructions we create here is the following:
    //  movz ip0, #:abs_g3:<addr>
    //  movk ip0, #:abs_g2_nc:<addr>
    //  movk ip0, #:abs_g1_nc:<addr>
    //  movk ip0, #:abs_g0_nc:<addr>
    //  br ip0
    MCInst Inst;
    Inst.setOpcode(AArch64::MOVZXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(
        MCSpecifierExpr::create(Target, AArch64::S_ABS_G3, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x30));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(
        MCSpecifierExpr::create(Target, AArch64::S_ABS_G2_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x20));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(
        MCSpecifierExpr::create(Target, AArch64::S_ABS_G1_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0x10));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::MOVKXi);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    Inst.addOperand(MCOperand::createExpr(
        MCSpecifierExpr::create(Target, AArch64::S_ABS_G0_NC, *Ctx)));
    Inst.addOperand(MCOperand::createImm(0));
    Seq.emplace_back(Inst);

    Inst.clear();
    Inst.setOpcode(AArch64::BR);
    Inst.addOperand(MCOperand::createReg(AArch64::X16));
    if (IsTailCall)
      setTailCall(Inst);
    Seq.emplace_back(Inst);
  }

  void createShortJmp(InstructionListType &Seq, const MCSymbol *Target,
                      MCContext *Ctx, bool IsTailCall) override {
    // ip0 (r16) is reserved to the linker (refer to 5.3.1.1 of "Procedure Call
    //   Standard for the ARM 64-bit Architecture (AArch64)".
    // The sequence of instructions we create here is the following:
    //  adrp ip0, imm
    //  add ip0, ip0, imm
    //  br ip0
    MCPhysReg Reg = AArch64::X16;
    InstructionListType Insts = materializeAddress(Target, Ctx, Reg);
    Insts.emplace_back();
    MCInst &Inst = Insts.back();
    Inst.clear();
    Inst.setOpcode(AArch64::BR);
    Inst.addOperand(MCOperand::createReg(Reg));
    if (IsTailCall)
      setTailCall(Inst);
    Seq.swap(Insts);
  }

  /// Matching pattern here is
  ///
  ///    ADRP  x16, imm
  ///    ADD   x16, x16, imm
  ///    BR    x16
  ///
  uint64_t matchLinkerVeneer(InstructionIterator Begin, InstructionIterator End,
                             uint64_t Address, const MCInst &CurInst,
                             MCInst *&TargetHiBits, MCInst *&TargetLowBits,
                             uint64_t &Target) const override {
    if (CurInst.getOpcode() != AArch64::BR || !CurInst.getOperand(0).isReg() ||
        CurInst.getOperand(0).getReg() != AArch64::X16)
      return 0;

    auto I = End;
    if (I == Begin)
      return 0;

    --I;
    Address -= 4;
    if (I == Begin || I->getOpcode() != AArch64::ADDXri ||
        MCPlus::getNumPrimeOperands(*I) < 3 || !I->getOperand(0).isReg() ||
        !I->getOperand(1).isReg() ||
        I->getOperand(0).getReg() != AArch64::X16 ||
        I->getOperand(1).getReg() != AArch64::X16 || !I->getOperand(2).isImm())
      return 0;
    TargetLowBits = &*I;
    uint64_t Addr = I->getOperand(2).getImm() & 0xFFF;

    --I;
    Address -= 4;
    if (I->getOpcode() != AArch64::ADRP ||
        MCPlus::getNumPrimeOperands(*I) < 2 || !I->getOperand(0).isReg() ||
        !I->getOperand(1).isImm() || I->getOperand(0).getReg() != AArch64::X16)
      return 0;
    TargetHiBits = &*I;
    Addr |= (Address + ((int64_t)I->getOperand(1).getImm() << 12)) &
            0xFFFFFFFFFFFFF000ULL;
    Target = Addr;
    return 3;
  }

  /// Match the following pattern:
  ///
  ///   LDR x16, .L1
  ///   BR  x16
  /// L1:
  ///   .quad Target
  ///
  /// Populate \p TargetAddress with the Target value on successful match.
  bool matchAbsLongVeneer(const BinaryFunction &BF,
                          uint64_t &TargetAddress) const override {
    if (BF.size() != 1 || BF.getMaxSize() < 16)
      return false;

    if (!BF.hasConstantIsland())
      return false;

    const BinaryBasicBlock &BB = BF.front();
    if (BB.size() != 2)
      return false;

    const MCInst &LDRInst = BB.getInstructionAtIndex(0);
    if (LDRInst.getOpcode() != AArch64::LDRXl)
      return false;

    if (!LDRInst.getOperand(0).isReg() ||
        LDRInst.getOperand(0).getReg() != AArch64::X16)
      return false;

    const MCSymbol *TargetSym = getTargetSymbol(LDRInst, 1);
    if (!TargetSym)
      return false;

    const MCInst &BRInst = BB.getInstructionAtIndex(1);
    if (BRInst.getOpcode() != AArch64::BR)
      return false;
    if (!BRInst.getOperand(0).isReg() ||
        BRInst.getOperand(0).getReg() != AArch64::X16)
      return false;

    const BinaryFunction::IslandInfo &IInfo = BF.getIslandInfo();
    if (IInfo.HasDynamicRelocations)
      return false;

    auto Iter = IInfo.Offsets.find(8);
    if (Iter == IInfo.Offsets.end() || Iter->second != TargetSym)
      return false;

    // Extract the absolute value stored inside the island.
    StringRef SectionContents = BF.getOriginSection()->getContents();
    StringRef FunctionContents = SectionContents.substr(
        BF.getAddress() - BF.getOriginSection()->getAddress(), BF.getMaxSize());

    const BinaryContext &BC = BF.getBinaryContext();
    DataExtractor DE(FunctionContents, BC.AsmInfo->isLittleEndian(),
                     BC.AsmInfo->getCodePointerSize());
    uint64_t Offset = 8;
    TargetAddress = DE.getAddress(&Offset);

    return true;
  }

  bool matchAdrpAddPair(const MCInst &Adrp, const MCInst &Add) const override {
    if (!isADRP(Adrp) || !isAddXri(Add))
      return false;

    assert(Adrp.getOperand(0).isReg() &&
           "Unexpected operand in ADRP instruction");
    MCPhysReg AdrpReg = Adrp.getOperand(0).getReg();
    assert(Add.getOperand(1).isReg() &&
           "Unexpected operand in ADDXri instruction");
    MCPhysReg AddReg = Add.getOperand(1).getReg();
    return AdrpReg == AddReg;
  }

  bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                               int64_t Addend, MCContext *Ctx, int64_t &Value,
                               uint32_t RelType) const override {
    unsigned ImmOpNo = -1U;
    for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
         ++Index) {
      if (Inst.getOperand(Index).isImm()) {
        ImmOpNo = Index;
        break;
      }
    }
    if (ImmOpNo == -1U)
      return false;

    Value = Inst.getOperand(ImmOpNo).getImm();

    setOperandToSymbolRef(Inst, ImmOpNo, Symbol, Addend, Ctx, RelType);

    return true;
  }

  void createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(AArch64::B);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        getTargetExprFor(Inst, MCSymbolRefExpr::create(TBB, *Ctx), *Ctx, 0)));
  }

  bool shouldRecordCodeRelocation(uint32_t RelType) const override {
    switch (RelType) {
    case ELF::R_AARCH64_ABS64:
    case ELF::R_AARCH64_ABS32:
    case ELF::R_AARCH64_ABS16:
    case ELF::R_AARCH64_ADD_ABS_LO12_NC:
    case ELF::R_AARCH64_ADR_GOT_PAGE:
    case ELF::R_AARCH64_ADR_PREL_LO21:
    case ELF::R_AARCH64_ADR_PREL_PG_HI21:
    case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC:
    case ELF::R_AARCH64_LD64_GOT_LO12_NC:
    case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
    case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
    case ELF::R_AARCH64_TLSDESC_ADD_LO12:
    case ELF::R_AARCH64_TLSDESC_ADR_PAGE21:
    case ELF::R_AARCH64_TLSDESC_ADR_PREL21:
    case ELF::R_AARCH64_TLSDESC_LD64_LO12:
    case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
    case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0:
    case ELF::R_AARCH64_TLSLE_MOVW_TPREL_G0_NC:
    case ELF::R_AARCH64_MOVW_UABS_G0:
    case ELF::R_AARCH64_MOVW_UABS_G0_NC:
    case ELF::R_AARCH64_MOVW_UABS_G1:
    case ELF::R_AARCH64_MOVW_UABS_G1_NC:
    case ELF::R_AARCH64_MOVW_UABS_G2:
    case ELF::R_AARCH64_MOVW_UABS_G2_NC:
    case ELF::R_AARCH64_MOVW_UABS_G3:
    case ELF::R_AARCH64_PREL16:
    case ELF::R_AARCH64_PREL32:
    case ELF::R_AARCH64_PREL64:
      return true;
    case ELF::R_AARCH64_CALL26:
    case ELF::R_AARCH64_JUMP26:
    case ELF::R_AARCH64_TSTBR14:
    case ELF::R_AARCH64_CONDBR19:
    case ELF::R_AARCH64_TLSDESC_CALL:
    case ELF::R_AARCH64_TLSLE_ADD_TPREL_HI12:
    case ELF::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC:
      return false;
    default:
      llvm_unreachable("Unexpected AArch64 relocation type in code");
    }
  }

  StringRef getTrapFillValue() const override {
    return StringRef("\0\0\0\0", 4);
  }

  void createReturn(MCInst &Inst) const override {
    Inst.setOpcode(AArch64::RET);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(AArch64::LR));
  }

  void createStackPointerIncrement(
      MCInst &Inst, int Size,
      bool NoFlagsClobber = false /*unused for AArch64*/) const override {
    Inst.setOpcode(AArch64::SUBXri);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(AArch64::SP));
    Inst.addOperand(MCOperand::createReg(AArch64::SP));
    Inst.addOperand(MCOperand::createImm(Size));
    Inst.addOperand(MCOperand::createImm(0));
  }

  void createStackPointerDecrement(
      MCInst &Inst, int Size,
      bool NoFlagsClobber = false /*unused for AArch64*/) const override {
    Inst.setOpcode(AArch64::ADDXri);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(AArch64::SP));
    Inst.addOperand(MCOperand::createReg(AArch64::SP));
    Inst.addOperand(MCOperand::createImm(Size));
    Inst.addOperand(MCOperand::createImm(0));
  }

  void createIndirectBranch(MCInst &Inst, MCPhysReg MemBaseReg,
                            int64_t Disp) const {
    Inst.setOpcode(AArch64::BR);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(MemBaseReg));
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    InstructionListType Insts(5);
    // Code sequence for instrumented indirect call handler:
    //   msr  nzcv, x1
    //   ldp  x0, x1, [sp], #16
    //   ldr  x16, [sp], #16
    //   ldp  x0, x1, [sp], #16
    //   br   x16
    setSystemFlag(Insts[0], AArch64::X1);
    createPopRegisters(Insts[1], AArch64::X0, AArch64::X1);
    // Here we load address of the next function which should be called in the
    // original binary to X16 register. Writing to X16 is permitted without
    // needing to restore.
    loadReg(Insts[2], AArch64::X16, AArch64::SP);
    createPopRegisters(Insts[3], AArch64::X0, AArch64::X1);
    createIndirectBranch(Insts[4], AArch64::X16, 0);
    return Insts;
  }

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    return createInstrumentedIndCallHandlerExitBB();
  }

  InstructionListType createGetter(MCContext *Ctx, const char *name) const {
    InstructionListType Insts(4);
    MCSymbol *Locs = Ctx->getOrCreateSymbol(name);
    InstructionListType Addr = materializeAddress(Locs, Ctx, AArch64::X0);
    std::copy(Addr.begin(), Addr.end(), Insts.begin());
    assert(Addr.size() == 2 && "Invalid Addr size");
    loadReg(Insts[2], AArch64::X0, AArch64::X0);
    createReturn(Insts[3]);
    return Insts;
  }

  InstructionListType createNumCountersGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_num_counters");
  }

  InstructionListType
  createInstrLocationsGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_locations");
  }

  InstructionListType createInstrTablesGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_tables");
  }

  InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_num_funcs");
  }

  void convertIndirectCallToLoad(MCInst &Inst, MCPhysReg Reg) override {
    bool IsTailCall = isTailCall(Inst);
    if (IsTailCall)
      removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (Inst.getOpcode() == AArch64::BR || Inst.getOpcode() == AArch64::BLR) {
      Inst.setOpcode(AArch64::ORRXrs);
      Inst.insert(Inst.begin(), MCOperand::createReg(Reg));
      Inst.insert(Inst.begin() + 1, MCOperand::createReg(AArch64::XZR));
      Inst.insert(Inst.begin() + 3, MCOperand::createImm(0));
      return;
    }
    llvm_unreachable("not implemented");
  }

  InstructionListType createLoadImmediate(const MCPhysReg Dest,
                                          uint64_t Imm) const override {
    InstructionListType Insts(4);
    int Shift = 48;
    for (int I = 0; I < 4; I++, Shift -= 16) {
      Insts[I].setOpcode(AArch64::MOVKXi);
      Insts[I].addOperand(MCOperand::createReg(Dest));
      Insts[I].addOperand(MCOperand::createReg(Dest));
      Insts[I].addOperand(MCOperand::createImm((Imm >> Shift) & 0xFFFF));
      Insts[I].addOperand(MCOperand::createImm(Shift));
    }
    return Insts;
  }

  void createIndirectCallInst(MCInst &Inst, bool IsTailCall,
                              MCPhysReg Reg) const {
    Inst.clear();
    Inst.setOpcode(IsTailCall ? AArch64::BR : AArch64::BLR);
    Inst.addOperand(MCOperand::createReg(Reg));
  }

  InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
                                                     MCSymbol *HandlerFuncAddr,
                                                     int CallSiteID,
                                                     MCContext *Ctx) override {
    InstructionListType Insts;
    // Code sequence used to enter indirect call instrumentation helper:
    //   stp x0, x1, [sp, #-16]! createPushRegisters
    //   mov target x0  convertIndirectCallToLoad -> orr x0 target xzr
    //   mov x1 CallSiteID createLoadImmediate ->
    //   movk    x1, #0x0, lsl #48
    //   movk    x1, #0x0, lsl #32
    //   movk    x1, #0x0, lsl #16
    //   movk    x1, #0x0
    //   stp x0, x1, [sp, #-16]!
    //   bl *HandlerFuncAddr createIndirectCall ->
    //   adr x0 *HandlerFuncAddr -> adrp + add
    //   blr x0
    Insts.emplace_back();
    createPushRegisters(Insts.back(), AArch64::X0, AArch64::X1);
    Insts.emplace_back(CallInst);
    convertIndirectCallToLoad(Insts.back(), AArch64::X0);
    InstructionListType LoadImm =
        createLoadImmediate(getIntArgRegister(1), CallSiteID);
    Insts.insert(Insts.end(), LoadImm.begin(), LoadImm.end());
    Insts.emplace_back();
    createPushRegisters(Insts.back(), AArch64::X0, AArch64::X1);
    Insts.resize(Insts.size() + 2);
    InstructionListType Addr =
        materializeAddress(HandlerFuncAddr, Ctx, AArch64::X0);
    assert(Addr.size() == 2 && "Invalid Addr size");
    std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());
    Insts.emplace_back();
    createIndirectCallInst(Insts.back(), isTailCall(CallInst), AArch64::X0);

    // Carry over metadata including tail call marker if present.
    stripAnnotations(Insts.back());
    moveAnnotations(std::move(CallInst), Insts.back());

    return Insts;
  }

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) override {
    // Code sequence used to check whether InstrTampoline was initialized
    // and call it if so, returns via IndCallHandler
    //   stp     x0, x1, [sp, #-16]!
    //   mrs     x1, nzcv
    //   adr     x0, InstrTrampoline -> adrp + add
    //   ldr     x0, [x0]
    //   subs    x0, x0, #0x0
    //   b.eq    IndCallHandler
    //   str     x30, [sp, #-16]!
    //   blr     x0
    //   ldr     x30, [sp], #16
    //   b       IndCallHandler
    InstructionListType Insts;
    Insts.emplace_back();
    createPushRegisters(Insts.back(), AArch64::X0, AArch64::X1);
    Insts.emplace_back();
    getSystemFlag(Insts.back(), getIntArgRegister(1));
    Insts.emplace_back();
    Insts.emplace_back();
    InstructionListType Addr =
        materializeAddress(InstrTrampoline, Ctx, AArch64::X0);
    std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());
    assert(Addr.size() == 2 && "Invalid Addr size");
    Insts.emplace_back();
    loadReg(Insts.back(), AArch64::X0, AArch64::X0);
    InstructionListType cmpJmp =
        createCmpJE(AArch64::X0, 0, IndCallHandler, Ctx);
    Insts.insert(Insts.end(), cmpJmp.begin(), cmpJmp.end());
    Insts.emplace_back();
    storeReg(Insts.back(), AArch64::LR, AArch64::SP);
    Insts.emplace_back();
    Insts.back().setOpcode(AArch64::BLR);
    Insts.back().addOperand(MCOperand::createReg(AArch64::X0));
    Insts.emplace_back();
    loadReg(Insts.back(), AArch64::LR, AArch64::SP);
    Insts.emplace_back();
    createDirectCall(Insts.back(), IndCallHandler, Ctx, /*IsTailCall*/ true);
    return Insts;
  }

  // Instrumentation code sequence using LSE atomic instruction has a total of
  // 6 instructions:
  //
  //     stp    x0, x1, [sp, #-0x10]!
  //     adrp   x0, page_address(counter)
  //     add    x0, x0, page_offset(counter)
  //     mov    x1, #0x1
  //     stadd  x1, [x0]
  //     ldp    x0, x1, [sp], #0x10
  //
  // Instrumentation code sequence without using LSE atomic instruction has
  // 8 instructions at instrumentation place, with 6 instructions in the helper:
  //
  //     stp    x0, x30, [sp, #-0x10]!
  //     stp    x1, x2, [sp, #-0x10]!
  //     adrp   x0, page_address(counter)
  //     add    x0, x0, page_offset(counter)
  //     adrp   x1, page_address(helper)
  //     add    x1, x1, page_offset(helper)
  //     blr    x1
  //     ldp    x0, x30, [sp], #0x10
  //
  //   <helper>:
  //     ldaxr  x1, [x0]
  //     add    x1, x1, #0x1
  //     stlxr  w2, x1, [x0]
  //     cbnz   w2, <helper>
  //     ldp    x1, x2, [sp], #0x10
  //     ret

  void createInstrCounterIncrFunc(BinaryContext &BC) override {
    assert(InstrCounterIncrFunc == nullptr &&
           "helper function of counter increment for instrumentation "
           "has already been created");

    if (!opts::NoLSEAtomics)
      return;

    MCContext *Ctx = BC.Ctx.get();
    InstrCounterIncrFunc = BC.createInjectedBinaryFunction(
        "__bolt_instr_counter_incr", /*IsSimple*/ false);
    std::vector<std::unique_ptr<BinaryBasicBlock>> BBs;

    BBs.emplace_back(InstrCounterIncrFunc->createBasicBlock());
    InstructionListType Instrs(4);
    Instrs[0].setOpcode(AArch64::LDAXRX);
    Instrs[0].clear();
    Instrs[0].addOperand(MCOperand::createReg(AArch64::X1));
    Instrs[0].addOperand(MCOperand::createReg(AArch64::X0));
    Instrs[1].setOpcode(AArch64::ADDXri);
    Instrs[1].clear();
    Instrs[1].addOperand(MCOperand::createReg(AArch64::X1));
    Instrs[1].addOperand(MCOperand::createReg(AArch64::X1));
    Instrs[1].addOperand(MCOperand::createImm(1));
    Instrs[1].addOperand(MCOperand::createImm(0));
    Instrs[2].setOpcode(AArch64::STLXRX);
    Instrs[2].clear();
    Instrs[2].addOperand(MCOperand::createReg(AArch64::W2));
    Instrs[2].addOperand(MCOperand::createReg(AArch64::X1));
    Instrs[2].addOperand(MCOperand::createReg(AArch64::X0));
    Instrs[3].setOpcode(AArch64::CBNZW);
    Instrs[3].clear();
    Instrs[3].addOperand(MCOperand::createReg(AArch64::W2));
    Instrs[3].addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(BBs.back()->getLabel(), *Ctx)));
    BBs.back()->addInstructions(Instrs.begin(), Instrs.end());
    BBs.back()->setCFIState(0);

    BBs.emplace_back(InstrCounterIncrFunc->createBasicBlock());
    InstructionListType InstrsEpilog(2);
    createPopRegisters(InstrsEpilog[0], AArch64::X1, AArch64::X2);
    createReturn(InstrsEpilog[1]);
    BBs.back()->addInstructions(InstrsEpilog.begin(), InstrsEpilog.end());
    BBs.back()->setCFIState(0);

    BBs[0]->addSuccessor(BBs[0].get());
    BBs[0]->addSuccessor(BBs[1].get());

    InstrCounterIncrFunc->insertBasicBlocks(nullptr, std::move(BBs),
                                            /*UpdateLayout*/ true,
                                            /*UpdateCFIState*/ false);
    InstrCounterIncrFunc->updateState(BinaryFunction::State::CFG_Finalized);

    LLVM_DEBUG({
      dbgs() << "BOLT-DEBUG: instrumentation counter increment helper:\n";
      InstrCounterIncrFunc->dump();
    });
  }

  InstructionListType createInstrIncMemory(const MCSymbol *Target,
                                           MCContext *Ctx, bool IsLeaf,
                                           unsigned CodePointerSize) override {
    unsigned int I = 0;
    InstructionListType Instrs(opts::NoLSEAtomics ? 8 : 6);

    if (opts::NoLSEAtomics) {
      createPushRegisters(Instrs[I++], AArch64::X0, AArch64::LR);
      createPushRegisters(Instrs[I++], AArch64::X1, AArch64::X2);
    } else {
      createPushRegisters(Instrs[I++], AArch64::X0, AArch64::X1);
    }

    InstructionListType Addr = materializeAddress(Target, Ctx, AArch64::X0);
    assert(Addr.size() == 2 && "Invalid Addr size");
    std::copy(Addr.begin(), Addr.end(), Instrs.begin() + I);
    I += Addr.size();

    if (opts::NoLSEAtomics) {
      const MCSymbol *Helper = InstrCounterIncrFunc->getSymbol();
      InstructionListType HelperAddr =
          materializeAddress(Helper, Ctx, AArch64::X1);
      assert(HelperAddr.size() == 2 && "Invalid HelperAddr size");
      std::copy(HelperAddr.begin(), HelperAddr.end(), Instrs.begin() + I);
      I += HelperAddr.size();
      createIndirectCallInst(Instrs[I++], /*IsTailCall*/ false, AArch64::X1);
    } else {
      InstructionListType Insts = createIncMemory(AArch64::X0, AArch64::X1);
      assert(Insts.size() == 2 && "Invalid Insts size");
      std::copy(Insts.begin(), Insts.end(), Instrs.begin() + I);
      I += Insts.size();
    }
    createPopRegisters(Instrs[I++], AArch64::X0,
                       opts::NoLSEAtomics ? AArch64::LR : AArch64::X1);
    return Instrs;
  }

  std::vector<MCInst> createSymbolTrampoline(const MCSymbol *TgtSym,
                                             MCContext *Ctx) override {
    std::vector<MCInst> Insts;
    createShortJmp(Insts, TgtSym, Ctx, /*IsTailCall*/ true);
    return Insts;
  }

  InstructionListType materializeAddress(const MCSymbol *Target, MCContext *Ctx,
                                         MCPhysReg RegName,
                                         int64_t Addend = 0) const override {
    // Get page-aligned address and add page offset
    InstructionListType Insts(2);
    Insts[0].setOpcode(AArch64::ADRP);
    Insts[0].clear();
    Insts[0].addOperand(MCOperand::createReg(RegName));
    Insts[0].addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(Insts[0], /* OpNum */ 1, Target, Addend, Ctx,
                          ELF::R_AARCH64_NONE);
    Insts[1].setOpcode(AArch64::ADDXri);
    Insts[1].clear();
    Insts[1].addOperand(MCOperand::createReg(RegName));
    Insts[1].addOperand(MCOperand::createReg(RegName));
    Insts[1].addOperand(MCOperand::createImm(0));
    Insts[1].addOperand(MCOperand::createImm(0));
    setOperandToSymbolRef(Insts[1], /* OpNum */ 2, Target, Addend, Ctx,
                          ELF::R_AARCH64_ADD_ABS_LO12_NC);
    return Insts;
  }

  std::optional<Relocation>
  createRelocation(const MCFixup &Fixup,
                   const MCAsmBackend &MAB) const override {
    MCFixupKindInfo FKI = MAB.getFixupKindInfo(Fixup.getKind());

    assert(FKI.TargetOffset == 0 && "0-bit relocation offset expected");
    const uint64_t RelOffset = Fixup.getOffset();

    uint32_t RelType;
    if (Fixup.getKind() == MCFixupKind(AArch64::fixup_aarch64_pcrel_call26))
      RelType = ELF::R_AARCH64_CALL26;
    else if (Fixup.getKind() ==
             MCFixupKind(AArch64::fixup_aarch64_pcrel_branch26))
      RelType = ELF::R_AARCH64_JUMP26;
    else if (Fixup.isPCRel()) {
      switch (FKI.TargetSize) {
      default:
        return std::nullopt;
      case 16:
        RelType = ELF::R_AARCH64_PREL16;
        break;
      case 32:
        RelType = ELF::R_AARCH64_PREL32;
        break;
      case 64:
        RelType = ELF::R_AARCH64_PREL64;
        break;
      }
    } else {
      switch (FKI.TargetSize) {
      default:
        return std::nullopt;
      case 16:
        RelType = ELF::R_AARCH64_ABS16;
        break;
      case 32:
        RelType = ELF::R_AARCH64_ABS32;
        break;
      case 64:
        RelType = ELF::R_AARCH64_ABS64;
        break;
      }
    }

    auto [RelSymbol, RelAddend] = extractFixupExpr(Fixup);

    return Relocation({RelOffset, RelSymbol, RelType, RelAddend, 0});
  }

  uint16_t getMinFunctionAlignment() const override { return 4; }

  std::optional<uint32_t>
  getInstructionSize(const MCInst &Inst) const override {
    return 4;
  }

  std::optional<uint64_t>
  extractMoveImmediate(const MCInst &Inst, MCPhysReg TargetReg) const override {
    // Match MOVZ instructions (both X and W register variants) with no shift.
    if ((Inst.getOpcode() == AArch64::MOVZXi ||
         Inst.getOpcode() == AArch64::MOVZWi) &&
        Inst.getOperand(2).getImm() == 0 &&
        getAliases(TargetReg)[Inst.getOperand(0).getReg()])
      return Inst.getOperand(1).getImm();
    return std::nullopt;
  }

  std::optional<uint64_t>
  findMemcpySizeInBytes(const BinaryBasicBlock &BB,
                        BinaryBasicBlock::iterator CallInst) const override {
    MCPhysReg SizeReg = getIntArgRegister(2);
    if (SizeReg == getNoRegister())
      return std::nullopt;

    BitVector WrittenRegs(RegInfo->getNumRegs());
    const BitVector &SizeRegAliases = getAliases(SizeReg);

    for (auto InstIt = BB.begin(); InstIt != CallInst; ++InstIt) {
      const MCInst &Inst = *InstIt;
      WrittenRegs.reset();
      getWrittenRegs(Inst, WrittenRegs);

      if (WrittenRegs.anyCommon(SizeRegAliases))
        return extractMoveImmediate(Inst, SizeReg);
    }
    return std::nullopt;
  }

  InstructionListType
  createInlineMemcpy(bool ReturnEnd,
                     std::optional<uint64_t> KnownSize) const override {
    assert(KnownSize.has_value() &&
           "AArch64 memcpy inlining requires known size");
    InstructionListType Code;
    uint64_t Size = *KnownSize;

    generateSizeSpecificMemcpy(Code, Size);

    // If _memcpy8, adjust X0 to return dest+size instead of dest.
    if (ReturnEnd)
      Code.emplace_back(MCInstBuilder(AArch64::ADDXri)
                            .addReg(AArch64::X0)
                            .addReg(AArch64::X0)
                            .addImm(Size)
                            .addImm(0));
    return Code;
  }

  InstructionListType generateSizeSpecificMemcpy(InstructionListType &Code,
                                                 uint64_t Size) const {
    auto AddLoadStorePair = [&](unsigned LoadOpc, unsigned StoreOpc,
                                unsigned Reg, unsigned Offset = 0) {
      Code.emplace_back(MCInstBuilder(LoadOpc)
                            .addReg(Reg)
                            .addReg(AArch64::X1)
                            .addImm(Offset));
      Code.emplace_back(MCInstBuilder(StoreOpc)
                            .addReg(Reg)
                            .addReg(AArch64::X0)
                            .addImm(Offset));
    };

    // Generate optimal instruction sequences based on exact size.
    switch (Size) {
    case 1:
      AddLoadStorePair(AArch64::LDRBBui, AArch64::STRBBui, AArch64::W9);
      break;
    case 2:
      AddLoadStorePair(AArch64::LDRHHui, AArch64::STRHHui, AArch64::W9);
      break;
    case 4:
      AddLoadStorePair(AArch64::LDRWui, AArch64::STRWui, AArch64::W9);
      break;
    case 8:
      AddLoadStorePair(AArch64::LDRXui, AArch64::STRXui, AArch64::X9);
      break;
    case 16:
      AddLoadStorePair(AArch64::LDRQui, AArch64::STRQui, AArch64::Q16);
      break;
    case 32:
      AddLoadStorePair(AArch64::LDRQui, AArch64::STRQui, AArch64::Q16, 0);
      AddLoadStorePair(AArch64::LDRQui, AArch64::STRQui, AArch64::Q17, 1);
      break;

    default:
      // For sizes up to 64 bytes, greedily use the largest possible loads.
      // Caller should have already filtered out sizes > 64 bytes.
      assert(Size <= 64 &&
             "Size should be <= 64 bytes for AArch64 memcpy inlining");

      uint64_t Remaining = Size;
      uint64_t Offset = 0;

      const std::array<std::tuple<uint64_t, unsigned, unsigned, unsigned>, 5>
          LoadStoreOps = {
              {{16, AArch64::LDRQui, AArch64::STRQui, AArch64::Q16},
               {8, AArch64::LDRXui, AArch64::STRXui, AArch64::X9},
               {4, AArch64::LDRWui, AArch64::STRWui, AArch64::W9},
               {2, AArch64::LDRHHui, AArch64::STRHHui, AArch64::W9},
               {1, AArch64::LDRBBui, AArch64::STRBBui, AArch64::W9}}};

      for (const auto &[OpSize, LoadOp, StoreOp, TempReg] : LoadStoreOps)
        while (Remaining >= OpSize) {
          AddLoadStorePair(LoadOp, StoreOp, TempReg, Offset / OpSize);
          Remaining -= OpSize;
          Offset += OpSize;
        }
      break;
    }
    return Code;
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createAArch64MCPlusBuilder(const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo,
                                          const MCSubtargetInfo *STI) {
  return new AArch64MCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
