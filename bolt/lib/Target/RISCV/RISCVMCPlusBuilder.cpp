//===- bolt/Target/RISCV/RISCVMCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class RISCVMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  bool equals(const MCSpecifierExpr &A, const MCSpecifierExpr &B,
              CompFuncTy Comp) const override {
    const auto &RISCVExprA = cast<MCSpecifierExpr>(A);
    const auto &RISCVExprB = cast<MCSpecifierExpr>(B);
    if (RISCVExprA.getSpecifier() != RISCVExprB.getSpecifier())
      return false;

    return MCPlusBuilder::equals(*RISCVExprA.getSubExpr(),
                                 *RISCVExprB.getSubExpr(), Comp);
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    Regs |= getAliases(RISCV::X2);
    Regs |= getAliases(RISCV::X8);
    Regs |= getAliases(RISCV::X9);
    Regs |= getAliases(RISCV::X18);
    Regs |= getAliases(RISCV::X19);
    Regs |= getAliases(RISCV::X20);
    Regs |= getAliases(RISCV::X21);
    Regs |= getAliases(RISCV::X22);
    Regs |= getAliases(RISCV::X23);
    Regs |= getAliases(RISCV::X24);
    Regs |= getAliases(RISCV::X25);
    Regs |= getAliases(RISCV::X26);
    Regs |= getAliases(RISCV::X27);
  }

  bool shouldRecordCodeRelocation(uint32_t RelType) const override {
    switch (RelType) {
    case ELF::R_RISCV_JAL:
    case ELF::R_RISCV_CALL:
    case ELF::R_RISCV_CALL_PLT:
    case ELF::R_RISCV_BRANCH:
    case ELF::R_RISCV_RVC_BRANCH:
    case ELF::R_RISCV_RVC_JUMP:
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_PCREL_HI20:
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
    case ELF::R_RISCV_HI20:
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
    case ELF::R_RISCV_TLS_GOT_HI20:
    case ELF::R_RISCV_TLS_GD_HI20:
      return true;
    default:
      llvm_unreachable("Unexpected RISCV relocation type in code");
    }
  }

  bool isNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::ADDI &&
           Inst.getOperand(0).getReg() == RISCV::X0 &&
           Inst.getOperand(1).getReg() == RISCV::X0 &&
           Inst.getOperand(2).getImm() == 0;
  }

  bool isCNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::C_NOP;
  }

  bool isNoop(const MCInst &Inst) const override {
    return isNop(Inst) || isCNop(Inst);
  }

  bool isPseudo(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return MCPlusBuilder::isPseudo(Inst);
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
      return false;
    }
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    if (!isCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JALR:
    case RISCV::C_JALR:
    case RISCV::C_JR:
      return true;
    }
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JAL:
    case RISCV::AUIPC:
      return true;
    }
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case RISCV::BEQ:
      return RISCV::BNE;
    case RISCV::BNE:
      return RISCV::BEQ;
    case RISCV::BLT:
      return RISCV::BGE;
    case RISCV::BGE:
      return RISCV::BLT;
    case RISCV::BLTU:
      return RISCV::BGEU;
    case RISCV::BGEU:
      return RISCV::BLTU;
    case RISCV::C_BEQZ:
      return RISCV::C_BNEZ;
    case RISCV::C_BNEZ:
      return RISCV::C_BEQZ;
    }
  }

  void reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    auto Opcode = getInvertedBranchOpcode(Inst.getOpcode());
    Inst.setOpcode(Opcode);
    replaceBranchTarget(Inst, TBB, Ctx);
  }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");

    unsigned SymOpIndex;
    auto Result = getSymbolRefOperandNum(Inst, SymOpIndex);
    (void)Result;
    assert(Result && "unimplemented branch");

    Inst.getOperand(SymOpIndex) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      MCInst *&PCRelBaseOut, MCInst *&FixedEntryLoadInst) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    PCRelBaseOut = nullptr;
    FixedEntryLoadInst = nullptr;

    // Check for the following long tail call sequence:
    // 1: auipc xi, %pcrel_hi(sym)
    // jalr zero, %pcrel_lo(1b)(xi)
    if (Instruction.getOpcode() == RISCV::JALR && Begin != End) {
      MCInst &PrevInst = *std::prev(End);
      if (isRISCVCall(PrevInst, Instruction) &&
          Instruction.getOperand(0).getReg() == RISCV::X0)
        return IndirectBranchType::POSSIBLE_TAIL_CALL;
    }

    return IndirectBranchType::UNKNOWN;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unsupported tail call opcode");
    case RISCV::JAL:
    case RISCV::JALR:
    case RISCV::C_J:
    case RISCV::C_JR:
      break;
    }

    setTailCall(Inst);
    return true;
  }

  void createReturn(MCInst &Inst) const override {
    // TODO "c.jr ra" when RVC is enabled
    Inst.setOpcode(RISCV::JALR);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createReg(RISCV::X1));
    Inst.addOperand(MCOperand::createImm(0));
  }

  void createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(RISCV::JAL);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx)));
  }

  StringRef getTrapFillValue() const override {
    return StringRef("\0\0\0\0", 4);
  }

  void createCall(unsigned Opcode, MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) {
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(MCSpecifierExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        ELF::R_RISCV_CALL_PLT, *Ctx)));
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    return createCall(RISCV::PseudoCALL, Inst, Target, Ctx);
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    createShortJmp(Seq, Target, Ctx, /*IsTailCall*/ true);
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createCall(RISCV::PseudoTAIL, Inst, Target, Ctx);
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

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::C_J:
      OpNum = 0;
      return true;
    case RISCV::AUIPC:
    case RISCV::JAL:
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
      OpNum = 1;
      return true;
    case RISCV::BEQ:
    case RISCV::BGE:
    case RISCV::BGEU:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BLTU:
      OpNum = 2;
      return true;
    }
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *RISCVExpr = dyn_cast<MCSpecifierExpr>(Expr);
    if (RISCVExpr && RISCVExpr->getSubExpr())
      return getTargetSymbol(RISCVExpr->getSubExpr());

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

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override {
    auto I = Begin;

    assert(I != End);
    auto &AUIPC = *I++;
    assert(AUIPC.getOpcode() == RISCV::AUIPC);
    assert(AUIPC.getOperand(0).getReg() == RISCV::X28);

    assert(I != End);
    auto &LD = *I++;
    assert(LD.getOpcode() == RISCV::LD);
    assert(LD.getOperand(0).getReg() == RISCV::X28);
    assert(LD.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &JALR = *I++;
    (void)JALR;
    assert(JALR.getOpcode() == RISCV::JALR);
    assert(JALR.getOperand(0).getReg() == RISCV::X6);
    assert(JALR.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &NOP = *I++;
    (void)NOP;
    assert(isNoop(NOP));

    assert(I == End);

    auto AUIPCOffset = AUIPC.getOperand(1).getImm() << 12;
    auto LDOffset = LD.getOperand(2).getImm();
    return BeginPC + AUIPCOffset + LDOffset;
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

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint32_t RelType) const override {
    switch (RelType) {
    default:
      return Expr;
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_TLS_GOT_HI20:
    case ELF::R_RISCV_TLS_GD_HI20:
      // The GOT is reused so no need to create GOT relocations
    case ELF::R_RISCV_PCREL_HI20:
      return MCSpecifierExpr::create(Expr, ELF::R_RISCV_PCREL_HI20, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return MCSpecifierExpr::create(Expr, RISCV::S_PCREL_LO, Ctx);
    case ELF::R_RISCV_HI20:
      return MCSpecifierExpr::create(Expr, ELF::R_RISCV_HI20, Ctx);
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
      return MCSpecifierExpr::create(Expr, RISCV::S_LO, Ctx);
    case ELF::R_RISCV_CALL:
      return MCSpecifierExpr::create(Expr, ELF::R_RISCV_CALL_PLT, Ctx);
    case ELF::R_RISCV_CALL_PLT:
      return MCSpecifierExpr::create(Expr, ELF::R_RISCV_CALL_PLT, Ctx);
    }
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    return false;
  }

  bool isCallAuipc(const MCInst &Inst) const {
    if (Inst.getOpcode() != RISCV::AUIPC)
      return false;

    const auto &ImmOp = Inst.getOperand(1);
    if (!ImmOp.isExpr())
      return false;

    const auto *ImmExpr = ImmOp.getExpr();
    if (!isa<MCSpecifierExpr>(ImmExpr))
      return false;

    switch (cast<MCSpecifierExpr>(ImmExpr)->getSpecifier()) {
    default:
      return false;
    case ELF::R_RISCV_CALL_PLT:
      return true;
    }
  }

  bool isRISCVCall(const MCInst &First, const MCInst &Second) const override {
    if (!isCallAuipc(First))
      return false;

    assert(Second.getOpcode() == RISCV::JALR);
    return true;
  }

  uint16_t getMinFunctionAlignment() const override {
    if (STI->hasFeature(RISCV::FeatureStdExtC) ||
        STI->hasFeature(RISCV::FeatureStdExtZca))
      return 2;
    return 4;
  }

  void createStackPointerIncrement(
      MCInst &Inst, int imm,
      bool NoFlagsClobber = false /*unused for RISCV*/) const override {
    Inst = MCInstBuilder(RISCV::ADDI)
               .addReg(RISCV::X2)
               .addReg(RISCV::X2)
               .addImm(-imm);
  }

  void createStackPointerDecrement(
      MCInst &Inst, int imm,
      bool NoFlagsClobber = false /*unused for RISCV*/) const override {
    Inst = MCInstBuilder(RISCV::ADDI)
               .addReg(RISCV::X2)
               .addReg(RISCV::X2)
               .addImm(imm);
  }

  void loadReg(MCInst &Inst, MCPhysReg To, MCPhysReg From,
               int64_t offset) const {
    Inst = MCInstBuilder(RISCV::LD).addReg(To).addReg(From).addImm(offset);
  }

  void storeReg(MCInst &Inst, MCPhysReg From, MCPhysReg To,
                int64_t offset) const {
    Inst = MCInstBuilder(RISCV::SD).addReg(From).addReg(To).addImm(offset);
  }

  void spillRegs(InstructionListType &Insts,
                 const SmallVector<unsigned> &Regs) const {
    Insts.emplace_back();
    createStackPointerIncrement(Insts.back(), Regs.size() * 8);

    int64_t Offset = 0;
    for (auto Reg : Regs) {
      Insts.emplace_back();
      storeReg(Insts.back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }
  }

  void reloadRegs(InstructionListType &Insts,
                  const SmallVector<unsigned> &Regs) const {
    int64_t Offset = 0;
    for (auto Reg : Regs) {
      Insts.emplace_back();
      loadReg(Insts.back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }

    Insts.emplace_back();
    createStackPointerDecrement(Insts.back(), Regs.size() * 8);
  }

  void atomicAdd(MCInst &Inst, MCPhysReg RegAtomic, MCPhysReg RegTo,
                 MCPhysReg RegCnt) const {
    Inst = MCInstBuilder(RISCV::AMOADD_D)
               .addReg(RegAtomic)
               .addReg(RegTo)
               .addReg(RegCnt);
  }

  InstructionListType createRegCmpJE(MCPhysReg RegNo, MCPhysReg RegTmp,
                                     const MCSymbol *Target,
                                     MCContext *Ctx) const {
    InstructionListType Insts;
    Insts.emplace_back(
        MCInstBuilder(RISCV::SUB).addReg(RegTmp).addReg(RegNo).addReg(RegNo));
    Insts.emplace_back(MCInstBuilder(RISCV::BEQ)
                           .addReg(RegNo)
                           .addReg(RegTmp)
                           .addExpr(MCSymbolRefExpr::create(
                               Target, MCSymbolRefExpr::VK_None, *Ctx)));
    return Insts;
  }

  void createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(RISCV::EBREAK);
  }

  void createShortJmp(InstructionListType &Seq, const MCSymbol *Target,
                      MCContext *Ctx, bool IsTailCall) override {
    // The sequence of instructions we create here is the following:
    //  auipc   a5, hi20(Target)
    //  addi    a5, a5, low12(Target)
    //  jr x5 => jalr x0, x5, 0
    MCPhysReg Reg = RISCV::X5;
    InstructionListType Insts = materializeAddress(Target, Ctx, Reg);
    Insts.emplace_back();
    MCInst &Inst = Insts.back();
    Inst.clear();
    Inst = MCInstBuilder(RISCV::JALR).addReg(RISCV::X0).addReg(Reg).addImm(0);
    if (IsTailCall)
      setTailCall(Inst);
    Seq.swap(Insts);
  }

  InstructionListType createGetter(MCContext *Ctx, const char *name) const {
    InstructionListType Insts(4);
    MCSymbol *Locs = Ctx->getOrCreateSymbol(name);
    InstructionListType Addr = materializeAddress(Locs, Ctx, RISCV::X10);
    std::copy(Addr.begin(), Addr.end(), Insts.begin());
    loadReg(Insts[2], RISCV::X10, RISCV::X10, 0);
    createReturn(Insts[3]);
    return Insts;
  }

  InstructionListType createIncMemory(MCPhysReg RegTo, MCPhysReg RegCnt,
                                      MCPhysReg RegAtomic) const {
    InstructionListType Insts;
    Insts.emplace_back();
    Insts.back() =
        MCInstBuilder(RISCV::ADDI).addReg(RegCnt).addReg(RegAtomic).addImm(1);
    Insts.emplace_back();
    atomicAdd(Insts.back(), RegAtomic, RegTo, RegCnt);
    return Insts;
  }

  InstructionListType materializeAddress(const MCSymbol *Target, MCContext *Ctx,
                                         MCPhysReg RegName,
                                         int64_t Addend = 0) const override {
    // Get the symbol address by auipc + addi
    InstructionListType Insts(2);
    MCSymbol *AuipcLabel = Ctx->createNamedTempSymbol("pcrel_hi");
    Insts[0] = MCInstBuilder(RISCV::AUIPC).addReg(RegName).addImm(0);
    setOperandToSymbolRef(Insts[0], /* OpNum */ 1, Target, Addend, Ctx,
                          ELF::R_RISCV_PCREL_HI20);
    setInstLabel(Insts[0], AuipcLabel);

    Insts[1] =
        MCInstBuilder(RISCV::ADDI).addReg(RegName).addReg(RegName).addImm(0);
    setOperandToSymbolRef(Insts[1], /* OpNum */ 2, AuipcLabel, Addend, Ctx,
                          ELF::R_RISCV_PCREL_LO12_I);
    return Insts;
  }

  InstructionListType
  createInstrIncMemory(const MCSymbol *Target, MCContext *Ctx, bool IsLeaf,
                       unsigned CodePointerSize) const override {
    // We need 2 scratch registers: one for the target address (x10), and one
    // for the increment value (x11).
    // addi sp, sp, -16
    // sd x10, 0(sp)
    // sd x11, 8(sp)
    // la x10, target         # 1: auipc x10, %pcrel_hi(target)
    //                        # addi x10, x10, %pcrel_lo(1b)
    // li x11, 1              # addi x11, zero, 1
    // amoadd.d zero, x10, x11
    // ld x10, 0(sp)
    // ld x11, 8(sp)
    // addi sp, sp, 16

    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    InstructionListType Addr = materializeAddress(Target, Ctx, RISCV::X10);
    Insts.insert(Insts.end(), Addr.begin(), Addr.end());
    InstructionListType IncInsts =
        createIncMemory(RISCV::X10, RISCV::X11, RISCV::X0);
    Insts.insert(Insts.end(), IncInsts.begin(), IncInsts.end());
    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    return Insts;
  }

  void createDirectCall(MCInst &Inst, const MCSymbol *Target, MCContext *Ctx,
                        bool IsTailCall) override {
    Inst.setOpcode(RISCV::JAL);
    Inst.clear();
    if (IsTailCall) {
      Inst.addOperand(MCOperand::createReg(RISCV::X0));
      Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
          Inst, MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
          *Ctx, 0)));
      convertJmpToTailCall(Inst);
    } else {
      Inst.addOperand(MCOperand::createReg(RISCV::X1));
      Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
          Inst, MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
          *Ctx, 0)));
    }
  }

  void createIndirectCallInst(MCInst &Inst, bool IsTailCall, MCPhysReg Reg,
                              int64_t Disp) const {
    Inst.clear();
    Inst.setOpcode(RISCV::JALR);
    Inst.clear();
    if (IsTailCall) {
      Inst.addOperand(MCOperand::createReg(RISCV::X0));
      Inst.addOperand(MCOperand::createReg(Reg));
      Inst.addOperand(MCOperand::createImm(Disp));
    } else {
      Inst.addOperand(MCOperand::createReg(RISCV::X1));
      Inst.addOperand(MCOperand::createReg(Reg));
      Inst.addOperand(MCOperand::createImm(Disp));
    }
  }

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) override {
    // Code sequence used to check whether InstrTampoline was initialized
    // and call it if so, returns via IndCallHandler
    //   sp      -16(sp)
    //   sd      x10, 0(sp)
    //   sd      x11, 0(sp)
    //   la      x10, InstrTrampoline -> auipc + addi
    //   ld      x10, [x10]
    //   beq     x10, x11, IndCallHandler
    //   sp      -16(sp)
    //   sd      x1, 0(sp)
    //   jalr    x1,x10,0
    //   ld      x1, [sp], #16
    //   sp      16(sp)
    //   jal     x0, IndCallHandler

    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    InstructionListType Addr =
        materializeAddress(InstrTrampoline, Ctx, RISCV::X10);
    Insts.insert(Insts.end(), Addr.begin(), Addr.end());
    Insts.emplace_back();
    loadReg(Insts.back(), RISCV::X10, RISCV::X10, 0);
    InstructionListType cmpJmp =
        createRegCmpJE(RISCV::X10, RISCV::X11, IndCallHandler, Ctx);
    Insts.insert(Insts.end(), cmpJmp.begin(), cmpJmp.end());
    Insts.emplace_back();
    createStackPointerIncrement(Insts.back(), 16);
    Insts.emplace_back();
    storeReg(Insts.back(), RISCV::X1, RISCV::X2, 0);
    Insts.emplace_back();
    createIndirectCallInst(Insts.back(), /*IsTailCall*/ false, RISCV::X10, 0);
    Insts.emplace_back();
    loadReg(Insts.back(), RISCV::X1, RISCV::X2, 0);
    Insts.emplace_back();
    createStackPointerDecrement(Insts.back(), 16);
    Insts.emplace_back();
    createDirectCall(Insts.back(), IndCallHandler, Ctx, /*IsTailCall*/ true);
    return Insts;
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    InstructionListType Insts;
    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.emplace_back();
    loadReg(Insts.back(), RISCV::X5, RISCV::X2, 0);
    Insts.emplace_back();
    createStackPointerDecrement(Insts.back(), 16);
    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.emplace_back();
    createIndirectCallInst(Insts.back(), /*IsTailCall*/ true, RISCV::X5, 0);
    return Insts;
  }

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    return createInstrumentedIndCallHandlerExitBB();
  }

  std::vector<MCInst> createSymbolTrampoline(const MCSymbol *TgtSym,
                                             MCContext *Ctx) override {
    std::vector<MCInst> Insts;
    createShortJmp(Insts, TgtSym, Ctx, /*IsTailCall*/ true);
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
    Inst.setOpcode(RISCV::ADD);
    Inst.insert(Inst.begin(), MCOperand::createReg(Reg));
    Inst.insert(Inst.begin() + 1, MCOperand::createReg(RISCV::X0));
  }

  InstructionListType createLoadImmediate(const MCPhysReg Dest,
                                          uint64_t Imm) const override {
    InstructionListType Insts;
    // get IMM higher 32bit
    Insts.emplace_back(
        MCInstBuilder(RISCV::LUI).addReg(Dest).addImm((Imm >> 44) & 0xFFFFF));
    Insts.emplace_back(MCInstBuilder(RISCV::LUI)
                           .addReg(RISCV::X5)
                           .addImm((Imm >> 32) & 0xFFF));
    Insts.emplace_back(MCInstBuilder(RISCV::SRLI)
                           .addReg(RISCV::X5)
                           .addReg(RISCV::X5)
                           .addImm(12));
    Insts.emplace_back(
        MCInstBuilder(RISCV::OR).addReg(Dest).addReg(Dest).addReg(RISCV::X5));
    Insts.emplace_back(
        MCInstBuilder(RISCV::SLLI).addReg(Dest).addReg(Dest).addImm(32));

    // get IMM lower 32bit
    Insts.emplace_back(MCInstBuilder(RISCV::LUI)
                           .addReg(RISCV::X5)
                           .addImm((Imm >> 12) & 0xFFFFF));
    Insts.emplace_back(
        MCInstBuilder(RISCV::LUI).addReg(RISCV::X6).addImm((Imm)&0xFFF));
    Insts.emplace_back(MCInstBuilder(RISCV::SRLI)
                           .addReg(RISCV::X6)
                           .addReg(RISCV::X6)
                           .addImm(12));
    Insts.emplace_back(
        MCInstBuilder(RISCV::OR).addReg(RISCV::X5).addReg(RISCV::X5).addReg(
            RISCV::X6));

    // get 64bit IMM
    Insts.emplace_back(
        MCInstBuilder(RISCV::OR).addReg(Dest).addReg(Dest).addReg(RISCV::X5));
    return Insts;
  }

  InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
                                                     MCSymbol *HandlerFuncAddr,
                                                     int CallSiteID,
                                                     MCContext *Ctx) override {
    // Code sequence used to enter indirect call instrumentation helper:
    //   addi  sp, sp, -0x10
    //   sd  a0, 0x0(sp)
    //   sd  a1, 0x8(sp)
    //   mov target x0  convertIndirectCallToLoad -> add a0, zero, target
    //   mov x1 CallSiteID createLoadImmediate
    //   addi  sp, sp, -0x10
    //   sd  a0, 0x0(sp)
    //   sd  a1, 0x8(sp)
    //   la x0 *HandlerFuncAddr -> auipc + addi
    //   jalr x0

    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.emplace_back(CallInst);
    convertIndirectCallToLoad(Insts.back(), RISCV::X10);
    InstructionListType LoadImm = createLoadImmediate(RISCV::X11, CallSiteID);
    Insts.insert(Insts.end(), LoadImm.begin(), LoadImm.end());
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    InstructionListType Addr =
        materializeAddress(HandlerFuncAddr, Ctx, RISCV::X5);
    Insts.insert(Insts.end(), Addr.begin(), Addr.end());
    Insts.emplace_back();
    createIndirectCallInst(Insts.back(), isTailCall(CallInst), RISCV::X5, 0);

    // // Carry over metadata including tail call marker if present.
    stripAnnotations(Insts.back());
    moveAnnotations(std::move(CallInst), Insts.back());

    return Insts;
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createRISCVMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                        const MCInstrInfo *Info,
                                        const MCRegisterInfo *RegInfo,
                                        const MCSubtargetInfo *STI) {
  return new RISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
