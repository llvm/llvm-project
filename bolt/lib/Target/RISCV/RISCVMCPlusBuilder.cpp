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

#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVMCSymbolizer.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
  using LocalUDChain = DenseMap<const MCInst *, SmallVector<MCInst *, 4>>;

  struct JumpTableLoad {
    const MCInst *Inst{nullptr};
    uint64_t EntrySize{0};
    bool EntrySigned{false};
    int64_t Offset{0};
  };

  struct ScaledAddress {
    const MCInst *BaseDef{nullptr};
    MCPhysReg IndexReg{MCRegister::NoRegister};
  };

  bool isRV64() const { return STI->hasFeature(RISCV::Feature64Bit); }
  bool isRVE() const { return STI->hasFeature(RISCV::FeatureStdExtE); }
  unsigned regSize() const { return isRV64() ? 8 : 4; }
  unsigned loadOpc() const { return isRV64() ? RISCV::LD : RISCV::LW; }
  unsigned storeOpc() const { return isRV64() ? RISCV::SD : RISCV::SW; }
  unsigned atomicAddOpc() const {
    return isRV64() ? RISCV::AMOADD_D : RISCV::AMOADD_W;
  }

  LocalUDChain computeLocalUDChain(const MCInst *CurInstr,
                                   InstructionIterator Begin,
                                   InstructionIterator End) const {
    DenseMap<int, MCInst *> RegAliasTable;
    LocalUDChain Uses;

    auto addInstrOperands = [&](const MCInst &Instr) {
      for (const MCOperand &Operand : MCPlus::primeOperands(Instr)) {
        if (!Operand.isReg())
          continue;
        Uses[&Instr].push_back(RegAliasTable[Operand.getReg()]);
      }
    };

    bool TerminatorSeen = false;
    for (auto II = Begin; II != End; ++II) {
      MCInst &Instr = *II;
      if (isPseudo(Instr) || isNoop(Instr))
        continue;
      if (TerminatorSeen) {
        RegAliasTable.clear();
        Uses.clear();
      }

      addInstrOperands(Instr);

      BitVector Regs(RegInfo->getNumRegs(), false);
      getWrittenRegs(Instr, Regs);
      for (int Idx : Regs.set_bits())
        RegAliasTable[Idx] = &Instr;

      TerminatorSeen = isTerminator(Instr);
    }

    if (CurInstr)
      addInstrOperands(*CurInstr);

    return Uses;
  }

  const MCInst *getOperandDef(const MCInst &Inst, unsigned OperandIndex,
                              const LocalUDChain &UDChain) const {
    if (OperandIndex >= MCPlus::getNumPrimeOperands(Inst) ||
        !Inst.getOperand(OperandIndex).isReg())
      return nullptr;

    const auto UsesIt = UDChain.find(&Inst);
    if (UsesIt == UDChain.end())
      return nullptr;

    unsigned RegOperandIndex = 0;
    for (unsigned Index = 0; Index < OperandIndex; ++Index)
      RegOperandIndex += Inst.getOperand(Index).isReg();

    if (RegOperandIndex >= UsesIt->second.size())
      return nullptr;
    return UsesIt->second[RegOperandIndex];
  }

  const MCInst *followCopies(const MCInst *Def,
                             const LocalUDChain &UDChain) const {
    SmallPtrSet<const MCInst *, 4> Visited;
    while (Def && Visited.insert(Def).second) {
      unsigned SourceOperand = 0;
      switch (Def->getOpcode()) {
      default:
        return Def;
      case RISCV::ADDI:
      case RISCV::ORI:
        if (!Def->getOperand(2).isImm() || Def->getOperand(2).getImm() != 0)
          return Def;
        SourceOperand = 1;
        break;
      case RISCV::ADD:
      case RISCV::OR:
        if (Def->getOperand(1).getReg() == RISCV::X0)
          SourceOperand = 2;
        else if (Def->getOperand(2).getReg() == RISCV::X0)
          SourceOperand = 1;
        else
          return Def;
        break;
      case RISCV::C_MV:
        SourceOperand = 1;
        break;
      }
      Def = getOperandDef(*Def, SourceOperand, UDChain);
    }
    return Def;
  }

  static const MCExpr *stripSpecifier(const MCExpr *Expr) {
    while (const auto *Specifier = dyn_cast_or_null<MCSpecifierExpr>(Expr))
      Expr = Specifier->getSubExpr();
    return Expr;
  }

  const MCExpr *matchJumpTableBase(const MCInst *Def,
                                   const LocalUDChain &UDChain) const {
    Def = followCopies(Def, UDChain);
    if (!Def)
      return nullptr;

    if (Def->getOpcode() == RISCV::ADDI || Def->getOpcode() == RISCV::C_ADDI) {
      Def = followCopies(getOperandDef(*Def, 1, UDChain), UDChain);
      if (!Def)
        return nullptr;
    }

    switch (Def->getOpcode()) {
    default:
      return nullptr;
    case RISCV::LUI:
    case RISCV::AUIPC:
    case RISCV::C_LUI:
      break;
    }

    if (!Def->getOperand(1).isExpr())
      return nullptr;
    const MCExpr *Expr = stripSpecifier(Def->getOperand(1).getExpr());
    return getTargetSymbolInfo(Expr).first ? Expr : nullptr;
  }

  bool matchJumpTableLoad(const MCInst *Def, const LocalUDChain &UDChain,
                          JumpTableLoad &Load) const {
    Def = followCopies(Def, UDChain);
    if (!Def)
      return false;

    switch (Def->getOpcode()) {
    default:
      return false;
    case RISCV::LW:
    case RISCV::C_LW:
      Load.EntrySize = 4;
      Load.EntrySigned = isRV64();
      break;
    case RISCV::LWU:
      Load.EntrySize = 4;
      Load.EntrySigned = false;
      break;
    case RISCV::LD:
    case RISCV::C_LD:
      // GCC uses full-width label-address arrays as a family of sub-tables
      // relative to one shared anchor. BOLT cannot move those safely until it
      // can retarget every LUI/AUIPC + ADDI reference to an interior label.
      return false;
    }

    if (!Def->getOperand(2).isImm())
      return false;
    Load.Inst = Def;
    Load.Offset = Def->getOperand(2).getImm();
    // A non-zero displacement can select an embedded table relative to a
    // larger anchor object. Moving that table requires retargeting the whole
    // LUI/AUIPC + ADDI pair to a new interior label, which is not represented
    // by MemLocInstr today. Reject it instead of moving the wrong sub-table.
    return Load.Offset == 0;
  }

  static unsigned getSHXADDScale(unsigned Opcode) {
    switch (Opcode) {
    default:
      return 0;
    case RISCV::SH1ADD:
    case RISCV::SH1ADD_UW:
      return 2;
    case RISCV::SH2ADD:
    case RISCV::SH2ADD_UW:
      return 4;
    case RISCV::SH3ADD:
    case RISCV::SH3ADD_UW:
      return 8;
    }
  }

  bool matchScaledAddress(const MCInst *Def, uint64_t EntrySize,
                          const LocalUDChain &UDChain,
                          ScaledAddress &Address) const {
    Def = followCopies(Def, UDChain);
    if (!Def)
      return false;

    if (getSHXADDScale(Def->getOpcode()) == EntrySize) {
      Address.IndexReg = Def->getOperand(1).getReg();
      Address.BaseDef = followCopies(getOperandDef(*Def, 2, UDChain), UDChain);
      return Address.BaseDef != nullptr;
    }

    if (Def->getOpcode() != RISCV::ADD && Def->getOpcode() != RISCV::C_ADD)
      return false;

    for (unsigned ShiftOperand : {1U, 2U}) {
      const unsigned BaseOperand = ShiftOperand == 1 ? 2 : 1;
      const MCInst *Shift =
          followCopies(getOperandDef(*Def, ShiftOperand, UDChain), UDChain);
      if (!Shift)
        continue;
      if (Shift->getOpcode() != RISCV::SLLI &&
          Shift->getOpcode() != RISCV::SLLI_UW &&
          Shift->getOpcode() != RISCV::C_SLLI)
        continue;
      if (!Shift->getOperand(2).isImm() ||
          (1ULL << Shift->getOperand(2).getImm()) != EntrySize)
        continue;

      Address.IndexReg = Shift->getOperand(1).getReg();
      Address.BaseDef =
          followCopies(getOperandDef(*Def, BaseOperand, UDChain), UDChain);
      if (Address.BaseDef)
        return true;
    }
    return false;
  }

  bool areSameJumpTable(const MCExpr *LHS, const MCExpr *RHS) const {
    return getTargetSymbolInfo(LHS) == getTargetSymbolInfo(RHS);
  }

  bool replaceJumpTableSymbol(MCInst &Inst, const MCSymbol *OldTarget,
                              const MCSymbol *NewTarget,
                              MCContext *Ctx) const {
    for (unsigned OpIndex = 0;
         OpIndex < MCPlus::getNumPrimeOperands(Inst); ++OpIndex) {
      MCOperand &Operand = Inst.getOperand(OpIndex);
      if (!Operand.isExpr())
        continue;

      const MCExpr *Expr = Operand.getExpr();
      const auto *Specifier = dyn_cast<MCSpecifierExpr>(Expr);
      const MCExpr *SubExpr = Specifier ? Specifier->getSubExpr() : Expr;
      const auto [Symbol, Addend] = getTargetSymbolInfo(SubExpr);
      if (Symbol != OldTarget)
        continue;

      const MCExpr *NewExpr = MCSymbolRefExpr::create(NewTarget, *Ctx);
      if (Addend)
        NewExpr = MCBinaryExpr::createAdd(
            NewExpr, MCConstantExpr::create(Addend, *Ctx), *Ctx);
      if (Specifier)
        NewExpr =
            MCSpecifierExpr::create(NewExpr, Specifier->getSpecifier(), *Ctx);
      Operand = MCOperand::createExpr(NewExpr);
      return true;
    }
    return false;
  }

public:
  using MCPlusBuilder::MCPlusBuilder;

  BitVector getRegsUsedAsParams() const override {
    BitVector Regs(RegInfo->getNumRegs(), false);
    const MCPhysReg LastArgReg = isRVE() ? RISCV::X15 : RISCV::X17;
    for (MCPhysReg Reg = RISCV::X10; Reg <= LastArgReg; ++Reg)
      Regs |= getAliases(Reg);
    return Regs;
  }

  std::unique_ptr<MCSymbolizer>
  createTargetSymbolizer(BinaryFunction &Function,
                         bool CreateNewSymbols) const override {
    return std::make_unique<RISCVMCSymbolizer>(Function, CreateNewSymbols);
  }

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
    if (isRVE())
      return;
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

  void getDefaultLiveOut(BitVector &Regs) const override {
    Regs |= getAliases(RISCV::X10);
    Regs |= getAliases(RISCV::X11);
  }

  void getGPRegs(BitVector &Regs, bool IncludeAlias = true) const override {
    const MCPhysReg LastGPR = isRVE() ? RISCV::X15 : RISCV::X31;
    for (MCPhysReg Reg = RISCV::X1; Reg <= LastGPR; ++Reg) {
      if (IncludeAlias)
        Regs |= getAliases(Reg);
      else
        Regs.set(Reg);
    }
  }

  void removeNonScavengeableRegs(BitVector &Regs) const override {
    BitVector ExclusionMask(RegInfo->getNumRegs(), false);
    ExclusionMask |= getAliases(RISCV::X1); // return address
    ExclusionMask |= getAliases(RISCV::X2); // stack pointer
    ExclusionMask |= getAliases(RISCV::X3); // global pointer
    ExclusionMask |= getAliases(RISCV::X4); // thread pointer
    ExclusionMask |= getAliases(RISCV::X8); // frame pointer
    ExclusionMask.flip();
    Regs &= ExclusionMask;
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

  InstructionListType
  reverseBranchCondition(MCInst Inst, const MCSymbol *TBB, MCContext *Ctx,
                         bool MustPreserveFlags = true) const override {
    auto Opcode = getInvertedBranchOpcode(Inst.getOpcode());
    Inst.setOpcode(Opcode);
    replaceBranchTarget(Inst, TBB, Ctx);
    return {Inst};
  }

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("Failed to get RISC-V PC-relative encoding size");
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
      return 9;
    case RISCV::C_J:
      return 12;
    case RISCV::BEQ:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BGE:
    case RISCV::BLTU:
    case RISCV::BGEU:
      return 13;
    case RISCV::JAL:
      return 21;
    }
  }

  int getUncondBranchEncodingSize() const override { return 21; }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");

    unsigned SymOpIndex;
    auto Result = getSymbolRefOperandNum(Inst, SymOpIndex);
    (void)Result;
    assert(Result && "unimplemented branch");

    Inst.getOperand(SymOpIndex) =
        MCOperand::createExpr(MCSymbolRefExpr::create(TBB, *Ctx));
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      uint64_t &EntrySize, bool &EntrySigned, MCInst *&PCRelBaseOut,
      MCInst *&FixedEntryLoadInst) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    EntrySize = 0;
    EntrySigned = false;
    PCRelBaseOut = nullptr;
    FixedEntryLoadInst = nullptr;

    (void)PtrSize;

    unsigned TargetOperand;
    switch (Instruction.getOpcode()) {
    default:
      return IndirectBranchType::UNKNOWN;
    case RISCV::JALR:
      if (Instruction.getOperand(0).getReg() != RISCV::X0)
        return IndirectBranchType::UNKNOWN;
      TargetOperand = 1;
      break;
    case RISCV::C_JR:
      TargetOperand = 0;
      break;
    }

    LocalUDChain UDChain = computeLocalUDChain(&Instruction, Begin, End);
    const MCInst *TargetDef =
        getOperandDef(Instruction, TargetOperand, UDChain);

    // Check for a long tail call. The local use-def chain makes this robust
    // against unrelated instructions between AUIPC and JALR.
    if (Instruction.getOpcode() == RISCV::JALR && TargetDef &&
        isRISCVCall(*TargetDef, Instruction))
      return IndirectBranchType::POSSIBLE_TAIL_CALL;

    // Jump-table dispatches use an unmodified register as the JALR target.
    if (Instruction.getOpcode() == RISCV::JALR &&
        (!Instruction.getOperand(2).isImm() ||
         Instruction.getOperand(2).getImm() != 0))
      return IndirectBranchType::UNKNOWN;

    const MCInst *Root = followCopies(TargetDef, UDChain);
    if (!Root)
      return IndirectBranchType::UNKNOWN;

    // PIC tables contain signed 32-bit offsets. Match
    //   add target, loaded-offset, table-base
    // before the absolute-address form, which branches directly to the load.
    if (Root->getOpcode() == RISCV::ADD || Root->getOpcode() == RISCV::C_ADD) {
      for (unsigned LoadOperand : {1U, 2U}) {
        const unsigned BaseOperand = LoadOperand == 1 ? 2 : 1;
        JumpTableLoad Load;
        if (!matchJumpTableLoad(getOperandDef(*Root, LoadOperand, UDChain),
                                UDChain, Load) ||
            Load.EntrySize != 4)
          continue;

        const MCExpr *TargetBase = matchJumpTableBase(
            getOperandDef(*Root, BaseOperand, UDChain), UDChain);
        if (!TargetBase)
          continue;

        ScaledAddress Address;
        if (!matchScaledAddress(getOperandDef(*Load.Inst, 1, UDChain),
                                Load.EntrySize, UDChain, Address))
          continue;
        const MCExpr *LoadBase = matchJumpTableBase(Address.BaseDef, UDChain);
        if (!LoadBase || !areSameJumpTable(TargetBase, LoadBase))
          continue;

        IndexRegNum = Address.IndexReg;
        DispValue = Load.Offset;
        DispExpr = LoadBase;
        EntrySize = Load.EntrySize;
        EntrySigned = true;
        return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
      }
    }

    // Absolute-address tables branch directly to a loaded 32/64-bit entry:
    //   load target, (table-base + index * entry-size)
    //   jr   target
    JumpTableLoad Load;
    if (!matchJumpTableLoad(Root, UDChain, Load))
      return IndirectBranchType::UNKNOWN;

    ScaledAddress Address;
    if (!matchScaledAddress(getOperandDef(*Load.Inst, 1, UDChain),
                            Load.EntrySize, UDChain, Address))
      return IndirectBranchType::UNKNOWN;

    const MCExpr *LoadBase = matchJumpTableBase(Address.BaseDef, UDChain);
    if (!LoadBase)
      return IndirectBranchType::UNKNOWN;

    BaseRegNum = getNoRegister();
    IndexRegNum = Address.IndexReg;
    DispValue = Load.Offset;
    DispExpr = LoadBase;
    EntrySize = Load.EntrySize;
    EntrySigned = Load.EntrySigned;
    return IndirectBranchType::POSSIBLE_JUMP_TABLE;
  }

  bool replaceJumpTableReference(
      MutableArrayRef<MCInst> InstrWindow, const MCSymbol *OldTarget,
      const MCSymbol *NewTarget, MCContext *Ctx) const override {
    for (MCInst &Inst : llvm::reverse(InstrWindow)) {
      switch (Inst.getOpcode()) {
      default:
        continue;
      case RISCV::AUIPC:
      case RISCV::LUI:
      case RISCV::C_LUI:
        break;
      }
      if (replaceJumpTableSymbol(Inst, OldTarget, NewTarget, Ctx))
        return true;
    }
    return false;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JAL:
    case RISCV::JALR:
    case RISCV::C_J:
    case RISCV::C_JR:
      break;
    }

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
    Inst.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(TBB, *Ctx)));
  }

  StringRef getTrapFillValue() const override {
    return StringRef("\0\0\0\0", 4);
  }

  void createCall(unsigned Opcode, MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) {
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(MCSpecifierExpr::create(
        MCSymbolRefExpr::create(Target, *Ctx), RISCV::S_CALL_PLT, *Ctx)));
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    return createCall(RISCV::PseudoCALL, Inst, Target, Ctx);
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    Seq.emplace_back();
    createTailCall(Seq.back(), Target, Ctx);
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createCall(RISCV::PseudoTAIL, Inst, Target, Ctx);
  }

  InstructionListType createIndirectPLTCall(MCInst &&DirectCall,
                                            const MCSymbol *TargetLocation,
                                            MCContext *Ctx) override {
    const bool IsTailCall = isTailCall(DirectCall);
    assert(((DirectCall.getOpcode() == RISCV::PseudoCALL && !IsTailCall) ||
            (DirectCall.getOpcode() == RISCV::PseudoTAIL && IsTailCall)) &&
           "RISC-V direct (tail) call instruction expected");

    // Load the resolved function address directly from its GOT slot:
    //
    //   auipc t3, %pcrel_hi(TargetLocation)
    //   l[dw] t3, %pcrel_lo(.Lpcrel_hi)(t3)
    //   jalr  ra, t3, 0
    //
    // A tail call uses zero instead of ra as the JALR destination.
    InstructionListType Code;
    // Use t3 (x28), the scratch register used by linker-generated RISC-V
    // PLT/IPLT entries. It is caller-saved, is not an argument register, and
    // the original call through the PLT already clobbers it.
    const MCPhysReg PLTScratchReg = RISCV::X28;
    MCSymbol *AUIPCLabel = Ctx->createNamedTempSymbol("pcrel_hi");

    MCInst InstAUIPC =
        MCInstBuilder(RISCV::AUIPC).addReg(PLTScratchReg).addImm(0);
    // TargetLocation is already registered at the existing GOT slot, so use a
    // direct PC-relative relocation to that slot instead of R_RISCV_GOT_HI20,
    // which is used when starting from the referenced function symbol.
    setOperandToSymbolRef(InstAUIPC, /*OpNum=*/1, TargetLocation,
                          /*Addend=*/0, Ctx, ELF::R_RISCV_PCREL_HI20);
    setInstLabel(InstAUIPC, AUIPCLabel);
    Code.emplace_back(std::move(InstAUIPC));

    MCInst InstLoad = MCInstBuilder(loadOpc())
                          .addReg(PLTScratchReg)
                          .addReg(PLTScratchReg)
                          .addImm(0);
    // Pair the I-type LD/LW immediate with the label on AUIPC. RISC-V
    // R_RISCV_PCREL_LO12_I relocations name the corresponding HI20 location,
    // not the final GOT-slot symbol.
    setOperandToSymbolRef(InstLoad, /*OpNum=*/2, AUIPCLabel,
                          /*Addend=*/0, Ctx, ELF::R_RISCV_PCREL_LO12_I);
    Code.emplace_back(std::move(InstLoad));

    MCInst InstCall = MCInstBuilder(RISCV::JALR)
                          .addReg(IsTailCall ? RISCV::X0 : RISCV::X1)
                          .addReg(PLTScratchReg)
                          .addImm(0);
    moveAnnotations(std::move(DirectCall), InstCall);
    Code.emplace_back(std::move(InstCall));

    return Code;
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

      // An indirect jump has no symbolic TBB operand. It may be a recognized
      // jump-table dispatch and must not enter the direct unconditional path.
      if (isIndirectBranch(*I))
        return false;

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
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
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
    assert(LD.getOpcode() == loadOpc());
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
      return MCSpecifierExpr::create(Expr, RISCV::S_PCREL_HI, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return MCSpecifierExpr::create(Expr, RISCV::S_PCREL_LO, Ctx);
    case ELF::R_RISCV_HI20:
      return MCSpecifierExpr::create(Expr, ELF::R_RISCV_HI20, Ctx);
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
      return MCSpecifierExpr::create(Expr, RISCV::S_LO, Ctx);
    case ELF::R_RISCV_CALL:
      return MCSpecifierExpr::create(Expr, RISCV::S_CALL_PLT, Ctx);
    case ELF::R_RISCV_CALL_PLT:
      return MCSpecifierExpr::create(Expr, RISCV::S_CALL_PLT, Ctx);
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
    case RISCV::S_CALL_PLT:
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
    Inst = MCInstBuilder(loadOpc()).addReg(To).addReg(From).addImm(offset);
  }

  void storeReg(MCInst &Inst, MCPhysReg From, MCPhysReg To,
                int64_t offset) const {
    Inst = MCInstBuilder(storeOpc()).addReg(From).addReg(To).addImm(offset);
  }

  void spillRegs(InstructionListType &Insts,
                 const SmallVector<unsigned> &Regs) const {
    Insts.emplace_back();
    createStackPointerIncrement(Insts.back(), Regs.size() * regSize());

    int64_t Offset = 0;
    for (auto Reg : Regs) {
      Insts.emplace_back();
      storeReg(Insts.back(), Reg, RISCV::X2, Offset);
      Offset += regSize();
    }
  }

  void reloadRegs(InstructionListType &Insts,
                  const SmallVector<unsigned> &Regs) const {
    int64_t Offset = 0;
    for (auto Reg : Regs) {
      Insts.emplace_back();
      loadReg(Insts.back(), Reg, RISCV::X2, Offset);
      Offset += regSize();
    }

    Insts.emplace_back();
    createStackPointerDecrement(Insts.back(), Regs.size() * regSize());
  }

  void atomicAdd(MCInst &Inst, MCPhysReg RegAtomic, MCPhysReg RegTo,
                 MCPhysReg RegCnt) const {
    // AMO operands are ordered as rd, rs2 (value), rs1 (address).
    Inst = MCInstBuilder(atomicAddOpc())
               .addReg(RegAtomic)
               .addReg(RegCnt)
               .addReg(RegTo);
  }

  InstructionListType createRegCmpJE(MCPhysReg RegNo, const MCSymbol *Target,
                                     MCContext *Ctx) const {
    InstructionListType Insts;
    Insts.emplace_back(MCInstBuilder(RISCV::BEQ)
                           .addReg(RegNo)
                           .addReg(RISCV::X0)
                           .addExpr(MCSymbolRefExpr::create(Target, *Ctx)));
    return Insts;
  }

  void createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(RISCV::EBREAK);
  }

  void createNoop(MCInst &Inst) const override {
    Inst.clear();
    Inst = MCInstBuilder(RISCV::ADDI)
               .addReg(RISCV::X0)
               .addReg(RISCV::X0)
               .addImm(0);
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

  void createLongJmp(InstructionListType &Seq, const MCSymbol *Target,
                     MCContext *Ctx, bool IsTailCall,
                     MCPhysReg ScratchReg) override {
    assert(ScratchReg && "RISC-V long jump requires a scratch register");
    MCSymbol *AuipcLabel = Ctx->createNamedTempSymbol("long_jmp");

    MCInst Inst = MCInstBuilder(RISCV::AUIPC).addReg(ScratchReg).addImm(0);
    setOperandToSymbolRef(Inst, /*OpNum=*/1, Target, /*Addend=*/0, Ctx,
                          ELF::R_RISCV_PCREL_HI20);
    setInstLabel(Inst, AuipcLabel);
    Seq.emplace_back(std::move(Inst));

    Inst = MCInstBuilder(RISCV::JALR)
               .addReg(RISCV::X0)
               .addReg(ScratchReg)
               .addImm(0);
    setOperandToSymbolRef(Inst, /*OpNum=*/2, AuipcLabel, /*Addend=*/0, Ctx,
                          ELF::R_RISCV_PCREL_LO12_I);
    if (IsTailCall)
      setTailCall(Inst);
    Seq.emplace_back(std::move(Inst));
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

  std::optional<Relocation>
  createRelocation(const MCFixup &Fixup,
                   const MCAsmBackend &MAB) const override {
    (void)MAB;
    const uint64_t RelOffset = Fixup.getOffset();

    uint32_t RelType;
    if (mc::isRelocation(Fixup.getKind())) {
      RelType = Fixup.getKind();
    } else if (Fixup.isPCRel()) {
      switch (Fixup.getKind()) {
      default:
        return std::nullopt;
      case FK_Data_4:
        RelType = ELF::R_RISCV_32_PCREL;
        break;
      case RISCV::fixup_riscv_pcrel_hi20:
        RelType = ELF::R_RISCV_PCREL_HI20;
        break;
      case RISCV::fixup_riscv_pcrel_lo12_i:
        RelType = ELF::R_RISCV_PCREL_LO12_I;
        break;
      case RISCV::fixup_riscv_pcrel_lo12_s:
        RelType = ELF::R_RISCV_PCREL_LO12_S;
        break;
      case RISCV::fixup_riscv_jal:
        RelType = ELF::R_RISCV_JAL;
        break;
      case RISCV::fixup_riscv_branch:
        RelType = ELF::R_RISCV_BRANCH;
        break;
      case RISCV::fixup_riscv_rvc_jump:
        RelType = ELF::R_RISCV_RVC_JUMP;
        break;
      case RISCV::fixup_riscv_rvc_branch:
        RelType = ELF::R_RISCV_RVC_BRANCH;
        break;
      case RISCV::fixup_riscv_call:
      case RISCV::fixup_riscv_call_plt:
        RelType = ELF::R_RISCV_CALL_PLT;
        break;
      }
    } else {
      switch (Fixup.getKind()) {
      default:
        return std::nullopt;
      case FK_Data_4:
        RelType = ELF::R_RISCV_32;
        break;
      case FK_Data_8:
        RelType = ELF::R_RISCV_64;
        break;
      case RISCV::fixup_riscv_hi20:
        RelType = ELF::R_RISCV_HI20;
        break;
      case RISCV::fixup_riscv_lo12_i:
        RelType = ELF::R_RISCV_LO12_I;
        break;
      case RISCV::fixup_riscv_lo12_s:
        RelType = ELF::R_RISCV_LO12_S;
        break;
      }
    }

    auto [RelSymbol, RelAddend] = extractFixupExpr(Fixup);
    return Relocation({RelOffset, RelSymbol, RelType, RelAddend, 0});
  }

  InstructionListType createInstrIncMemory(const MCSymbol *Target,
                                           MCContext *Ctx, bool IsLeaf,
                                           unsigned CodePointerSize) override {
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
          Inst, MCSymbolRefExpr::create(Target, *Ctx), *Ctx, 0)));
      convertJmpToTailCall(Inst);
    } else {
      Inst.addOperand(MCOperand::createReg(RISCV::X1));
      Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
          Inst, MCSymbolRefExpr::create(Target, *Ctx), *Ctx, 0)));
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
        createRegCmpJE(RISCV::X10, IndCallHandler, Ctx);
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
                                                     size_t CallSiteID,
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
