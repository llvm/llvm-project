#pragma once

#include "bolt/Core/MCPlusBuilder.h"

namespace llvm {
namespace bolt {

class PPCMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  static void createPushRegisters(MCInst &Inst1, MCInst &Inst2, MCPhysReg Reg1,
                                  MCPhysReg Reg2);

  bool shouldRecordCodeRelocation(unsigned Type) const override;

  bool hasPCRelOperand(const MCInst &I) const override;
  int getPCRelOperandNum(const MCInst &Inst) const;

  int getMemoryOperandNo(const MCInst &Inst) const override;

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override;
  bool isIndirectBranch(const MCInst &I) const override;

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override;

  bool convertJmpToTailCall(MCInst &Inst) override;

  bool isCall(const MCInst &Inst) const override;

  bool isTailCall(const MCInst &Inst) const;
  bool isReturn(const MCInst &Inst) const override;
  bool isConditionalBranch(const MCInst &Inst) const override;
  bool isUnconditionalBranch(const MCInst &Inst) const override;

  const MCInst *getConditionalTailCall(const MCInst &Inst) const;

  IndirectBranchType
  analyzeIndirectBranch(MCInst &Instruction, InstructionIterator Begin,
                        InstructionIterator End, const unsigned PtrSize,
                        MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
                        unsigned &IndexRegNumOut, int64_t &DispValueOut,
                        const MCExpr *&DispExprOut, MCInst *&PCRelBaseOut,
                        MCInst *&FixedEntryLoadInstr) const override;

  bool isNoop(const MCInst &Inst) const override;

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const llvm::MCSymbol *&Tgt,
                     const llvm::MCSymbol *&Fallthrough, llvm::MCInst *&CondBr,
                     llvm::MCInst *&UncondBr) const override;

  bool lowerTailCall(llvm::MCInst &Inst) override;

  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override;

  void createLongTailCall(std::vector<MCInst> &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override;

  std::optional<Relocation>
  createRelocation(const MCFixup &Fixup,
                   const MCAsmBackend &MAB) const override;
};

} // namespace bolt
} // namespace llvm