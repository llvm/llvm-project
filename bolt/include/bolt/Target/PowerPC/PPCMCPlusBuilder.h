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
  IndirectBranchType
  analyzeIndirectBranch(MCInst &Instruction, InstructionIterator Begin,
                        InstructionIterator End, const unsigned PtrSize,
                        MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
                        unsigned &IndexRegNumOut, int64_t &DispValueOut,
                        const MCExpr *&DispExprOut, MCInst *&PCRelBaseOut,
                        MCInst *&FixedEntryLoadInstr) const override;
};

} // namespace bolt
} // namespace llvm