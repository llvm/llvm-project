#pragma once

#include "bolt/Core/MCPlusBuilder.h"

namespace llvm {
namespace bolt {

class PPCMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  static void createPushRegisters(MCInst &Inst1, MCInst &Inst2, MCPhysReg Reg1,
                                  MCPhysReg Reg2);
};

} // namespace bolt
} // namespace llvm