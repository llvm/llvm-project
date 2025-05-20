//===- bolt/Target/PPC/PPCMCSymbolizer.h ------------------------*- C++ -*-===//
//
// Minimal PowerPC Symbolizer for BOLT "Hello World" Programs
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_TARGET_PPC_PPCMCSYMBOLIZER_H
#define BOLT_TARGET_PPC_PPCMCSYMBOLIZER_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"

namespace llvm {
namespace bolt {

class PPCMCSymbolizer : public MCSymbolizer {
protected:
  BinaryFunction &Function;

public:
  PPCMCSymbolizer(BinaryFunction &Function)
      : MCSymbolizer(*Function.getBinaryContext().Ctx, nullptr),
        Function(Function) {}

  PPCMCSymbolizer(const PPCMCSymbolizer &) = delete;
  PPCMCSymbolizer &operator=(const PPCMCSymbolizer &) = delete;
  virtual ~PPCMCSymbolizer();

  /// Minimal: Try to add a symbolic operand if there is a matching relocation
  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &CStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t OpSize,
                                uint64_t InstSize) override;

  void tryAddingPcLoadReferenceComment(raw_ostream &CStream, int64_t Value,
                                       uint64_t Address) override;
};

} // namespace bolt
} // namespace llvm

#endif
