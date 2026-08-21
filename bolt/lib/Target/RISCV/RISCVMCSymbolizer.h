//===- bolt/Target/RISCV/RISCVMCSymbolizer.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_TARGET_RISCV_RISCVMCSYMBOLIZER_H
#define BOLT_TARGET_RISCV_RISCVMCSYMBOLIZER_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include <map>

namespace llvm {
namespace bolt {

class RISCVMCSymbolizer : public MCSymbolizer {
protected:
  BinaryFunction &Function;
  bool CreateNewSymbols{true};

  /// Map function offsets referenced by %pcrel_lo relocations to labels that
  /// must remain attached to the corresponding %pcrel_hi instructions.
  std::map<uint64_t, MCSymbol *> InstructionLabels;

  /// Map referenced instruction offsets to their %pcrel_lo relocations.
  std::map<uint64_t, const Relocation *> InstructionReferences;

  MCSymbol *getOrCreateInstructionLabel(uint64_t Offset);

  /// Return the complete PC-relative value for a GOT relocation. The value
  /// recorded when relocations are read assumes that the low instruction
  /// immediately follows AUIPC. Reconstruct it from the matching low
  /// relocation so linker scheduling does not affect symbolization.
  uint64_t getGOTValue(const Relocation &Rel) const;

public:
  RISCVMCSymbolizer(BinaryFunction &Function, bool CreateNewSymbols = true);

  RISCVMCSymbolizer(const RISCVMCSymbolizer &) = delete;
  RISCVMCSymbolizer &operator=(const RISCVMCSymbolizer &) = delete;
  ~RISCVMCSymbolizer() override;

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
