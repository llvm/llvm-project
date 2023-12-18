//===- bolt/Target/X86/X86MCSymbolizer.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_X86MCSYMBOLIZER_H
#define BOLT_CORE_X86MCSYMBOLIZER_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"

namespace llvm {
namespace bolt {

class X86MCSymbolizer : public MCSymbolizer {
protected:
  BinaryFunction &Function;
  bool CreateNewSymbols{true};

  std::pair<MCSymbol *, uint64_t> handleGOTPC64(const Relocation &R,
                                                uint64_t InstrAddr);

public:
  X86MCSymbolizer(BinaryFunction &Function, bool CreateNewSymbols = true)
      : MCSymbolizer(*Function.getBinaryContext().Ctx.get(), nullptr),
        Function(Function), CreateNewSymbols(CreateNewSymbols) {}

  X86MCSymbolizer(const X86MCSymbolizer &) = delete;
  X86MCSymbolizer &operator=(const X86MCSymbolizer &) = delete;
  virtual ~X86MCSymbolizer();

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
