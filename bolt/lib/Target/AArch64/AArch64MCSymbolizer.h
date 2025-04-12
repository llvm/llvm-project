//===- bolt/Target/AArch64/AArch64MCSymbolizer.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_TARGET_AARCH64_AARCH64MCSYMBOLIZER_H
#define BOLT_TARGET_AARCH64_AARCH64MCSYMBOLIZER_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include <optional>

namespace llvm {
namespace bolt {

class AArch64MCSymbolizer : public MCSymbolizer {
protected:
  BinaryFunction &Function;
  bool CreateNewSymbols{true};

  /// Modify relocation \p Rel based on type of the relocation and the
  /// instruction it was applied to. Return the new relocation info, or
  /// std::nullopt if the relocation should be ignored, e.g. in the case the
  /// instruction was modified by the linker.
  std::optional<Relocation> adjustRelocation(const Relocation &Rel,
                                             const MCInst &Inst) const;

  /// Return true if \p PageAddress is a valid page address for .got section.
  bool isPageAddressValidForGOT(uint64_t PageAddress) const;

public:
  AArch64MCSymbolizer(BinaryFunction &Function, bool CreateNewSymbols = true)
      : MCSymbolizer(*Function.getBinaryContext().Ctx.get(), nullptr),
        Function(Function), CreateNewSymbols(CreateNewSymbols) {}

  AArch64MCSymbolizer(const AArch64MCSymbolizer &) = delete;
  AArch64MCSymbolizer &operator=(const AArch64MCSymbolizer &) = delete;
  virtual ~AArch64MCSymbolizer();

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
