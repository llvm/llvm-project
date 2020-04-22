//===- SyntheticSections.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYNTHETIC_SECTIONS_H
#define LLD_MACHO_SYNTHETIC_SECTIONS_H

#include "InputSection.h"
#include "Target.h"
#include "llvm/ADT/SetVector.h"

namespace lld {
namespace macho {

class DylibSymbol;

// This section will be populated by dyld with addresses to non-lazily-loaded
// dylib symbols.
class GotSection : public InputSection {
public:
  GotSection();

  void addEntry(DylibSymbol &sym);
  const llvm::SetVector<const DylibSymbol *> &getEntries() const {
    return entries;
  }

  size_t getSize() const override { return entries.size() * WordSize; }

  void writeTo(uint8_t *buf) override {
    // Nothing to write, GOT contains all zeros at link time; it's populated at
    // runtime by dyld.
  }

private:
  llvm::SetVector<const DylibSymbol *> entries;
};

struct InStruct {
  GotSection *got;
};

extern InStruct in;

} // namespace macho
} // namespace lld

#endif
