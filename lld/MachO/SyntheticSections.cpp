//===- SyntheticSections.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"
#include "Symbols.h"

using namespace llvm::MachO;

namespace lld {
namespace macho {

GotSection::GotSection() {
  segname = "__DATA_CONST";
  name = "__got";
  align = 8;
  flags = S_NON_LAZY_SYMBOL_POINTERS;

  // TODO: section_64::reserved1 should be an index into the indirect symbol
  // table, which we do not currently emit
}

void GotSection::addEntry(DylibSymbol &sym) {
  if (entries.insert(&sym)) {
    sym.gotIndex = entries.size() - 1;
  }
}

InStruct in;

} // namespace macho
} // namespace lld
