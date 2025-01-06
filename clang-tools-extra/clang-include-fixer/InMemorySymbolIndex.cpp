//===-- InMemorySymbolIndex.cpp--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InMemorySymbolIndex.h"

using clang::find_all_symbols::SymbolAndSignals;

namespace clang {
namespace include_fixer {

InMemorySymbolIndex::InMemorySymbolIndex(
    const std::vector<SymbolAndSignals> &Symbols) {
  for (const auto &Symbol : Symbols)
    LookupTable[std::string(Symbol.Symbol.getName())].push_back(Symbol);
}

std::vector<SymbolAndSignals>
InMemorySymbolIndex::search(llvm::StringRef Identifier) {
  auto I = LookupTable.find(Identifier);
  if (I != LookupTable.end())
    return I->second;
  return {};
}

} // namespace include_fixer
} // namespace clang
