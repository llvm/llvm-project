//===-- InMemorySymbolIndex.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYSYMBOLINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYSYMBOLINDEX_H

#include "SymbolIndex.h"
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

/// Xref database with fixed content.
class InMemorySymbolIndex : public SymbolIndex {
public:
  InMemorySymbolIndex(
      const std::vector<find_all_symbols::SymbolAndSignals> &Symbols);

  std::vector<find_all_symbols::SymbolAndSignals>
  search(llvm::StringRef Identifier) override;

private:
  std::map<std::string, std::vector<find_all_symbols::SymbolAndSignals>>
      LookupTable;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYSYMBOLINDEX_H
