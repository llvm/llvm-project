//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SYMBOL_TABLE_H
#define LLD_MACHO_SYMBOL_TABLE_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/Object/Archive.h"

namespace lld {
namespace macho {

class InputFile;
class InputSection;
class ArchiveFile;
class Symbol;

class SymbolTable {
public:
  Symbol *addDefined(StringRef name, InputSection *isec, uint32_t value);

  Symbol *addUndefined(StringRef name);

  ArrayRef<Symbol *> getSymbols() const { return symVector; }
  Symbol *find(StringRef name);

private:
  std::pair<Symbol *, bool> insert(StringRef name);
  llvm::DenseMap<llvm::CachedHashStringRef, int> symMap;
  std::vector<Symbol *> symVector;
};

extern SymbolTable *symtab;

} // namespace macho
} // namespace lld

#endif
