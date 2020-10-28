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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/Archive.h"

namespace lld {
namespace macho {

class ArchiveFile;
class DylibFile;
class InputFile;
class InputSection;
class MachHeaderSection;
class Symbol;

/*
 * Note that the SymbolTable handles name collisions by calling
 * replaceSymbol(), which does an in-place update of the Symbol via `placement
 * new`. Therefore, there is no need to update any relocations that hold
 * pointers the "old" Symbol -- they will automatically point to the new one.
 */
class SymbolTable {
public:
  Symbol *addDefined(StringRef name, InputSection *isec, uint32_t value,
                     bool isWeakDef);

  Symbol *addUndefined(StringRef name);

  Symbol *addCommon(StringRef name, InputFile *, uint64_t size, uint32_t align);

  Symbol *addDylib(StringRef name, DylibFile *file, bool isWeakDef, bool isTlv);

  Symbol *addLazy(StringRef name, ArchiveFile *file,
                  const llvm::object::Archive::Symbol &sym);

  Symbol *addDSOHandle(const MachHeaderSection *);

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
