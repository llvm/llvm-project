//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

Symbol *SymbolTable::find(StringRef name) {
  auto it = symMap.find(llvm::CachedHashStringRef(name));
  if (it == symMap.end())
    return nullptr;
  return symVector[it->second];
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef name) {
  auto p = symMap.insert({CachedHashStringRef(name), (int)symVector.size()});

  // Name already present in the symbol table.
  if (!p.second)
    return {symVector[p.first->second], false};

  // Name is a new symbol.
  Symbol *sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
  symVector.push_back(sym);
  return {sym, true};
}

Symbol *SymbolTable::addDefined(StringRef name, InputSection *isec,
                                uint32_t value) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (!wasInserted && isa<Defined>(s))
    error("duplicate symbol: " + name);

  replaceSymbol<Defined>(s, name, isec, value);
  return s;
}

Symbol *SymbolTable::addUndefined(StringRef name) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (wasInserted)
    replaceSymbol<Undefined>(s, name);
  else if (LazySymbol *lazy = dyn_cast<LazySymbol>(s))
    lazy->fetchArchiveMember();
  return s;
}

Symbol *SymbolTable::addDylib(StringRef name, DylibFile *file) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (wasInserted || isa<Undefined>(s))
    replaceSymbol<DylibSymbol>(s, file, name);
  return s;
}

Symbol *SymbolTable::addLazy(StringRef name, ArchiveFile *file,
                             const llvm::object::Archive::Symbol &sym) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (wasInserted)
    replaceSymbol<LazySymbol>(s, file, sym);
  else if (isa<Undefined>(s))
    file->fetch(sym);
  return s;
}

SymbolTable *macho::symtab;
