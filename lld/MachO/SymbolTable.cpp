//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

Symbol *SymbolTable::find(StringRef name) {
  auto it = symMap.find(CachedHashStringRef(name));
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

Defined *SymbolTable::addDefined(StringRef name, InputFile *file,
                                 InputSection *isec, uint32_t value,
                                 bool isWeakDef, bool isPrivateExtern) {
  Symbol *s;
  bool wasInserted;
  bool overridesWeakDef = false;
  std::tie(s, wasInserted) = insert(name);

  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef) {
        // Both old and new symbol weak (e.g. inline function in two TUs):
        // If one of them isn't private extern, the merged symbol isn't.
        if (defined->isWeakDef())
          defined->privateExtern &= isPrivateExtern;
        return defined;
      }
      if (!defined->isWeakDef()) {
        error("duplicate symbol: " + name + "\n>>> defined in " +
              toString(defined->getFile()) + "\n>>> defined in " +
              toString(file));
      }
    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      overridesWeakDef = !isWeakDef && dysym->isWeakDef();
    }
    // Defined symbols take priority over other types of symbols, so in case
    // of a name conflict, we fall through to the replaceSymbol() call below.
  }

  Defined *defined =
      replaceSymbol<Defined>(s, name, file, isec, value, isWeakDef,
                             /*isExternal=*/true, isPrivateExtern);
  defined->overridesWeakDef = overridesWeakDef;
  return defined;
}

Symbol *SymbolTable::addUndefined(StringRef name, InputFile *file,
                                  bool isWeakRef) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  RefState refState = isWeakRef ? RefState::Weak : RefState::Strong;

  if (wasInserted)
    replaceSymbol<Undefined>(s, name, file, refState);
  else if (auto *lazy = dyn_cast<LazySymbol>(s))
    lazy->fetchArchiveMember();
  else if (auto *dynsym = dyn_cast<DylibSymbol>(s))
    dynsym->refState = std::max(dynsym->refState, refState);
  else if (auto *undefined = dyn_cast<Undefined>(s))
    undefined->refState = std::max(undefined->refState, refState);
  return s;
}

Symbol *SymbolTable::addCommon(StringRef name, InputFile *file, uint64_t size,
                               uint32_t align, bool isPrivateExtern) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (!wasInserted) {
    if (auto *common = dyn_cast<CommonSymbol>(s)) {
      if (size < common->size)
        return s;
    } else if (isa<Defined>(s)) {
      return s;
    }
    // Common symbols take priority over all non-Defined symbols, so in case of
    // a name conflict, we fall through to the replaceSymbol() call below.
  }

  replaceSymbol<CommonSymbol>(s, name, file, size, align, isPrivateExtern);
  return s;
}

Symbol *SymbolTable::addDylib(StringRef name, DylibFile *file, bool isWeakDef,
                              bool isTlv) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  RefState refState = RefState::Unreferenced;
  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef && !defined->isWeakDef())
        defined->overridesWeakDef = true;
    } else if (auto *undefined = dyn_cast<Undefined>(s)) {
      refState = undefined->refState;
    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      refState = dysym->refState;
    }
  }

  bool isDynamicLookup = file == nullptr;
  if (wasInserted || isa<Undefined>(s) ||
      (isa<DylibSymbol>(s) &&
       ((!isWeakDef && s->isWeakDef()) ||
        (!isDynamicLookup && cast<DylibSymbol>(s)->isDynamicLookup()))))
    replaceSymbol<DylibSymbol>(s, file, name, isWeakDef, refState, isTlv);

  return s;
}

Symbol *SymbolTable::addDynamicLookup(StringRef name) {
  return addDylib(name, /*file=*/nullptr, /*isWeakDef=*/false, /*isTlv=*/false);
}

Symbol *SymbolTable::addLazy(StringRef name, ArchiveFile *file,
                             const object::Archive::Symbol &sym) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name);

  if (wasInserted)
    replaceSymbol<LazySymbol>(s, file, sym);
  else if (isa<Undefined>(s) || (isa<DylibSymbol>(s) && s->isWeakDef()))
    file->fetch(sym);
  return s;
}

Defined *SymbolTable::addSynthetic(StringRef name, InputSection *isec,
                                   uint32_t value, bool isPrivateExtern,
                                   bool isLinkerInternal) {
  Defined *s = addDefined(name, nullptr, isec, value, /*isWeakDef=*/false,
                          isPrivateExtern);
  s->linkerInternal = isLinkerInternal;
  return s;
}

void lld::macho::treatUndefinedSymbol(const Undefined &sym) {
  auto message = [](const Undefined &sym) {
    std::string message = "undefined symbol: " + toString(sym);
    std::string fileName = toString(sym.getFile());
    if (!fileName.empty())
      message += "\n>>> referenced by " + fileName;
    return message;
  };
  switch (config->undefinedSymbolTreatment) {
  case UndefinedSymbolTreatment::error:
    error(message(sym));
    break;
  case UndefinedSymbolTreatment::warning:
    warn(message(sym));
    LLVM_FALLTHROUGH;
  case UndefinedSymbolTreatment::dynamic_lookup:
  case UndefinedSymbolTreatment::suppress:
    symtab->addDynamicLookup(sym.getName());
    break;
  case UndefinedSymbolTreatment::unknown:
    llvm_unreachable("unknown -undefined TREATMENT");
  }
}

SymbolTable *macho::symtab;
