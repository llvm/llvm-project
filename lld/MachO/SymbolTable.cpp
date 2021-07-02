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

Symbol *SymbolTable::find(CachedHashStringRef cachedName) {
  auto it = symMap.find(cachedName);
  if (it == symMap.end())
    return nullptr;
  return symVector[it->second];
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef name,
                                              const InputFile *file) {
  auto p = symMap.insert({CachedHashStringRef(name), (int)symVector.size()});

  Symbol *sym;
  if (!p.second) {
    // Name already present in the symbol table.
    sym = symVector[p.first->second];
  } else {
    // Name is a new symbol.
    sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
    symVector.push_back(sym);
  }

  sym->isUsedInRegularObj |= !file || isa<ObjFile>(file);
  return {sym, p.second};
}

Defined *SymbolTable::addDefined(StringRef name, InputFile *file,
                                 InputSection *isec, uint64_t value,
                                 uint64_t size, bool isWeakDef,
                                 bool isPrivateExtern, bool isThumb,
                                 bool isReferencedDynamically,
                                 bool noDeadStrip) {
  Symbol *s;
  bool wasInserted;
  bool overridesWeakDef = false;
  std::tie(s, wasInserted) = insert(name, file);

  assert(!isWeakDef || (isa<BitcodeFile>(file) && !isec) ||
         (isa<ObjFile>(file) && file == isec->getFile()));

  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef) {
        if (defined->isWeakDef()) {
          // Both old and new symbol weak (e.g. inline function in two TUs):
          // If one of them isn't private extern, the merged symbol isn't.
          defined->privateExtern &= isPrivateExtern;
          defined->referencedDynamically |= isReferencedDynamically;
          defined->noDeadStrip |= noDeadStrip;

          // FIXME: Handle this for bitcode files.
          // FIXME: We currently only do this if both symbols are weak.
          //        We could do this if either is weak (but getting the
          //        case where !isWeakDef && defined->isWeakDef() right
          //        requires some care and testing).
          if (auto concatIsec = dyn_cast_or_null<ConcatInputSection>(isec))
            concatIsec->wasCoalesced = true;
        }

        return defined;
      }
      if (!defined->isWeakDef())
        error("duplicate symbol: " + name + "\n>>> defined in " +
              toString(defined->getFile()) + "\n>>> defined in " +
              toString(file));
    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      overridesWeakDef = !isWeakDef && dysym->isWeakDef();
      dysym->unreference();
    }
    // Defined symbols take priority over other types of symbols, so in case
    // of a name conflict, we fall through to the replaceSymbol() call below.
  }

  Defined *defined = replaceSymbol<Defined>(
      s, name, file, isec, value, size, isWeakDef, /*isExternal=*/true,
      isPrivateExtern, isThumb, isReferencedDynamically, noDeadStrip);
  defined->overridesWeakDef = overridesWeakDef;
  return defined;
}

Symbol *SymbolTable::addUndefined(StringRef name, InputFile *file,
                                  bool isWeakRef) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  RefState refState = isWeakRef ? RefState::Weak : RefState::Strong;

  if (wasInserted)
    replaceSymbol<Undefined>(s, name, file, refState);
  else if (auto *lazy = dyn_cast<LazySymbol>(s))
    lazy->fetchArchiveMember();
  else if (auto *dynsym = dyn_cast<DylibSymbol>(s))
    dynsym->reference(refState);
  else if (auto *undefined = dyn_cast<Undefined>(s))
    undefined->refState = std::max(undefined->refState, refState);
  return s;
}

Symbol *SymbolTable::addCommon(StringRef name, InputFile *file, uint64_t size,
                               uint32_t align, bool isPrivateExtern) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

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
  std::tie(s, wasInserted) = insert(name, file);

  RefState refState = RefState::Unreferenced;
  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef && !defined->isWeakDef())
        defined->overridesWeakDef = true;
    } else if (auto *undefined = dyn_cast<Undefined>(s)) {
      refState = undefined->refState;
    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      refState = dysym->getRefState();
    }
  }

  bool isDynamicLookup = file == nullptr;
  if (wasInserted || isa<Undefined>(s) ||
      (isa<DylibSymbol>(s) &&
       ((!isWeakDef && s->isWeakDef()) ||
        (!isDynamicLookup && cast<DylibSymbol>(s)->isDynamicLookup())))) {
    if (auto *dynsym = dyn_cast<DylibSymbol>(s))
      dynsym->unreference();
    replaceSymbol<DylibSymbol>(s, file, name, isWeakDef, refState, isTlv);
  }

  return s;
}

Symbol *SymbolTable::addDynamicLookup(StringRef name) {
  return addDylib(name, /*file=*/nullptr, /*isWeakDef=*/false, /*isTlv=*/false);
}

Symbol *SymbolTable::addLazy(StringRef name, ArchiveFile *file,
                             const object::Archive::Symbol &sym) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  if (wasInserted)
    replaceSymbol<LazySymbol>(s, file, sym);
  else if (isa<Undefined>(s) || (isa<DylibSymbol>(s) && s->isWeakDef()))
    file->fetch(sym);
  return s;
}

Defined *SymbolTable::addSynthetic(StringRef name, InputSection *isec,
                                   uint64_t value, bool isPrivateExtern,
                                   bool includeInSymtab,
                                   bool referencedDynamically) {
  Defined *s = addDefined(name, nullptr, isec, value, /*size=*/0,
                          /*isWeakDef=*/false, isPrivateExtern,
                          /*isThumb=*/false, referencedDynamically,
                          /*noDeadStrip=*/false);
  s->includeInSymtab = includeInSymtab;
  return s;
}

void lld::macho::treatUndefinedSymbol(const Undefined &sym, StringRef source) {
  auto message = [source, &sym]() {
    std::string message = "undefined symbol: " + toString(sym);
    if (!source.empty())
      message += "\n>>> referenced by " + source.str();
    else
      message += "\n>>> referenced by " + toString(sym.getFile());
    return message;
  };
  switch (config->undefinedSymbolTreatment) {
  case UndefinedSymbolTreatment::error:
    error(message());
    break;
  case UndefinedSymbolTreatment::warning:
    warn(message());
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
