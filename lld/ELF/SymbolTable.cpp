//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbol table is a bag of all known symbols. We put all symbols of
// all input files to the symbol table. The symbol table is basically
// a hash table with the logic to resolve symbol name conflicts using
// the symbol types.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "LinkerScript.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

SymbolTable *elf::symtab;

void SymbolTable::wrap(Symbol *sym, Symbol *real, Symbol *wrap) {
  // Swap symbols as instructed by -wrap.
  int &idx1 = symMap[CachedHashStringRef(sym->getName())];
  int &idx2 = symMap[CachedHashStringRef(real->getName())];
  int &idx3 = symMap[CachedHashStringRef(wrap->getName())];

  idx2 = idx1;
  idx1 = idx3;

  if (real->exportDynamic)
    sym->exportDynamic = true;
  if (!real->isUsedInRegularObj && sym->isUndefined())
    sym->isUsedInRegularObj = false;

  // Now renaming is complete, and no one refers to real. We drop real from
  // .symtab and .dynsym. If real is undefined, it is important that we don't
  // leave it in .dynsym, because otherwise it might lead to an undefined symbol
  // error in a subsequent link. If real is defined, we could emit real as an
  // alias for sym, but that could degrade the user experience of some tools
  // that can print out only one symbol for each location: sym is a preferred
  // name than real, but they might print out real instead.
  memcpy(real, sym, sizeof(SymbolUnion));
  real->isUsedInRegularObj = false;
}

// Find an existing symbol or create a new one.
Symbol *SymbolTable::insert(StringRef name) {
  // <name>@@<version> means the symbol is the default version. In that
  // case <name>@@<version> will be used to resolve references to <name>.
  //
  // Since this is a hot path, the following string search code is
  // optimized for speed. StringRef::find(char) is much faster than
  // StringRef::find(StringRef).
  size_t pos = name.find('@');
  if (pos != StringRef::npos && pos + 1 < name.size() && name[pos + 1] == '@')
    name = name.take_front(pos);

  auto p = symMap.insert({CachedHashStringRef(name), (int)symVector.size()});
  int &symIndex = p.first->second;
  bool isNew = p.second;

  if (!isNew)
    return symVector[symIndex];

  Symbol *sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
  symVector.push_back(sym);

  // *sym was not initialized by a constructor. Fields that may get referenced
  // when it is a placeholder must be initialized here.
  sym->setName(name);
  sym->symbolKind = Symbol::PlaceholderKind;
  sym->versionId = VER_NDX_GLOBAL;
  sym->visibility = STV_DEFAULT;
  sym->isUsedInRegularObj = false;
  sym->exportDynamic = false;
  sym->inDynamicList = false;
  sym->canInline = true;
  sym->referenced = false;
  sym->traced = false;
  sym->scriptDefined = false;
  sym->partition = 1;
  return sym;
}

Symbol *SymbolTable::addSymbol(const Symbol &newSym) {
  Symbol *sym = insert(newSym.getName());
  sym->resolve(newSym);
  return sym;
}

Symbol *SymbolTable::find(StringRef name) {
  auto it = symMap.find(CachedHashStringRef(name));
  if (it == symMap.end())
    return nullptr;
  Symbol *sym = symVector[it->second];
  if (sym->isPlaceholder())
    return nullptr;
  return sym;
}

// A version script/dynamic list is only meaningful for a Defined symbol.
// A CommonSymbol will be converted to a Defined in replaceCommonSymbols().
// A lazy symbol may be made Defined if an LTO libcall fetches it.
static bool canBeVersioned(const Symbol &sym) {
  return sym.isDefined() || sym.isCommon() || sym.isLazy();
}

// Initialize demangledSyms with a map from demangled symbols to symbol
// objects. Used to handle "extern C++" directive in version scripts.
//
// The map will contain all demangled symbols. That can be very large,
// and in LLD we generally want to avoid do anything for each symbol.
// Then, why are we doing this? Here's why.
//
// Users can use "extern C++ {}" directive to match against demangled
// C++ symbols. For example, you can write a pattern such as
// "llvm::*::foo(int, ?)". Obviously, there's no way to handle this
// other than trying to match a pattern against all demangled symbols.
// So, if "extern C++" feature is used, we need to demangle all known
// symbols.
StringMap<std::vector<Symbol *>> &SymbolTable::getDemangledSyms() {
  if (!demangledSyms) {
    demangledSyms.emplace();
    for (Symbol *sym : symVector)
      if (canBeVersioned(*sym))
        (*demangledSyms)[demangleItanium(sym->getName())].push_back(sym);
  }
  return *demangledSyms;
}

std::vector<Symbol *> SymbolTable::findByVersion(SymbolVersion ver) {
  if (ver.isExternCpp)
    return getDemangledSyms().lookup(ver.name);
  if (Symbol *sym = find(ver.name))
    if (canBeVersioned(*sym))
      return {sym};
  return {};
}

std::vector<Symbol *> SymbolTable::findAllByVersion(SymbolVersion ver) {
  std::vector<Symbol *> res;
  SingleStringMatcher m(ver.name);

  if (ver.isExternCpp) {
    for (auto &p : getDemangledSyms())
      if (m.match(p.first()))
        res.insert(res.end(), p.second.begin(), p.second.end());
    return res;
  }

  for (Symbol *sym : symVector)
    if (canBeVersioned(*sym) && m.match(sym->getName()))
      res.push_back(sym);
  return res;
}

// Handles -dynamic-list.
void SymbolTable::handleDynamicList() {
  for (SymbolVersion &ver : config->dynamicList) {
    std::vector<Symbol *> syms;
    if (ver.hasWildcard)
      syms = findAllByVersion(ver);
    else
      syms = findByVersion(ver);

    for (Symbol *sym : syms)
      sym->inDynamicList = true;
  }
}

// Set symbol versions to symbols. This function handles patterns
// containing no wildcard characters.
void SymbolTable::assignExactVersion(SymbolVersion ver, uint16_t versionId,
                                     StringRef versionName) {
  if (ver.hasWildcard)
    return;

  // Get a list of symbols which we need to assign the version to.
  std::vector<Symbol *> syms = findByVersion(ver);
  if (syms.empty()) {
    if (!config->undefinedVersion)
      error("version script assignment of '" + versionName + "' to symbol '" +
            ver.name + "' failed: symbol not defined");
    return;
  }

  auto getName = [](uint16_t ver) -> std::string {
    if (ver == VER_NDX_LOCAL)
      return "VER_NDX_LOCAL";
    if (ver == VER_NDX_GLOBAL)
      return "VER_NDX_GLOBAL";
    return ("version '" + config->versionDefinitions[ver].name + "'").str();
  };

  // Assign the version.
  for (Symbol *sym : syms) {
    // Skip symbols containing version info because symbol versions
    // specified by symbol names take precedence over version scripts.
    // See parseSymbolVersion().
    if (sym->getName().contains('@'))
      continue;

    // If the version has not been assigned, verdefIndex is -1. Use an arbitrary
    // number (0) to indicate the version has been assigned.
    if (sym->verdefIndex == UINT32_C(-1)) {
      sym->verdefIndex = 0;
      sym->versionId = versionId;
    }
    if (sym->versionId == versionId)
      continue;

    warn("attempt to reassign symbol '" + ver.name + "' of " +
         getName(sym->versionId) + " to " + getName(versionId));
  }
}

void SymbolTable::assignWildcardVersion(SymbolVersion ver, uint16_t versionId) {
  // Exact matching takes precedence over fuzzy matching,
  // so we set a version to a symbol only if no version has been assigned
  // to the symbol. This behavior is compatible with GNU.
  for (Symbol *sym : findAllByVersion(ver))
    if (sym->verdefIndex == UINT32_C(-1)) {
      sym->verdefIndex = 0;
      sym->versionId = versionId;
    }
}

// This function processes version scripts by updating the versionId
// member of symbols.
// If there's only one anonymous version definition in a version
// script file, the script does not actually define any symbol version,
// but just specifies symbols visibilities.
void SymbolTable::scanVersionScript() {
  // First, we assign versions to exact matching symbols,
  // i.e. version definitions not containing any glob meta-characters.
  for (VersionDefinition &v : config->versionDefinitions)
    for (SymbolVersion &pat : v.patterns)
      assignExactVersion(pat, v.id, v.name);

  // Next, assign versions to wildcards that are not "*". Note that because the
  // last match takes precedence over previous matches, we iterate over the
  // definitions in the reverse order.
  for (VersionDefinition &v : llvm::reverse(config->versionDefinitions))
    for (SymbolVersion &pat : v.patterns)
      if (pat.hasWildcard && pat.name != "*")
        assignWildcardVersion(pat, v.id);

  // Then, assign versions to "*". In GNU linkers they have lower priority than
  // other wildcards.
  for (VersionDefinition &v : config->versionDefinitions)
    for (SymbolVersion &pat : v.patterns)
      if (pat.hasWildcard && pat.name == "*")
        assignWildcardVersion(pat, v.id);

  // Symbol themselves might know their versions because symbols
  // can contain versions in the form of <name>@<version>.
  // Let them parse and update their names to exclude version suffix.
  for (Symbol *sym : symVector)
    sym->parseSymbolVersion();

  // isPreemptible is false at this point. To correctly compute the binding of a
  // Defined (which is used by includeInDynsym()), we need to know if it is
  // VER_NDX_LOCAL or not. Compute symbol versions before handling
  // --dynamic-list.
  handleDynamicList();
}
