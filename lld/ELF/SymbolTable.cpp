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

SymbolTable *elf::Symtab;

void SymbolTable::wrap(Symbol *Sym, Symbol *Real, Symbol *Wrap) {
  // Swap symbols as instructed by -wrap.
  int &Idx1 = SymMap[CachedHashStringRef(Sym->getName())];
  int &Idx2 = SymMap[CachedHashStringRef(Real->getName())];
  int &Idx3 = SymMap[CachedHashStringRef(Wrap->getName())];

  Idx2 = Idx1;
  Idx1 = Idx3;

  // Now renaming is complete. No one refers Real symbol. We could leave
  // Real as-is, but if Real is written to the symbol table, that may
  // contain irrelevant values. So, we copy all values from Sym to Real.
  StringRef S = Real->getName();
  memcpy(Real, Sym, sizeof(SymbolUnion));
  Real->setName(S);
}

// Find an existing symbol or create a new one.
Symbol *SymbolTable::insert(StringRef Name) {
  // <name>@@<version> means the symbol is the default version. In that
  // case <name>@@<version> will be used to resolve references to <name>.
  //
  // Since this is a hot path, the following string search code is
  // optimized for speed. StringRef::find(char) is much faster than
  // StringRef::find(StringRef).
  size_t Pos = Name.find('@');
  if (Pos != StringRef::npos && Pos + 1 < Name.size() && Name[Pos + 1] == '@')
    Name = Name.take_front(Pos);

  auto P = SymMap.insert({CachedHashStringRef(Name), (int)SymVector.size()});
  int &SymIndex = P.first->second;
  bool IsNew = P.second;

  if (!IsNew)
    return SymVector[SymIndex];

  Symbol *Sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
  SymVector.push_back(Sym);

  Sym->setName(Name);
  Sym->SymbolKind = Symbol::PlaceholderKind;
  Sym->VersionId = Config->DefaultSymbolVersion;
  Sym->Visibility = STV_DEFAULT;
  Sym->IsUsedInRegularObj = false;
  Sym->ExportDynamic = false;
  Sym->CanInline = true;
  Sym->ScriptDefined = false;
  Sym->Partition = 1;
  return Sym;
}

Symbol *SymbolTable::addSymbol(const Symbol &New) {
  Symbol *Sym = Symtab->insert(New.getName());
  Sym->resolve(New);
  return Sym;
}

Symbol *SymbolTable::find(StringRef Name) {
  auto It = SymMap.find(CachedHashStringRef(Name));
  if (It == SymMap.end())
    return nullptr;
  Symbol *Sym = SymVector[It->second];
  if (Sym->isPlaceholder())
    return nullptr;
  return Sym;
}

// Initialize DemangledSyms with a map from demangled symbols to symbol
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
  if (!DemangledSyms) {
    DemangledSyms.emplace();
    for (Symbol *Sym : SymVector) {
      if (!Sym->isDefined() && !Sym->isCommon())
        continue;
      if (Optional<std::string> S = demangleItanium(Sym->getName()))
        (*DemangledSyms)[*S].push_back(Sym);
      else
        (*DemangledSyms)[Sym->getName()].push_back(Sym);
    }
  }
  return *DemangledSyms;
}

std::vector<Symbol *> SymbolTable::findByVersion(SymbolVersion Ver) {
  if (Ver.IsExternCpp)
    return getDemangledSyms().lookup(Ver.Name);
  if (Symbol *B = find(Ver.Name))
    if (B->isDefined() || B->isCommon())
      return {B};
  return {};
}

std::vector<Symbol *> SymbolTable::findAllByVersion(SymbolVersion Ver) {
  std::vector<Symbol *> Res;
  StringMatcher M(Ver.Name);

  if (Ver.IsExternCpp) {
    for (auto &P : getDemangledSyms())
      if (M.match(P.first()))
        Res.insert(Res.end(), P.second.begin(), P.second.end());
    return Res;
  }

  for (Symbol *Sym : SymVector)
    if ((Sym->isDefined() || Sym->isCommon()) && M.match(Sym->getName()))
      Res.push_back(Sym);
  return Res;
}

// If there's only one anonymous version definition in a version
// script file, the script does not actually define any symbol version,
// but just specifies symbols visibilities.
void SymbolTable::handleAnonymousVersion() {
  for (SymbolVersion &Ver : Config->VersionScriptGlobals)
    assignExactVersion(Ver, VER_NDX_GLOBAL, "global");
  for (SymbolVersion &Ver : Config->VersionScriptGlobals)
    assignWildcardVersion(Ver, VER_NDX_GLOBAL);
  for (SymbolVersion &Ver : Config->VersionScriptLocals)
    assignExactVersion(Ver, VER_NDX_LOCAL, "local");
  for (SymbolVersion &Ver : Config->VersionScriptLocals)
    assignWildcardVersion(Ver, VER_NDX_LOCAL);
}

// Handles -dynamic-list.
void SymbolTable::handleDynamicList() {
  for (SymbolVersion &Ver : Config->DynamicList) {
    std::vector<Symbol *> Syms;
    if (Ver.HasWildcard)
      Syms = findAllByVersion(Ver);
    else
      Syms = findByVersion(Ver);

    for (Symbol *B : Syms) {
      if (!Config->Shared)
        B->ExportDynamic = true;
      else if (B->includeInDynsym())
        B->IsPreemptible = true;
    }
  }
}

// Set symbol versions to symbols. This function handles patterns
// containing no wildcard characters.
void SymbolTable::assignExactVersion(SymbolVersion Ver, uint16_t VersionId,
                                     StringRef VersionName) {
  if (Ver.HasWildcard)
    return;

  // Get a list of symbols which we need to assign the version to.
  std::vector<Symbol *> Syms = findByVersion(Ver);
  if (Syms.empty()) {
    if (!Config->UndefinedVersion)
      error("version script assignment of '" + VersionName + "' to symbol '" +
            Ver.Name + "' failed: symbol not defined");
    return;
  }

  // Assign the version.
  for (Symbol *Sym : Syms) {
    // Skip symbols containing version info because symbol versions
    // specified by symbol names take precedence over version scripts.
    // See parseSymbolVersion().
    if (Sym->getName().contains('@'))
      continue;

    if (Sym->VersionId != Config->DefaultSymbolVersion &&
        Sym->VersionId != VersionId)
      error("duplicate symbol '" + Ver.Name + "' in version script");
    Sym->VersionId = VersionId;
  }
}

void SymbolTable::assignWildcardVersion(SymbolVersion Ver, uint16_t VersionId) {
  if (!Ver.HasWildcard)
    return;

  // Exact matching takes precendence over fuzzy matching,
  // so we set a version to a symbol only if no version has been assigned
  // to the symbol. This behavior is compatible with GNU.
  for (Symbol *B : findAllByVersion(Ver))
    if (B->VersionId == Config->DefaultSymbolVersion)
      B->VersionId = VersionId;
}

// This function processes version scripts by updating VersionId
// member of symbols.
void SymbolTable::scanVersionScript() {
  // Handle edge cases first.
  handleAnonymousVersion();
  handleDynamicList();

  // Now we have version definitions, so we need to set version ids to symbols.
  // Each version definition has a glob pattern, and all symbols that match
  // with the pattern get that version.

  // First, we assign versions to exact matching symbols,
  // i.e. version definitions not containing any glob meta-characters.
  for (VersionDefinition &V : Config->VersionDefinitions)
    for (SymbolVersion &Ver : V.Globals)
      assignExactVersion(Ver, V.Id, V.Name);

  // Next, we assign versions to fuzzy matching symbols,
  // i.e. version definitions containing glob meta-characters.
  // Note that because the last match takes precedence over previous matches,
  // we iterate over the definitions in the reverse order.
  for (VersionDefinition &V : llvm::reverse(Config->VersionDefinitions))
    for (SymbolVersion &Ver : V.Globals)
      assignWildcardVersion(Ver, V.Id);

  // Symbol themselves might know their versions because symbols
  // can contain versions in the form of <name>@<version>.
  // Let them parse and update their names to exclude version suffix.
  for (Symbol *Sym : SymVector)
    Sym->parseSymbolVersion();
}
