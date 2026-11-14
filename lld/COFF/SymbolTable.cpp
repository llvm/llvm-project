//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "COFFLinkerContext.h"
#include "Config.h"
#include "Driver.h"
#include "LTO.h"
#include "PDB.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Timer.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/COFFModuleDefinition.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;
using namespace llvm::COFF;
using namespace llvm::object;
using namespace llvm::support;

namespace lld::coff {

StringRef ltrim1(StringRef s, const char *chars) {
  if (!s.empty() && strchr(chars, s[0]))
    return s.substr(1);
  return s;
}

static COFFSyncStream errorOrWarn(COFFLinkerContext &ctx) {
  return {ctx, ctx.config.forceUnresolved ? DiagLevel::Warn : DiagLevel::Err};
}

// Causes the file associated with a lazy symbol to be linked in.
static void forceLazy(Symbol *s) {
  s->pendingArchiveLoad = true;
  switch (s->kind()) {
  case Symbol::Kind::LazyArchiveKind: {
    auto *l = cast<LazyArchive>(s);
    l->file->addMember(l->sym);
    break;
  }
  case Symbol::Kind::LazyObjectKind: {
    InputFile *file = cast<LazyObject>(s)->file;
    // FIXME: Remove this once we resolve all defineds before all undefineds in
    //        ObjFile::initializeSymbols().
    if (!file->lazy)
      return;
    file->lazy = false;
    file->symtab.ctx.driver.addFile(file);
    break;
  }
  case Symbol::Kind::LazyDLLSymbolKind: {
    auto *l = cast<LazyDLLSymbol>(s);
    l->file->makeImport(l->sym);
    break;
  }
  default:
    llvm_unreachable(
        "symbol passed to forceLazy is not a LazyArchive or LazyObject");
  }
}

// Returns the symbol in SC whose value is <= Addr that is closest to Addr.
// This is generally the global variable or function whose definition contains
// Addr.
static Symbol *getSymbol(SectionChunk *sc, uint32_t addr) {
  DefinedRegular *candidate = nullptr;

  for (Symbol *s : sc->file->getSymbols()) {
    auto *d = dyn_cast_or_null<DefinedRegular>(s);
    if (!d || !d->data || d->file != sc->file || d->getChunk() != sc ||
        d->getValue() > addr ||
        (candidate && d->getValue() < candidate->getValue()))
      continue;

    candidate = d;
  }

  return candidate;
}

static std::vector<std::string> getSymbolLocations(BitcodeFile *file) {
  std::string res("\n>>> referenced by ");
  StringRef source = file->obj->getSourceFileName();
  if (!source.empty())
    res += source.str() + "\n>>>               ";
  res += toString(file);
  return {res};
}

static std::optional<std::pair<StringRef, uint32_t>>
getFileLineDwarf(const SectionChunk *c, uint32_t addr) {
  std::optional<DILineInfo> optionalLineInfo =
      c->file->getDILineInfo(addr, c->getSectionNumber() - 1);
  if (!optionalLineInfo)
    return std::nullopt;
  const DILineInfo &lineInfo = *optionalLineInfo;
  if (lineInfo.FileName == DILineInfo::BadString)
    return std::nullopt;
  return std::make_pair(saver().save(lineInfo.FileName), lineInfo.Line);
}

static std::optional<std::pair<StringRef, uint32_t>>
getFileLine(const SectionChunk *c, uint32_t addr) {
  // MinGW can optionally use codeview, even if the default is dwarf.
  std::optional<std::pair<StringRef, uint32_t>> fileLine =
      getFileLineCodeView(c, addr);
  // If codeview didn't yield any result, check dwarf in MinGW mode.
  if (!fileLine && c->file->symtab.ctx.config.mingw)
    fileLine = getFileLineDwarf(c, addr);
  return fileLine;
}

// Given a file and the index of a symbol in that file, returns a description
// of all references to that symbol from that file. If no debug information is
// available, returns just the name of the file, else one string per actual
// reference as described in the debug info.
// Returns up to maxStrings string descriptions, along with the total number of
// locations found.
static std::pair<std::vector<std::string>, size_t>
getSymbolLocations(ObjFile *file, uint32_t symIndex, size_t maxStrings) {
  struct Location {
    Symbol *sym;
    std::pair<StringRef, uint32_t> fileLine;
  };
  std::vector<Location> locations;
  size_t numLocations = 0;

  for (Chunk *c : file->getChunks()) {
    auto *sc = dyn_cast<SectionChunk>(c);
    if (!sc)
      continue;
    for (const coff_relocation &r : sc->getRelocs()) {
      if (r.SymbolTableIndex != symIndex)
        continue;
      numLocations++;
      if (locations.size() >= maxStrings)
        continue;

      std::optional<std::pair<StringRef, uint32_t>> fileLine =
          getFileLine(sc, r.VirtualAddress);
      Symbol *sym = getSymbol(sc, r.VirtualAddress);
      if (fileLine)
        locations.push_back({sym, *fileLine});
      else if (sym)
        locations.push_back({sym, {"", 0}});
    }
  }

  if (maxStrings == 0)
    return std::make_pair(std::vector<std::string>(), numLocations);

  if (numLocations == 0)
    return std::make_pair(
        std::vector<std::string>{"\n>>> referenced by " + toString(file)}, 1);

  std::vector<std::string> symbolLocations(locations.size());
  size_t i = 0;
  for (Location loc : locations) {
    llvm::raw_string_ostream os(symbolLocations[i++]);
    os << "\n>>> referenced by ";
    if (!loc.fileLine.first.empty())
      os << loc.fileLine.first << ":" << loc.fileLine.second
         << "\n>>>               ";
    os << toString(file);
    if (loc.sym)
      os << ":(" << toString(file->symtab.ctx, *loc.sym) << ')';
  }
  return std::make_pair(symbolLocations, numLocations);
}

std::vector<std::string> getSymbolLocations(ObjFile *file, uint32_t symIndex) {
  return getSymbolLocations(file, symIndex, SIZE_MAX).first;
}

static std::pair<std::vector<std::string>, size_t>
getSymbolLocations(InputFile *file, uint32_t symIndex, size_t maxStrings) {
  if (auto *o = dyn_cast<ObjFile>(file))
    return getSymbolLocations(o, symIndex, maxStrings);
  if (auto *b = dyn_cast<BitcodeFile>(file)) {
    std::vector<std::string> symbolLocations = getSymbolLocations(b);
    size_t numLocations = symbolLocations.size();
    if (symbolLocations.size() > maxStrings)
      symbolLocations.resize(maxStrings);
    return std::make_pair(symbolLocations, numLocations);
  }
  llvm_unreachable("unsupported file type passed to getSymbolLocations");
  return std::make_pair(std::vector<std::string>(), (size_t)0);
}

// For an undefined symbol, stores all files referencing it and the index of
// the undefined symbol in each file.
struct UndefinedDiag {
  Symbol *sym;
  struct File {
    InputFile *file;
    uint32_t symIndex;
  };
  std::vector<File> files;
};

void SymbolTable::reportUndefinedSymbol(const UndefinedDiag &undefDiag) {
  auto diag = errorOrWarn(ctx);
  diag << "undefined symbol: " << printSymbol(undefDiag.sym);

  const size_t maxUndefReferences = 3;
  size_t numDisplayedRefs = 0, numRefs = 0;
  for (const UndefinedDiag::File &ref : undefDiag.files) {
    auto [symbolLocations, totalLocations] = getSymbolLocations(
        ref.file, ref.symIndex, maxUndefReferences - numDisplayedRefs);

    numRefs += totalLocations;
    numDisplayedRefs += symbolLocations.size();
    for (const std::string &s : symbolLocations)
      diag << s;
  }
  if (numDisplayedRefs < numRefs)
    diag << "\n>>> referenced " << numRefs - numDisplayedRefs << " more times";

  // Hints
  StringRef name = undefDiag.sym->getName();
  if (name.consume_front("__imp_")) {
    Symbol *imp = find(name);
    if (imp && imp->isLazy()) {
      diag << "\nNOTE: a relevant symbol '" << imp->getName()
           << "' is available in " << toString(imp->getFile())
           << " but cannot be used because it is not an import library.";
    }
  }
}

void SymbolTable::loadMinGWSymbols(bool loadStdcallFixups,
                                   bool loadAutoImports) {
  SmallPtrSet<Symbol *, 8> seen;
  std::vector<Symbol *> undefs;
  auto addUndefined = [&](Symbol *sym) {
    auto *undef = dyn_cast_or_null<Undefined>(sym);
    if (!undef || undef->getWeakAlias() || !seen.insert(sym).second)
      return;
    undefs.push_back(sym);
  };
  for (auto &i : symMap) {
    Symbol *sym = i.second;
    if (sym->hasUndefinedReference)
      addUndefined(sym);
  }

  // References from replaceable COMDAT groups are provisional until the
  // final prevailing group is known. Let the MinGW fixup pass inspect them so
  // that it can load a provider which may affect COMDAT selection, but do not
  // publish them through markReferenced(). If their group survives, they are
  // attributed normally by resolveDeferredReferences().
  for (SectionChunk *chunk : deferredReferenceChunks)
    if (!chunk->isDiscardedByCOMDAT())
      for (Symbol *sym : chunk->symbols())
        addUndefined(sym);

  for (auto sym : undefs) {
    auto *undef = dyn_cast<Undefined>(sym);
    if (!undef)
      continue;
    if (undef->getWeakAlias())
      continue;
    StringRef name = undef->getName();

    if (loadStdcallFixups && machine == I386 && ctx.config.stdcallFixup) {
      // Check if we can resolve an undefined decorated symbol by finding
      // the intended target as an undecorated symbol (only with a leading
      // underscore).
      StringRef origName = name;
      StringRef baseName = name;
      // Trim down stdcall/fastcall/vectorcall symbols to the base name.
      baseName = ltrim1(baseName, "_@");
      baseName = baseName.substr(0, baseName.find('@'));
      // Add a leading underscore, as it would be in cdecl form.
      std::string newName = ("_" + baseName).str();
      Symbol *l;
      if (newName != origName && (l = find(newName)) != nullptr) {
        // If we found a symbol and it is lazy; load it.
        if (l->isLazy() && !l->pendingArchiveLoad) {
          Log(ctx) << "Loading lazy " << l->getName() << " from "
                   << l->getFile()->getName() << " for stdcall fixup";
          forceLazy(l);
        }
        // If it's lazy or already defined, hook it up as weak alias.
        if (l->isLazy() || isa<Defined>(l)) {
          if (ctx.config.warnStdcallFixup)
            Warn(ctx) << "Resolving " << origName << " by linking to "
                      << newName;
          else
            Log(ctx) << "Resolving " << origName << " by linking to "
                     << newName;
          undef->setWeakAlias(l);
          continue;
        }
      }
    }

    if (loadAutoImports && ctx.config.autoImport) {
      if (name.starts_with("__imp_"))
        continue;
      // If we have an undefined symbol, but we have a lazy symbol we could
      // load, load it.
      Symbol *l = find(("__imp_" + name).str());
      if (!l || l->pendingArchiveLoad || !l->isLazy())
        continue;

      Log(ctx) << "Loading lazy " << l->getName() << " from "
               << l->getFile()->getName() << " for automatic import";
      forceLazy(l);
    }
  }
}

Defined *SymbolTable::impSymbol(StringRef name) {
  if (name.starts_with("__imp_"))
    return nullptr;
  return dyn_cast_or_null<Defined>(find(("__imp_" + name).str()));
}

bool SymbolTable::handleMinGWAutomaticImport(Symbol *sym, StringRef name) {
  Defined *imp = impSymbol(name);
  if (!imp)
    return false;

  // Replace the reference directly to a variable with a reference
  // to the import address table instead. This obviously isn't right,
  // but we mark the symbol as isRuntimePseudoReloc, and a later pass
  // will add runtime pseudo relocations for every relocation against
  // this Symbol. The runtime pseudo relocation framework expects the
  // reference itself to point at the IAT entry.
  size_t impSize = 0;
  if (isa<DefinedImportData>(imp)) {
    Log(ctx) << "Automatically importing " << name << " from "
             << cast<DefinedImportData>(imp)->getDLLName();
    impSize = sizeof(DefinedImportData);
  } else if (isa<DefinedRegular>(imp)) {
    Log(ctx) << "Automatically importing " << name << " from "
             << toString(cast<DefinedRegular>(imp)->file);
    impSize = sizeof(DefinedRegular);
  } else {
    Warn(ctx) << "unable to automatically import " << name << " from "
              << imp->getName() << " from " << cast<DefinedRegular>(imp)->file
              << "; unexpected symbol type";
    return false;
  }
  sym->replaceKeepingName(imp, impSize);
  sym->isRuntimePseudoReloc = true;

  // There may exist symbols named .refptr.<name> which only consist
  // of a single pointer to <name>. If it turns out <name> is
  // automatically imported, we don't need to keep the .refptr.<name>
  // pointer at all, but redirect all accesses to it to the IAT entry
  // for __imp_<name> instead, and drop the whole .refptr.<name> chunk.
  DefinedRegular *refptr =
      dyn_cast_or_null<DefinedRegular>(find((".refptr." + name).str()));
  if (refptr && refptr->getChunk()->getSize() == ctx.config.wordsize) {
    SectionChunk *sc = dyn_cast_or_null<SectionChunk>(refptr->getChunk());
    if (sc && sc->getRelocs().size() == 1 && *sc->symbols().begin() == sym) {
      Log(ctx) << "Replacing .refptr." << name << " with " << imp->getName();
      refptr->getChunk()->live = false;
      refptr->replaceKeepingName(imp, impSize);
    }
  }
  return true;
}

/// Helper function for reportUnresolvable and resolveRemainingUndefines.
/// This function emits an "undefined symbol" diagnostic for each symbol in
/// undefs. If localImports is not nullptr, it also emits a "locally
/// defined symbol imported" diagnostic for symbols in localImports.
/// objFiles and bitcodeFiles (if not nullptr) are used to report where
/// undefined symbols are referenced.
void SymbolTable::reportProblemSymbols(
    const SmallPtrSetImpl<Symbol *> &undefs,
    const DenseMap<Symbol *, Symbol *> *localImports, bool needBitcodeFiles) {
  // Return early if there is nothing to report (which should be
  // the common case).
  if (undefs.empty() && (!localImports || localImports->empty()))
    return;

  for (Symbol *b : ctx.config.gcroot) {
    if (undefs.contains(b))
      errorOrWarn(ctx) << "<root>: undefined symbol: " << printSymbol(b);
    if (localImports)
      if (Symbol *imp = localImports->lookup(b))
        Warn(ctx) << "<root>: locally defined symbol imported: "
                  << printSymbol(imp) << " (defined in "
                  << toString(imp->getFile()) << ") [LNK4217]";
  }

  std::vector<UndefinedDiag> undefDiags;
  DenseMap<Symbol *, int> firstDiag;

  auto processFile = [&](InputFile *file, ArrayRef<Symbol *> symbols) {
    uint32_t symIndex = (uint32_t)-1;
    for (Symbol *sym : symbols) {
      ++symIndex;
      if (!sym)
        continue;
      if (undefs.contains(sym)) {
        auto [it, inserted] = firstDiag.try_emplace(sym, undefDiags.size());
        if (inserted)
          undefDiags.push_back({sym, {{file, symIndex}}});
        else
          undefDiags[it->second].files.push_back({file, symIndex});
      }
      if (localImports)
        if (Symbol *imp = localImports->lookup(sym))
          Warn(ctx) << file
                    << ": locally defined symbol imported: " << printSymbol(imp)
                    << " (defined in " << imp->getFile() << ") [LNK4217]";
    }
  };

  for (ObjFile *file : ctx.objFileInstances)
    processFile(file, file->getSymbols());

  if (needBitcodeFiles)
    for (BitcodeFile *file : bitcodeFileInstances)
      processFile(file, file->getSymbols());

  for (const UndefinedDiag &undefDiag : undefDiags)
    reportUndefinedSymbol(undefDiag);
}

void SymbolTable::reportUnresolvable() {
  SmallPtrSet<Symbol *, 8> undefs;
  for (auto &i : symMap) {
    Symbol *sym = i.second;
    auto *undef = dyn_cast<Undefined>(sym);
    // Definitions from discarded COMDATs are represented as Undefined so
    // that they no longer participate in symbol-table consumers. Do not
    // diagnose them unless a real undefined reference was also observed.
    if (!undef || !sym->hasUndefinedReference || sym->deferUndefined)
      continue;
    if (undef->getWeakAlias())
      continue;
    StringRef name = undef->getName();
    if (name.starts_with("__imp_")) {
      Symbol *imp = find(name.substr(strlen("__imp_")));
      if (Defined *def = dyn_cast_or_null<Defined>(imp)) {
        def->isUsedInRegularObj = true;
        continue;
      }
    }
    if (name.contains("_PchSym_"))
      continue;
    if (ctx.config.autoImport && impSymbol(name))
      continue;
    undefs.insert(sym);
  }

  reportProblemSymbols(undefs, /*localImports=*/nullptr, true);
}

void SymbolTable::resolveRemainingUndefines(std::vector<Undefined *> &aliases) {
  llvm::TimeTraceScope timeScope("Resolve remaining undefined symbols");
  SmallPtrSet<Symbol *, 8> undefs;
  DenseMap<Symbol *, Symbol *> localImports;

  for (auto &i : symMap) {
    Symbol *sym = i.second;
    auto *undef = dyn_cast<Undefined>(sym);
    if (!undef)
      continue;
    if (!sym->isUsedInRegularObj)
      continue;

    StringRef name = undef->getName();

    // A weak alias may have been resolved, so check for that.
    if (undef->getWeakAlias()) {
      aliases.push_back(undef);
      continue;
    }

    // If we can resolve a symbol by removing __imp_ prefix, do that.
    // This odd rule is for compatibility with MSVC linker.
    if (name.starts_with("__imp_")) {
      auto findLocalSym = [&](StringRef n) {
        Symbol *sym = find(n);
        return sym ? sym->getDefined() : nullptr;
      };

      StringRef impName = name.substr(strlen("__imp_"));
      Defined *imp = findLocalSym(impName);
      if (!imp && isEC()) {
        // Try to use the mangled symbol on ARM64EC.
        std::optional<std::string> mangledName =
            getArm64ECMangledFunctionName(impName);
        if (mangledName)
          imp = findLocalSym(*mangledName);
        if (!imp && impName.consume_front("aux_")) {
          // If it's a __imp_aux_ symbol, try skipping the aux_ prefix.
          imp = findLocalSym(impName);
          if (!imp && (mangledName = getArm64ECMangledFunctionName(impName)))
            imp = findLocalSym(*mangledName);
        }
      }
      if (imp) {
        replaceSymbol<DefinedLocalImport>(sym, ctx, name, imp);
        localImportChunks.push_back(cast<DefinedLocalImport>(sym)->getChunk());
        localImports[sym] = imp;
        continue;
      }
    }

    // We don't want to report missing Microsoft precompiled headers symbols.
    // A proper message will be emitted instead in PDBLinker::aquirePrecompObj
    if (name.contains("_PchSym_"))
      continue;

    if (ctx.config.autoImport && handleMinGWAutomaticImport(sym, name))
      continue;

    // Remaining undefined symbols are not fatal if /force is specified.
    // They are replaced with dummy defined symbols.
    if (ctx.config.forceUnresolved)
      replaceSymbol<DefinedAbsolute>(sym, ctx, name, 0);
    undefs.insert(sym);
  }

  reportProblemSymbols(
      undefs, ctx.config.warnLocallyDefinedImported ? &localImports : nullptr,
      false);
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef name) {
  Symbol *&sym = symMap[CachedHashStringRef(name)];
  if (sym)
    return {sym, false};

  // Give the symbol slot a valid dynamic type immediately. Besides making the
  // default state explicit, this ensures replaceSymbol() never reads fields
  // through a pointer to storage in which no Symbol object exists yet.
  sym = new (make<SymbolUnion>()) Undefined(name);
  if (isEC() && name.starts_with("EXP+"))
    expSymbols.push_back(sym);
  return {sym, true};
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef name, InputFile *file) {
  std::pair<Symbol *, bool> result = insert(name);
  if (!file || !isa<BitcodeFile>(file))
    result.first->isUsedInRegularObj = true;
  return result;
}

void SymbolTable::initializeLoadConfig() {
  auto sym =
      dyn_cast_or_null<DefinedRegular>(findUnderscore("_load_config_used"));
  if (!sym) {
    if (isEC()) {
      Warn(ctx) << "EC version of '_load_config_used' is missing";
      return;
    }
    if (ctx.config.machine == ARM64X) {
      Warn(ctx) << "native version of '_load_config_used' is missing for "
                   "ARM64X target";
      return;
    }
    if (ctx.config.guardCF != GuardCFLevel::Off)
      Warn(ctx)
          << "Control Flow Guard is enabled but '_load_config_used' is missing";
    if (ctx.config.dependentLoadFlags)
      Warn(ctx) << "_load_config_used not found, /dependentloadflag will have "
                   "no effect";
    return;
  }

  SectionChunk *sc = sym->getChunk();
  if (!sc->hasData) {
    Err(ctx) << "_load_config_used points to uninitialized data";
    return;
  }
  uint64_t offsetInChunk = sym->getValue();
  if (offsetInChunk + 4 > sc->getSize()) {
    Err(ctx) << "_load_config_used section chunk is too small";
    return;
  }

  ArrayRef<uint8_t> secContents = sc->getContents();
  loadConfigSize =
      *reinterpret_cast<const ulittle32_t *>(&secContents[offsetInChunk]);
  if (offsetInChunk + loadConfigSize > sc->getSize()) {
    Err(ctx) << "_load_config_used specifies a size larger than its containing "
                "section chunk";
    return;
  }

  uint32_t expectedAlign = ctx.config.is64() ? 8 : 4;
  if (sc->getAlignment() < expectedAlign)
    Warn(ctx) << "'_load_config_used' is misaligned (expected alignment to be "
              << expectedAlign << " bytes, got " << sc->getAlignment()
              << " instead)";
  else if (!isAligned(Align(expectedAlign), offsetInChunk))
    Warn(ctx) << "'_load_config_used' is misaligned (section offset is 0x"
              << Twine::utohexstr(sym->getValue()) << " not aligned to "
              << expectedAlign << " bytes)";

  loadConfigSym = sym;
}

void SymbolTable::addEntryThunk(Symbol *from, Symbol *to) {
  entryThunks.push_back({from, to});
}

void SymbolTable::addExitThunk(Symbol *from, Symbol *to) {
  exitThunks[from] = to;
}

void SymbolTable::initializeECThunks() {
  if (!isArm64EC(ctx.config.machine))
    return;

  for (auto it : entryThunks) {
    Defined *to = it.second->getDefined();
    if (!to)
      continue;
    auto *from = dyn_cast_or_null<DefinedRegular>(it.first->getDefined());
    // We need to be able to add padding to the function and fill it with an
    // offset to its entry thunks. To ensure that padding the function is
    // feasible, functions are required to be COMDAT symbols with no offset.
    if (!from || !from->getChunk()->isCOMDAT() ||
        cast<DefinedRegular>(from)->getValue()) {
      Err(ctx) << "non COMDAT symbol '" << from->getName() << "' in hybrid map";
      continue;
    }
    from->getChunk()->setEntryThunk(to);
  }

  for (ImportFile *file : ctx.importFileInstances) {
    if (!file->impchkThunk)
      continue;

    Symbol *sym = exitThunks.lookup(file->thunkSym);
    if (!sym)
      sym = exitThunks.lookup(file->impECSym);
    if (sym)
      file->impchkThunk->exitThunk = sym->getDefined();
  }

  // On ARM64EC, the __imp_ symbol references the auxiliary IAT, while the
  // __imp_aux_ symbol references the regular IAT. However, x86_64 code expects
  // both to reference the regular IAT, so adjust the symbol if necessary.
  parallelForEach(ctx.objFileInstances, [&](ObjFile *file) {
    if (file->getMachineType() != AMD64)
      return;
    for (auto &sym : file->getMutableSymbols()) {
      auto impSym = dyn_cast_or_null<DefinedImportData>(sym);
      if (impSym && impSym->file->impchkThunk && sym == impSym->file->impECSym)
        sym = impSym->file->impSym;
    }
  });
}

void SymbolTable::initializeSameAddressThunks() {
  for (auto iter : ctx.config.sameAddresses) {
    auto sym = dyn_cast_or_null<DefinedRegular>(iter.first->getDefined());
    if (!sym || !sym->isLive())
      continue;
    auto nativeSym =
        dyn_cast_or_null<DefinedRegular>(iter.second->getDefined());
    if (!nativeSym || !nativeSym->isLive())
      continue;
    Defined *entryThunk = sym->getChunk()->getEntryThunk();
    if (!entryThunk)
      continue;

    // Replace symbols with symbols referencing the thunk. Store the original
    // symbol as equivalent DefinedSynthetic instances for use in the thunk
    // itself.
    auto symClone = make<DefinedSynthetic>(sym->getName(), sym->getChunk(),
                                           sym->getValue());
    auto nativeSymClone = make<DefinedSynthetic>(
        nativeSym->getName(), nativeSym->getChunk(), nativeSym->getValue());
    SameAddressThunkARM64EC *thunk =
        make<SameAddressThunkARM64EC>(nativeSymClone, symClone, entryThunk);
    sameAddressThunks.push_back(thunk);

    replaceSymbol<DefinedSynthetic>(sym, sym->getName(), thunk);
    replaceSymbol<DefinedSynthetic>(nativeSym, nativeSym->getName(), thunk);
  }
}

Symbol *SymbolTable::addUndefined(StringRef name, InputFile *f,
                                  bool overrideLazy, bool markReference) {
  Symbol *s = markReference ? insert(name, f).first : insert(name).first;
  if (markReference)
    s->hasUndefinedReference = true;

  if (!s->isLazy())
    return s;

  if (overrideLazy)
    return replaceSymbol<Undefined>(s, name);

  if (markReference)
    forceLazy(s);
  return s;
}

Symbol *SymbolTable::addDiscardedDefinition(StringRef name) {
  return insert(name).first;
}

Symbol *SymbolTable::canonicalizeDeferredCOMDATSymbol(Symbol *symbol) const {
  if (Symbol *source = deferredComdatLeaderSources.lookup(symbol))
    return source;
  return symbol;
}

bool SymbolTable::hasDeferredCOMDATWork() const {
  return !deferredSymbols.empty() || !deferredSymbolSources.empty() ||
         !deferredSymbolsToResolve.empty() ||
         !deferredReferenceChunks.empty() || !deferredDuplicates.empty();
}

static bool mayBeReplacedByLargerCOMDAT(Symbol *s) {
  auto *d = dyn_cast<DefinedRegular>(s);
  // Bitcode definitions use shared fake ANY chunks for all symbols, not just
  // members of a replaceable COMDAT. Treating those chunks as provisional
  // would defer unrelated strong/weak resolution and can mark multiple LTO
  // definitions as prevailing.
  if (!d || !d->data || !isa<ObjFile>(d->file))
    return false;
  SectionChunk *chunk = d->getChunk();
  if (chunk->sym == d)
    return false;
  return chunk->isInReplaceableCOMDATGroup();
}

static bool shouldReplaceWeakAlias(SymbolTable &symtab, InputFile *f,
                                   Symbol *source, Symbol *currentTarget,
                                   bool currentIsAntiDep, Symbol *target,
                                   bool isAntiDep) {
  if (currentTarget && currentTarget != target) {
    // Ignore duplicated anti-dependency symbols.
    if (isAntiDep)
      return false;
    if (!currentIsAntiDep) {
      // Weak aliases as produced by GCC are named in the form
      // .weak.<weaksymbol>.<othersymbol>, where <othersymbol> is the name of
      // another symbol emitted near the weak symbol.
      if (symtab.ctx.config.allowDuplicateWeak) {
        auto isAbsZero = [](Symbol *sym) -> bool {
          auto *absolute = dyn_cast<DefinedAbsolute>(sym);
          return absolute && absolute->getVA() == 0;
        };
        // If the alias we had points at absolute zero, and we get another weak
        // symbol which isn't absolute zero, prefer that one.
        return isAbsZero(currentTarget) && !isAbsZero(target);
      }
      symtab.reportDuplicate(source, f);
    }
  }
  return true;
}

void SymbolTable::addWeakAlias(InputFile *f, Symbol *source, Symbol *target,
                               bool isAntiDep) {
  auto *u = dyn_cast<Undefined>(source);
  if (!u) {
    // A definition suppresses weak aliases. Remember every alias for a regular
    // definition that may later be discarded with its COMDAT group. If that
    // happens, replaying the aliases in input order preserves the usual
    // conflict diagnostics and allowDuplicateWeak precedence.
    if (mayBeReplacedByLargerCOMDAT(source))
      deferredWeakAliases[source].suppressed.push_back({f, target, isAntiDep});
    return;
  }

  if (shouldReplaceWeakAlias(*this, f, source, u->weakAlias, u->isAntiDep,
                             target, isAntiDep)) {
    u->setWeakAlias(target, isAntiDep);
    if (source->hasUndefinedReference && u->isECAlias(machine))
      markReferenced(target);
  }
}

void SymbolTable::markReferenced(Symbol *symbol, bool forceLazyProvider) {
  bool wasReferenced = symbol->hasUndefinedReference;
  symbol->hasUndefinedReference = true;
  symbol->isUsedInRegularObj = true;
  if (forceLazyProvider && symbol->isLazy() && !symbol->pendingArchiveLoad)
    forceLazy(symbol);
  if (!wasReferenced)
    if (auto *undefined = dyn_cast<Undefined>(symbol);
        undefined && undefined->weakAlias && undefined->isECAlias(machine))
      markReferenced(undefined->weakAlias, forceLazyProvider);
}

void SymbolTable::addReference(Symbol *symbol) {
  if (symbol)
    markReferenced(symbol);
}

void SymbolTable::deferSectionReferences(SectionChunk *chunk) {
  if (!chunk->isDiscardedByCOMDAT())
    deferredReferenceChunks.push_back(chunk);
}

void SymbolTable::addUnrelocatedReference(Symbol *symbol) {
  if (symbol)
    markReferenced(symbol);
}

void SymbolTable::resolveDeferredReferences() {
  SmallVector<SectionChunk *, 0> chunks;
  chunks.swap(deferredReferenceChunks);
  for (SectionChunk *chunk : chunks) {
    if (chunk->isDiscardedByCOMDAT())
      continue;
    for (Symbol *symbol : chunk->symbols())
      if (symbol)
        markReferenced(symbol);
  }
}

void SymbolTable::finalizeDeferredSymbols() {
  SmallVector<Symbol *, 0> sources;
  sources.swap(deferredSymbolSources);
  replayDeferredSources(sources, /*importThunks=*/false, /*final=*/true);
  replayDeferredSources(sources, /*importThunks=*/true, /*final=*/true);
  assert(deferredSymbols.empty() && "deferred providers were not replayed");
  deferredProviders.clear();
  updateDeferredImportLiveness();

  SmallVector<DeferredDuplicate, 0> duplicates;
  duplicates.swap(deferredDuplicates);
  for (const DeferredDuplicate &duplicate : duplicates)
    if (!duplicate.chunk->isDiscardedByCOMDAT())
      reportDuplicate(duplicate.existing, duplicate.newFile, duplicate.chunk,
                      duplicate.sectionOffset);
}

void SymbolTable::restoreDeferredSymbolState(Undefined *source) {
  if (deferredWeakAliases.empty() && deferredSymbols.empty())
    return;

  auto it = deferredWeakAliases.find(source);
  if (it != deferredWeakAliases.end()) {
    DeferredWeakAliases aliases = std::move(it->second);
    deferredWeakAliases.erase(it);
    if (aliases.initialTarget)
      source->setWeakAlias(aliases.initialTarget, aliases.initialIsAntiDep);
    for (const DeferredWeakAlias &alias : aliases.suppressed)
      addWeakAlias(alias.file, source, alias.target, alias.isAntiDep);
  }

  if (deferredSymbols.contains(source) &&
      deferredSymbolsQueued.insert(source).second)
    deferredSymbolsToResolve.push_back(source);
}

Symbol *SymbolTable::cloneDeferredSymbol(Symbol *symbol) {
  switch (symbol->kind()) {
  case Symbol::DefinedRegularKind:
    llvm_unreachable("regular deferred providers use descriptors");
  case Symbol::DefinedCommonKind:
    return make<DefinedCommon>(*cast<DefinedCommon>(symbol));
  case Symbol::DefinedLocalImportKind:
    return make<DefinedLocalImport>(*cast<DefinedLocalImport>(symbol));
  case Symbol::DefinedImportThunkKind: {
    auto *thunk = cast<DefinedImportThunk>(symbol);
    auto *copy = make<DefinedImportThunk>(*thunk);
    deferredImportThunks[copy] = {thunk->wrappedSym->file,
                                  thunk->wrappedSym->getName()};
    return copy;
  }
  case Symbol::DefinedImportDataKind:
    return make<DefinedImportData>(*cast<DefinedImportData>(symbol));
  case Symbol::DefinedAbsoluteKind:
    return make<DefinedAbsolute>(*cast<DefinedAbsolute>(symbol));
  case Symbol::DefinedSyntheticKind:
    return make<DefinedSynthetic>(*cast<DefinedSynthetic>(symbol));
  case Symbol::LazyArchiveKind:
    return make<LazyArchive>(*cast<LazyArchive>(symbol));
  case Symbol::LazyObjectKind:
    return make<LazyObject>(*cast<LazyObject>(symbol));
  case Symbol::LazyDLLSymbolKind:
    return make<LazyDLLSymbol>(*cast<LazyDLLSymbol>(symbol));
  case Symbol::UndefinedKind:
    llvm_unreachable("an undefined symbol is not a deferred provider");
  }
  llvm_unreachable("unknown deferred symbol kind");
}

void SymbolTable::appendDeferredProvider(Symbol *source,
                                         DeferredProvider provider) {
  auto [it, inserted] = deferredSymbols.try_emplace(source);
  if (inserted)
    deferredSymbolSources.push_back(source);

  if (deferredProviders.size() >= noDeferredProvider)
    Fatal(ctx) << "too many deferred COMDAT providers";
  DeferredProviderId id =
      static_cast<DeferredProviderId>(deferredProviders.size());
  deferredProviders.push_back(provider);
  DeferredSourceState &state = it->second;
  if (state.last == noDeferredProvider)
    state.first = id;
  else
    deferredProviders[state.last].next = id;
  state.last = id;
}

void SymbolTable::deferDefinedSymbol(Symbol *source, Symbol *provider,
                                     bool isUsedInRegularObj, Symbol *parent) {
  provider->isUsedInRegularObj = isUsedInRegularObj;
  if (auto *common = dyn_cast<DefinedCommon>(provider))
    if (CommonChunk *chunk = common->getChunk())
      chunk->live = false;
  if (auto *data = dyn_cast<DefinedImportData>(provider)) {
    deferredImportFiles.insert(data->file);
    data->file->live = false;
  }
  if (auto *thunk = dyn_cast<DefinedImportThunk>(provider)) {
    thunk->getChunk()->live = false;
    auto it = deferredImportThunks.find(provider);
    assert(it != deferredImportThunks.end() &&
           "deferred import thunk has no import metadata");
    deferredImportFiles.insert(it->second.file);
  }

  DeferredProvider deferred;
  deferred.symbol = provider;
  deferred.parent = parent;
  appendDeferredProvider(source, deferred);
}

void SymbolTable::deferRegularSymbol(Symbol *source, InputFile *file,
                                     const coff_symbol_generic *coffSym,
                                     SectionChunk *chunk, uint32_t value,
                                     bool isWeak, Symbol *parent) {
  DeferredProvider provider;
  provider.regularFile = file;
  provider.regularSym = coffSym;
  provider.regularChunk = chunk;
  provider.parent = parent;
  provider.regularValue = value;
  provider.isRegular = true;
  provider.regularIsWeak = isWeak;
  appendDeferredProvider(source, provider);
}

template <typename T>
static void updateDeferredImportField(T *&field, const Symbol *source,
                                      const Symbol *provider, T *selected) {
  // ImportFile caches derived pointers into replaceable SymbolUnion storage.
  // Compare addresses without dereferencing an object whose lifetime may have
  // ended in replaceSymbol().
  const void *address = field;
  if (address == source || address == provider)
    field = selected;
}

void SymbolTable::replayDeferredSymbol(Symbol *source,
                                       DeferredProviderId providerId) {
  DeferredProvider &deferred = deferredProviders[providerId];
  if (Symbol *parent = deferred.parent) {
    auto selection = deferredComdatSelections.find(parent);
    assert(selection != deferredComdatSelections.end() &&
           "deferred COMDAT child replayed before its leader");
    if (!selection->second)
      return;
  }

  StringRef name = source->getName();
  if (deferred.isRegular) {
    if (deferred.regularChunk && deferred.regularChunk->isDiscardedByCOMDAT())
      return;
    addRegular(deferred.regularFile, name, deferred.regularSym,
               deferred.regularChunk, deferred.regularValue,
               deferred.regularIsWeak);
    return;
  }

  Symbol *provider = deferred.symbol;
  assert(provider && "non-regular deferred provider has no symbol payload");
  switch (provider->kind()) {
  case Symbol::DefinedRegularKind: {
    auto *regular = cast<DefinedRegular>(provider);
    SectionChunk *chunk = regular->data ? regular->getChunk() : nullptr;
    if (chunk && chunk->isDiscardedByCOMDAT())
      return;
    const coff_symbol_generic *coffSym = nullptr;
    uint32_t value = 0;
    if (isa<ObjFile>(regular->file)) {
      coffSym = regular->getCOFFSymbol().getGeneric();
      value = regular->getValue();
    }
    if (regular->isCOMDAT) {
      auto recordSelection = [&](bool selected) {
        deferredComdatSelections[provider] = selected;
        if (!selected)
          return;
        if (Symbol *old = selectedDeferredComdats.lookup(source))
          deferredComdatSelections[old] = false;
        selectedDeferredComdats[source] = provider;
      };

      if (!isa<DefinedRegular>(source)) {
        replaceSymbol<DefinedRegular>(source, *regular);
        if (chunk && isa<ObjFile>(regular->file))
          chunk->sym = cast<DefinedRegular>(source);
        recordSelection(true);
      } else {
        auto *existing = cast<DefinedRegular>(source);
        auto *file = dyn_cast_or_null<ObjFile>(regular->file);
        if (!existing->isCOMDAT) {
          reportDuplicate(source, regular->file, chunk, value);
          if (chunk && file)
            file->discardCOMDATGroup(chunk);
          recordSelection(false);
        } else if (chunk && file) {
          COMDATType selection = chunk->selection;
          if (SectionChunk *replaced = file->handleComdatSelection(
                  chunk, selection, existing, value, /*def=*/nullptr)) {
            replaced->file->discardCOMDATGroup(replaced);
            replaceSymbol<DefinedRegular>(source, *regular);
            chunk->setSelection(selection);
            chunk->sym = cast<DefinedRegular>(source);
            recordSelection(true);
          } else {
            file->discardCOMDATGroup(chunk);
            recordSelection(false);
          }
        } else {
          recordSelection(false);
        }
      }
      deferredComdatLeaderSources.erase(provider);
      return;
    }
    addRegular(regular->file, name, coffSym, chunk, value, regular->isWeak);
    return;
  }
  case Symbol::DefinedCommonKind: {
    auto *common = cast<DefinedCommon>(provider);
    const coff_symbol_generic *coffSym = nullptr;
    if (isa<ObjFile>(common->file))
      coffSym = common->getCOFFSymbol().getGeneric();
    addCommon(common->file, name, common->getSize(), coffSym,
              common->getChunk());
    return;
  }
  case Symbol::DefinedImportDataKind: {
    auto *data = cast<DefinedImportData>(provider);
    DefinedImportData *selected =
        addImportData(name, data->file, data->location);
    if (selected && selected->file != data->file)
      selected = nullptr;
    updateDeferredImportField(data->file->impSym, source, provider, selected);
    updateDeferredImportField(data->file->impECSym, source, provider, selected);
    updateDeferredImportField(data->file->auxImpCopySym, source, provider,
                              selected);
    return;
  }
  case Symbol::DefinedImportThunkKind: {
    auto *thunk = cast<DefinedImportThunk>(provider);
    auto infoIt = deferredImportThunks.find(provider);
    assert(infoIt != deferredImportThunks.end() &&
           "deferred import thunk replay has no import metadata");
    DeferredImportThunk info = infoIt->second;
    deferredImportThunks.erase(infoIt);
    auto *importData =
        dyn_cast_or_null<DefinedImportData>(find(info.importName));
    if (importData && importData->file != info.file)
      importData = nullptr;
    Defined *selected =
        importData ? addImportThunk(name, importData, thunk->getChunk())
                   : nullptr;
    if (selected &&
        (!isa<DefinedImportThunk>(selected) ||
         cast<DefinedImportThunk>(selected)->wrappedSym->file != info.file))
      selected = nullptr;
    thunk->getChunk()->live = selected && !ctx.config.doGC;
    updateDeferredImportField(info.file->thunkSym, source, provider, selected);
    updateDeferredImportField(info.file->auxThunkSym, source, provider,
                              selected);
    if (info.file->impchkThunk == thunk->getChunk()) {
      if (selected)
        info.file->impchkThunk->sym = selected;
      else
        info.file->impchkThunk = nullptr;
    }
    return;
  }
  case Symbol::DefinedAbsoluteKind:
    addAbsolute(name, cast<DefinedAbsolute>(provider)->getVA());
    return;
  case Symbol::DefinedSyntheticKind:
    addSynthetic(name, cast<DefinedSynthetic>(provider)->getChunk());
    return;
  case Symbol::DefinedLocalImportKind:
    if (isa<Undefined>(source) || source->isLazy())
      replaceSymbol<DefinedLocalImport>(source,
                                        *cast<DefinedLocalImport>(provider));
    else
      reportDuplicate(source, nullptr);
    return;
  case Symbol::LazyArchiveKind: {
    auto *lazy = cast<LazyArchive>(provider);
    addLazyArchive(lazy->file, lazy->sym);
    return;
  }
  case Symbol::LazyObjectKind: {
    auto *lazy = cast<LazyObject>(provider);
    addLazyObject(lazy->file, name);
    return;
  }
  case Symbol::LazyDLLSymbolKind: {
    auto *lazy = cast<LazyDLLSymbol>(provider);
    addLazyDLLSymbol(lazy->file, lazy->sym, name);
    return;
  }
  case Symbol::UndefinedKind:
    llvm_unreachable("an undefined symbol is not a deferred provider");
  }
  llvm_unreachable("unknown deferred symbol kind");
}

void SymbolTable::replayDeferredSources(ArrayRef<Symbol *> sources,
                                        bool importThunks, bool final) {
  auto isComdatLeader = [&](DeferredProviderId id) {
    Symbol *provider = deferredProviders[id].symbol;
    auto *regular = dyn_cast_or_null<DefinedRegular>(provider);
    return regular && regular->isCOMDAT;
  };
  auto isImportThunk = [&](DeferredProviderId id) {
    return isa_and_nonnull<DefinedImportThunk>(deferredProviders[id].symbol);
  };
  auto canReplay = [&](Symbol *source) {
    return final || isa<Undefined>(source) ||
           !mayBeReplacedByLargerCOMDAT(source);
  };

  auto replayProviders = [&](Symbol *source, bool replayComdatLeaders) {
    auto deferred = deferredSymbols.find(source);
    if (deferred == deferredSymbols.end() || !canReplay(source))
      return;

    DeferredSourceState state = deferred->second;
    DeferredProviderId replayFirst = noDeferredProvider;
    DeferredProviderId replayLast = noDeferredProvider;
    DeferredProviderId remainingFirst = noDeferredProvider;
    DeferredProviderId remainingLast = noDeferredProvider;

    auto append = [&](DeferredProviderId id, DeferredProviderId &first,
                      DeferredProviderId &last) {
      if (last == noDeferredProvider)
        first = id;
      else
        deferredProviders[last].next = id;
      last = id;
    };

    for (DeferredProviderId id = state.first; id != noDeferredProvider;) {
      assert(id < deferredProviders.size() &&
             "deferred provider list contains an invalid id");
      DeferredProvider &provider = deferredProviders[id];
      DeferredProviderId next = provider.next;
      Symbol *parent = provider.parent;
      bool awaitsParent = parent && !deferredComdatSelections.contains(parent);
      if (isImportThunk(id) != importThunks ||
          isComdatLeader(id) != replayComdatLeaders ||
          (!replayComdatLeaders && awaitsParent))
        append(id, remainingFirst, remainingLast);
      else
        append(id, replayFirst, replayLast);
      id = next;
    }

    if (remainingLast != noDeferredProvider)
      deferredProviders[remainingLast].next = noDeferredProvider;
    if (replayLast != noDeferredProvider)
      deferredProviders[replayLast].next = noDeferredProvider;

    if (remainingFirst == noDeferredProvider) {
      deferredSymbols.erase(deferred);
    } else {
      deferred->second.first = remainingFirst;
      deferred->second.last = remainingLast;
    }

    if (replayFirst == noDeferredProvider)
      return;

    replayingDeferredSymbols = true;
    for (DeferredProviderId id = replayFirst; id != noDeferredProvider;) {
      DeferredProviderId next = deferredProviders[id].next;
      replayDeferredSymbol(source, id);
      id = next;
    }
    replayingDeferredSymbols = false;
    updateDeferredCommonLiveness(replayFirst, source);
    if (!deferredSymbols.contains(source))
      deferredWeakAliases.erase(source);
  };

  auto hasComdatLeader = [&](const DeferredSourceState &state) {
    for (DeferredProviderId id = state.first; id != noDeferredProvider;
         id = deferredProviders[id].next)
      if (isComdatLeader(id))
        return true;
    return false;
  };

  // Collect leader sources first. The common no-dependency case should not pay
  // for a DenseMap or graph arrays.
  SmallVector<Symbol *, 8> leaderSources;
  SmallPtrSet<Symbol *, 8> leaderSourceSet;
  if (!importThunks) {
    for (Symbol *source : sources) {
      auto deferred = deferredSymbols.find(source);
      if (deferred == deferredSymbols.end())
        continue;
      if (canReplay(source) && hasComdatLeader(deferred->second) &&
          leaderSourceSet.insert(source).second)
        leaderSources.push_back(source);
    }
  }

  auto getDependency = [&](Symbol *source) -> Symbol * {
    Symbol *dependency = nullptr;
    if (auto *regular = dyn_cast<DefinedRegular>(source);
        regular && regular->data) {
      SectionChunk *chunk = regular->getChunk();
      if (chunk->isInReplaceableCOMDATGroup() && chunk->sym != regular)
        dependency = chunk->sym;
    }
    if (Symbol *dependencySource =
            deferredComdatLeaderSources.lookup(dependency))
      dependency = dependencySource;
    return dependency;
  };

  if (!leaderSources.empty()) {
    bool hasDependencies = false;
    for (Symbol *source : leaderSources) {
      Symbol *dependency = getDependency(source);
      if (dependency && leaderSourceSet.contains(dependency)) {
        hasDependencies = true;
        break;
      }
    }

    if (!hasDependencies) {
      for (Symbol *source : leaderSources)
        replayProviders(source, /*replayComdatLeaders=*/true);
    } else {
      if (leaderSources.size() >= noDeferredProvider)
        Fatal(ctx) << "too many deferred COMDAT leader sources";

      DenseMap<Symbol *, uint32_t> leaderIndex;
      leaderIndex.reserve(leaderSources.size());
      for (uint32_t i = 0; i != leaderSources.size(); ++i)
        leaderIndex.try_emplace(leaderSources[i], i);

      static constexpr uint32_t noLeader = ~uint32_t{0};
      SmallVector<uint32_t, 0> firstChild(leaderSources.size(), noLeader);
      SmallVector<uint32_t, 0> lastChild(leaderSources.size(), noLeader);
      SmallVector<uint32_t, 0> nextSibling(leaderSources.size(), noLeader);
      SmallVector<uint32_t, 8> worklist;

      for (uint32_t i = 0; i != leaderSources.size(); ++i) {
        Symbol *dependency = getDependency(leaderSources[i]);
        auto dependencyIt = leaderIndex.find(dependency);
        if (!dependency || dependencyIt == leaderIndex.end()) {
          worklist.push_back(i);
          continue;
        }

        uint32_t parent = dependencyIt->second;
        if (lastChild[parent] == noLeader)
          firstChild[parent] = i;
        else
          nextSibling[lastChild[parent]] = i;
        lastChild[parent] = i;
      }

      SmallVector<uint8_t, 0> replayed(leaderSources.size(), 0);
      for (size_t i = 0; i != worklist.size(); ++i) {
        uint32_t sourceIndex = worklist[i];
        if (replayed[sourceIndex])
          continue;
        replayed[sourceIndex] = 1;
        replayProviders(leaderSources[sourceIndex],
                        /*replayComdatLeaders=*/true);

        for (uint32_t child = firstChild[sourceIndex]; child != noLeader;
             child = nextSibling[child])
          worklist.push_back(child);
      }

      // Cyclic dependencies cannot be topologically ordered. Preserve input
      // order for every source that was not reached from a root.
      for (uint32_t i = 0; i != leaderSources.size(); ++i)
        if (!replayed[i])
          replayProviders(leaderSources[i], /*replayComdatLeaders=*/true);
    }
  }

  // Leader selection is now final, so child providers can be published in one
  // linear pass.
  for (Symbol *source : sources)
    replayProviders(source, /*replayComdatLeaders=*/false);
}

void SymbolTable::updateDeferredCommonLiveness(DeferredProviderId firstProvider,
                                               Symbol *source) {
  CommonChunk *selected = nullptr;
  if (auto *common = dyn_cast<DefinedCommon>(source))
    selected = common->getChunk();
  for (DeferredProviderId id = firstProvider; id != noDeferredProvider;
       id = deferredProviders[id].next) {
    const DeferredProvider &deferred = deferredProviders[id];
    if (deferred.isRegular)
      continue;
    Symbol *provider = deferred.symbol;
    if (auto *common = dyn_cast<DefinedCommon>(provider))
      if (CommonChunk *chunk = common->getChunk())
        chunk->live = chunk == selected;
  }
}

void SymbolTable::updateDeferredImportLiveness() {
  for (ImportFile *file : deferredImportFiles) {
    auto isSelected = [&](DefinedImportData *symbol) {
      return symbol && isa<DefinedImportData>(symbol) && symbol->file == file;
    };
    bool selected = isSelected(file->impSym);
    if (isArm64EC(file->getMachineType()))
      selected = selected && isSelected(file->impECSym) &&
                 isSelected(file->auxImpCopySym);
    file->live = selected && !ctx.config.doGC;
  }
}

void SymbolTable::resolveDeferredSymbols() {
  SmallVector<Symbol *, 0> symbols;
  symbols.swap(deferredSymbolsToResolve);
  deferredSymbolsQueued.clear();
  replayDeferredSources(symbols, /*importThunks=*/false, /*final=*/false);
  replayDeferredSources(symbols, /*importThunks=*/true, /*final=*/false);
  updateDeferredImportLiveness();
}

Symbol *SymbolTable::addGCRoot(StringRef name, bool aliasEC) {
  Symbol *b = addUndefined(name);
  if (!b->isGCRoot) {
    b->isGCRoot = true;
    ctx.config.gcroot.push_back(b);
  }

  // On ARM64EC, a symbol may be defined in either its mangled or demangled form
  // (or both). Define an anti-dependency symbol that binds both forms, similar
  // to how compiler-generated code references external functions.
  if (aliasEC && isEC()) {
    if (std::optional<std::string> mangledName =
            getArm64ECMangledFunctionName(name)) {
      auto u = dyn_cast<Undefined>(b);
      if (u && !u->weakAlias) {
        Symbol *t = addUndefined(saver().save(*mangledName));
        u->setWeakAlias(t, true);
      }
    } else if (std::optional<std::string> demangledName =
                   getArm64ECDemangledFunctionName(name)) {
      Symbol *us = addUndefined(saver().save(*demangledName));
      auto u = dyn_cast<Undefined>(us);
      if (u && !u->weakAlias)
        u->setWeakAlias(b, true);
    }
  }
  return b;
}

// On ARM64EC, a function symbol may appear in both mangled and demangled forms:
// - ARM64EC archives contain only the mangled name, while the demangled symbol
//   is defined by the object file as an alias.
// - x86_64 archives contain only the demangled name (the mangled name is
//   usually defined by an object referencing the symbol as an alias to a guess
//   exit thunk).
// - ARM64EC import files contain both the mangled and demangled names for
//   thunks.
// If more than one archive defines the same function, this could lead
// to different libraries being used for the same function depending on how they
// are referenced. Avoid this by checking if the paired symbol is already
// defined before adding a symbol to the table.
template <typename T>
bool checkLazyECPair(SymbolTable *symtab, StringRef name, InputFile *f) {
  if (name.starts_with("__imp_"))
    return true;
  std::string pairName;
  if (std::optional<std::string> mangledName =
          getArm64ECMangledFunctionName(name))
    pairName = std::move(*mangledName);
  else if (std::optional<std::string> demangledName =
               getArm64ECDemangledFunctionName(name))
    pairName = std::move(*demangledName);
  else
    return true;

  Symbol *sym = symtab->find(pairName);
  if (!sym)
    return true;
  if (mayBeReplacedByLargerCOMDAT(sym))
    return true;
  if (sym->pendingArchiveLoad)
    return false;
  if (auto u = dyn_cast<Undefined>(sym))
    return !u->weakAlias || u->isAntiDep;
  // If the symbol is lazy, allow it only if it originates from the same
  // archive.
  auto lazy = dyn_cast<T>(sym);
  return lazy && lazy->file == f;
}

void SymbolTable::addLazyArchive(ArchiveFile *f, const Archive::Symbol &sym) {
  StringRef name = sym.getName();
  if (isEC() && !checkLazyECPair<LazyArchive>(this, name, f))
    return;
  Symbol *s = insert(name).first;
  if (isa<Undefined>(s) && !s->hasUndefinedReference) {
    replaceSymbol<LazyArchive>(s, f, sym);
    return;
  }
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<LazyArchive>(f, sym), false);
    return;
  }
  auto *u = dyn_cast<Undefined>(s);
  if (!u || (u->weakAlias && !u->isECAlias(machine)) || s->pendingArchiveLoad)
    return;
  s->pendingArchiveLoad = true;
  f->addMember(sym);
}

void SymbolTable::addLazyObject(InputFile *f, StringRef n) {
  assert(f->lazy);
  if (isEC() && !checkLazyECPair<LazyObject>(this, n, f))
    return;
  Symbol *s = insert(n, f).first;
  if (isa<Undefined>(s) && !s->hasUndefinedReference) {
    replaceSymbol<LazyObject>(s, f, n);
    return;
  }
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<LazyObject>(f, n), false);
    return;
  }
  auto *u = dyn_cast<Undefined>(s);
  if (!u || (u->weakAlias && !u->isECAlias(machine)) || s->pendingArchiveLoad)
    return;
  s->pendingArchiveLoad = true;
  f->lazy = false;
  ctx.driver.addFile(f);
}

void SymbolTable::addLazyDLLSymbol(DLLFile *f, DLLFile::Symbol *sym,
                                   StringRef n) {
  Symbol *s = insert(n).first;
  if (isa<Undefined>(s) && !s->hasUndefinedReference) {
    replaceSymbol<LazyDLLSymbol>(s, f, sym, n);
    return;
  }
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<LazyDLLSymbol>(f, sym, n), false);
    return;
  }
  auto *u = dyn_cast<Undefined>(s);
  if (!u || (u->weakAlias && !u->isECAlias(machine)) || s->pendingArchiveLoad)
    return;
  s->pendingArchiveLoad = true;
  f->makeImport(sym);
}

static std::string getSourceLocationBitcode(BitcodeFile *file) {
  std::string res("\n>>> defined at ");
  StringRef source = file->obj->getSourceFileName();
  if (!source.empty())
    res += source.str() + "\n>>>            ";
  res += toString(file);
  return res;
}

static std::string getSourceLocationObj(ObjFile *file, SectionChunk *sc,
                                        uint32_t offset, StringRef name) {
  std::optional<std::pair<StringRef, uint32_t>> fileLine;
  if (sc)
    fileLine = getFileLine(sc, offset);
  if (!fileLine)
    fileLine = file->getVariableLocation(name);

  std::string res;
  llvm::raw_string_ostream os(res);
  os << "\n>>> defined at ";
  if (fileLine)
    os << fileLine->first << ":" << fileLine->second << "\n>>>            ";
  os << toString(file);
  return res;
}

static std::string getSourceLocation(InputFile *file, SectionChunk *sc,
                                     uint32_t offset, StringRef name) {
  if (!file)
    return "";
  if (auto *o = dyn_cast<ObjFile>(file))
    return getSourceLocationObj(o, sc, offset, name);
  if (auto *b = dyn_cast<BitcodeFile>(file))
    return getSourceLocationBitcode(b);
  return "\n>>> defined at " + toString(file);
}

// Construct and print an error message in the form of:
//
//   lld-link: error: duplicate symbol: foo
//   >>> defined at bar.c:30
//   >>>            bar.o
//   >>> defined at baz.c:563
//   >>>            baz.o
void SymbolTable::reportDuplicate(Symbol *existing, InputFile *newFile,
                                  SectionChunk *newSc,
                                  uint32_t newSectionOffset) {
  COFFSyncStream diag(ctx, ctx.config.forceMultiple ? DiagLevel::Warn
                                                    : DiagLevel::Err);
  diag << "duplicate symbol: " << printSymbol(existing);

  DefinedRegular *d = dyn_cast<DefinedRegular>(existing);
  if (d && isa<ObjFile>(d->getFile())) {
    diag << getSourceLocation(d->getFile(), d->getChunk(), d->getValue(),
                              existing->getName());
  } else {
    diag << getSourceLocation(existing->getFile(), nullptr, 0, "");
  }
  diag << getSourceLocation(newFile, newSc, newSectionOffset,
                            existing->getName());
}

Symbol *SymbolTable::addAbsolute(StringRef n, COFFSymbolRef sym) {
  auto [s, wasInserted] = insert(n, nullptr);
  s->isUsedInRegularObj = true;
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<DefinedAbsolute>(ctx, n, sym), true);
    return s;
  }
  if (wasInserted || isa<Undefined>(s) || s->isLazy())
    replaceSymbol<DefinedAbsolute>(s, ctx, n, sym);
  else if (auto *da = dyn_cast<DefinedAbsolute>(s)) {
    if (da->getVA() != sym.getValue())
      reportDuplicate(s, nullptr);
  } else if (!isa<DefinedCOFF>(s))
    reportDuplicate(s, nullptr);
  return s;
}

Symbol *SymbolTable::addAbsolute(StringRef n, uint64_t va) {
  auto [s, wasInserted] = insert(n, nullptr);
  s->isUsedInRegularObj = true;
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<DefinedAbsolute>(ctx, n, va), true);
    return s;
  }
  if (wasInserted || isa<Undefined>(s) || s->isLazy())
    replaceSymbol<DefinedAbsolute>(s, ctx, n, va);
  else if (auto *da = dyn_cast<DefinedAbsolute>(s)) {
    if (da->getVA() != va)
      reportDuplicate(s, nullptr);
  } else if (!isa<DefinedCOFF>(s))
    reportDuplicate(s, nullptr);
  return s;
}

Symbol *SymbolTable::addSynthetic(StringRef n, Chunk *c) {
  auto [s, wasInserted] = insert(n, nullptr);
  s->isUsedInRegularObj = true;
  if (wasInserted || isa<Undefined>(s) || s->isLazy())
    replaceSymbol<DefinedSynthetic>(s, n, c);
  else if (!isa<DefinedCOFF>(s))
    reportDuplicate(s, nullptr);
  return s;
}

Symbol *SymbolTable::addRegular(InputFile *f, StringRef n,
                                const coff_symbol_generic *sym, SectionChunk *c,
                                uint32_t sectionOffset, bool isWeak) {
  Symbol *s = insert(n, f).first;
  if (auto *undefined = dyn_cast<Undefined>(s);
      undefined && !undefined->weakAlias)
    return replaceSymbol<DefinedRegular>(s, f, n, /*IsCOMDAT=*/false,
                                         /*IsExternal=*/true, sym, c, isWeak);

  if (replayingDeferredSymbols) {
    if (!isa<DefinedRegular>(s) || s->isWeak)
      return replaceSymbol<DefinedRegular>(s, f, n, /*IsCOMDAT=*/false,
                                           /*IsExternal=*/true, sym, c, isWeak);
    if (!isWeak)
      reportDuplicate(s, f, c, sectionOffset);
    return s;
  }

  bool provisional =
      isa<ObjFile>(f) && c && c->isInReplaceableCOMDATGroup() && c->sym != s;
  bool existingProvisional = mayBeReplacedByLargerCOMDAT(s);

  // Keep the normal symbol insertion path compact. In particular, a new
  // secondary COMDAT definition normally replaces an unaliased Undefined and
  // needs no deferred state at all.
  if (LLVM_LIKELY(!existingProvisional && !provisional)) {
    if (!isa<DefinedRegular>(s) || s->isWeak)
      return replaceSymbol<DefinedRegular>(s, f, n, /*IsCOMDAT=*/false,
                                           /*IsExternal=*/true, sym, c, isWeak);
    if (!isWeak)
      reportDuplicate(s, f, c, sectionOffset);
    return s;
  }

  return addRegularWithDeferredCOMDAT(f, n, sym, c, sectionOffset, isWeak, s,
                                      provisional, existingProvisional);
}

LLVM_ATTRIBUTE_NOINLINE Symbol *SymbolTable::addRegularWithDeferredCOMDAT(
    InputFile *f, StringRef n, const coff_symbol_generic *sym, SectionChunk *c,
    uint32_t sectionOffset, bool isWeak, Symbol *s, bool provisional,
    bool existingProvisional) {
  assert(!replayingDeferredSymbols && (provisional || existingProvisional) &&
         "deferred COMDAT path requires a provisional definition");

  // Replacing an undefined symbol destroys its selected weak alias. Preserve
  // that state only when the replacement can itself be discarded later.
  if (provisional) {
    if (auto *undefined = dyn_cast<Undefined>(s);
        undefined && undefined->weakAlias) {
      auto [it, inserted] = deferredWeakAliases.try_emplace(s);
      if (inserted) {
        it->second.initialTarget = undefined->weakAlias;
        it->second.initialIsAntiDep = undefined->isAntiDep;
      }
    }
  }

  // If a provider was registered before this provisional definition, retain
  // it before replacing the symbol-table slot. DefinedRegular providers stay
  // in the slot (the normal first-definition rule) and therefore cannot be
  // lost when this group is discarded.
  if (provisional && !isa<Undefined>(s) &&
      (!isa<DefinedRegular>(s) || s->isWeak)) {
    if (auto *regular = dyn_cast<DefinedRegular>(s)) {
      const coff_symbol_generic *providerSym = nullptr;
      uint32_t providerValue = 0;
      if (isa<ObjFile>(regular->file)) {
        providerSym = regular->getCOFFSymbol().getGeneric();
        providerValue = regular->getValue();
      }
      deferRegularSymbol(s, regular->file, providerSym,
                         regular->data ? regular->getChunk() : nullptr,
                         providerValue, regular->isWeak);
    } else {
      deferDefinedSymbol(s, cloneDeferredSymbol(s), s->isUsedInRegularObj);
    }
  }

  // A permanent strong regular/COMDAT provider that precedes the provisional
  // group remains the prevailing definition. Delay its duplicate diagnostic
  // until we know whether the new group survives. If the existing definition
  // is itself provisional, retain the new provider below: the existing group
  // may be discarded by a later LARGEST selection.
  if (provisional && isa<DefinedRegular>(s) && !s->isWeak &&
      !existingProvisional) {
    deferredDuplicates.push_back({c, s, f, sectionOffset});
    return s;
  }

  if (existingProvisional) {
    deferRegularSymbol(s, f, sym, c, sectionOffset, isWeak);
    return s;
  }

  if (!isa<DefinedRegular>(s) || s->isWeak) {
    s = replaceSymbol<DefinedRegular>(s, f, n, /*IsCOMDAT=*/false,
                                      /*IsExternal=*/true, sym, c, isWeak);
  } else {
    if (!isWeak)
      reportDuplicate(s, f, c, sectionOffset);
  }
  return s;
}

Symbol *SymbolTable::addDeferredComdatRegular(BitcodeFile *f, StringRef n,
                                              SectionChunk *c, bool isWeak,
                                              Symbol *comdatLeader) {
  Symbol *source = insert(n, f).first;
  deferRegularSymbol(source, f, /*coffSym=*/nullptr, c, /*value=*/0, isWeak,
                     comdatLeader);
  return source;
}

std::pair<DefinedRegular *, bool>
SymbolTable::addComdat(InputFile *f, StringRef n,
                       const coff_symbol_generic *sym) {
  Symbol *s = insert(n, f).first;
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    auto *provider = make<DefinedRegular>(f, n, /*IsCOMDAT=*/true,
                                          /*IsExternal=*/true, sym, nullptr);
    deferredComdatLeaderSources[provider] = s;
    deferDefinedSymbol(s, provider, !isa<BitcodeFile>(f));
    return {provider, true};
  }
  if (!isa<DefinedRegular>(s))
    return {replaceSymbol<DefinedRegular>(s, f, n, /*IsCOMDAT=*/true,
                                          /*IsExternal=*/true, sym, nullptr),
            true};
  auto *existingSymbol = cast<DefinedRegular>(s);
  if (!existingSymbol->isCOMDAT)
    reportDuplicate(s, f);
  return {existingSymbol, false};
}

Symbol *SymbolTable::addCommon(InputFile *f, StringRef n, uint64_t size,
                               const coff_symbol_generic *sym, CommonChunk *c) {
  auto [s, wasInserted] = insert(n, f);
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    deferDefinedSymbol(s, make<DefinedCommon>(f, n, size, sym, c), true);
    return s;
  }
  if (wasInserted || !isa<DefinedCOFF>(s)) {
    replaceSymbol<DefinedCommon>(s, f, n, size, sym, c);
  } else if (auto *dc = dyn_cast<DefinedCommon>(s)) {
    if (size > dc->getSize())
      replaceSymbol<DefinedCommon>(s, f, n, size, sym, c);
  }
  return s;
}

DefinedImportData *SymbolTable::addImportData(StringRef n, ImportFile *f,
                                              Chunk *&location) {
  auto [s, wasInserted] = insert(n, nullptr);
  s->isUsedInRegularObj = true;
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    auto *provider = make<DefinedImportData>(n, f, location);
    deferDefinedSymbol(s, provider, true);
    return provider;
  }
  if (wasInserted || isa<Undefined>(s) || s->isLazy()) {
    replaceSymbol<DefinedImportData>(s, n, f, location);
    return cast<DefinedImportData>(s);
  }

  reportDuplicate(s, f);
  return nullptr;
}

Defined *SymbolTable::addImportThunk(StringRef name, DefinedImportData *id,
                                     ImportThunkChunk *chunk) {
  auto [s, wasInserted] = insert(name, nullptr);
  s->isUsedInRegularObj = true;
  if (!replayingDeferredSymbols && mayBeReplacedByLargerCOMDAT(s)) {
    auto *provider = make<DefinedImportThunk>(ctx, name, id, chunk);
    deferredImportThunks[provider] = {id->file, id->getName()};
    deferDefinedSymbol(s, provider, true);
    return provider;
  }
  if (wasInserted || isa<Undefined>(s) || s->isLazy()) {
    replaceSymbol<DefinedImportThunk>(s, ctx, name, id, chunk);
    return cast<Defined>(s);
  }

  reportDuplicate(s, id->file);
  return nullptr;
}

void SymbolTable::addLibcall(StringRef name) {
  Symbol *sym = findUnderscore(name);
  if (!sym)
    return;

  if (auto *l = dyn_cast<LazyArchive>(sym)) {
    MemoryBufferRef mb = l->getMemberBuffer();
    if (isBitcode(mb))
      addUndefined(sym->getName());
  } else if (LazyObject *o = dyn_cast<LazyObject>(sym)) {
    if (isBitcode(o->file->mb))
      addUndefined(sym->getName());
  }
}

Symbol *SymbolTable::find(StringRef name) const {
  return symMap.lookup(CachedHashStringRef(name));
}

Symbol *SymbolTable::findUnderscore(StringRef name) const {
  if (machine == I386)
    return find(("_" + name).str());
  return find(name);
}

// Return all symbols that start with Prefix, possibly ignoring the first
// character of Prefix or the first character symbol.
std::vector<Symbol *> SymbolTable::getSymsWithPrefix(StringRef prefix) {
  std::vector<Symbol *> syms;
  for (auto pair : symMap) {
    StringRef name = pair.first.val();
    if (name.starts_with(prefix) || name.starts_with(prefix.drop_front()) ||
        name.drop_front().starts_with(prefix) ||
        name.drop_front().starts_with(prefix.drop_front())) {
      syms.push_back(pair.second);
    }
  }
  return syms;
}

Symbol *SymbolTable::findMangle(StringRef name) {
  if (Symbol *sym = find(name)) {
    if (auto *u = dyn_cast<Undefined>(sym)) {
      // We're specifically looking for weak aliases that ultimately resolve to
      // defined symbols, hence the call to getWeakAlias() instead of just using
      // the weakAlias member variable. This matches link.exe's behavior.
      if (Symbol *weakAlias = u->getWeakAlias())
        return weakAlias;
    } else {
      return sym;
    }
  }

  // Efficient fuzzy string lookup is impossible with a hash table, so iterate
  // the symbol table once and collect all possibly matching symbols into this
  // vector. Then compare each possibly matching symbol with each possible
  // mangling.
  std::vector<Symbol *> syms = getSymsWithPrefix(name);
  auto findByPrefix = [&syms](const Twine &t) -> Symbol * {
    std::string prefix = t.str();
    for (auto *s : syms)
      if (s->getName().starts_with(prefix))
        return s;
    return nullptr;
  };

  // For non-x86, just look for C++ functions.
  if (machine != I386)
    return findByPrefix("?" + name + "@@Y");

  if (!name.starts_with("_"))
    return nullptr;
  // Search for x86 stdcall function.
  if (Symbol *s = findByPrefix(name + "@"))
    return s;
  // Search for x86 fastcall function.
  if (Symbol *s = findByPrefix("@" + name.substr(1) + "@"))
    return s;
  // Search for x86 vectorcall function.
  if (Symbol *s = findByPrefix(name.substr(1) + "@@"))
    return s;
  // Search for x86 C++ non-member function.
  return findByPrefix("?" + name.substr(1) + "@@Y");
}

bool SymbolTable::findUnderscoreMangle(StringRef sym) {
  Symbol *s = findMangle(mangle(sym));
  return s && !isa<Undefined>(s);
}

// Symbol names are mangled by prepending "_" on x86.
StringRef SymbolTable::mangle(StringRef sym) {
  assert(machine != IMAGE_FILE_MACHINE_UNKNOWN);
  if (machine == I386)
    return saver().save("_" + sym);
  return sym;
}

StringRef SymbolTable::mangleMaybe(Symbol *s) {
  // If the plain symbol name has already been resolved, do nothing.
  Undefined *unmangled = dyn_cast<Undefined>(s);
  if (!unmangled)
    return "";

  // Otherwise, see if a similar, mangled symbol exists in the symbol table.
  Symbol *mangled = findMangle(unmangled->getName());
  if (!mangled)
    return "";

  // If we find a similar mangled symbol, make this an alias to it and return
  // its name.
  Log(ctx) << unmangled->getName() << " aliased to " << mangled->getName();
  unmangled->setWeakAlias(addUndefined(mangled->getName()));
  return mangled->getName();
}

// Windows specific -- find default entry point name.
//
// There are four different entry point functions for Windows executables,
// each of which corresponds to a user-defined "main" function. This function
// infers an entry point from a user-defined "main" function.
StringRef SymbolTable::findDefaultEntry() {
  assert(ctx.config.subsystem != IMAGE_SUBSYSTEM_UNKNOWN &&
         "must handle /subsystem before calling this");

  if (ctx.config.mingw)
    return mangle(ctx.config.subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI
                      ? "WinMainCRTStartup"
                      : "mainCRTStartup");

  if (ctx.config.subsystem == IMAGE_SUBSYSTEM_WINDOWS_GUI) {
    if (findUnderscoreMangle("wWinMain")) {
      if (!findUnderscoreMangle("WinMain"))
        return mangle("wWinMainCRTStartup");
      Warn(ctx) << "found both wWinMain and WinMain; using latter";
    }
    return mangle("WinMainCRTStartup");
  }
  if (findUnderscoreMangle("wmain")) {
    if (!findUnderscoreMangle("main"))
      return mangle("wmainCRTStartup");
    Warn(ctx) << "found both wmain and main; using latter";
  }
  return mangle("mainCRTStartup");
}

WindowsSubsystem SymbolTable::inferSubsystem() {
  if (ctx.config.dll)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  if (ctx.config.mingw)
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  // Note that link.exe infers the subsystem from the presence of these
  // functions even if /entry: or /nodefaultlib are passed which causes them
  // to not be called.
  bool haveMain = findUnderscoreMangle("main");
  bool haveWMain = findUnderscoreMangle("wmain");
  bool haveWinMain = findUnderscoreMangle("WinMain");
  bool haveWWinMain = findUnderscoreMangle("wWinMain");
  if (haveMain || haveWMain) {
    if (haveWinMain || haveWWinMain) {
      Warn(ctx) << "found " << (haveMain ? "main" : "wmain") << " and "
                << (haveWinMain ? "WinMain" : "wWinMain")
                << "; defaulting to /subsystem:console";
    }
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  }
  if (haveWinMain || haveWWinMain)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  return IMAGE_SUBSYSTEM_UNKNOWN;
}

void SymbolTable::addUndefinedGlob(StringRef arg) {
  Expected<GlobPattern> pat = GlobPattern::create(arg);
  if (!pat) {
    Err(ctx) << "/includeglob: " << toString(pat.takeError());
    return;
  }

  SmallVector<Symbol *, 0> syms;
  forEachSymbol([&syms, &pat](Symbol *sym) {
    if (pat->match(sym->getName())) {
      syms.push_back(sym);
    }
  });

  for (Symbol *sym : syms)
    addGCRoot(sym->getName());
}

// Convert stdcall/fastcall style symbols into unsuffixed symbols,
// with or without a leading underscore. (MinGW specific.)
static StringRef killAt(StringRef sym, bool prefix) {
  if (sym.empty())
    return sym;
  // Strip any trailing stdcall suffix
  sym = sym.substr(0, sym.find('@', 1));
  if (!sym.starts_with("@")) {
    if (prefix && !sym.starts_with("_"))
      return saver().save("_" + sym);
    return sym;
  }
  // For fastcall, remove the leading @ and replace it with an
  // underscore, if prefixes are used.
  sym = sym.substr(1);
  if (prefix)
    sym = saver().save("_" + sym);
  return sym;
}

static StringRef exportSourceName(ExportSource s) {
  switch (s) {
  case ExportSource::Directives:
    return "source file (directives)";
  case ExportSource::Export:
    return "/export";
  case ExportSource::ModuleDefinition:
    return "/def";
  case ExportSource::ExportAll:
    return "/export-all-symbols";
  default:
    llvm_unreachable("unknown ExportSource");
  }
}

static int exportSourcePriority(ExportSource s) {
  switch (s) {
  case ExportSource::Directives:
    return 1;
  case ExportSource::ExportAll:
    // Give directives (embedded in object files, from dllexport attributes)
    // and linker generated exports (from /export-all-symbols) differing
    // priority, to avoid warnings about conflicts between the two. In
    // practice, there shouldn't be any conflicts between the two, as both
    // should set the DATA flag consistently. Both have lower priority than
    // def files and explicit export arguments.
    return 2;
  case ExportSource::Export:
  case ExportSource::ModuleDefinition:
    // Give the same priority to export arguments and def files; this produces
    // warnings if they are duplicate and if they differ.
    return 3;
  default:
    llvm_unreachable("unknown ExportSource");
  }
}

// Performs error checking on all /export arguments.
// It also sets ordinals.
void SymbolTable::fixupExports() {
  llvm::TimeTraceScope timeScope("Fixup exports");
  // Symbol ordinals must be unique.
  std::set<uint16_t> ords;
  for (Export &e : exports) {
    if (e.ordinal == 0)
      continue;
    if (!ords.insert(e.ordinal).second)
      Fatal(ctx) << "duplicate export ordinal: " << e.name;
  }

  for (Export &e : exports) {
    if (!e.exportAs.empty()) {
      e.exportName = e.exportAs;
      continue;
    }

    StringRef sym =
        !e.forwardTo.empty() || e.extName.empty() ? e.name : e.extName;
    if (machine == I386 && sym.starts_with("_")) {
      // In MSVC mode, a fully decorated stdcall function is exported
      // as-is with the leading underscore (with type IMPORT_NAME).
      // In MinGW mode, a decorated stdcall function gets the underscore
      // removed, just like normal cdecl functions.
      if (ctx.config.mingw || !sym.contains('@')) {
        e.exportName = sym.substr(1);
        continue;
      }
    }
    if (isEC() && !e.data && !e.constant) {
      if (std::optional<std::string> demangledName =
              getArm64ECDemangledFunctionName(sym)) {
        e.exportName = saver().save(*demangledName);
        continue;
      }
    }
    e.exportName = sym;
  }

  if (ctx.config.killAt && machine == I386) {
    for (Export &e : exports) {
      e.name = killAt(e.name, true);
      e.exportName = killAt(e.exportName, false);
      e.extName = killAt(e.extName, true);
      e.symbolName = killAt(e.symbolName, true);
    }
  }

  // Uniquefy by name.
  DenseMap<StringRef, std::pair<Export *, unsigned>> map(exports.size());
  std::vector<Export> v;
  for (Export &e : exports) {
    auto pair = map.insert(std::make_pair(e.exportName, std::make_pair(&e, 0)));
    bool inserted = pair.second;
    if (inserted) {
      pair.first->second.second = v.size();
      v.push_back(e);
      continue;
    }
    Export *existing = pair.first->second.first;
    if (e == *existing || e.name != existing->name)
      continue;
    // If the existing export comes from .OBJ directives, we are allowed to
    // overwrite it with /DEF: or /EXPORT without any warning, as MSVC link.exe
    // does.
    // Also silently override exports from /export-all-symbols with ones from
    // def files and /EXPORT.
    int existingPriority = exportSourcePriority(existing->source);
    int newPriority = exportSourcePriority(e.source);
    if (existingPriority < newPriority) {
      // New definition with higher priority; don't warn, and replace the
      // existing definition with this one.
      *existing = e;
      v[pair.first->second.second] = e;
      continue;
    }
    if (existingPriority > newPriority) {
      // New definition with lower priority; ignore the new one silently
      // without warning.
      continue;
    }

    if (existing->source == e.source) {
      Warn(ctx) << "duplicate " << exportSourceName(existing->source)
                << " option: " << e.name;
    } else {
      Warn(ctx) << "duplicate export: " << e.name << " first seen in "
                << exportSourceName(existing->source) << ", now in "
                << exportSourceName(e.source);
    }
  }
  exports = std::move(v);

  // Sort by name.
  llvm::sort(exports, [](const Export &a, const Export &b) {
    return a.exportName < b.exportName;
  });
}

void SymbolTable::assignExportOrdinals() {
  // Assign unique ordinals if default (= 0).
  uint32_t max = 0;
  for (Export &e : exports)
    max = std::max(max, (uint32_t)e.ordinal);
  for (Export &e : exports)
    if (e.ordinal == 0)
      e.ordinal = ++max;
  if (max > std::numeric_limits<uint16_t>::max())
    Fatal(ctx) << "too many exported symbols (got " << max << ", max "
               << Twine(std::numeric_limits<uint16_t>::max()) << ")";
}

void SymbolTable::parseModuleDefs(StringRef path) {
  llvm::TimeTraceScope timeScope("Parse def file");
  std::unique_ptr<MemoryBuffer> mb =
      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false,
                                  /*IsVolatile=*/true),
            "could not open " + path);
  COFFModuleDefinition m = check(parseCOFFModuleDefinition(
      mb->getMemBufferRef(), machine, ctx.config.mingw));

  // Include in /reproduce: output if applicable.
  ctx.driver.takeBuffer(std::move(mb));

  if (ctx.config.outputFile.empty())
    ctx.config.outputFile = std::string(saver().save(m.OutputFile));
  ctx.config.importName = std::string(saver().save(m.ImportName));
  if (m.ImageBase)
    ctx.config.imageBase = m.ImageBase;
  if (m.StackReserve)
    ctx.config.stackReserve = m.StackReserve;
  if (m.StackCommit)
    ctx.config.stackCommit = m.StackCommit;
  if (m.HeapReserve)
    ctx.config.heapReserve = m.HeapReserve;
  if (m.HeapCommit)
    ctx.config.heapCommit = m.HeapCommit;
  if (m.MajorImageVersion)
    ctx.config.majorImageVersion = m.MajorImageVersion;
  if (m.MinorImageVersion)
    ctx.config.minorImageVersion = m.MinorImageVersion;
  if (m.MajorOSVersion)
    ctx.config.majorOSVersion = m.MajorOSVersion;
  if (m.MinorOSVersion)
    ctx.config.minorOSVersion = m.MinorOSVersion;

  for (COFFShortExport e1 : m.Exports) {
    Export e2;
    // Renamed exports are parsed and set as "ExtName = Name". If Name has
    // the form "OtherDll.Func", it shouldn't be a normal exported
    // function but a forward to another DLL instead. This is supported
    // by both MS and GNU linkers.
    if (!e1.ExtName.empty() && e1.ExtName != e1.Name &&
        StringRef(e1.Name).contains('.')) {
      e2.name = saver().save(e1.ExtName);
      e2.forwardTo = saver().save(e1.Name);
    } else {
      e2.name = saver().save(e1.Name);
      e2.extName = saver().save(e1.ExtName);
    }
    e2.exportAs = saver().save(e1.ExportAs);
    e2.importName = saver().save(e1.ImportName);
    e2.ordinal = e1.Ordinal;
    e2.noname = e1.Noname;
    e2.data = e1.Data;
    e2.isPrivate = e1.Private;
    e2.constant = e1.Constant;
    e2.source = ExportSource::ModuleDefinition;
    exports.push_back(e2);
  }
}

// Parse a string of the form of "<from>=<to>".
void SymbolTable::parseAlternateName(StringRef s) {
  auto [from, to] = s.split('=');
  if (from.empty() || to.empty())
    Fatal(ctx) << "/alternatename: invalid argument: " << s;
  auto it = alternateNames.find(from);
  if (it != alternateNames.end() && it->second != to)
    Fatal(ctx) << "/alternatename: conflicts: " << s;
  alternateNames.insert(it, std::make_pair(from, to));
}

void SymbolTable::resolveAlternateNames() {
  // Add weak aliases. Weak aliases is a mechanism to give remaining
  // undefined symbols final chance to be resolved successfully.
  for (auto pair : alternateNames) {
    StringRef from = pair.first;
    StringRef to = pair.second;
    Symbol *sym = find(from);
    if (!sym)
      continue;
    if (!sym->hasUndefinedReference)
      continue;
    if (auto *u = dyn_cast<Undefined>(sym)) {
      if (u->weakAlias) {
        // On ARM64EC, anti-dependency aliases are treated as undefined
        // symbols unless a demangled symbol aliases a defined one, which
        // is part of the implementation.
        if (!isEC() || !u->isAntiDep)
          continue;
        if (!isa<Undefined>(u->weakAlias) &&
            !isArm64ECMangledFunctionName(u->getName()))
          continue;
      }

      // Check if the destination symbol is defined. If not, skip it.
      // It may still be resolved later if more input files are added.
      // Also skip anti-dependency targets, as they can't be chained anyway.
      Symbol *toSym = find(to);
      if (!toSym)
        continue;
      auto toUndef = dyn_cast<Undefined>(toSym);
      if (toUndef && (!toUndef->weakAlias || toUndef->isAntiDep))
        continue;
      toSym->isUsedInRegularObj = true;
      if (toSym->isLazy())
        forceLazy(toSym);
      u->setWeakAlias(toSym);
    }
  }
}

// Parses /aligncomm option argument.
void SymbolTable::parseAligncomm(StringRef s) {
  auto [name, align] = s.split(',');
  if (name.empty() || align.empty()) {
    Err(ctx) << "/aligncomm: invalid argument: " << s;
    return;
  }
  int v;
  if (align.getAsInteger(0, v)) {
    Err(ctx) << "/aligncomm: invalid argument: " << s;
    return;
  }
  alignComm[std::string(name)] = std::max(alignComm[std::string(name)], 1 << v);
}

Symbol *SymbolTable::addUndefined(StringRef name) {
  return addUndefined(name, nullptr, false);
}

std::string SymbolTable::printSymbol(Symbol *sym) const {
  std::string name = maybeDemangleSymbol(ctx, sym->getName());
  if (ctx.hybridSymtab)
    return name + (isEC() ? " (EC symbol)" : " (native symbol)");
  return name;
}

void SymbolTable::compileBitcodeFiles() {
  if (bitcodeFileInstances.empty())
    return;

  // Collect the bitcode library functions that are not safe to call because
  // they were not yet brought in the link. (Such symbols are lazy.)
  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver saver(alloc);
  SmallVector<StringRef> bitcodeLibFuncs;
  // Triple must be captured before the bitcode is moved into the compiler.
  // Note that the below assumes that the set of possible libfuncs is roughly
  // equivalent for all bitcode translation units.
  llvm::Triple tt =
      llvm::Triple(bitcodeFileInstances.front()->obj->getTargetTriple());
  for (StringRef libFunc : lto::LTO::getLibFuncSymbols(tt, saver)) {
    if (Symbol *sym = find(libFunc)) {
      if (auto *l = dyn_cast<LazyArchive>(sym)) {
        if (isBitcode(l->getMemberBuffer()))
          bitcodeLibFuncs.push_back(libFunc);
      } else if (auto *o = dyn_cast<LazyObject>(sym)) {
        if (isBitcode(o->file->mb))
          bitcodeLibFuncs.push_back(libFunc);
      }
    }
  }

  ScopedTimer t(ctx.ltoTimer);
  lto.reset(new BitcodeCompiler(ctx));
  lto->setBitcodeLibFuncs(bitcodeLibFuncs);
  {
    llvm::TimeTraceScope addScope("Add bitcode file instances");
    for (BitcodeFile *f : bitcodeFileInstances)
      lto->add(*f);
  }

  for (InputFile *newObj : lto->compile()) {
    ObjFile *obj = cast<ObjFile>(newObj);
    obj->parse();
    ctx.objFileInstances.push_back(obj);
  }
}

void SymbolTable::waitForLTOCleanup() {
  if (lto)
    lto->waitForLTOCleanup();
}

} // namespace lld::coff
