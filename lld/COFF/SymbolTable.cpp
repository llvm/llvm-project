//===- SymbolTable.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "Driver.h"
#include "LTO.h"
#include "Memory.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

namespace lld {
namespace coff {

enum SymbolPreference {
  SP_EXISTING = -1,
  SP_CONFLICT = 0,
  SP_NEW = 1,
};

/// Checks if an existing symbol S should be kept or replaced by a new symbol.
/// Returns SP_EXISTING when S should be kept, SP_NEW when the new symbol
/// should be kept, and SP_CONFLICT if no valid resolution exists.
static SymbolPreference compareDefined(SymbolBody *S, bool WasInserted,
                                       bool NewIsCOMDAT) {
  // If the symbol wasn't previously known, the new symbol wins by default.
  if (WasInserted || !isa<Defined>(S))
    return SP_NEW;

  // If the existing symbol is a DefinedRegular, both it and the new symbol
  // must be comdats. In that case, we have no reason to prefer one symbol
  // over the other, and we keep the existing one. If one of the symbols
  // is not a comdat, we report a conflict.
  if (auto *R = dyn_cast<DefinedRegular>(S)) {
    if (NewIsCOMDAT && R->isCOMDAT())
      return SP_EXISTING;
    else
      return SP_CONFLICT;
  }

  // Existing symbol is not a DefinedRegular; new symbol wins.
  return SP_NEW;
}

SymbolTable *Symtab;

void SymbolTable::addFile(InputFile *File) {
  log("Reading " + toString(File));
  File->parse();

  MachineTypes MT = File->getMachineType();
  if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    Config->Machine = MT;
  } else if (MT != IMAGE_FILE_MACHINE_UNKNOWN && Config->Machine != MT) {
    fatal(toString(File) + ": machine type " + machineToStr(MT) +
          " conflicts with " + machineToStr(Config->Machine));
  }

  if (auto *F = dyn_cast<ObjFile>(File)) {
    ObjFile::Instances.push_back(F);
  } else if (auto *F = dyn_cast<BitcodeFile>(File)) {
    BitcodeFile::Instances.push_back(F);
  } else if (auto *F = dyn_cast<ImportFile>(File)) {
    ImportFile::Instances.push_back(F);
  }

  StringRef S = File->getDirectives();
  if (S.empty())
    return;

  log("Directives: " + toString(File) + ": " + S);
  Driver->parseDirectives(S);
}

static void errorOrWarn(const Twine &S) {
  if (Config->Force)
    warn(S);
  else
    error(S);
}

void SymbolTable::reportRemainingUndefines() {
  SmallPtrSet<SymbolBody *, 8> Undefs;

  for (auto &I : Symtab) {
    SymbolBody *Sym = I.second;
    auto *Undef = dyn_cast<Undefined>(Sym);
    if (!Undef)
      continue;
    if (!Sym->IsUsedInRegularObj)
      continue;

    StringRef Name = Undef->getName();

    // A weak alias may have been resolved, so check for that.
    if (Defined *D = Undef->getWeakAlias()) {
      // We want to replace Sym with D. However, we can't just blindly
      // copy sizeof(SymbolUnion) bytes from D to Sym because D may be an
      // internal symbol, and internal symbols are stored as "unparented"
      // Symbols. For that reason we need to check which type of symbol we
      // are dealing with and copy the correct number of bytes.
      if (isa<DefinedRegular>(D))
        memcpy(Sym, D, sizeof(DefinedRegular));
      else if (isa<DefinedAbsolute>(D))
        memcpy(Sym, D, sizeof(DefinedAbsolute));
      else
        memcpy(Sym, D, sizeof(SymbolUnion));
      continue;
    }

    // If we can resolve a symbol by removing __imp_ prefix, do that.
    // This odd rule is for compatibility with MSVC linker.
    if (Name.startswith("__imp_")) {
      SymbolBody *Imp = find(Name.substr(strlen("__imp_")));
      if (Imp && isa<Defined>(Imp)) {
        auto *D = cast<Defined>(Imp);
        replaceBody<DefinedLocalImport>(Sym, Name, D);
        LocalImportChunks.push_back(cast<DefinedLocalImport>(Sym)->getChunk());
        continue;
      }
    }

    // Remaining undefined symbols are not fatal if /force is specified.
    // They are replaced with dummy defined symbols.
    if (Config->Force)
      replaceBody<DefinedAbsolute>(Sym, Name, 0);
    Undefs.insert(Sym);
  }

  if (Undefs.empty())
    return;

  for (SymbolBody *B : Config->GCRoot)
    if (Undefs.count(B))
      errorOrWarn("<root>: undefined symbol: " + B->getName());

  for (ObjFile *File : ObjFile::Instances)
    for (SymbolBody *Sym : File->getSymbols())
      if (Undefs.count(Sym))
        errorOrWarn(toString(File) + ": undefined symbol: " + Sym->getName());
}

std::pair<SymbolBody *, bool> SymbolTable::insert(StringRef Name) {
  SymbolBody *&Sym = Symtab[CachedHashStringRef(Name)];
  if (Sym)
    return {Sym, false};
  Sym = (SymbolBody *)make<SymbolUnion>();
  Sym->IsUsedInRegularObj = false;
  Sym->PendingArchiveLoad = false;
  return {Sym, true};
}

SymbolBody *SymbolTable::addUndefined(StringRef Name, InputFile *F,
                                      bool IsWeakAlias) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  if (!F || !isa<BitcodeFile>(F))
    S->IsUsedInRegularObj = true;
  if (WasInserted || (isa<Lazy>(S) && IsWeakAlias)) {
    replaceBody<Undefined>(S, Name);
    return S;
  }
  if (auto *L = dyn_cast<Lazy>(S)) {
    if (!S->PendingArchiveLoad) {
      S->PendingArchiveLoad = true;
      L->File->addMember(&L->Sym);
    }
  }
  return S;
}

void SymbolTable::addLazy(ArchiveFile *F, const Archive::Symbol Sym) {
  StringRef Name = Sym.getName();
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  if (WasInserted) {
    replaceBody<Lazy>(S, F, Sym);
    return;
  }
  auto *U = dyn_cast<Undefined>(S);
  if (!U || U->WeakAlias || S->PendingArchiveLoad)
    return;
  S->PendingArchiveLoad = true;
  F->addMember(&Sym);
}

void SymbolTable::reportDuplicate(SymbolBody *Existing, InputFile *NewFile) {
  error("duplicate symbol: " + toString(*Existing) + " in " +
        toString(Existing->getFile()) + " and in " +
        (NewFile ? toString(NewFile) : "(internal)"));
}

SymbolBody *SymbolTable::addAbsolute(StringRef N, COFFSymbolRef Sym) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S) || isa<Lazy>(S))
    replaceBody<DefinedAbsolute>(S, N, Sym);
  else if (!isa<DefinedCOFF>(S))
    reportDuplicate(S, nullptr);
  return S;
}

SymbolBody *SymbolTable::addAbsolute(StringRef N, uint64_t VA) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S) || isa<Lazy>(S))
    replaceBody<DefinedAbsolute>(S, N, VA);
  else if (!isa<DefinedCOFF>(S))
    reportDuplicate(S, nullptr);
  return S;
}

SymbolBody *SymbolTable::addSynthetic(StringRef N, Chunk *C) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S) || isa<Lazy>(S))
    replaceBody<DefinedSynthetic>(S, N, C);
  else if (!isa<DefinedCOFF>(S))
    reportDuplicate(S, nullptr);
  return S;
}

SymbolBody *SymbolTable::addRegular(InputFile *F, StringRef N, bool IsCOMDAT,
                                    const coff_symbol_generic *Sym,
                                    SectionChunk *C) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  if (!isa<BitcodeFile>(F))
    S->IsUsedInRegularObj = true;
  SymbolPreference SP = compareDefined(S, WasInserted, IsCOMDAT);
  if (SP == SP_CONFLICT) {
    reportDuplicate(S, F);
  } else if (SP == SP_NEW) {
    replaceBody<DefinedRegular>(S, F, N, IsCOMDAT, /*IsExternal*/ true, Sym, C);
  } else if (SP == SP_EXISTING && IsCOMDAT && C) {
    C->markDiscarded();
    // Discard associative chunks that we've parsed so far. No need to recurse
    // because an associative section cannot have children.
    for (SectionChunk *Child : C->children())
      Child->markDiscarded();
  }
  return S;
}

SymbolBody *SymbolTable::addCommon(InputFile *F, StringRef N, uint64_t Size,
                                   const coff_symbol_generic *Sym,
                                   CommonChunk *C) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  if (!isa<BitcodeFile>(F))
    S->IsUsedInRegularObj = true;
  if (WasInserted || !isa<DefinedCOFF>(S))
    replaceBody<DefinedCommon>(S, F, N, Size, Sym, C);
  else if (auto *DC = dyn_cast<DefinedCommon>(S))
    if (Size > DC->getSize())
      replaceBody<DefinedCommon>(S, F, N, Size, Sym, C);
  return S;
}

DefinedImportData *SymbolTable::addImportData(StringRef N, ImportFile *F) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(N);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S) || isa<Lazy>(S)) {
    replaceBody<DefinedImportData>(S, N, F);
    return cast<DefinedImportData>(S);
  }

  reportDuplicate(S, F);
  return nullptr;
}

DefinedImportThunk *SymbolTable::addImportThunk(StringRef Name,
                                               DefinedImportData *ID,
                                               uint16_t Machine) {
  SymbolBody *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);
  S->IsUsedInRegularObj = true;
  if (WasInserted || isa<Undefined>(S) || isa<Lazy>(S)) {
    replaceBody<DefinedImportThunk>(S, Name, ID, Machine);
    return cast<DefinedImportThunk>(S);
  }

  reportDuplicate(S, ID->File);
  return nullptr;
}

std::vector<Chunk *> SymbolTable::getChunks() {
  std::vector<Chunk *> Res;
  for (ObjFile *File : ObjFile::Instances) {
    std::vector<Chunk *> &V = File->getChunks();
    Res.insert(Res.end(), V.begin(), V.end());
  }
  return Res;
}

SymbolBody *SymbolTable::find(StringRef Name) {
  auto It = Symtab.find(CachedHashStringRef(Name));
  if (It == Symtab.end())
    return nullptr;
  return It->second;
}

SymbolBody *SymbolTable::findUnderscore(StringRef Name) {
  if (Config->Machine == I386)
    return find(("_" + Name).str());
  return find(Name);
}

StringRef SymbolTable::findByPrefix(StringRef Prefix) {
  for (auto Pair : Symtab) {
    StringRef Name = Pair.first.val();
    if (Name.startswith(Prefix))
      return Name;
  }
  return "";
}

StringRef SymbolTable::findMangle(StringRef Name) {
  if (SymbolBody *Sym = find(Name))
    if (!isa<Undefined>(Sym))
      return Name;
  if (Config->Machine != I386)
    return findByPrefix(("?" + Name + "@@Y").str());
  if (!Name.startswith("_"))
    return "";
  // Search for x86 stdcall function.
  StringRef S = findByPrefix((Name + "@").str());
  if (!S.empty())
    return S;
  // Search for x86 fastcall function.
  S = findByPrefix(("@" + Name.substr(1) + "@").str());
  if (!S.empty())
    return S;
  // Search for x86 vectorcall function.
  S = findByPrefix((Name.substr(1) + "@@").str());
  if (!S.empty())
    return S;
  // Search for x86 C++ non-member function.
  return findByPrefix(("?" + Name.substr(1) + "@@Y").str());
}

void SymbolTable::mangleMaybe(SymbolBody *B) {
  auto *U = dyn_cast<Undefined>(B);
  if (!U || U->WeakAlias)
    return;
  StringRef Alias = findMangle(U->getName());
  if (!Alias.empty()) {
    log(U->getName() + " aliased to " + Alias);
    U->WeakAlias = addUndefined(Alias);
  }
}

SymbolBody *SymbolTable::addUndefined(StringRef Name) {
  return addUndefined(Name, nullptr, false);
}

std::vector<StringRef> SymbolTable::compileBitcodeFiles() {
  LTO.reset(new BitcodeCompiler);
  for (BitcodeFile *F : BitcodeFile::Instances)
    LTO->add(*F);
  return LTO->compile();
}

void SymbolTable::addCombinedLTOObjects() {
  if (BitcodeFile::Instances.empty())
    return;
  for (StringRef Object : compileBitcodeFiles()) {
    auto *Obj = make<ObjFile>(MemoryBufferRef(Object, "lto.tmp"));
    Obj->parse();
    ObjFile::Instances.push_back(Obj);
  }
}

} // namespace coff
} // namespace lld
