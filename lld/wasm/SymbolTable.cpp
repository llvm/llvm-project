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
#include "InputChunks.h"
#include "InputGlobal.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

SymbolTable *lld::wasm::Symtab;

void SymbolTable::addFile(InputFile *File) {
  log("Processing: " + toString(File));
  File->parse();

  if (auto *F = dyn_cast<ObjFile>(File))
    ObjectFiles.push_back(F);
}

void SymbolTable::reportRemainingUndefines() {
  SetVector<Symbol *> Undefs;
  for (Symbol *Sym : SymVector) {
    if (Sym->isUndefined() && !Sym->isWeak() &&
        Config->AllowUndefinedSymbols.count(Sym->getName()) == 0) {
      Undefs.insert(Sym);
    }
  }

  if (Undefs.empty())
    return;

  for (ObjFile *File : ObjectFiles)
    for (Symbol *Sym : File->getSymbols())
      if (Undefs.count(Sym))
        error(toString(File) + ": undefined symbol: " + toString(*Sym));

  for (Symbol *Sym : Undefs)
    if (!Sym->getFile())
      error("undefined symbol: " + toString(*Sym));
}

Symbol *SymbolTable::find(StringRef Name) {
  return SymMap.lookup(CachedHashStringRef(Name));
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef Name) {
  Symbol *&Sym = SymMap[CachedHashStringRef(Name)];
  if (Sym)
    return {Sym, false};
  Sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
  SymVector.emplace_back(Sym);
  return {Sym, true};
}

static void reportTypeError(const Symbol *Existing, const InputFile *File,
                            StringRef Type) {
  error("symbol type mismatch: " + toString(*Existing) + "\n>>> defined as " +
        toString(Existing->getWasmType()) + " in " +
        toString(Existing->getFile()) + "\n>>> defined as " + Type + " in " +
        toString(File));
}

static void checkFunctionType(const Symbol *Existing, const InputFile *File,
                              const WasmSignature *NewSig) {
  if (!isa<FunctionSymbol>(Existing)) {
    reportTypeError(Existing, File, "Function");
    return;
  }

  if (!Config->CheckSignatures)
    return;

  const WasmSignature *OldSig =
      cast<FunctionSymbol>(Existing)->getFunctionType();
  if (OldSig && *NewSig != *OldSig) {
    error("Function type mismatch: " + Existing->getName() +
          "\n>>> defined as " + toString(*OldSig) + " in " +
          toString(Existing->getFile()) + "\n>>> defined as " +
          toString(*NewSig) + " in " + toString(File));
  }
}

// Check the type of new symbol matches that of the symbol is replacing.
// For functions this can also involve verifying that the signatures match.
static void checkGlobalType(const Symbol *Existing, const InputFile *File,
                            const WasmGlobalType *NewType) {
  if (!isa<GlobalSymbol>(Existing)) {
    reportTypeError(Existing, File, "Global");
    return;
  }

  const WasmGlobalType *OldType = cast<GlobalSymbol>(Existing)->getGlobalType();
  if (*NewType != *OldType) {
    error("Global type mismatch: " + Existing->getName() + "\n>>> defined as " +
          toString(*OldType) + " in " + toString(Existing->getFile()) +
          "\n>>> defined as " + toString(*NewType) + " in " + toString(File));
  }
}

static void checkDataType(const Symbol *Existing, const InputFile *File) {
  if (!isa<DataSymbol>(Existing))
    reportTypeError(Existing, File, "Data");
}

DefinedFunction *SymbolTable::addSyntheticFunction(StringRef Name,
                                                   uint32_t Flags,
                                                   InputFunction *Function) {
  DEBUG(dbgs() << "addSyntheticFunction: " << Name << "\n");
  assert(!find(Name));
  SyntheticFunctions.emplace_back(Function);
  return replaceSymbol<DefinedFunction>(insert(Name).first, Name, Flags,
                                        nullptr, Function);
}

DefinedData *SymbolTable::addSyntheticDataSymbol(StringRef Name,
                                                 uint32_t Flags) {
  DEBUG(dbgs() << "addSyntheticDataSymbol: " << Name << "\n");
  assert(!find(Name));
  return replaceSymbol<DefinedData>(insert(Name).first, Name, Flags);
}

DefinedGlobal *SymbolTable::addSyntheticGlobal(StringRef Name, uint32_t Flags,
                                               InputGlobal *Global) {
  DEBUG(dbgs() << "addSyntheticGlobal: " << Name << " -> " << Global << "\n");
  assert(!find(Name));
  SyntheticGlobals.emplace_back(Global);
  return replaceSymbol<DefinedGlobal>(insert(Name).first, Name, Flags, nullptr,
                                      Global);
}

static bool shouldReplace(const Symbol *Existing, InputFile *NewFile,
                          uint32_t NewFlags) {
  // If existing symbol is undefined, replace it.
  if (!Existing->isDefined()) {
    DEBUG(dbgs() << "resolving existing undefined symbol: "
                 << Existing->getName() << "\n");
    return true;
  }

  // Now we have two defined symbols. If the new one is weak, we can ignore it.
  if ((NewFlags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_WEAK) {
    DEBUG(dbgs() << "existing symbol takes precedence\n");
    return false;
  }

  // If the existing symbol is weak, we should replace it.
  if (Existing->isWeak()) {
    DEBUG(dbgs() << "replacing existing weak symbol\n");
    return true;
  }

  // Neither symbol is week. They conflict.
  error("duplicate symbol: " + toString(*Existing) + "\n>>> defined in " +
        toString(Existing->getFile()) + "\n>>> defined in " +
        toString(NewFile));
  return true;
}

Symbol *SymbolTable::addDefinedFunction(StringRef Name, uint32_t Flags,
                                        InputFile *File,
                                        InputFunction *Function) {
  DEBUG(dbgs() << "addDefinedFunction: " << Name << "\n");
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedFunction>(S, Name, Flags, File, Function);
    return S;
  }

  checkFunctionType(S, File, &Function->Signature);

  if (shouldReplace(S, File, Flags))
    replaceSymbol<DefinedFunction>(S, Name, Flags, File, Function);
  return S;
}

Symbol *SymbolTable::addDefinedData(StringRef Name, uint32_t Flags,
                                    InputFile *File, InputSegment *Segment,
                                    uint32_t Address, uint32_t Size) {
  DEBUG(dbgs() << "addDefinedData:" << Name << " addr:" << Address << "\n");
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedData>(S, Name, Flags, File, Segment, Address, Size);
    return S;
  }

  checkDataType(S, File);

  if (shouldReplace(S, File, Flags))
    replaceSymbol<DefinedData>(S, Name, Flags, File, Segment, Address, Size);
  return S;
}

Symbol *SymbolTable::addDefinedGlobal(StringRef Name, uint32_t Flags,
                                      InputFile *File, InputGlobal *Global) {
  DEBUG(dbgs() << "addDefinedGlobal:" << Name << "\n");
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedGlobal>(S, Name, Flags, File, Global);
    return S;
  }

  checkGlobalType(S, File, &Global->getType());

  if (shouldReplace(S, File, Flags))
    replaceSymbol<DefinedGlobal>(S, Name, Flags, File, Global);
  return S;
}

Symbol *SymbolTable::addUndefinedFunction(StringRef Name, uint32_t Flags,
                                          InputFile *File,
                                          const WasmSignature *Sig) {
  DEBUG(dbgs() << "addUndefinedFunction: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted)
    replaceSymbol<UndefinedFunction>(S, Name, Flags, File, Sig);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else if (S->isDefined())
    checkFunctionType(S, File, Sig);
  return S;
}

Symbol *SymbolTable::addUndefinedData(StringRef Name, uint32_t Flags,
                                      InputFile *File) {
  DEBUG(dbgs() << "addUndefinedData: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted)
    replaceSymbol<UndefinedData>(S, Name, Flags, File);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else if (S->isDefined())
    checkDataType(S, File);
  return S;
}

Symbol *SymbolTable::addUndefinedGlobal(StringRef Name, uint32_t Flags,
                                        InputFile *File,
                                        const WasmGlobalType *Type) {
  DEBUG(dbgs() << "addUndefinedGlobal: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted)
    replaceSymbol<UndefinedGlobal>(S, Name, Flags, File, Type);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else if (S->isDefined())
    checkGlobalType(S, File, Type);
  return S;
}

void SymbolTable::addLazy(ArchiveFile *File, const Archive::Symbol *Sym) {
  DEBUG(dbgs() << "addLazy: " << Sym->getName() << "\n");
  StringRef Name = Sym->getName();

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name);

  if (WasInserted) {
    replaceSymbol<LazySymbol>(S, Name, File, *Sym);
    return;
  }

  // If there is an existing undefined symbol, load a new one from the archive.
  if (S->isUndefined()) {
    DEBUG(dbgs() << "replacing existing undefined\n");
    File->addMember(Sym);
  }
}

bool SymbolTable::addComdat(StringRef Name) {
  return Comdats.insert(CachedHashStringRef(Name)).second;
}
