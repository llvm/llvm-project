//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputEvent.h"
#include "InputGlobal.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace llvm::object;
using namespace lld;
using namespace lld::wasm;

SymbolTable *lld::wasm::Symtab;

void SymbolTable::addFile(InputFile *File) {
  log("Processing: " + toString(File));
  if (Config->Trace)
    message(toString(File));
  File->parse();

  // LLVM bitcode file
  if (auto *F = dyn_cast<BitcodeFile>(File))
    BitcodeFiles.push_back(F);
  else if (auto *F = dyn_cast<ObjFile>(File))
    ObjectFiles.push_back(F);
}

// This function is where all the optimizations of link-time
// optimization happens. When LTO is in use, some input files are
// not in native object file format but in the LLVM bitcode format.
// This function compiles bitcode files into a few big native files
// using LLVM functions and replaces bitcode symbols with the results.
// Because all bitcode files that the program consists of are passed
// to the compiler at once, it can do whole-program optimization.
void SymbolTable::addCombinedLTOObject() {
  if (BitcodeFiles.empty())
    return;

  // Compile bitcode files and replace bitcode symbols.
  LTO.reset(new BitcodeCompiler);
  for (BitcodeFile *F : BitcodeFiles)
    LTO->add(*F);

  for (StringRef Filename : LTO->compile()) {
    auto *Obj = make<ObjFile>(MemoryBufferRef(Filename, "lto.tmp"));
    Obj->parse();
    ObjectFiles.push_back(Obj);
  }
}

void SymbolTable::reportRemainingUndefines() {
  for (Symbol *Sym : SymVector) {
    if (!Sym->isUndefined() || Sym->isWeak())
      continue;
    if (Config->AllowUndefinedSymbols.count(Sym->getName()) != 0)
      continue;
    if (!Sym->IsUsedInRegularObj)
      continue;
    error(toString(Sym->getFile()) + ": undefined symbol: " + toString(*Sym));
  }
}

Symbol *SymbolTable::find(StringRef Name) {
  auto It = SymMap.find(CachedHashStringRef(Name));
  if (It == SymMap.end() || It->second == -1)
    return nullptr;
  return SymVector[It->second];
}

std::pair<Symbol *, bool> SymbolTable::insertName(StringRef Name) {
  bool Trace = false;
  auto P = SymMap.insert({CachedHashStringRef(Name), (int)SymVector.size()});
  int &SymIndex = P.first->second;
  bool IsNew = P.second;
  if (SymIndex == -1) {
    SymIndex = SymVector.size();
    Trace = true;
    IsNew = true;
  }

  if (!IsNew)
    return {SymVector[SymIndex], false};

  Symbol *Sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
  Sym->IsUsedInRegularObj = false;
  Sym->Traced = Trace;
  SymVector.emplace_back(Sym);
  return {Sym, true};
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef Name,
                                              const InputFile *File) {
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insertName(Name);

  if (!File || File->kind() == InputFile::ObjectKind)
    S->IsUsedInRegularObj = true;

  return {S, WasInserted};
}

static void reportTypeError(const Symbol *Existing, const InputFile *File,
                            llvm::wasm::WasmSymbolType Type) {
  error("symbol type mismatch: " + toString(*Existing) + "\n>>> defined as " +
        toString(Existing->getWasmType()) + " in " +
        toString(Existing->getFile()) + "\n>>> defined as " + toString(Type) +
        " in " + toString(File));
}

// Check the type of new symbol matches that of the symbol is replacing.
// For functions this can also involve verifying that the signatures match.
static void checkFunctionType(Symbol *Existing, const InputFile *File,
                              const WasmSignature *NewSig) {
  auto ExistingFunction = dyn_cast<FunctionSymbol>(Existing);
  if (!ExistingFunction) {
    reportTypeError(Existing, File, WASM_SYMBOL_TYPE_FUNCTION);
    return;
  }

  if (!NewSig)
    return;

  const WasmSignature *OldSig = ExistingFunction->Signature;
  if (!OldSig) {
    ExistingFunction->Signature = NewSig;
    return;
  }

  if (*NewSig != *OldSig)
    warn("function signature mismatch: " + Existing->getName() +
         "\n>>> defined as " + toString(*OldSig) + " in " +
         toString(Existing->getFile()) + "\n>>> defined as " +
         toString(*NewSig) + " in " + toString(File));
}

static void checkGlobalType(const Symbol *Existing, const InputFile *File,
                            const WasmGlobalType *NewType) {
  if (!isa<GlobalSymbol>(Existing)) {
    reportTypeError(Existing, File, WASM_SYMBOL_TYPE_GLOBAL);
    return;
  }

  const WasmGlobalType *OldType = cast<GlobalSymbol>(Existing)->getGlobalType();
  if (*NewType != *OldType) {
    error("Global type mismatch: " + Existing->getName() + "\n>>> defined as " +
          toString(*OldType) + " in " + toString(Existing->getFile()) +
          "\n>>> defined as " + toString(*NewType) + " in " + toString(File));
  }
}

static void checkEventType(const Symbol *Existing, const InputFile *File,
                           const WasmEventType *NewType,
                           const WasmSignature *NewSig) {
  auto ExistingEvent = dyn_cast<EventSymbol>(Existing);
  if (!isa<EventSymbol>(Existing)) {
    reportTypeError(Existing, File, WASM_SYMBOL_TYPE_EVENT);
    return;
  }

  const WasmEventType *OldType = cast<EventSymbol>(Existing)->getEventType();
  const WasmSignature *OldSig = ExistingEvent->Signature;
  if (NewType->Attribute != OldType->Attribute)
    error("Event type mismatch: " + Existing->getName() + "\n>>> defined as " +
          toString(*OldType) + " in " + toString(Existing->getFile()) +
          "\n>>> defined as " + toString(*NewType) + " in " + toString(File));
  if (*NewSig != *OldSig)
    warn("Event signature mismatch: " + Existing->getName() +
         "\n>>> defined as " + toString(*OldSig) + " in " +
         toString(Existing->getFile()) + "\n>>> defined as " +
         toString(*NewSig) + " in " + toString(File));
}

static void checkDataType(const Symbol *Existing, const InputFile *File) {
  if (!isa<DataSymbol>(Existing))
    reportTypeError(Existing, File, WASM_SYMBOL_TYPE_DATA);
}

DefinedFunction *SymbolTable::addSyntheticFunction(StringRef Name,
                                                   uint32_t Flags,
                                                   InputFunction *Function) {
  LLVM_DEBUG(dbgs() << "addSyntheticFunction: " << Name << "\n");
  assert(!find(Name));
  SyntheticFunctions.emplace_back(Function);
  return replaceSymbol<DefinedFunction>(insert(Name, nullptr).first, Name,
                                        Flags, nullptr, Function);
}

DefinedData *SymbolTable::addSyntheticDataSymbol(StringRef Name,
                                                 uint32_t Flags) {
  LLVM_DEBUG(dbgs() << "addSyntheticDataSymbol: " << Name << "\n");
  assert(!find(Name));
  return replaceSymbol<DefinedData>(insert(Name, nullptr).first, Name, Flags);
}

DefinedGlobal *SymbolTable::addSyntheticGlobal(StringRef Name, uint32_t Flags,
                                               InputGlobal *Global) {
  LLVM_DEBUG(dbgs() << "addSyntheticGlobal: " << Name << " -> " << Global
                    << "\n");
  assert(!find(Name));
  SyntheticGlobals.emplace_back(Global);
  return replaceSymbol<DefinedGlobal>(insert(Name, nullptr).first, Name, Flags,
                                      nullptr, Global);
}

static bool shouldReplace(const Symbol *Existing, InputFile *NewFile,
                          uint32_t NewFlags) {
  // If existing symbol is undefined, replace it.
  if (!Existing->isDefined()) {
    LLVM_DEBUG(dbgs() << "resolving existing undefined symbol: "
                      << Existing->getName() << "\n");
    return true;
  }

  // Now we have two defined symbols. If the new one is weak, we can ignore it.
  if ((NewFlags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_WEAK) {
    LLVM_DEBUG(dbgs() << "existing symbol takes precedence\n");
    return false;
  }

  // If the existing symbol is weak, we should replace it.
  if (Existing->isWeak()) {
    LLVM_DEBUG(dbgs() << "replacing existing weak symbol\n");
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
  LLVM_DEBUG(dbgs() << "addDefinedFunction: " << Name << " ["
                    << (Function ? toString(Function->Signature) : "none")
                    << "]\n");
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedFunction>(S, Name, Flags, File, Function);
    return S;
  }

  if (Function)
    checkFunctionType(S, File, &Function->Signature);

  if (shouldReplace(S, File, Flags)) {
    // If the new defined function doesn't have signture (i.e. bitcode
    // functions) but the old symbol does then preserve the old signature
    const WasmSignature *OldSig = nullptr;
    if (auto* F = dyn_cast<FunctionSymbol>(S))
      OldSig = F->Signature;
    if (auto *L = dyn_cast<LazySymbol>(S))
      OldSig = L->Signature;
    auto NewSym = replaceSymbol<DefinedFunction>(S, Name, Flags, File, Function);
    if (!NewSym->Signature)
      NewSym->Signature = OldSig;
  }
  return S;
}

Symbol *SymbolTable::addDefinedData(StringRef Name, uint32_t Flags,
                                    InputFile *File, InputSegment *Segment,
                                    uint32_t Address, uint32_t Size) {
  LLVM_DEBUG(dbgs() << "addDefinedData:" << Name << " addr:" << Address
                    << "\n");
  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

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
  LLVM_DEBUG(dbgs() << "addDefinedGlobal:" << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedGlobal>(S, Name, Flags, File, Global);
    return S;
  }

  checkGlobalType(S, File, &Global->getType());

  if (shouldReplace(S, File, Flags))
    replaceSymbol<DefinedGlobal>(S, Name, Flags, File, Global);
  return S;
}

Symbol *SymbolTable::addDefinedEvent(StringRef Name, uint32_t Flags,
                                     InputFile *File, InputEvent *Event) {
  LLVM_DEBUG(dbgs() << "addDefinedEvent:" << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted || S->isLazy()) {
    replaceSymbol<DefinedEvent>(S, Name, Flags, File, Event);
    return S;
  }

  checkEventType(S, File, &Event->getType(), &Event->Signature);

  if (shouldReplace(S, File, Flags))
    replaceSymbol<DefinedEvent>(S, Name, Flags, File, Event);
  return S;
}

Symbol *SymbolTable::addUndefinedFunction(StringRef Name, StringRef ImportName,
                                          StringRef ImportModule,
                                          uint32_t Flags, InputFile *File,
                                          const WasmSignature *Sig) {
  LLVM_DEBUG(dbgs() << "addUndefinedFunction: " << Name <<
             " [" << (Sig ? toString(*Sig) : "none") << "]\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted)
    replaceSymbol<UndefinedFunction>(S, Name, ImportName, ImportModule, Flags,
                                     File, Sig);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else
    checkFunctionType(S, File, Sig);

  return S;
}

Symbol *SymbolTable::addUndefinedData(StringRef Name, uint32_t Flags,
                                      InputFile *File) {
  LLVM_DEBUG(dbgs() << "addUndefinedData: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted)
    replaceSymbol<UndefinedData>(S, Name, Flags, File);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else if (S->isDefined())
    checkDataType(S, File);
  return S;
}

Symbol *SymbolTable::addUndefinedGlobal(StringRef Name, StringRef ImportName,
                                        StringRef ImportModule, uint32_t Flags,
                                        InputFile *File,
                                        const WasmGlobalType *Type) {
  LLVM_DEBUG(dbgs() << "addUndefinedGlobal: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  if (WasInserted)
    replaceSymbol<UndefinedGlobal>(S, Name, ImportName, ImportModule, Flags,
                                   File, Type);
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else if (S->isDefined())
    checkGlobalType(S, File, Type);
  return S;
}

void SymbolTable::addLazy(ArchiveFile *File, const Archive::Symbol *Sym) {
  LLVM_DEBUG(dbgs() << "addLazy: " << Sym->getName() << "\n");
  StringRef Name = Sym->getName();

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, nullptr);

  if (WasInserted) {
    replaceSymbol<LazySymbol>(S, Name, 0, File, *Sym);
    return;
  }

  if (!S->isUndefined())
    return;

  // The existing symbol is undefined, load a new one from the archive,
  // unless the the existing symbol is weak in which case replace the undefined
  // symbols with a LazySymbol.
  if (S->isWeak()) {
    const WasmSignature *OldSig = nullptr;
    // In the case of an UndefinedFunction we need to preserve the expected
    // signature.
    if (auto *F = dyn_cast<UndefinedFunction>(S))
      OldSig = F->Signature;
    LLVM_DEBUG(dbgs() << "replacing existing weak undefined symbol\n");
    auto NewSym = replaceSymbol<LazySymbol>(S, Name, WASM_SYMBOL_BINDING_WEAK,
                                            File, *Sym);
    NewSym->Signature = OldSig;
    return;
  }

  LLVM_DEBUG(dbgs() << "replacing existing undefined\n");
  File->addMember(Sym);
}

bool SymbolTable::addComdat(StringRef Name) {
  return Comdats.insert(CachedHashStringRef(Name)).second;
}

// Set a flag for --trace-symbol so that we can print out a log message
// if a new symbol with the same name is inserted into the symbol table.
void SymbolTable::trace(StringRef Name) {
  SymMap.insert({CachedHashStringRef(Name), -1});
}

static const uint8_t UnreachableFn[] = {
    0x03 /* ULEB length */, 0x00 /* ULEB num locals */,
    0x00 /* opcode unreachable */, 0x0b /* opcode end */
};

// Replace the given symbol body with an unreachable function.
// This is used by handleWeakUndefines in order to generate a callable
// equivalent of an undefined function.
InputFunction *SymbolTable::replaceWithUnreachable(Symbol *Sym,
                                                   const WasmSignature &Sig,
                                                   StringRef DebugName) {
  auto *Func = make<SyntheticFunction>(Sig, Sym->getName(), DebugName);
  Func->setBody(UnreachableFn);
  SyntheticFunctions.emplace_back(Func);
  replaceSymbol<DefinedFunction>(Sym, Sym->getName(), Sym->getFlags(), nullptr, Func);
  return Func;
}

// For weak undefined functions, there may be "call" instructions that reference
// the symbol. In this case, we need to synthesise a dummy/stub function that
// will abort at runtime, so that relocations can still provided an operand to
// the call instruction that passes Wasm validation.
void SymbolTable::handleWeakUndefines() {
  for (Symbol *Sym : getSymbols()) {
    if (!Sym->isUndefWeak())
      continue;

    const WasmSignature *Sig = nullptr;

    if (auto *FuncSym = dyn_cast<FunctionSymbol>(Sym)) {
      // It is possible for undefined functions not to have a signature (eg. if
      // added via "--undefined"), but weak undefined ones do have a signature.
      assert(FuncSym->Signature);
      Sig = FuncSym->Signature;
    } else if (auto *LazySym = dyn_cast<LazySymbol>(Sym)) {
      // Lazy symbols may not be functions and therefore can have a null
      // signature.
      Sig = LazySym->Signature;
    }

    if (!Sig)
      continue;

    // Add a synthetic dummy for weak undefined functions.  These dummies will
    // be GC'd if not used as the target of any "call" instructions.
    StringRef DebugName = Saver.save("undefined:" + toString(*Sym));
    InputFunction* Func = replaceWithUnreachable(Sym, *Sig, DebugName);
    // Ensure it compares equal to the null pointer, and so that table relocs
    // don't pull in the stub body (only call-operand relocs should do that).
    Func->setTableIndex(0);
    // Hide our dummy to prevent export.
    Sym->setHidden(true);
  }
}
