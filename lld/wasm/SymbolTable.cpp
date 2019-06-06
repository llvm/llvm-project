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

  // .a file
  if (auto *F = dyn_cast<ArchiveFile>(File)) {
    F->parse();
    return;
  }

  // .so file
  if (auto *F = dyn_cast<SharedFile>(File)) {
    SharedFiles.push_back(F);
    return;
  }

  if (Config->Trace)
    message(toString(File));

  // LLVM bitcode file
  if (auto *F = dyn_cast<BitcodeFile>(File)) {
    F->parse();
    BitcodeFiles.push_back(F);
    return;
  }

  // Regular object file
  auto *F = cast<ObjFile>(File);
  F->parse(false);
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
    auto *Obj = make<ObjFile>(MemoryBufferRef(Filename, "lto.tmp"), "");
    Obj->parse(true);
    ObjectFiles.push_back(Obj);
  }
}

void SymbolTable::reportRemainingUndefines() {
  for (const auto& Pair : SymMap) {
    const Symbol *Sym = SymVector[Pair.second];
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

void SymbolTable::replace(StringRef Name, Symbol* Sym) {
  auto It = SymMap.find(CachedHashStringRef(Name));
  SymVector[It->second] = Sym;
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
  Sym->CanInline = true;
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
// Returns true if the function types match, false is there is a singature
// mismatch.
static bool signatureMatches(FunctionSymbol *Existing,
                             const WasmSignature *NewSig) {
  const WasmSignature *OldSig = Existing->Signature;

  // If either function is missing a signature (this happend for bitcode
  // symbols) then assume they match.  Any mismatch will be reported later
  // when the LTO objects are added.
  if (!NewSig || !OldSig)
    return true;

  return *NewSig == *OldSig;
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
  return replaceSymbol<DefinedFunction>(insertName(Name).first, Name,
                                        Flags, nullptr, Function);
}

// Adds an optional, linker generated, data symbols.  The symbol will only be
// added if there is an undefine reference to it, or if it is explictly exported
// via the --export flag.  Otherwise we don't add the symbol and return nullptr.
DefinedData *SymbolTable::addOptionalDataSymbol(StringRef Name, uint32_t Value,
                                                uint32_t Flags) {
  Symbol *S = find(Name);
  if (!S && (Config->ExportAll || Config->ExportedSymbols.count(Name) != 0))
    S = insertName(Name).first;
  else if (!S || S->isDefined())
    return nullptr;
  LLVM_DEBUG(dbgs() << "addOptionalDataSymbol: " << Name << "\n");
  auto *rtn = replaceSymbol<DefinedData>(S, Name, Flags);
  rtn->setVirtualAddress(Value);
  return rtn;
}

DefinedData *SymbolTable::addSyntheticDataSymbol(StringRef Name,
                                                 uint32_t Flags) {
  LLVM_DEBUG(dbgs() << "addSyntheticDataSymbol: " << Name << "\n");
  assert(!find(Name));
  return replaceSymbol<DefinedData>(insertName(Name).first, Name, Flags);
}

DefinedGlobal *SymbolTable::addSyntheticGlobal(StringRef Name, uint32_t Flags,
                                               InputGlobal *Global) {
  LLVM_DEBUG(dbgs() << "addSyntheticGlobal: " << Name << " -> " << Global
                    << "\n");
  assert(!find(Name));
  SyntheticGlobals.emplace_back(Global);
  return replaceSymbol<DefinedGlobal>(insertName(Name).first, Name, Flags,
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

  auto Replace = [&](Symbol* Sym) {
    // If the new defined function doesn't have signture (i.e. bitcode
    // functions) but the old symbol does, then preserve the old signature
    const WasmSignature *OldSig = S->getSignature();
    auto* NewSym = replaceSymbol<DefinedFunction>(Sym, Name, Flags, File, Function);
    if (!NewSym->Signature)
      NewSym->Signature = OldSig;
  };

  if (WasInserted || S->isLazy()) {
    Replace(S);
    return S;
  }

  auto ExistingFunction = dyn_cast<FunctionSymbol>(S);
  if (!ExistingFunction) {
    reportTypeError(S, File, WASM_SYMBOL_TYPE_FUNCTION);
    return S;
  }

  bool CheckSig = true;
  if (auto UD = dyn_cast<UndefinedFunction>(ExistingFunction))
    CheckSig = UD->IsCalledDirectly;

  if (CheckSig && Function && !signatureMatches(ExistingFunction, &Function->Signature)) {
    Symbol* Variant;
    if (getFunctionVariant(S, &Function->Signature, File, &Variant))
      // New variant, always replace
      Replace(Variant);
    else if (shouldReplace(S, File, Flags))
      // Variant already exists, replace it after checking shouldReplace
      Replace(Variant);

    // This variant we found take the place in the symbol table as the primary
    // variant.
    replace(Name, Variant);
    return Variant;
  }

  // Existing function with matching signature.
  if (shouldReplace(S, File, Flags))
    Replace(S);

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

  auto Replace = [&]() {
    replaceSymbol<DefinedData>(S, Name, Flags, File, Segment, Address, Size);
  };

  if (WasInserted || S->isLazy()) {
    Replace();
    return S;
  }

  checkDataType(S, File);

  if (shouldReplace(S, File, Flags))
    Replace();
  return S;
}

Symbol *SymbolTable::addDefinedGlobal(StringRef Name, uint32_t Flags,
                                      InputFile *File, InputGlobal *Global) {
  LLVM_DEBUG(dbgs() << "addDefinedGlobal:" << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  auto Replace = [&]() {
    replaceSymbol<DefinedGlobal>(S, Name, Flags, File, Global);
  };

  if (WasInserted || S->isLazy()) {
    Replace();
    return S;
  }

  checkGlobalType(S, File, &Global->getType());

  if (shouldReplace(S, File, Flags))
    Replace();
  return S;
}

Symbol *SymbolTable::addDefinedEvent(StringRef Name, uint32_t Flags,
                                     InputFile *File, InputEvent *Event) {
  LLVM_DEBUG(dbgs() << "addDefinedEvent:" << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);

  auto Replace = [&]() {
    replaceSymbol<DefinedEvent>(S, Name, Flags, File, Event);
  };

  if (WasInserted || S->isLazy()) {
    Replace();
    return S;
  }

  checkEventType(S, File, &Event->getType(), &Event->Signature);

  if (shouldReplace(S, File, Flags))
    Replace();
  return S;
}

Symbol *SymbolTable::addUndefinedFunction(StringRef Name, StringRef ImportName,
                                          StringRef ImportModule,
                                          uint32_t Flags, InputFile *File,
                                          const WasmSignature *Sig,
                                          bool IsCalledDirectly) {
  LLVM_DEBUG(dbgs() << "addUndefinedFunction: " << Name << " ["
                    << (Sig ? toString(*Sig) : "none")
                    << "] IsCalledDirectly:" << IsCalledDirectly << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);
  if (S->Traced)
    printTraceSymbolUndefined(Name, File);

  auto Replace = [&]() {
    replaceSymbol<UndefinedFunction>(S, Name, ImportName, ImportModule, Flags,
                                     File, Sig, IsCalledDirectly);
  };

  if (WasInserted)
    Replace();
  else if (auto *Lazy = dyn_cast<LazySymbol>(S))
    Lazy->fetch();
  else {
    auto ExistingFunction = dyn_cast<FunctionSymbol>(S);
    if (!ExistingFunction) {
      reportTypeError(S, File, WASM_SYMBOL_TYPE_FUNCTION);
      return S;
    }
    if (!ExistingFunction->Signature && Sig)
      ExistingFunction->Signature = Sig;
    if (IsCalledDirectly && !signatureMatches(ExistingFunction, Sig))
      if (getFunctionVariant(S, Sig, File, &S))
        Replace();
  }

  return S;
}

Symbol *SymbolTable::addUndefinedData(StringRef Name, uint32_t Flags,
                                      InputFile *File) {
  LLVM_DEBUG(dbgs() << "addUndefinedData: " << Name << "\n");

  Symbol *S;
  bool WasInserted;
  std::tie(S, WasInserted) = insert(Name, File);
  if (S->Traced)
    printTraceSymbolUndefined(Name, File);

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
  if (S->Traced)
    printTraceSymbolUndefined(Name, File);

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
  std::tie(S, WasInserted) = insertName(Name);

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
  return ComdatGroups.insert(CachedHashStringRef(Name)).second;
}

// The new signature doesn't match.  Create a variant to the symbol with the
// signature encoded in the name and return that instead.  These symbols are
// then unified later in handleSymbolVariants.
bool SymbolTable::getFunctionVariant(Symbol* Sym, const WasmSignature *Sig,
                                     const InputFile *File, Symbol **Out) {
  LLVM_DEBUG(dbgs() << "getFunctionVariant: " << Sym->getName() << " -> "
                    << " " << toString(*Sig) << "\n");
  Symbol *Variant = nullptr;

  // Linear search through symbol variants.  Should never be more than two
  // or three entries here.
  auto &Variants = SymVariants[CachedHashStringRef(Sym->getName())];
  if (Variants.empty())
    Variants.push_back(Sym);

  for (Symbol* V : Variants) {
    if (*V->getSignature() == *Sig) {
      Variant = V;
      break;
    }
  }

  bool WasAdded = !Variant;
  if (WasAdded) {
    // Create a new variant;
    LLVM_DEBUG(dbgs() << "added new variant\n");
    Variant = reinterpret_cast<Symbol *>(make<SymbolUnion>());
    Variants.push_back(Variant);
  } else {
    LLVM_DEBUG(dbgs() << "variant already exists: " << toString(*Variant) << "\n");
    assert(*Variant->getSignature() == *Sig);
  }

  *Out = Variant;
  return WasAdded;
}

// Set a flag for --trace-symbol so that we can print out a log message
// if a new symbol with the same name is inserted into the symbol table.
void SymbolTable::trace(StringRef Name) {
  SymMap.insert({CachedHashStringRef(Name), -1});
}

void SymbolTable::wrap(Symbol *Sym, Symbol *Real, Symbol *Wrap) {
  // Swap symbols as instructed by -wrap.
  int &OrigIdx = SymMap[CachedHashStringRef(Sym->getName())];
  int &RealIdx= SymMap[CachedHashStringRef(Real->getName())];
  int &WrapIdx = SymMap[CachedHashStringRef(Wrap->getName())];
  LLVM_DEBUG(dbgs() << "wrap: " << Sym->getName() << "\n");

  // Anyone looking up __real symbols should get the original
  RealIdx = OrigIdx;
  // Anyone looking up the original should get the __wrap symbol
  OrigIdx = WrapIdx;
}

static const uint8_t UnreachableFn[] = {
    0x03 /* ULEB length */, 0x00 /* ULEB num locals */,
    0x00 /* opcode unreachable */, 0x0b /* opcode end */
};

// Replace the given symbol body with an unreachable function.
// This is used by handleWeakUndefines in order to generate a callable
// equivalent of an undefined function and also handleSymbolVariants for
// undefined functions that don't match the signature of the definition.
InputFunction *SymbolTable::replaceWithUnreachable(Symbol *Sym,
                                                   const WasmSignature &Sig,
                                                   StringRef DebugName) {
  auto *Func = make<SyntheticFunction>(Sig, Sym->getName(), DebugName);
  Func->setBody(UnreachableFn);
  SyntheticFunctions.emplace_back(Func);
  replaceSymbol<DefinedFunction>(Sym, Sym->getName(), Sym->getFlags(), nullptr,
                                 Func);
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

    const WasmSignature *Sig = Sym->getSignature();
    if (!Sig) {
      // It is possible for undefined functions not to have a signature (eg. if
      // added via "--undefined"), but weak undefined ones do have a signature.
      // Lazy symbols may not be functions and therefore Sig can still be null
      // in some circumstantce.
      assert(!isa<FunctionSymbol>(Sym));
      continue;
    }

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

static void reportFunctionSignatureMismatch(StringRef SymName,
                                            FunctionSymbol *A,
                                            FunctionSymbol *B, bool Error) {
  std::string msg = ("function signature mismatch: " + SymName +
                     "\n>>> defined as " + toString(*A->Signature) + " in " +
                     toString(A->getFile()) + "\n>>> defined as " +
                     toString(*B->Signature) + " in " + toString(B->getFile()))
                        .str();
  if (Error)
    error(msg);
  else
    warn(msg);
}

// Remove any variant symbols that were created due to function signature
// mismatches.
void SymbolTable::handleSymbolVariants() {
  for (auto Pair : SymVariants) {
    // Push the initial symbol onto the list of variants.
    StringRef SymName = Pair.first.val();
    std::vector<Symbol *> &Variants = Pair.second;

#ifndef NDEBUG
    LLVM_DEBUG(dbgs() << "symbol with (" << Variants.size()
                      << ") variants: " << SymName << "\n");
    for (auto *S: Variants) {
      auto *F = cast<FunctionSymbol>(S);
      LLVM_DEBUG(dbgs() << " variant: " + F->getName() << " "
                        << toString(*F->Signature) << "\n");
    }
#endif

    // Find the one definition.
    DefinedFunction *Defined = nullptr;
    for (auto *Symbol : Variants) {
      if (auto F = dyn_cast<DefinedFunction>(Symbol)) {
        Defined = F;
        break;
      }
    }

    // If there are no definitions, and the undefined symbols disagree on
    // the signature, there is not we can do since we don't know which one
    // to use as the signature on the import.
    if (!Defined) {
      reportFunctionSignatureMismatch(SymName,
                                      cast<FunctionSymbol>(Variants[0]),
                                      cast<FunctionSymbol>(Variants[1]), true);
      return;
    }

    for (auto *Symbol : Variants) {
      if (Symbol != Defined) {
        auto *F = cast<FunctionSymbol>(Symbol);
        reportFunctionSignatureMismatch(SymName, F, Defined, false);
        StringRef DebugName = Saver.save("unreachable:" + toString(*F));
        replaceWithUnreachable(F, *F->Signature, DebugName);
      }
    }
  }
}
