//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputFiles.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

DefinedFunction *WasmSym::CallCtors;
DefinedData *WasmSym::DsoHandle;
DefinedData *WasmSym::DataEnd;
DefinedData *WasmSym::HeapBase;
DefinedData *WasmSym::StackPointer;

bool Symbol::hasOutputIndex() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    if (F->Function)
      return F->Function->hasOutputIndex();
  return OutputIndex != INVALID_INDEX;
}

uint32_t Symbol::getOutputIndex() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    if (F->Function)
      return F->Function->getOutputIndex();
  assert(OutputIndex != INVALID_INDEX);
  return OutputIndex;
}

InputChunk *Symbol::getChunk() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    return F->Function;
  if (auto *G = dyn_cast<DefinedData>(this))
    return G->Segment;
  return nullptr;
}

void Symbol::setOutputIndex(uint32_t Index) {
  DEBUG(dbgs() << "setOutputIndex " << Name << " -> " << Index << "\n");
  assert(OutputIndex == INVALID_INDEX);
  OutputIndex = Index;
}

bool Symbol::isWeak() const {
  return (Flags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_WEAK;
}

bool Symbol::isLocal() const {
  return (Flags & WASM_SYMBOL_BINDING_MASK) == WASM_SYMBOL_BINDING_LOCAL;
}

bool Symbol::isHidden() const {
  return (Flags & WASM_SYMBOL_VISIBILITY_MASK) == WASM_SYMBOL_VISIBILITY_HIDDEN;
}

void Symbol::setHidden(bool IsHidden) {
  DEBUG(dbgs() << "setHidden: " << Name << " -> " << IsHidden << "\n");
  Flags &= ~WASM_SYMBOL_VISIBILITY_MASK;
  if (IsHidden)
    Flags |= WASM_SYMBOL_VISIBILITY_HIDDEN;
  else
    Flags |= WASM_SYMBOL_VISIBILITY_DEFAULT;
}

uint32_t FunctionSymbol::getTableIndex() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    return F->Function->getTableIndex();
  assert(TableIndex != INVALID_INDEX);
  return TableIndex;
}

bool FunctionSymbol::hasTableIndex() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    return F->Function->hasTableIndex();
  return TableIndex != INVALID_INDEX;
}

void FunctionSymbol::setTableIndex(uint32_t Index) {
  // For imports, we set the table index here on the Symbol; for defined
  // functions we set the index on the InputFunction so that we don't export
  // the same thing twice (keeps the table size down).
  if (auto *F = dyn_cast<DefinedFunction>(this)) {
    F->Function->setTableIndex(Index);
    return;
  }
  DEBUG(dbgs() << "setTableIndex " << Name << " -> " << Index << "\n");
  assert(TableIndex == INVALID_INDEX);
  TableIndex = Index;
}

DefinedFunction::DefinedFunction(StringRef Name, uint32_t Flags, InputFile *F,
                                 InputFunction *Function)
    : FunctionSymbol(Name, DefinedFunctionKind, Flags, F,
                     Function ? &Function->Signature : nullptr),
      Function(Function) {}

uint32_t DefinedData::getVirtualAddress() const {
  DEBUG(dbgs() << "getVirtualAddress: " << getName() << "\n");
  return Segment ? Segment->translateVA(VirtualAddress) : VirtualAddress;
}

void DefinedData::setVirtualAddress(uint32_t Value) {
  DEBUG(dbgs() << "setVirtualAddress " << Name << " -> " << Value << "\n");
  VirtualAddress = Value;
}

std::string lld::toString(const wasm::Symbol &Sym) {
  if (Config->Demangle)
    if (Optional<std::string> S = demangleItanium(Sym.getName()))
      return "`" + *S + "'";
  return Sym.getName();
}

std::string lld::toString(wasm::Symbol::Kind Kind) {
  switch (Kind) {
  case wasm::Symbol::DefinedFunctionKind:
    return "DefinedFunction";
  case wasm::Symbol::DefinedDataKind:
    return "DefinedData";
  case wasm::Symbol::UndefinedFunctionKind:
    return "UndefinedFunction";
  case wasm::Symbol::UndefinedDataKind:
    return "UndefinedData";
  case wasm::Symbol::LazyKind:
    return "LazyKind";
  }
  llvm_unreachable("Invalid symbol kind!");
}
