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
#include "InputGlobal.h"
#include "OutputSegment.h"
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
DefinedGlobal *WasmSym::StackPointer;

WasmSymbolType Symbol::getWasmType() const {
  switch (SymbolKind) {
  case Symbol::DefinedFunctionKind:
  case Symbol::UndefinedFunctionKind:
    return llvm::wasm::WASM_SYMBOL_TYPE_FUNCTION;
  case Symbol::DefinedDataKind:
  case Symbol::UndefinedDataKind:
    return llvm::wasm::WASM_SYMBOL_TYPE_DATA;
  case Symbol::DefinedGlobalKind:
  case Symbol::UndefinedGlobalKind:
    return llvm::wasm::WASM_SYMBOL_TYPE_GLOBAL;
  default:
    llvm_unreachable("invalid symbol kind");
  }
}

bool Symbol::hasOutputIndex() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    if (F->Function)
      return F->Function->hasOutputIndex();
  if (auto *G = dyn_cast<DefinedGlobal>(this))
    if (G->Global)
      return G->Global->hasOutputIndex();
  return OutputIndex != INVALID_INDEX;
}

uint32_t Symbol::getOutputIndex() const {
  assert(!isa<DataSymbol>(this));
  if (auto *F = dyn_cast<DefinedFunction>(this))
    if (F->Function)
      return F->Function->getOutputIndex();
  if (auto *G = dyn_cast<DefinedGlobal>(this))
    if (G->Global)
      return G->Global->getOutputIndex();
  assert(OutputIndex != INVALID_INDEX);
  return OutputIndex;
}

InputChunk *Symbol::getChunk() const {
  if (auto *F = dyn_cast<DefinedFunction>(this))
    return F->Function;
  if (auto *D = dyn_cast<DefinedData>(this))
    return D->Segment;
  return nullptr;
}

bool Symbol::isLive() const {
  if (auto *G = dyn_cast<DefinedGlobal>(this))
    return G->Global->Live;
  if (InputChunk *C = getChunk())
    return C->Live;
  // Assume any other kind of symbol is live.
  return true;
}

uint32_t Symbol::getOutputSymbolIndex() const {
  assert(OutputSymbolIndex != INVALID_INDEX);
  return OutputSymbolIndex;
}

void Symbol::setOutputSymbolIndex(uint32_t Index) {
  DEBUG(dbgs() << "setOutputSymbolIndex " << Name << " -> " << Index << "\n");
  assert(OutputSymbolIndex == INVALID_INDEX);
  OutputSymbolIndex = Index;
}

void Symbol::setOutputIndex(uint32_t Index) {
  DEBUG(dbgs() << "setOutputIndex " << Name << " -> " << Index << "\n");
  assert(!isa<DataSymbol>(this));
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
  return Segment ? Segment->translateVA(Offset) : Offset;
}

void DefinedData::setVirtualAddress(uint32_t Value) {
  DEBUG(dbgs() << "setVirtualAddress " << Name << " -> " << Value << "\n");
  assert(!Segment);
  Offset = Value;
}

uint32_t DefinedData::getOutputSegmentOffset() const {
  DEBUG(dbgs() << "getOutputSegmentOffset: " << getName() << "\n");
  return Segment->OutputSegmentOffset + Offset;
}

uint32_t DefinedData::getOutputSegmentIndex() const {
  DEBUG(dbgs() << "getOutputSegmentIndex: " << getName() << "\n");
  return Segment->getOutputSegment()->Index;
}

DefinedGlobal::DefinedGlobal(StringRef Name, uint32_t Flags, InputFile *File,
                             InputGlobal *Global)
    : GlobalSymbol(Name, DefinedGlobalKind, Flags, File,
                   Global ? &Global->getType() : nullptr),
      Global(Global) {}

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
  case wasm::Symbol::DefinedGlobalKind:
    return "DefinedGlobal";
  case wasm::Symbol::UndefinedFunctionKind:
    return "UndefinedFunction";
  case wasm::Symbol::UndefinedDataKind:
    return "UndefinedData";
  case wasm::Symbol::UndefinedGlobalKind:
    return "UndefinedGlobal";
  case wasm::Symbol::LazyKind:
    return "LazyKind";
  }
  llvm_unreachable("invalid symbol kind");
}

std::string lld::toString(WasmSymbolType Type) {
  switch (Type) {
  case llvm::wasm::WASM_SYMBOL_TYPE_FUNCTION:
    return "Function";
  case llvm::wasm::WASM_SYMBOL_TYPE_DATA:
    return "Data";
  case llvm::wasm::WASM_SYMBOL_TYPE_GLOBAL:
    return "Global";
  }
  llvm_unreachable("invalid symbol type");
}
