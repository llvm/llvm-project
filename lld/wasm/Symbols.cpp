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

const WasmSignature &Symbol::getFunctionType() const {
  if (Function != nullptr)
    return Function->Signature;

  assert(FunctionType != nullptr);
  return *FunctionType;
}

void Symbol::setFunctionType(const WasmSignature *Type) {
  assert(FunctionType == nullptr);
  assert(Function == nullptr);
  FunctionType = Type;
}

uint32_t Symbol::getVirtualAddress() const {
  assert(isGlobal());
  DEBUG(dbgs() << "getVirtualAddress: " << getName() << "\n");
  return Segment ? Segment->translateVA(VirtualAddress) : VirtualAddress;
}

bool Symbol::hasOutputIndex() const {
  if (Function)
    return Function->hasOutputIndex();
  return OutputIndex.hasValue();
}

uint32_t Symbol::getOutputIndex() const {
  if (Function)
    return Function->getOutputIndex();
  return OutputIndex.getValue();
}

void Symbol::setVirtualAddress(uint32_t Value) {
  DEBUG(dbgs() << "setVirtualAddress " << Name << " -> " << Value << "\n");
  assert(isGlobal());
  VirtualAddress = Value;
}

void Symbol::setOutputIndex(uint32_t Index) {
  DEBUG(dbgs() << "setOutputIndex " << Name << " -> " << Index << "\n");
  assert(!Function);
  assert(!OutputIndex.hasValue());
  OutputIndex = Index;
}

uint32_t Symbol::getTableIndex() const {
  if (Function)
    return Function->getTableIndex();
  return TableIndex.getValue();
}

bool Symbol::hasTableIndex() const {
  if (Function)
    return Function->hasTableIndex();
  return TableIndex.hasValue();
}

void Symbol::setTableIndex(uint32_t Index) {
  // For imports, we set the table index here on the Symbol; for defined
  // functions we set the index on the InputFunction so that we don't export
  // the same thing twice (keeps the table size down).
  if (Function) {
    Function->setTableIndex(Index);
    return;
  }
  DEBUG(dbgs() << "setTableIndex " << Name << " -> " << Index << "\n");
  assert(!TableIndex.hasValue());
  TableIndex = Index;
}

void Symbol::update(Kind K, InputFile *F, uint32_t Flags_,
                    const InputSegment *Seg, InputFunction *Func,
                    uint32_t Address) {
  SymbolKind = K;
  File = F;
  Flags = Flags_;
  Segment = Seg;
  Function = Func;
  if (Address != UINT32_MAX)
    setVirtualAddress(Address);
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
  case wasm::Symbol::DefinedGlobalKind:
    return "DefinedGlobal";
  case wasm::Symbol::UndefinedFunctionKind:
    return "UndefinedFunction";
  case wasm::Symbol::UndefinedGlobalKind:
    return "UndefinedGlobal";
  case wasm::Symbol::LazyKind:
    return "LazyKind";
  }
  llvm_unreachable("Invalid symbol kind!");
}
