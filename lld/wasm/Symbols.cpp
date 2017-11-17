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
#include "InputFiles.h"
#include "InputSegment.h"
#include "Strings.h"
#include "lld/Common/ErrorHandler.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace lld;
using namespace lld::wasm;

uint32_t Symbol::getGlobalIndex() const {
  assert(!Sym->isFunction());
  return Sym->ElementIndex;
}

uint32_t Symbol::getFunctionIndex() const {
  assert(Sym->isFunction());
  return Sym->ElementIndex;
}

uint32_t Symbol::getFunctionTypeIndex() const {
  assert(Sym->isFunction());
  ObjFile *Obj = cast<ObjFile>(File);
  if (Obj->isImportedFunction(Sym->ElementIndex)) {
    const WasmImport &Import = Obj->getWasmObj()->imports()[Sym->ImportIndex];
    DEBUG(dbgs() << "getFunctionTypeIndex: import: " << Sym->ImportIndex
                 << " -> " << Import.SigIndex << "\n");
    return Import.SigIndex;
  }
  DEBUG(dbgs() << "getFunctionTypeIndex: non import: " << Sym->ElementIndex
               << "\n");
  uint32_t FuntionIndex = Sym->ElementIndex - Obj->NumFunctionImports();
  return Obj->getWasmObj()->functionTypes()[FuntionIndex];
}

uint32_t Symbol::getVirtualAddress() const {
  assert(isGlobal());
  DEBUG(dbgs() << "getVirtualAddress: " << getName() << "\n");
  if (isUndefined())
    return UINT32_MAX;

  assert(Sym != nullptr);
  ObjFile *Obj = cast<ObjFile>(File);
  const WasmGlobal &Global =
      Obj->getWasmObj()->globals()[getGlobalIndex() - Obj->NumGlobalImports()];
  assert(Global.Type == llvm::wasm::WASM_TYPE_I32);
  assert(Segment);
  return Segment->translateVA(Global.InitExpr.Value.Int32);
}

uint32_t Symbol::getOutputIndex() const {
  if (isUndefined() && isWeak())
    return 0;
  return OutputIndex.getValue();
}

void Symbol::setOutputIndex(uint32_t Index) {
  DEBUG(dbgs() << "setOutputIndex " << Name << " -> " << Index << "\n");
  assert(!hasOutputIndex());
  OutputIndex = Index;
}

void Symbol::update(Kind K, InputFile *F, const WasmSymbol *WasmSym,
                    const InputSegment *Seg) {
  SymbolKind = K;
  File = F;
  Sym = WasmSym;
  Segment = Seg;
}

bool Symbol::isWeak() const { return Sym && Sym->isWeak(); }

std::string lld::toString(wasm::Symbol &Sym) {
  return wasm::displayName(Sym.getName());
}

std::string lld::toString(wasm::Symbol::Kind &Kind) {
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
