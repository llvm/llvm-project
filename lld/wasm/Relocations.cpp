//===- Relocations.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Relocations.h"

#include "InputChunks.h"
#include "SyntheticSections.h"

using namespace llvm;
using namespace llvm::wasm;

using namespace lld;
using namespace lld::wasm;

static bool requiresGOTAccess(const Symbol *Sym) {
  return Config->Pic && !Sym->isHidden() && !Sym->isLocal();
}

void lld::wasm::scanRelocations(InputChunk *Chunk) {
  if (!Chunk->Live)
    return;
  ObjFile *File = Chunk->File;
  ArrayRef<WasmSignature> Types = File->getWasmObj()->types();
  for (const WasmRelocation &Reloc : Chunk->getRelocations()) {
    if (Reloc.Type == R_WASM_TYPE_INDEX_LEB) {
      // Mark target type as live
      File->TypeMap[Reloc.Index] =
          Out.TypeSec->registerType(Types[Reloc.Index]);
      File->TypeIsUsed[Reloc.Index] = true;
      continue;
    }

    // Other relocation types all have a corresponding symbol
    Symbol *Sym = File->getSymbols()[Reloc.Index];

    switch (Reloc.Type) {
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_REL_SLEB:
      if (requiresGOTAccess(Sym))
        break;
      Out.ElemSec->addEntry(cast<FunctionSymbol>(Sym));
      break;
    case R_WASM_GLOBAL_INDEX_LEB:
      if (!isa<GlobalSymbol>(Sym))
        Out.ImportSec->addGOTEntry(Sym);
      break;
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_LEB:
    case R_WASM_MEMORY_ADDR_REL_SLEB:
      if (!Config->Relocatable && Sym->isUndefined() && !Sym->isWeak()) {
        error(toString(File) + ": cannot resolve relocation of type " +
              relocTypeToString(Reloc.Type) +
              " against undefined (non-weak) data symbol: " + toString(*Sym));
      }
      break;
    }

    if (Config->Pic) {
      switch (Reloc.Type) {
      case R_WASM_TABLE_INDEX_SLEB:
      case R_WASM_MEMORY_ADDR_SLEB:
      case R_WASM_MEMORY_ADDR_LEB:
        // Certain relocation types can't be used when building PIC output,
        // since they would require absolute symbol addresses at link time.
        error(toString(File) + ": relocation " + relocTypeToString(Reloc.Type) +
              " cannot be used againt symbol " + toString(*Sym) +
              "; recompile with -fPIC");
        break;
      case R_WASM_TABLE_INDEX_I32:
      case R_WASM_MEMORY_ADDR_I32:
        // These relocation types are only present in the data section and
        // will be converted into code by `generateRelocationCode`.  This code
        // requires the symbols to have GOT entires.
        if (requiresGOTAccess(Sym))
          Out.ImportSec->addGOTEntry(Sym);
        break;
      }
    }
  }
}
