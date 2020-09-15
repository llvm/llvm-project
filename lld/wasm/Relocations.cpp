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

namespace lld {
namespace wasm {
static bool requiresGOTAccess(const Symbol *sym) {
  return config->isPic && !sym->isHidden() && !sym->isLocal();
}

static bool allowUndefined(const Symbol* sym) {
  // Undefined functions and globals with explicit import name are allowed to be
  // undefined at link time.
  if (auto *f = dyn_cast<UndefinedFunction>(sym))
    if (f->importName)
      return true;
  if (auto *g = dyn_cast<UndefinedGlobal>(sym))
    if (g->importName)
      return true;
  return (config->allowUndefined ||
          config->allowUndefinedSymbols.count(sym->getName()) != 0);
}

static void reportUndefined(const Symbol* sym) {
  assert(sym->isUndefined());
  assert(!sym->isWeak());
  if (!allowUndefined(sym))
    error(toString(sym->getFile()) + ": undefined symbol: " + toString(*sym));
}

static void addGOTEntry(Symbol *sym) {
  // In PIC mode a GOT entry is an imported global that the dynamic linker
  // will assign.
  // In non-PIC mode (i.e. when code compiled as fPIC is linked into a static
  // binary) we create an internal wasm global with a fixed value that takes the
  // place of th GOT entry and effectivly acts as an i32 const. This can
  // potentially be optimized away at runtime or with a post-link tool.
  // TODO(sbc): Linker relaxation might also be able to optimize this away.
  if (config->isPic)
    out.importSec->addGOTEntry(sym);
  else
    out.globalSec->addStaticGOTEntry(sym);
}

void scanRelocations(InputChunk *chunk) {
  if (!chunk->live)
    return;
  ObjFile *file = chunk->file;
  ArrayRef<WasmSignature> types = file->getWasmObj()->types();
  for (const WasmRelocation &reloc : chunk->getRelocations()) {
    if (reloc.Type == R_WASM_TYPE_INDEX_LEB) {
      // Mark target type as live
      file->typeMap[reloc.Index] =
          out.typeSec->registerType(types[reloc.Index]);
      file->typeIsUsed[reloc.Index] = true;
      continue;
    }

    // Other relocation types all have a corresponding symbol
    Symbol *sym = file->getSymbols()[reloc.Index];

    switch (reloc.Type) {
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_TABLE_INDEX_I64:
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_SLEB64:
    case R_WASM_TABLE_INDEX_REL_SLEB:
      if (requiresGOTAccess(sym))
        break;
      out.elemSec->addEntry(cast<FunctionSymbol>(sym));
      break;
    case R_WASM_GLOBAL_INDEX_LEB:
    case R_WASM_GLOBAL_INDEX_I32:
      if (!isa<GlobalSymbol>(sym))
        addGOTEntry(sym);
      break;
    }

    if (config->isPic) {
      switch (reloc.Type) {
      case R_WASM_TABLE_INDEX_SLEB:
      case R_WASM_TABLE_INDEX_SLEB64:
      case R_WASM_MEMORY_ADDR_SLEB:
      case R_WASM_MEMORY_ADDR_LEB:
      case R_WASM_MEMORY_ADDR_SLEB64:
      case R_WASM_MEMORY_ADDR_LEB64:
        // Certain relocation types can't be used when building PIC output,
        // since they would require absolute symbol addresses at link time.
        error(toString(file) + ": relocation " + relocTypeToString(reloc.Type) +
              " cannot be used against symbol " + toString(*sym) +
              "; recompile with -fPIC");
        break;
      case R_WASM_TABLE_INDEX_I32:
      case R_WASM_TABLE_INDEX_I64:
      case R_WASM_MEMORY_ADDR_I32:
      case R_WASM_MEMORY_ADDR_I64:
        // These relocation types are only present in the data section and
        // will be converted into code by `generateRelocationCode`.  This code
        // requires the symbols to have GOT entires.
        if (requiresGOTAccess(sym))
          addGOTEntry(sym);
        break;
      }
    } else {
      // Report undefined symbols
      if (sym->isUndefined() && !config->relocatable && !sym->isWeak())
        reportUndefined(sym);
    }

  }
}

} // namespace wasm
} // namespace lld
