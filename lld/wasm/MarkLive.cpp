//===- MarkLive.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements --gc-sections, which is a feature to remove unused
// chunks from the output. Unused chunks are those that are not reachable from
// known root symbols or chunks. This feature is implemented as a mark-sweep
// garbage collector.
//
// Here's how it works. Each InputChunk has a "Live" bit. The bit is off by
// default. Starting with the GC-roots, visit all reachable chunks and set their
// Live bits. The Writer will then ignore chunks whose Live bits are off, so
// that such chunk are not appear in the output.
//
//===----------------------------------------------------------------------===//

#include "MarkLive.h"
#include "Config.h"
#include "InputChunks.h"
#include "SymbolTable.h"
#include "Symbols.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

void lld::wasm::markLive() {
  if (!Config->GcSections)
    return;

  DEBUG(dbgs() << "markLive\n");
  SmallVector<InputChunk *, 256> Q;

  auto Enqueue = [&](Symbol *Sym) {
    if (!Sym)
      return;
    InputChunk *Chunk = Sym->getChunk();
    if (!Chunk || Chunk->Live)
      return;
    Chunk->Live = true;
    Q.push_back(Chunk);
  };

  // Add GC root symbols.
  if (!Config->Entry.empty())
    Enqueue(Symtab->find(Config->Entry));
  Enqueue(WasmSym::CallCtors);

  // By default we export all non-hidden, so they are gc roots too
  for (Symbol *Sym : Symtab->getSymbols())
    if (!Sym->isHidden())
      Enqueue(Sym);

  // The ctor functions are all used in the synthetic __wasm_call_ctors
  // function, but since this function is created in-place it doesn't contain
  // relocations which mean we have to manually mark the ctors.
  for (const ObjFile *Obj : Symtab->ObjectFiles) {
    const WasmLinkingData &L = Obj->getWasmObj()->linkingData();
    for (const WasmInitFunc &F : L.InitFunctions)
      Enqueue(Obj->getFunctionSymbol(F.FunctionIndex));
  }

  // Follow relocations to mark all reachable chunks.
  while (!Q.empty()) {
    InputChunk *C = Q.pop_back_val();

    for (const WasmRelocation Reloc : C->getRelocations()) {
      switch (Reloc.Type) {
      case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
      case R_WEBASSEMBLY_TABLE_INDEX_I32:
      case R_WEBASSEMBLY_TABLE_INDEX_SLEB:
        Enqueue(C->File->getFunctionSymbol(Reloc.Index));
        break;
      case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
      case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
      case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
      case R_WEBASSEMBLY_MEMORY_ADDR_I32:
        Enqueue(C->File->getGlobalSymbol(Reloc.Index));
        break;
      }
    }
  }

  // Report garbage-collected sections.
  if (Config->PrintGcSections) {
    for (const ObjFile *Obj : Symtab->ObjectFiles) {
      for (InputChunk *C : Obj->Functions)
        if (!C->Live)
          message("removing unused section " + toString(C));
      for (InputChunk *C : Obj->Segments)
        if (!C->Live)
          message("removing unused section " + toString(C));
    }
  }
}
