//===- MarkLive.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "InputEvent.h"
#include "InputGlobal.h"
#include "SymbolTable.h"
#include "Symbols.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;

namespace lld {
namespace wasm {

namespace {

class MarkLive {
public:
  void run();

private:
  void enqueue(Symbol *sym);
  void enqueueInitFunctions(const ObjFile *sym);
  void markSymbol(Symbol *sym);
  void mark();
  bool isCallCtorsLive();

  // A list of chunks to visit.
  SmallVector<InputChunk *, 256> queue;
};

} // namespace

void MarkLive::enqueue(Symbol *sym) {
  if (!sym || sym->isLive())
    return;
  LLVM_DEBUG(dbgs() << "markLive: " << sym->getName() << "\n");

  InputFile *file = sym->getFile();
  bool needInitFunctions = file && !file->isLive() && sym->isDefined();

  sym->markLive();

  // Mark ctor functions in the object that defines this symbol live.
  // The ctor functions are all referenced by the synthetic callCtors
  // function. However, this function does not contain relocations so we
  // have to manually mark the ctors as live.
  if (needInitFunctions)
    enqueueInitFunctions(cast<ObjFile>(file));

  if (InputChunk *chunk = sym->getChunk())
    queue.push_back(chunk);
}

// The ctor functions are all referenced by the synthetic callCtors
// function.  However, this function does not contain relocations so we
// have to manually mark the ctors as live.
void MarkLive::enqueueInitFunctions(const ObjFile *obj) {
  const WasmLinkingData &l = obj->getWasmObj()->linkingData();
  for (const WasmInitFunc &f : l.InitFunctions) {
    auto *initSym = obj->getFunctionSymbol(f.Symbol);
    if (!initSym->isDiscarded())
      enqueue(initSym);
  }
}

void MarkLive::run() {
  // Add GC root symbols.
  if (!config->entry.empty())
    enqueue(symtab->find(config->entry));

  // We need to preserve any no-strip or exported symbol
  for (Symbol *sym : symtab->getSymbols())
    if (sym->isNoStrip() || sym->isExported())
      enqueue(sym);

  // If we'll be calling the user's `__wasm_call_dtors` function, mark it live.
  if (Symbol *callDtors = WasmSym::callDtors)
    enqueue(callDtors);

  // In Emscripten-style PIC, `__wasm_call_ctors` calls `__wasm_apply_relocs`.
  if (config->isPic)
    enqueue(WasmSym::applyRelocs);

  if (config->sharedMemory && !config->shared)
    enqueue(WasmSym::initMemory);

  // Enqueue constructors in objects explicitly live from the command-line.
  for (const ObjFile *obj : symtab->objectFiles)
    if (obj->isLive())
      enqueueInitFunctions(obj);

  mark();

  // If we have any non-discarded init functions, mark `__wasm_call_ctors` as
  // live so that we assign it an index and call it.
  if (isCallCtorsLive())
    WasmSym::callCtors->markLive();
}

void MarkLive::mark() {
  // Follow relocations to mark all reachable chunks.
  while (!queue.empty()) {
    InputChunk *c = queue.pop_back_val();

    for (const WasmRelocation reloc : c->getRelocations()) {
      if (reloc.Type == R_WASM_TYPE_INDEX_LEB)
        continue;
      Symbol *sym = c->file->getSymbol(reloc.Index);

      // If the function has been assigned the special index zero in the table,
      // the relocation doesn't pull in the function body, since the function
      // won't actually go in the table (the runtime will trap attempts to call
      // that index, since we don't use it).  A function with a table index of
      // zero is only reachable via "call", not via "call_indirect".  The stub
      // functions used for weak-undefined symbols have this behaviour (compare
      // equal to null pointer, only reachable via direct call).
      if (reloc.Type == R_WASM_TABLE_INDEX_SLEB ||
          reloc.Type == R_WASM_TABLE_INDEX_SLEB64 ||
          reloc.Type == R_WASM_TABLE_INDEX_I32 ||
          reloc.Type == R_WASM_TABLE_INDEX_I64) {
        auto *funcSym = cast<FunctionSymbol>(sym);
        if (funcSym->hasTableIndex() && funcSym->getTableIndex() == 0)
          continue;
      }

      enqueue(sym);
    }
  }
}

void markLive() {
  if (!config->gcSections)
    return;

  LLVM_DEBUG(dbgs() << "markLive\n");

  MarkLive marker;
  marker.run();

  // Report garbage-collected sections.
  if (config->printGcSections) {
    for (const ObjFile *obj : symtab->objectFiles) {
      for (InputChunk *c : obj->functions)
        if (!c->live)
          message("removing unused section " + toString(c));
      for (InputChunk *c : obj->segments)
        if (!c->live)
          message("removing unused section " + toString(c));
      for (InputGlobal *g : obj->globals)
        if (!g->live)
          message("removing unused section " + toString(g));
      for (InputEvent *e : obj->events)
        if (!e->live)
          message("removing unused section " + toString(e));
    }
    for (InputChunk *c : symtab->syntheticFunctions)
      if (!c->live)
        message("removing unused section " + toString(c));
    for (InputGlobal *g : symtab->syntheticGlobals)
      if (!g->live)
        message("removing unused section " + toString(g));
  }
}

bool MarkLive::isCallCtorsLive() {
  // In a reloctable link, we don't call `__wasm_call_ctors`.
  if (config->relocatable)
    return false;

  // In Emscripten-style PIC, we call `__wasm_call_ctors` which calls
  // `__wasm_apply_relocs`.
  if (config->isPic)
    return true;

  // If there are any init functions, mark `__wasm_call_ctors` live so that
  // it can call them.
  for (const ObjFile *file : symtab->objectFiles) {
    const WasmLinkingData &l = file->getWasmObj()->linkingData();
    for (const WasmInitFunc &f : l.InitFunctions) {
      auto *sym = file->getFunctionSymbol(f.Symbol);
      if (!sym->isDiscarded() && sym->isLive())
        return true;
    }
  }

  return false;
}

} // namespace wasm
} // namespace lld
