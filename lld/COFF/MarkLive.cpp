//===- MarkLive.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Symbols.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>

namespace lld {
namespace coff {

// Set live bit for each reachable COMDAT chunk and dllimported symbol.
// Unmarked (or unreachable) chunks or symbols are ignored by Writer,
// so they are not included in the final output.
void markLive(const std::vector<Chunk *> &Chunks) {
  // We build up a worklist of sections which have been marked as live. We only
  // push into the worklist when we discover an unmarked section, and we mark
  // as we push, so sections never appear twice in the list.
  SmallVector<SectionChunk *, 256> Worklist;

  // Non-COMDAT chunks are never be gc'ed, so they are gc-root.
  for (Chunk *C : Chunks)
    if (auto *SC = dyn_cast<SectionChunk>(C))
      if (SC->isLive())
        Worklist.push_back(SC);

  auto Enqueue = [&](SectionChunk *C) {
    if (C->isLive())
      return;
    C->markLive();
    Worklist.push_back(C);
  };

  // Mark a given symbol as reachable.
  std::function<void(SymbolBody * B)> AddSym = [&](SymbolBody *B) {
    if (auto *Sym = dyn_cast<DefinedRegular>(B)) {
      Enqueue(Sym->getChunk());
    } else if (auto *Sym = dyn_cast<DefinedImportData>(B)) {
      if (Sym->Live)
        return;
      Sym->Live = true;
      if (Sym->Sibling)
        Sym->Sibling->Live = true;
    } else if (auto *Sym = dyn_cast<DefinedImportThunk>(B)) {
      if (Sym->Live)
        return;
      Sym->Live = true;
      AddSym(Sym->WrappedSym);
    } else if (auto *Sym = dyn_cast<DefinedLocalImport>(B)) {
      AddSym(Sym->WrappedSym);
    }
  };

  // Add gc-root symbols.
  for (SymbolBody *B : Config->GCRoot)
    AddSym(B);

  while (!Worklist.empty()) {
    SectionChunk *SC = Worklist.pop_back_val();
    assert(SC->isLive() && "We mark as live when pushing onto the worklist!");

    // Mark all symbols listed in the relocation table for this section.
    for (SymbolBody *B : SC->symbols())
      AddSym(B);

    // Mark associative sections if any.
    for (SectionChunk *C : SC->children())
      Enqueue(C);
  }
}

}
}
