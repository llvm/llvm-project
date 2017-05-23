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

// Set live bit on for each reachable chunk. Unmarked (unreachable)
// COMDAT chunks will be ignored by Writer, so they will be excluded
// from the final output.
void markLive(const std::vector<Chunk *> &Chunks) {
  // We build up a worklist of sections which have been marked as live. We only
  // push into the worklist when we discover an unmarked section, and we mark
  // as we push, so sections never appear twice in the list.
  SmallVector<SectionChunk *, 256> Worklist;

  // COMDAT section chunks are dead by default. Add non-COMDAT chunks.
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

  // Add GC root chunks.
  for (SymbolBody *B : Config->GCRoot)
    if (auto *D = dyn_cast<DefinedRegular>(B))
      Enqueue(D->getChunk());

  while (!Worklist.empty()) {
    SectionChunk *SC = Worklist.pop_back_val();
    assert(SC->isLive() && "We mark as live when pushing onto the worklist!");

    // Mark all symbols listed in the relocation table for this section.
    for (SymbolBody *S : SC->symbols())
      if (auto *D = dyn_cast<DefinedRegular>(S))
        Enqueue(D->getChunk());

    // Mark associative sections if any.
    for (SectionChunk *C : SC->children())
      Enqueue(C);
  }
}

}
}
