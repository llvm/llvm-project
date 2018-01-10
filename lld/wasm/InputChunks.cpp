//===- InputSegment.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputChunks.h"
#include "OutputSegment.h"
#include "lld/Common/LLVM.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace lld::wasm;

uint32_t InputSegment::translateVA(uint32_t Address) const {
  assert(Address >= startVA() && Address < endVA());
  int32_t Delta = OutputSeg->StartVA + OutputOffset - startVA();
  DEBUG(dbgs() << "translateVA: " << getName() << " Delta=" << Delta
               << " Address=" << Address << "\n");
  return Address + Delta;
}

void InputChunk::copyRelocations(const WasmSection &Section) {
  size_t Start = getInputSectionOffset();
  size_t Size = getSize();
  for (const WasmRelocation &R : Section.Relocations)
    if (R.Offset >= Start && R.Offset < Start + Size)
      Relocations.push_back(R);
}
