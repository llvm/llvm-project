//===- InputChunks.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputChunks.h"
#include "Config.h"
#include "OutputSegment.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "llvm/Support/LEB128.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace lld;
using namespace lld::wasm;

uint32_t InputSegment::translateVA(uint32_t Address) const {
  assert(Address >= startVA() && Address < endVA());
  int32_t Delta = OutputSeg->StartVA + OutputSegmentOffset - startVA();
  DEBUG(dbgs() << "translateVA: " << getName() << " Delta=" << Delta
               << " Address=" << Address << "\n");
  return Address + Delta;
}

void InputChunk::copyRelocations(const WasmSection &Section) {
  if (Section.Relocations.empty())
    return;
  size_t Start = getInputSectionOffset();
  size_t Size = getSize();
  for (const WasmRelocation &R : Section.Relocations)
    if (R.Offset >= Start && R.Offset < Start + Size)
      Relocations.push_back(R);
}

static void applyRelocation(uint8_t *Buf, const OutputRelocation &Reloc) {
  DEBUG(dbgs() << "write reloc: type=" << Reloc.Reloc.Type
               << " index=" << Reloc.Reloc.Index << " value=" << Reloc.Value
               << " offset=" << Reloc.Reloc.Offset << "\n");
  Buf += Reloc.Reloc.Offset;
  int64_t ExistingValue;
  switch (Reloc.Reloc.Type) {
  case R_WEBASSEMBLY_TYPE_INDEX_LEB:
  case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
  case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
    ExistingValue = decodeULEB128(Buf);
    // Additional check to verify that the existing value that the location
    // matches our expectations.
    if (ExistingValue != Reloc.Reloc.Index) {
      DEBUG(dbgs() << "existing value: " << decodeULEB128(Buf) << "\n");
      assert(decodeULEB128(Buf) == Reloc.Reloc.Index);
    }
    LLVM_FALLTHROUGH;
  case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    encodeULEB128(Reloc.Value, Buf, 5);
    break;
  case R_WEBASSEMBLY_TABLE_INDEX_SLEB:
  case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
    encodeSLEB128(static_cast<int32_t>(Reloc.Value), Buf, 5);
    break;
  case R_WEBASSEMBLY_TABLE_INDEX_I32:
  case R_WEBASSEMBLY_MEMORY_ADDR_I32:
    support::endian::write32<support::little>(Buf, Reloc.Value);
    break;
  default:
    llvm_unreachable("unknown relocation type");
  }
}

static void applyRelocations(uint8_t *Buf, ArrayRef<OutputRelocation> Relocs) {
  if (!Relocs.size())
    return;
  DEBUG(dbgs() << "applyRelocations: count=" << Relocs.size() << "\n");
  for (const OutputRelocation &Reloc : Relocs)
    applyRelocation(Buf, Reloc);
}

void InputChunk::writeTo(uint8_t *SectionStart) const {
  memcpy(SectionStart + getOutputOffset(), data().data(), data().size());
  applyRelocations(SectionStart, OutRelocations);
}

// Populate OutRelocations based on the input relocations and offset within the
// output section.  Calculates the updated index and offset for each relocation
// as well as the value to write out in the final binary.
void InputChunk::calcRelocations() {
  if (Relocations.empty())
    return;
  int32_t Off = getOutputOffset() - getInputSectionOffset();
  DEBUG(dbgs() << "calcRelocations: " << File->getName()
               << " offset=" << Twine(Off) << "\n");
  for (const WasmRelocation &Reloc : Relocations) {
    OutputRelocation NewReloc;
    NewReloc.Reloc = Reloc;
    assert(Reloc.Offset + Off > 0);
    NewReloc.Reloc.Offset += Off;
    DEBUG(dbgs() << "reloc: type=" << Reloc.Type << " index=" << Reloc.Index
                 << " offset=" << Reloc.Offset
                 << " newOffset=" << NewReloc.Reloc.Offset << "\n");

    if (Config->Relocatable)
      NewReloc.NewIndex = File->calcNewIndex(Reloc);

    NewReloc.Value = File->calcNewValue(Reloc);
    OutRelocations.emplace_back(NewReloc);
  }
}

void InputFunction::setOutputIndex(uint32_t Index) {
  DEBUG(dbgs() << "InputFunction::setOutputIndex: " << Index << "\n");
  assert(!hasOutputIndex());
  OutputIndex = Index;
}

void InputFunction::setTableIndex(uint32_t Index) {
  DEBUG(dbgs() << "InputFunction::setTableIndex " << Index << "\n");
  assert(!hasTableIndex());
  TableIndex = Index;
}
