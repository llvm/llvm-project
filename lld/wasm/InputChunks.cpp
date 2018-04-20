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
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "llvm/Support/LEB128.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::wasm;

StringRef ReloctTypeToString(uint8_t RelocType) {
  switch (RelocType) {
#define WASM_RELOC(NAME, REL) case REL: return #NAME;
#include "llvm/BinaryFormat/WasmRelocs.def"
#undef WASM_RELOC
  }
  llvm_unreachable("unknown reloc type");
}

std::string lld::toString(const InputChunk *C) {
  return (toString(C->File) + ":(" + C->getName() + ")").str();
}

StringRef InputChunk::getComdatName() const {
  uint32_t Index = getComdat();
  if (Index == UINT32_MAX)
    return StringRef();
  return File->getWasmObj()->linkingData().Comdats[Index];
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

// Copy this input chunk to an mmap'ed output file and apply relocations.
void InputChunk::writeTo(uint8_t *Buf) const {
  // Copy contents
  memcpy(Buf + OutputOffset, data().data(), data().size());

  // Apply relocations
  if (Relocations.empty())
    return;

  DEBUG(dbgs() << "applying relocations: " << getName()
               << " count=" << Relocations.size() << "\n");
  int32_t Off = OutputOffset - getInputSectionOffset();

  for (const WasmRelocation &Rel : Relocations) {
    uint8_t *Loc = Buf + Rel.Offset + Off;
    uint32_t Value = File->calcNewValue(Rel);
    uint32_t ExistingValue;
    DEBUG(dbgs() << "apply reloc: type=" << ReloctTypeToString(Rel.Type)
                 << " addend=" << Rel.Addend << " index=" << Rel.Index
                 << " value=" << Value << " offset=" << Rel.Offset << "\n");

    switch (Rel.Type) {
    case R_WEBASSEMBLY_TYPE_INDEX_LEB:
    case R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    case R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
    case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
      ExistingValue = decodeULEB128(Loc);
      encodeULEB128(Value, Loc, 5);
      break;
    case R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
      ExistingValue = static_cast<uint32_t>(decodeSLEB128(Loc));
      encodeSLEB128(static_cast<int32_t>(Value), Loc, 5);
      break;
    case R_WEBASSEMBLY_TABLE_INDEX_I32:
    case R_WEBASSEMBLY_MEMORY_ADDR_I32:
      ExistingValue = static_cast<uint32_t>(read32le(Loc));
      write32le(Loc, Value);
      break;
    default:
      llvm_unreachable("unknown relocation type");
    }

    uint32_t ExpectedValue = File->calcExpectedValue(Rel);
    if (ExpectedValue != ExistingValue)
      error("unexpected existing value for " + ReloctTypeToString(Rel.Type) +
            ": existing=" + Twine(ExistingValue) +
            " expected=" + Twine(ExpectedValue));
  }
}

// Copy relocation entries to a given output stream.
// This function is used only when a user passes "-r". For a regular link,
// we consume relocations instead of copying them to an output file.
void InputChunk::writeRelocations(raw_ostream &OS) const {
  if (Relocations.empty())
    return;

  int32_t Off = OutputOffset - getInputSectionOffset();
  DEBUG(dbgs() << "writeRelocations: " << File->getName()
               << " offset=" << Twine(Off) << "\n");

  for (const WasmRelocation &Rel : Relocations) {
    writeUleb128(OS, Rel.Type, "reloc type");
    writeUleb128(OS, Rel.Offset + Off, "reloc offset");
    writeUleb128(OS, File->calcNewIndex(Rel), "reloc index");

    switch (Rel.Type) {
    case R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    case R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
    case R_WEBASSEMBLY_MEMORY_ADDR_I32:
      writeSleb128(OS, Rel.Addend, "reloc addend");
      break;
    }
  }
}

void InputFunction::setFunctionIndex(uint32_t Index) {
  DEBUG(dbgs() << "InputFunction::setFunctionIndex: " << getName() << " -> "
               << Index << "\n");
  assert(!hasFunctionIndex());
  FunctionIndex = Index;
}

void InputFunction::setTableIndex(uint32_t Index) {
  DEBUG(dbgs() << "InputFunction::setTableIndex: " << getName() << " -> "
               << Index << "\n");
  assert(!hasTableIndex());
  TableIndex = Index;
}
