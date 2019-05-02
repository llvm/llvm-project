//===- InputChunks.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

StringRef lld::relocTypeToString(uint8_t RelocType) {
  switch (RelocType) {
#define WASM_RELOC(NAME, REL)                                                  \
  case REL:                                                                    \
    return #NAME;
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

void InputChunk::verifyRelocTargets() const {
  for (const WasmRelocation &Rel : Relocations) {
    uint32_t ExistingValue;
    unsigned BytesRead = 0;
    uint32_t Offset = Rel.Offset - getInputSectionOffset();
    const uint8_t *Loc = data().data() + Offset;
    switch (Rel.Type) {
    case R_WASM_TYPE_INDEX_LEB:
    case R_WASM_FUNCTION_INDEX_LEB:
    case R_WASM_GLOBAL_INDEX_LEB:
    case R_WASM_EVENT_INDEX_LEB:
    case R_WASM_MEMORY_ADDR_LEB:
      ExistingValue = decodeULEB128(Loc, &BytesRead);
      break;
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_REL_SLEB:
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_REL_SLEB:
      ExistingValue = static_cast<uint32_t>(decodeSLEB128(Loc, &BytesRead));
      break;
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_MEMORY_ADDR_I32:
    case R_WASM_FUNCTION_OFFSET_I32:
    case R_WASM_SECTION_OFFSET_I32:
      ExistingValue = static_cast<uint32_t>(read32le(Loc));
      break;
    default:
      llvm_unreachable("unknown relocation type");
    }

    if (BytesRead && BytesRead != 5)
      warn("expected LEB at relocation site be 5-byte padded");

    if (Rel.Type != R_WASM_GLOBAL_INDEX_LEB) {
      uint32_t ExpectedValue = File->calcExpectedValue(Rel);
      if (ExpectedValue != ExistingValue)
        warn("unexpected existing value for " + relocTypeToString(Rel.Type) +
             ": existing=" + Twine(ExistingValue) +
             " expected=" + Twine(ExpectedValue));
    }
  }
}

// Copy this input chunk to an mmap'ed output file and apply relocations.
void InputChunk::writeTo(uint8_t *Buf) const {
  // Copy contents
  memcpy(Buf + OutputOffset, data().data(), data().size());

  // Apply relocations
  if (Relocations.empty())
    return;

#ifndef NDEBUG
  verifyRelocTargets();
#endif

  LLVM_DEBUG(dbgs() << "applying relocations: " << getName()
                    << " count=" << Relocations.size() << "\n");
  int32_t Off = OutputOffset - getInputSectionOffset();

  for (const WasmRelocation &Rel : Relocations) {
    uint8_t *Loc = Buf + Rel.Offset + Off;
    uint32_t Value = File->calcNewValue(Rel);
    LLVM_DEBUG(dbgs() << "apply reloc: type=" << relocTypeToString(Rel.Type)
                      << " addend=" << Rel.Addend << " index=" << Rel.Index
                      << " value=" << Value << " offset=" << Rel.Offset
                      << "\n");

    switch (Rel.Type) {
    case R_WASM_TYPE_INDEX_LEB:
    case R_WASM_FUNCTION_INDEX_LEB:
    case R_WASM_GLOBAL_INDEX_LEB:
    case R_WASM_EVENT_INDEX_LEB:
    case R_WASM_MEMORY_ADDR_LEB:
      encodeULEB128(Value, Loc, 5);
      break;
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_REL_SLEB:
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_REL_SLEB:
      encodeSLEB128(static_cast<int32_t>(Value), Loc, 5);
      break;
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_MEMORY_ADDR_I32:
    case R_WASM_FUNCTION_OFFSET_I32:
    case R_WASM_SECTION_OFFSET_I32:
      write32le(Loc, Value);
      break;
    default:
      llvm_unreachable("unknown relocation type");
    }
  }
}

// Copy relocation entries to a given output stream.
// This function is used only when a user passes "-r". For a regular link,
// we consume relocations instead of copying them to an output file.
void InputChunk::writeRelocations(raw_ostream &OS) const {
  if (Relocations.empty())
    return;

  int32_t Off = OutputOffset - getInputSectionOffset();
  LLVM_DEBUG(dbgs() << "writeRelocations: " << File->getName()
                    << " offset=" << Twine(Off) << "\n");

  for (const WasmRelocation &Rel : Relocations) {
    writeUleb128(OS, Rel.Type, "reloc type");
    writeUleb128(OS, Rel.Offset + Off, "reloc offset");
    writeUleb128(OS, File->calcNewIndex(Rel), "reloc index");

    switch (Rel.Type) {
    case R_WASM_MEMORY_ADDR_LEB:
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_I32:
    case R_WASM_FUNCTION_OFFSET_I32:
    case R_WASM_SECTION_OFFSET_I32:
      writeSleb128(OS, File->calcNewAddend(Rel), "reloc addend");
      break;
    }
  }
}

void InputFunction::setFunctionIndex(uint32_t Index) {
  LLVM_DEBUG(dbgs() << "InputFunction::setFunctionIndex: " << getName()
                    << " -> " << Index << "\n");
  assert(!hasFunctionIndex());
  FunctionIndex = Index;
}

void InputFunction::setTableIndex(uint32_t Index) {
  LLVM_DEBUG(dbgs() << "InputFunction::setTableIndex: " << getName() << " -> "
                    << Index << "\n");
  assert(!hasTableIndex());
  TableIndex = Index;
}

// Write a relocation value without padding and return the number of bytes
// witten.
static unsigned writeCompressedReloc(uint8_t *Buf, const WasmRelocation &Rel,
                                     uint32_t Value) {
  switch (Rel.Type) {
  case R_WASM_TYPE_INDEX_LEB:
  case R_WASM_FUNCTION_INDEX_LEB:
  case R_WASM_GLOBAL_INDEX_LEB:
  case R_WASM_EVENT_INDEX_LEB:
  case R_WASM_MEMORY_ADDR_LEB:
    return encodeULEB128(Value, Buf);
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_MEMORY_ADDR_SLEB:
    return encodeSLEB128(static_cast<int32_t>(Value), Buf);
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

static unsigned getRelocWidthPadded(const WasmRelocation &Rel) {
  switch (Rel.Type) {
  case R_WASM_TYPE_INDEX_LEB:
  case R_WASM_FUNCTION_INDEX_LEB:
  case R_WASM_GLOBAL_INDEX_LEB:
  case R_WASM_EVENT_INDEX_LEB:
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_MEMORY_ADDR_SLEB:
    return 5;
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

static unsigned getRelocWidth(const WasmRelocation &Rel, uint32_t Value) {
  uint8_t Buf[5];
  return writeCompressedReloc(Buf, Rel, Value);
}

// Relocations of type LEB and SLEB in the code section are padded to 5 bytes
// so that a fast linker can blindly overwrite them without needing to worry
// about the number of bytes needed to encode the values.
// However, for optimal output the code section can be compressed to remove
// the padding then outputting non-relocatable files.
// In this case we need to perform a size calculation based on the value at each
// relocation.  At best we end up saving 4 bytes for each relocation entry.
//
// This function only computes the final output size.  It must be called
// before getSize() is used to calculate of layout of the code section.
void InputFunction::calculateSize() {
  if (!File || !Config->CompressRelocations)
    return;

  LLVM_DEBUG(dbgs() << "calculateSize: " << getName() << "\n");

  const uint8_t *SecStart = File->CodeSection->Content.data();
  const uint8_t *FuncStart = SecStart + getInputSectionOffset();
  uint32_t FunctionSizeLength;
  decodeULEB128(FuncStart, &FunctionSizeLength);

  uint32_t Start = getInputSectionOffset();
  uint32_t End = Start + Function->Size;

  uint32_t LastRelocEnd = Start + FunctionSizeLength;
  for (const WasmRelocation &Rel : Relocations) {
    LLVM_DEBUG(dbgs() << "  region: " << (Rel.Offset - LastRelocEnd) << "\n");
    CompressedFuncSize += Rel.Offset - LastRelocEnd;
    CompressedFuncSize += getRelocWidth(Rel, File->calcNewValue(Rel));
    LastRelocEnd = Rel.Offset + getRelocWidthPadded(Rel);
  }
  LLVM_DEBUG(dbgs() << "  final region: " << (End - LastRelocEnd) << "\n");
  CompressedFuncSize += End - LastRelocEnd;

  // Now we know how long the resulting function is we can add the encoding
  // of its length
  uint8_t Buf[5];
  CompressedSize = CompressedFuncSize + encodeULEB128(CompressedFuncSize, Buf);

  LLVM_DEBUG(dbgs() << "  calculateSize orig: " << Function->Size << "\n");
  LLVM_DEBUG(dbgs() << "  calculateSize  new: " << CompressedSize << "\n");
}

// Override the default writeTo method so that we can (optionally) write the
// compressed version of the function.
void InputFunction::writeTo(uint8_t *Buf) const {
  if (!File || !Config->CompressRelocations)
    return InputChunk::writeTo(Buf);

  Buf += OutputOffset;
  uint8_t *Orig = Buf;
  (void)Orig;

  const uint8_t *SecStart = File->CodeSection->Content.data();
  const uint8_t *FuncStart = SecStart + getInputSectionOffset();
  const uint8_t *End = FuncStart + Function->Size;
  uint32_t Count;
  decodeULEB128(FuncStart, &Count);
  FuncStart += Count;

  LLVM_DEBUG(dbgs() << "write func: " << getName() << "\n");
  Buf += encodeULEB128(CompressedFuncSize, Buf);
  const uint8_t *LastRelocEnd = FuncStart;
  for (const WasmRelocation &Rel : Relocations) {
    unsigned ChunkSize = (SecStart + Rel.Offset) - LastRelocEnd;
    LLVM_DEBUG(dbgs() << "  write chunk: " << ChunkSize << "\n");
    memcpy(Buf, LastRelocEnd, ChunkSize);
    Buf += ChunkSize;
    Buf += writeCompressedReloc(Buf, Rel, File->calcNewValue(Rel));
    LastRelocEnd = SecStart + Rel.Offset + getRelocWidthPadded(Rel);
  }

  unsigned ChunkSize = End - LastRelocEnd;
  LLVM_DEBUG(dbgs() << "  write final chunk: " << ChunkSize << "\n");
  memcpy(Buf, LastRelocEnd, ChunkSize);
  LLVM_DEBUG(dbgs() << "  total: " << (Buf + ChunkSize - Orig) << "\n");
}

// Generate code to apply relocations to the data section at runtime.
// This is only called when generating shared libaries (PIC) where address are
// not known at static link time.
void InputSegment::generateRelocationCode(raw_ostream &OS) const {
  LLVM_DEBUG(dbgs() << "generating runtime relocations: " << getName()
                    << " count=" << Relocations.size() << "\n");

  // TODO(sbc): Encode the relocations in the data section and write a loop
  // here to apply them.
  uint32_t SegmentVA = OutputSeg->StartVA + OutputSegmentOffset;
  for (const WasmRelocation &Rel : Relocations) {
    uint32_t Offset = Rel.Offset - getInputSectionOffset();
    uint32_t OutputOffset = SegmentVA + Offset;

    LLVM_DEBUG(dbgs() << "gen reloc: type=" << relocTypeToString(Rel.Type)
                      << " addend=" << Rel.Addend << " index=" << Rel.Index
                      << " output offset=" << OutputOffset << "\n");

    // Get __memory_base
    writeU8(OS, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
    writeUleb128(OS, WasmSym::MemoryBase->getGlobalIndex(), "memory_base");

    // Add the offset of the relocation
    writeU8(OS, WASM_OPCODE_I32_CONST, "I32_CONST");
    writeSleb128(OS, OutputOffset, "offset");
    writeU8(OS, WASM_OPCODE_I32_ADD, "ADD");

    Symbol *Sym = File->getSymbol(Rel);
    // Now figure out what we want to store
    if (Sym->hasGOTIndex()) {
      writeU8(OS, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(OS, Sym->getGOTIndex(), "global index");
      if (Rel.Addend) {
        writeU8(OS, WASM_OPCODE_I32_CONST, "CONST");
        writeSleb128(OS, Rel.Addend, "addend");
        writeU8(OS, WASM_OPCODE_I32_ADD, "ADD");
      }
    } else {
      const GlobalSymbol* BaseSymbol = WasmSym::MemoryBase;
      if (Rel.Type == R_WASM_TABLE_INDEX_I32)
        BaseSymbol = WasmSym::TableBase;
      writeU8(OS, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(OS, BaseSymbol->getGlobalIndex(), "base");
      writeU8(OS, WASM_OPCODE_I32_CONST, "CONST");
      writeSleb128(OS, File->calcNewValue(Rel), "offset");
      writeU8(OS, WASM_OPCODE_I32_ADD, "ADD");
    }

    // Store that value at the virtual address
    writeU8(OS, WASM_OPCODE_I32_STORE, "I32_STORE");
    writeUleb128(OS, 2, "align");
    writeUleb128(OS, 0, "offset");
  }
}
