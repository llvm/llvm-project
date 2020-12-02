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

namespace lld {
StringRef relocTypeToString(uint8_t relocType) {
  switch (relocType) {
#define WASM_RELOC(NAME, REL)                                                  \
  case REL:                                                                    \
    return #NAME;
#include "llvm/BinaryFormat/WasmRelocs.def"
#undef WASM_RELOC
  }
  llvm_unreachable("unknown reloc type");
}

bool relocIs64(uint8_t relocType) {
  switch (relocType) {
  case R_WASM_MEMORY_ADDR_LEB64:
  case R_WASM_MEMORY_ADDR_SLEB64:
  case R_WASM_MEMORY_ADDR_REL_SLEB64:
  case R_WASM_MEMORY_ADDR_I64:
    return true;
  default:
    return false;
  }
}

std::string toString(const wasm::InputChunk *c) {
  return (toString(c->file) + ":(" + c->getName() + ")").str();
}

namespace wasm {
StringRef InputChunk::getComdatName() const {
  uint32_t index = getComdat();
  if (index == UINT32_MAX)
    return StringRef();
  return file->getWasmObj()->linkingData().Comdats[index];
}

void InputChunk::verifyRelocTargets() const {
  for (const WasmRelocation &rel : relocations) {
    uint64_t existingValue;
    unsigned bytesRead = 0;
    unsigned paddedLEBWidth = 5;
    auto offset = rel.Offset - getInputSectionOffset();
    const uint8_t *loc = data().data() + offset;
    switch (rel.Type) {
    case R_WASM_TYPE_INDEX_LEB:
    case R_WASM_FUNCTION_INDEX_LEB:
    case R_WASM_GLOBAL_INDEX_LEB:
    case R_WASM_EVENT_INDEX_LEB:
    case R_WASM_MEMORY_ADDR_LEB:
      existingValue = decodeULEB128(loc, &bytesRead);
      break;
    case R_WASM_MEMORY_ADDR_LEB64:
      existingValue = decodeULEB128(loc, &bytesRead);
      paddedLEBWidth = 10;
      break;
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_REL_SLEB:
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_REL_SLEB:
    case R_WASM_MEMORY_ADDR_TLS_SLEB:
      existingValue = static_cast<uint64_t>(decodeSLEB128(loc, &bytesRead));
      break;
    case R_WASM_TABLE_INDEX_SLEB64:
    case R_WASM_MEMORY_ADDR_SLEB64:
    case R_WASM_MEMORY_ADDR_REL_SLEB64:
      existingValue = static_cast<uint64_t>(decodeSLEB128(loc, &bytesRead));
      paddedLEBWidth = 10;
      break;
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_MEMORY_ADDR_I32:
    case R_WASM_FUNCTION_OFFSET_I32:
    case R_WASM_SECTION_OFFSET_I32:
    case R_WASM_GLOBAL_INDEX_I32:
      existingValue = read32le(loc);
      break;
    case R_WASM_TABLE_INDEX_I64:
    case R_WASM_MEMORY_ADDR_I64:
    case R_WASM_FUNCTION_OFFSET_I64:
      existingValue = read64le(loc);
      break;
    default:
      llvm_unreachable("unknown relocation type");
    }

    if (bytesRead && bytesRead != paddedLEBWidth)
      warn("expected LEB at relocation site be 5/10-byte padded");

    if (rel.Type != R_WASM_GLOBAL_INDEX_LEB &&
        rel.Type != R_WASM_GLOBAL_INDEX_I32) {
      auto expectedValue = file->calcExpectedValue(rel);
      if (expectedValue != existingValue)
        warn(toString(this) + ": unexpected existing value for " +
             relocTypeToString(rel.Type) + ": existing=" +
             Twine(existingValue) + " expected=" + Twine(expectedValue));
    }
  }
}

// Copy this input chunk to an mmap'ed output file and apply relocations.
void InputChunk::writeTo(uint8_t *buf) const {
  // Copy contents
  memcpy(buf + outputOffset, data().data(), data().size());

  // Apply relocations
  if (relocations.empty())
    return;

#ifndef NDEBUG
  verifyRelocTargets();
#endif

  LLVM_DEBUG(dbgs() << "applying relocations: " << toString(this)
                    << " count=" << relocations.size() << "\n");
  int32_t off = outputOffset - getInputSectionOffset();
  auto tombstone = getTombstone();

  for (const WasmRelocation &rel : relocations) {
    uint8_t *loc = buf + rel.Offset + off;
    auto value = file->calcNewValue(rel, tombstone);
    LLVM_DEBUG(dbgs() << "apply reloc: type=" << relocTypeToString(rel.Type));
    if (rel.Type != R_WASM_TYPE_INDEX_LEB)
      LLVM_DEBUG(dbgs() << " sym=" << file->getSymbols()[rel.Index]->getName());
    LLVM_DEBUG(dbgs() << " addend=" << rel.Addend << " index=" << rel.Index
                      << " value=" << value << " offset=" << rel.Offset
                      << "\n");

    switch (rel.Type) {
    case R_WASM_TYPE_INDEX_LEB:
    case R_WASM_FUNCTION_INDEX_LEB:
    case R_WASM_GLOBAL_INDEX_LEB:
    case R_WASM_EVENT_INDEX_LEB:
    case R_WASM_MEMORY_ADDR_LEB:
      encodeULEB128(value, loc, 5);
      break;
    case R_WASM_MEMORY_ADDR_LEB64:
      encodeULEB128(value, loc, 10);
      break;
    case R_WASM_TABLE_INDEX_SLEB:
    case R_WASM_TABLE_INDEX_REL_SLEB:
    case R_WASM_MEMORY_ADDR_SLEB:
    case R_WASM_MEMORY_ADDR_REL_SLEB:
    case R_WASM_MEMORY_ADDR_TLS_SLEB:
      encodeSLEB128(static_cast<int32_t>(value), loc, 5);
      break;
    case R_WASM_TABLE_INDEX_SLEB64:
    case R_WASM_MEMORY_ADDR_SLEB64:
    case R_WASM_MEMORY_ADDR_REL_SLEB64:
      encodeSLEB128(static_cast<int64_t>(value), loc, 10);
      break;
    case R_WASM_TABLE_INDEX_I32:
    case R_WASM_MEMORY_ADDR_I32:
    case R_WASM_FUNCTION_OFFSET_I32:
    case R_WASM_SECTION_OFFSET_I32:
    case R_WASM_GLOBAL_INDEX_I32:
      write32le(loc, value);
      break;
    case R_WASM_TABLE_INDEX_I64:
    case R_WASM_MEMORY_ADDR_I64:
    case R_WASM_FUNCTION_OFFSET_I64:
      write64le(loc, value);
      break;
    default:
      llvm_unreachable("unknown relocation type");
    }
  }
}

// Copy relocation entries to a given output stream.
// This function is used only when a user passes "-r". For a regular link,
// we consume relocations instead of copying them to an output file.
void InputChunk::writeRelocations(raw_ostream &os) const {
  if (relocations.empty())
    return;

  int32_t off = outputOffset - getInputSectionOffset();
  LLVM_DEBUG(dbgs() << "writeRelocations: " << file->getName()
                    << " offset=" << Twine(off) << "\n");

  for (const WasmRelocation &rel : relocations) {
    writeUleb128(os, rel.Type, "reloc type");
    writeUleb128(os, rel.Offset + off, "reloc offset");
    writeUleb128(os, file->calcNewIndex(rel), "reloc index");

    if (relocTypeHasAddend(rel.Type))
      writeSleb128(os, file->calcNewAddend(rel), "reloc addend");
  }
}

void InputFunction::setFunctionIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "InputFunction::setFunctionIndex: " << getName()
                    << " -> " << index << "\n");
  assert(!hasFunctionIndex());
  functionIndex = index;
}

void InputFunction::setTableIndex(uint32_t index) {
  LLVM_DEBUG(dbgs() << "InputFunction::setTableIndex: " << getName() << " -> "
                    << index << "\n");
  assert(!hasTableIndex());
  tableIndex = index;
}

// Write a relocation value without padding and return the number of bytes
// witten.
static unsigned writeCompressedReloc(uint8_t *buf, const WasmRelocation &rel,
                                     uint64_t value) {
  switch (rel.Type) {
  case R_WASM_TYPE_INDEX_LEB:
  case R_WASM_FUNCTION_INDEX_LEB:
  case R_WASM_GLOBAL_INDEX_LEB:
  case R_WASM_EVENT_INDEX_LEB:
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_MEMORY_ADDR_LEB64:
    return encodeULEB128(value, buf);
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_TABLE_INDEX_SLEB64:
  case R_WASM_MEMORY_ADDR_SLEB:
  case R_WASM_MEMORY_ADDR_SLEB64:
    return encodeSLEB128(static_cast<int64_t>(value), buf);
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

static unsigned getRelocWidthPadded(const WasmRelocation &rel) {
  switch (rel.Type) {
  case R_WASM_TYPE_INDEX_LEB:
  case R_WASM_FUNCTION_INDEX_LEB:
  case R_WASM_GLOBAL_INDEX_LEB:
  case R_WASM_EVENT_INDEX_LEB:
  case R_WASM_MEMORY_ADDR_LEB:
  case R_WASM_TABLE_INDEX_SLEB:
  case R_WASM_MEMORY_ADDR_SLEB:
    return 5;
  case R_WASM_TABLE_INDEX_SLEB64:
  case R_WASM_MEMORY_ADDR_LEB64:
  case R_WASM_MEMORY_ADDR_SLEB64:
    return 10;
  default:
    llvm_unreachable("unexpected relocation type");
  }
}

static unsigned getRelocWidth(const WasmRelocation &rel, uint64_t value) {
  uint8_t buf[10];
  return writeCompressedReloc(buf, rel, value);
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
  if (!file || !config->compressRelocations)
    return;

  LLVM_DEBUG(dbgs() << "calculateSize: " << getName() << "\n");

  const uint8_t *secStart = file->codeSection->Content.data();
  const uint8_t *funcStart = secStart + getInputSectionOffset();
  uint32_t functionSizeLength;
  decodeULEB128(funcStart, &functionSizeLength);

  uint32_t start = getInputSectionOffset();
  uint32_t end = start + function->Size;

  auto tombstone = getTombstone();

  uint32_t lastRelocEnd = start + functionSizeLength;
  for (const WasmRelocation &rel : relocations) {
    LLVM_DEBUG(dbgs() << "  region: " << (rel.Offset - lastRelocEnd) << "\n");
    compressedFuncSize += rel.Offset - lastRelocEnd;
    compressedFuncSize += getRelocWidth(rel, file->calcNewValue(rel, tombstone));
    lastRelocEnd = rel.Offset + getRelocWidthPadded(rel);
  }
  LLVM_DEBUG(dbgs() << "  final region: " << (end - lastRelocEnd) << "\n");
  compressedFuncSize += end - lastRelocEnd;

  // Now we know how long the resulting function is we can add the encoding
  // of its length
  uint8_t buf[5];
  compressedSize = compressedFuncSize + encodeULEB128(compressedFuncSize, buf);

  LLVM_DEBUG(dbgs() << "  calculateSize orig: " << function->Size << "\n");
  LLVM_DEBUG(dbgs() << "  calculateSize  new: " << compressedSize << "\n");
}

// Override the default writeTo method so that we can (optionally) write the
// compressed version of the function.
void InputFunction::writeTo(uint8_t *buf) const {
  if (!file || !config->compressRelocations)
    return InputChunk::writeTo(buf);

  buf += outputOffset;
  uint8_t *orig = buf;
  (void)orig;

  const uint8_t *secStart = file->codeSection->Content.data();
  const uint8_t *funcStart = secStart + getInputSectionOffset();
  const uint8_t *end = funcStart + function->Size;
  auto tombstone = getTombstone();
  uint32_t count;
  decodeULEB128(funcStart, &count);
  funcStart += count;

  LLVM_DEBUG(dbgs() << "write func: " << getName() << "\n");
  buf += encodeULEB128(compressedFuncSize, buf);
  const uint8_t *lastRelocEnd = funcStart;
  for (const WasmRelocation &rel : relocations) {
    unsigned chunkSize = (secStart + rel.Offset) - lastRelocEnd;
    LLVM_DEBUG(dbgs() << "  write chunk: " << chunkSize << "\n");
    memcpy(buf, lastRelocEnd, chunkSize);
    buf += chunkSize;
    buf += writeCompressedReloc(buf, rel, file->calcNewValue(rel, tombstone));
    lastRelocEnd = secStart + rel.Offset + getRelocWidthPadded(rel);
  }

  unsigned chunkSize = end - lastRelocEnd;
  LLVM_DEBUG(dbgs() << "  write final chunk: " << chunkSize << "\n");
  memcpy(buf, lastRelocEnd, chunkSize);
  LLVM_DEBUG(dbgs() << "  total: " << (buf + chunkSize - orig) << "\n");
}

// Generate code to apply relocations to the data section at runtime.
// This is only called when generating shared libaries (PIC) where address are
// not known at static link time.
void InputSegment::generateRelocationCode(raw_ostream &os) const {
  LLVM_DEBUG(dbgs() << "generating runtime relocations: " << getName()
                    << " count=" << relocations.size() << "\n");

  unsigned opcode_ptr_const = config->is64.getValueOr(false)
                                  ? WASM_OPCODE_I64_CONST
                                  : WASM_OPCODE_I32_CONST;
  unsigned opcode_ptr_add = config->is64.getValueOr(false)
                                ? WASM_OPCODE_I64_ADD
                                : WASM_OPCODE_I32_ADD;

  auto tombstone = getTombstone();
  // TODO(sbc): Encode the relocations in the data section and write a loop
  // here to apply them.
  uint64_t segmentVA = outputSeg->startVA + outputSegmentOffset;
  for (const WasmRelocation &rel : relocations) {
    uint64_t offset = rel.Offset - getInputSectionOffset();
    uint64_t outputOffset = segmentVA + offset;

    LLVM_DEBUG(dbgs() << "gen reloc: type=" << relocTypeToString(rel.Type)
                      << " addend=" << rel.Addend << " index=" << rel.Index
                      << " output offset=" << outputOffset << "\n");

    // Get __memory_base
    writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
    writeUleb128(os, WasmSym::memoryBase->getGlobalIndex(), "memory_base");

    // Add the offset of the relocation
    writeU8(os, opcode_ptr_const, "CONST");
    writeSleb128(os, outputOffset, "offset");
    writeU8(os, opcode_ptr_add, "ADD");

    bool is64 = relocIs64(rel.Type);
    unsigned opcode_reloc_const =
        is64 ? WASM_OPCODE_I64_CONST : WASM_OPCODE_I32_CONST;
    unsigned opcode_reloc_add =
        is64 ? WASM_OPCODE_I64_ADD : WASM_OPCODE_I32_ADD;
    unsigned opcode_reloc_store =
        is64 ? WASM_OPCODE_I64_STORE : WASM_OPCODE_I32_STORE;

    Symbol *sym = file->getSymbol(rel);
    // Now figure out what we want to store
    if (sym->hasGOTIndex()) {
      writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(os, sym->getGOTIndex(), "global index");
      if (rel.Addend) {
        writeU8(os, opcode_reloc_const, "CONST");
        writeSleb128(os, rel.Addend, "addend");
        writeU8(os, opcode_reloc_add, "ADD");
      }
    } else {
      const GlobalSymbol* baseSymbol = WasmSym::memoryBase;
      if (rel.Type == R_WASM_TABLE_INDEX_I32 ||
          rel.Type == R_WASM_TABLE_INDEX_I64)
        baseSymbol = WasmSym::tableBase;
      writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
      writeUleb128(os, baseSymbol->getGlobalIndex(), "base");
      writeU8(os, opcode_reloc_const, "CONST");
      writeSleb128(os, file->calcNewValue(rel, tombstone), "offset");
      writeU8(os, opcode_reloc_add, "ADD");
    }

    // Store that value at the virtual address
    writeU8(os, opcode_reloc_store, "I32_STORE");
    writeUleb128(os, 2, "align");
    writeUleb128(os, 0, "offset");
  }
}

uint64_t InputSection::getTombstoneForSection(StringRef name) {
  // When a function is not live we need to update relocations referring to it.
  // If they occur in DWARF debug symbols, we want to change the pc of the
  // function to -1 to avoid overlapping with a valid range. However for the
  // debug_ranges and debug_loc sections that would conflict with the existing
  // meaning of -1 so we use -2.
  // Returning 0 means there is no tombstone value for this section, and relocation
  // will just use the addend.
  if (!name.startswith(".debug_"))
    return 0;
  if (name.equals(".debug_ranges") || name.equals(".debug_loc"))
    return UINT64_C(-2);
  return UINT64_C(-1);
}

} // namespace wasm
} // namespace lld
