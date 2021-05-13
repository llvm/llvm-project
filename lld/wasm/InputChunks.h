//===- InputChunks.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An InputChunks represents an indivisible opaque region of a input wasm file.
// i.e. a single wasm data segment or a single wasm function.
//
// They are written directly to the mmap'd output file after which relocations
// are applied.  Because each Chunk is independent they can be written in
// parallel.
//
// Chunks are also unit on which garbage collection (--gc-sections) operates.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_CHUNKS_H
#define LLD_WASM_INPUT_CHUNKS_H

#include "Config.h"
#include "InputFiles.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

class ObjFile;
class OutputSegment;
class OutputSection;

class InputChunk {
public:
  enum Kind {
    DataSegment,
    Merge,
    MergedSegment,
    Function,
    SyntheticFunction,
    Section
  };

  Kind kind() const { return sectionKind; }

  virtual uint32_t getSize() const { return data().size(); }
  virtual uint32_t getInputSize() const { return getSize(); };

  virtual void writeTo(uint8_t *buf) const;
  void relocate(uint8_t *buf) const;

  ArrayRef<WasmRelocation> getRelocations() const { return relocations; }
  void setRelocations(ArrayRef<WasmRelocation> rs) { relocations = rs; }

  uint64_t getOffset(uint64_t offset) const { return outSecOff + offset; }
  virtual StringRef getName() const = 0;
  virtual StringRef getDebugName() const = 0;
  virtual uint32_t getComdat() const = 0;
  StringRef getComdatName() const;
  virtual uint32_t getInputSectionOffset() const = 0;

  size_t getNumRelocations() const { return relocations.size(); }
  void writeRelocations(llvm::raw_ostream &os) const;

  ObjFile *file;
  OutputSection *outputSec = nullptr;

  // After assignAddresses is called, this represents the offset from
  // the beginning of the output section this chunk was assigned to.
  int32_t outSecOff = 0;

  // Signals that the section is part of the output.  The garbage collector,
  // and COMDAT handling can set a sections' Live bit.
  // If GC is disabled, all sections start out as live by default.
  unsigned live : 1;

  // Signals the chunk was discarded by COMDAT handling.
  unsigned discarded : 1;

protected:
  InputChunk(ObjFile *f, Kind k)
      : file(f), live(!config->gcSections), discarded(false), sectionKind(k) {}
  virtual ~InputChunk() = default;
  virtual ArrayRef<uint8_t> data() const = 0;
  virtual uint64_t getTombstone() const { return 0; }

  // Verifies the existing data at relocation targets matches our expectations.
  // This is performed only debug builds as an extra sanity check.
  void verifyRelocTargets() const;

  ArrayRef<WasmRelocation> relocations;
  Kind sectionKind;
};

// Represents a WebAssembly data segment which can be included as part of
// an output data segments.  Note that in WebAssembly, unlike ELF and other
// formats, used the term "data segment" to refer to the continuous regions of
// memory that make on the data section. See:
// https://webassembly.github.io/spec/syntax/modules.html#syntax-data
//
// For example, by default, clang will produce a separate data section for
// each global variable.
class InputSegment : public InputChunk {
public:
  InputSegment(const WasmSegment *seg, ObjFile *f)
      : InputChunk(f, InputChunk::DataSegment), segment(seg) {
    alignment = segment->Data.Alignment;
    flags = segment->Data.LinkingFlags;
  }

  InputSegment(uint32_t alignment, uint32_t flags)
      : InputChunk(nullptr, InputChunk::DataSegment), alignment(alignment),
        flags(flags) {}

  static bool classof(const InputChunk *c) {
    return c->kind() == DataSegment || c->kind() == Merge ||
           c->kind() == MergedSegment;
  }

  void generateRelocationCode(raw_ostream &os) const;

  StringRef getName() const override { return segment->Data.Name; }
  StringRef getDebugName() const override { return StringRef(); }
  uint32_t getComdat() const override { return segment->Data.Comdat; }
  uint32_t getInputSectionOffset() const override {
    return segment->SectionOffset;
  }

  // Translate an offset in the input section to an offset in the output
  // section.
  uint64_t getOffset(uint64_t offset) const;

  uint64_t getVA(uint64_t offset = 0) const;

  bool isTLS() const {
    // Older object files don't include WASM_SEG_FLAG_TLS and instead
    // relied on the naming convention.
    return flags & llvm::wasm::WASM_SEG_FLAG_TLS ||
           getName().startswith(".tdata") || getName().startswith(".tbss");
  }

  const OutputSegment *outputSeg = nullptr;
  uint32_t outputSegmentOffset = 0;
  uint32_t alignment = 0;
  uint32_t flags = 0;

protected:
  ArrayRef<uint8_t> data() const override { return segment->Data.Content; }

  const WasmSegment *segment = nullptr;
};

class SyntheticMergedDataSegment;

// Merge segment handling copied from lld/ELF/InputSection.h.  Keep in sync
// where possible.

// SegmentPiece represents a piece of splittable segment contents.
// We allocate a lot of these and binary search on them. This means that they
// have to be as compact as possible, which is why we don't store the size (can
// be found by looking at the next one).
struct SegmentPiece {
  SegmentPiece(size_t off, uint32_t hash, bool live)
      : inputOff(off), live(live || !config->gcSections), hash(hash >> 1) {}

  uint32_t inputOff;
  uint32_t live : 1;
  uint32_t hash : 31;
  uint64_t outputOff = 0;
};

static_assert(sizeof(SegmentPiece) == 16, "SectionPiece is too big");

// This corresponds segments marked as WASM_SEG_FLAG_STRINGS.
class MergeInputSegment : public InputSegment {
public:
  MergeInputSegment(const WasmSegment *seg, ObjFile *f) : InputSegment(seg, f) {
    sectionKind = Merge;
  }

  static bool classof(const InputChunk *s) { return s->kind() == Merge; }
  void splitIntoPieces();

  // Translate an offset in the input section to an offset in the parent
  // MergeSyntheticSection.
  uint64_t getParentOffset(uint64_t offset) const;

  // Splittable sections are handled as a sequence of data
  // rather than a single large blob of data.
  std::vector<SegmentPiece> pieces;

  // Returns I'th piece's data. This function is very hot when
  // string merging is enabled, so we want to inline.
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  llvm::CachedHashStringRef getData(size_t i) const {
    size_t begin = pieces[i].inputOff;
    size_t end =
        (pieces.size() - 1 == i) ? data().size() : pieces[i + 1].inputOff;
    return {toStringRef(data().slice(begin, end - begin)), pieces[i].hash};
  }

  // Returns the SectionPiece at a given input section offset.
  SegmentPiece *getSegmentPiece(uint64_t offset);
  const SegmentPiece *getSegmentPiece(uint64_t offset) const {
    return const_cast<MergeInputSegment *>(this)->getSegmentPiece(offset);
  }

  SyntheticMergedDataSegment *parent = nullptr;

private:
  void splitStrings(ArrayRef<uint8_t> a);
};

// SyntheticMergedDataSegment is a class that allows us to put mergeable
// sections with different attributes in a single output sections. To do that we
// put them into SyntheticMergedDataSegment synthetic input sections which are
// attached to regular output sections.
class SyntheticMergedDataSegment : public InputSegment {
public:
  SyntheticMergedDataSegment(StringRef name, uint32_t alignment, uint32_t flags)
      : InputSegment(alignment, flags), name(name),
        builder(llvm::StringTableBuilder::RAW, 1ULL << alignment) {
    sectionKind = InputChunk::MergedSegment;
  }

  static bool classof(const InputChunk *c) {
    return c->kind() == InputChunk::MergedSegment;
  }

  uint32_t getSize() const override;

  StringRef getName() const override { return name; }

  uint32_t getComdat() const override { return segments[0]->getComdat(); }

  void writeTo(uint8_t *buf) const override;

  void addMergeSegment(MergeInputSegment *ms) {
    ms->parent = this;
    segments.push_back(ms);
  }

  void finalizeContents();

protected:
  std::vector<MergeInputSegment *> segments;
  StringRef name;
  llvm::StringTableBuilder builder;
};

// Represents a single wasm function within and input file.  These are
// combined to create the final output CODE section.
class InputFunction : public InputChunk {
public:
  InputFunction(const WasmSignature &s, const WasmFunction *func, ObjFile *f)
      : InputChunk(f, InputChunk::Function), signature(s), function(func),
        exportName(func && func->ExportName.hasValue()
                       ? (*func->ExportName).str()
                       : llvm::Optional<std::string>()) {}

  static bool classof(const InputChunk *c) {
    return c->kind() == InputChunk::Function ||
           c->kind() == InputChunk::SyntheticFunction;
  }

  void writeTo(uint8_t *buf) const override;
  StringRef getName() const override { return function->SymbolName; }
  StringRef getDebugName() const override { return function->DebugName; }
  llvm::Optional<StringRef> getExportName() const {
    return exportName.hasValue() ? llvm::Optional<StringRef>(*exportName)
                                 : llvm::Optional<StringRef>();
  }
  void setExportName(std::string exportName) { this->exportName = exportName; }
  uint32_t getComdat() const override { return function->Comdat; }
  uint32_t getFunctionInputOffset() const { return getInputSectionOffset(); }
  uint32_t getFunctionCodeOffset() const { return function->CodeOffset; }
  uint32_t getSize() const override {
    if (config->compressRelocations && file) {
      assert(compressedSize);
      return compressedSize;
    }
    return data().size();
  }
  uint32_t getInputSize() const override { return function->Size; }
  uint32_t getFunctionIndex() const { return functionIndex.getValue(); }
  bool hasFunctionIndex() const { return functionIndex.hasValue(); }
  void setFunctionIndex(uint32_t index);
  uint32_t getInputSectionOffset() const override {
    return function->CodeSectionOffset;
  }
  uint32_t getTableIndex() const { return tableIndex.getValue(); }
  bool hasTableIndex() const { return tableIndex.hasValue(); }
  void setTableIndex(uint32_t index);

  // The size of a given input function can depend on the values of the
  // LEB relocations within it.  This finalizeContents method is called after
  // all the symbol values have be calculated but before getSize() is ever
  // called.
  void calculateSize();

  const WasmSignature &signature;

protected:
  ArrayRef<uint8_t> data() const override {
    assert(!config->compressRelocations);
    return file->codeSection->Content.slice(getInputSectionOffset(),
                                            function->Size);
  }

  const WasmFunction *function;
  llvm::Optional<std::string> exportName;
  llvm::Optional<uint32_t> functionIndex;
  llvm::Optional<uint32_t> tableIndex;
  uint32_t compressedFuncSize = 0;
  uint32_t compressedSize = 0;
};

class SyntheticFunction : public InputFunction {
public:
  SyntheticFunction(const WasmSignature &s, StringRef name,
                    StringRef debugName = {})
      : InputFunction(s, nullptr, nullptr), name(name), debugName(debugName) {
    sectionKind = InputChunk::SyntheticFunction;
  }

  static bool classof(const InputChunk *c) {
    return c->kind() == InputChunk::SyntheticFunction;
  }

  StringRef getName() const override { return name; }
  StringRef getDebugName() const override { return debugName; }
  uint32_t getComdat() const override { return UINT32_MAX; }

  void setBody(ArrayRef<uint8_t> body_) { body = body_; }

protected:
  ArrayRef<uint8_t> data() const override { return body; }

  StringRef name;
  StringRef debugName;
  ArrayRef<uint8_t> body;
};

// Represents a single Wasm Section within an input file.
class InputSection : public InputChunk {
public:
  InputSection(const WasmSection &s, ObjFile *f)
      : InputChunk(f, InputChunk::Section), section(s), tombstoneValue(getTombstoneForSection(s.Name)) {
    assert(section.Type == llvm::wasm::WASM_SEC_CUSTOM);
  }

  StringRef getName() const override { return section.Name; }
  StringRef getDebugName() const override { return StringRef(); }
  uint32_t getComdat() const override { return section.Comdat; }

protected:
  ArrayRef<uint8_t> data() const override { return section.Content; }

  // Offset within the input section.  This is only zero since this chunk
  // type represents an entire input section, not part of one.
  uint32_t getInputSectionOffset() const override { return 0; }
  uint64_t getTombstone() const override { return tombstoneValue; }
  static uint64_t getTombstoneForSection(StringRef name);

  const WasmSection &section;
  const uint64_t tombstoneValue;
};

} // namespace wasm

std::string toString(const wasm::InputChunk *);
StringRef relocTypeToString(uint8_t relocType);

} // namespace lld

#endif // LLD_WASM_INPUT_CHUNKS_H
