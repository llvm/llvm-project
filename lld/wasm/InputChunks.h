//===- InputChunks.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Object/Wasm.h"

using llvm::object::WasmSegment;
using llvm::wasm::WasmFunction;
using llvm::wasm::WasmRelocation;
using llvm::wasm::WasmSignature;
using llvm::object::WasmSection;

namespace llvm {
class raw_ostream;
}

namespace lld {
namespace wasm {

class ObjFile;
class OutputSegment;

class InputChunk {
public:
  enum Kind { DataSegment, Function };

  Kind kind() const { return SectionKind; }

  uint32_t getSize() const { return data().size(); }

  void copyRelocations(const WasmSection &Section);

  void writeTo(uint8_t *SectionStart) const;

  ArrayRef<WasmRelocation> getRelocations() const { return Relocations; }

  virtual StringRef getComdat() const = 0;
  virtual StringRef getName() const = 0;

  size_t NumRelocations() const { return Relocations.size(); }
  void writeRelocations(llvm::raw_ostream &OS) const;

  ObjFile *File;
  int32_t OutputOffset = 0;

  // Signals that the section is part of the output.  The garbage collector,
  // and COMDAT handling can set a sections' Live bit.
  // If GC is disabled, all sections start out as live by default.
  unsigned Live : 1;

protected:
  InputChunk(ObjFile *F, Kind K)
      : File(F), Live(!Config->GcSections), SectionKind(K) {}
  virtual ~InputChunk() = default;
  virtual ArrayRef<uint8_t> data() const = 0;
  virtual uint32_t getInputSectionOffset() const = 0;

  std::vector<WasmRelocation> Relocations;
  Kind SectionKind;
};

// Represents a WebAssembly data segment which can be included as part of
// an output data segments.  Note that in WebAssembly, unlike ELF and other
// formats, used the term "data segment" to refer to the continous regions of
// memory that make on the data section. See:
// https://webassembly.github.io/spec/syntax/modules.html#syntax-data
//
// For example, by default, clang will produce a separate data section for
// each global variable.
class InputSegment : public InputChunk {
public:
  InputSegment(const WasmSegment &Seg, ObjFile *F)
      : InputChunk(F, InputChunk::DataSegment), Segment(Seg) {}

  static bool classof(const InputChunk *C) { return C->kind() == DataSegment; }

  uint32_t getAlignment() const { return Segment.Data.Alignment; }
  StringRef getName() const override { return Segment.Data.Name; }
  StringRef getComdat() const override { return Segment.Data.Comdat; }

  const OutputSegment *OutputSeg = nullptr;
  int32_t OutputSegmentOffset = 0;

protected:
  ArrayRef<uint8_t> data() const override { return Segment.Data.Content; }
  uint32_t getInputSectionOffset() const override {
    return Segment.SectionOffset;
  }

  const WasmSegment &Segment;
};

// Represents a single wasm function within and input file.  These are
// combined to create the final output CODE section.
class InputFunction : public InputChunk {
public:
  InputFunction(const WasmSignature &S, const WasmFunction *Func,
                ObjFile *F)
      : InputChunk(F, InputChunk::Function), Signature(S), Function(Func) {}

  static bool classof(const InputChunk *C) {
    return C->kind() == InputChunk::Function;
  }

  StringRef getName() const override { return Function->Name; }
  StringRef getComdat() const override { return Function->Comdat; }
  uint32_t getOutputIndex() const { return OutputIndex.getValue(); }
  bool hasOutputIndex() const { return OutputIndex.hasValue(); }
  void setOutputIndex(uint32_t Index);
  uint32_t getTableIndex() const { return TableIndex.getValue(); }
  bool hasTableIndex() const { return TableIndex.hasValue(); }
  void setTableIndex(uint32_t Index);

  const WasmSignature &Signature;

protected:
  ArrayRef<uint8_t> data() const override {
    return File->CodeSection->Content.slice(getInputSectionOffset(),
                                            Function->Size);
  }
  uint32_t getInputSectionOffset() const override {
    return Function->CodeSectionOffset;
  }

  const WasmFunction *Function;
  llvm::Optional<uint32_t> OutputIndex;
  llvm::Optional<uint32_t> TableIndex;
};

class SyntheticFunction : public InputFunction {
public:
  SyntheticFunction(const WasmSignature &S, ArrayRef<uint8_t> Body,
                    StringRef Name)
      : InputFunction(S, nullptr, nullptr), Name(Name), Body(Body) {}

  StringRef getName() const override { return Name; }

protected:
  ArrayRef<uint8_t> data() const override { return Body; }

  StringRef Name;
  ArrayRef<uint8_t> Body;
};

} // namespace wasm

std::string toString(const wasm::InputChunk *);
} // namespace lld

#endif // LLD_WASM_INPUT_CHUNKS_H
