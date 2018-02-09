//===- InputChunks.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// An input chunk represents an indivisible blocks of code or data from an input
// file.  i.e. a single wasm data segment or a single wasm function.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_CHUNKS_H
#define LLD_WASM_INPUT_CHUNKS_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

using llvm::object::WasmSegment;
using llvm::wasm::WasmFunction;
using llvm::wasm::WasmRelocation;
using llvm::wasm::WasmSignature;
using llvm::object::WasmSection;

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

  void setOutputOffset(uint32_t Offset) {
    OutputOffset = Offset;
    calcRelocations();
  }

  uint32_t getOutputOffset() const { return OutputOffset; }
  ArrayRef<WasmRelocation> getRelocations() const { return Relocations; }
  StringRef getFileName() const { return File->getName(); }

  virtual StringRef getComdat() const = 0;
  virtual StringRef getName() const = 0;

  bool Discarded = false;
  std::vector<OutputRelocation> OutRelocations;
  ObjFile *File;

  // The garbage collector sets sections' Live bits.
  // If GC is disabled, all sections are considered live by default.
  unsigned Live : 1;

protected:
  InputChunk(ObjFile *F, Kind K)
      : File(F), Live(!Config->GcSections), SectionKind(K) {}
  virtual ~InputChunk() = default;
  void calcRelocations();
  virtual ArrayRef<uint8_t> data() const = 0;
  virtual uint32_t getInputSectionOffset() const = 0;

  std::vector<WasmRelocation> Relocations;
  int32_t OutputOffset = 0;
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

  // Translate an offset in the input segment to an offset in the output
  // segment.
  uint32_t translateVA(uint32_t Address) const;

  const OutputSegment *getOutputSegment() const { return OutputSeg; }

  void setOutputSegment(const OutputSegment *Segment, uint32_t Offset) {
    OutputSeg = Segment;
    OutputSegmentOffset = Offset;
  }

  uint32_t getAlignment() const { return Segment.Data.Alignment; }
  uint32_t startVA() const { return Segment.Data.Offset.Value.Int32; }
  uint32_t endVA() const { return startVA() + getSize(); }
  StringRef getName() const override { return Segment.Data.Name; }
  StringRef getComdat() const override { return Segment.Data.Comdat; }

  int32_t OutputSegmentOffset = 0;

protected:
  ArrayRef<uint8_t> data() const override { return Segment.Data.Content; }
  uint32_t getInputSectionOffset() const override {
    return Segment.SectionOffset;
  }

  const WasmSegment &Segment;
  const OutputSegment *OutputSeg = nullptr;
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
} // namespace lld

#endif // LLD_WASM_INPUT_CHUNKS_H
