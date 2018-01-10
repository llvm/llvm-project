//===- InpuFunction.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represents a WebAssembly function in an input file which could also be
// assigned a function index in the output.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_FUNCTION_H
#define LLD_WASM_INPUT_FUNCTION_H

#include "WriterUtils.h"
#include "llvm/Object/Wasm.h"

using llvm::wasm::WasmRelocation;
using llvm::wasm::WasmFunction;

namespace lld {
namespace wasm {

class ObjFile;

class InputFunction {
public:
  InputFunction(const WasmSignature &S, const WasmFunction &Func,
                const ObjFile &F)
      : Signature(S), Function(Func), File(F) {}

  uint32_t getOutputIndex() const { return OutputIndex.getValue(); };
  bool hasOutputIndex() const { return OutputIndex.hasValue(); };

  void setOutputIndex(uint32_t Index) {
    assert(!hasOutputIndex());
    OutputIndex = Index;
  };

  const WasmSignature &Signature;
  const WasmFunction &Function;
  int32_t OutputOffset = 0;
  std::vector<WasmRelocation> Relocations;
  std::vector<OutputRelocation> OutRelocations;
  const ObjFile &File;

protected:
  llvm::Optional<uint32_t> OutputIndex;
};

} // namespace wasm
} // namespace lld

#endif // LLD_WASM_INPUT_FUNCTION_H
