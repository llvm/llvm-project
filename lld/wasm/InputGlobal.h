//===- InputGlobal.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_GLOBAL_H
#define LLD_WASM_INPUT_GLOBAL_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

using llvm::wasm::WasmGlobal;
using llvm::wasm::WasmInitExpr;

namespace lld {
namespace wasm {

// Represents a single Wasm Global Variable within an input file. These are
// combined to form the final GLOBALS section.
class InputGlobal {
public:
  InputGlobal(const WasmGlobal &G) : Global(G) {}

  const WasmGlobalType &getType() const { return Global.Type; }

  uint32_t getOutputIndex() const { return OutputIndex.getValue(); }
  bool hasOutputIndex() const { return OutputIndex.hasValue(); }
  void setOutputIndex(uint32_t Index) {
    assert(!hasOutputIndex());
    OutputIndex = Index;
  }

  bool Live = false;

  WasmGlobal Global;

protected:
  llvm::Optional<uint32_t> OutputIndex;
};

} // namespace wasm

} // namespace lld

#endif // LLD_WASM_INPUT_GLOBAL_H
