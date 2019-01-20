//===- InputGlobal.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_GLOBAL_H
#define LLD_WASM_INPUT_GLOBAL_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

// Represents a single Wasm Global Variable within an input file. These are
// combined to form the final GLOBALS section.
class InputGlobal {
public:
  InputGlobal(const WasmGlobal &G, ObjFile *F)
      : File(F), Global(G), Live(!Config->GcSections) {}

  StringRef getName() const { return Global.SymbolName; }
  const WasmGlobalType &getType() const { return Global.Type; }

  uint32_t getGlobalIndex() const { return GlobalIndex.getValue(); }
  bool hasGlobalIndex() const { return GlobalIndex.hasValue(); }
  void setGlobalIndex(uint32_t Index) {
    assert(!hasGlobalIndex());
    GlobalIndex = Index;
  }

  ObjFile *File;
  WasmGlobal Global;

  bool Live = false;

protected:
  llvm::Optional<uint32_t> GlobalIndex;
};

} // namespace wasm

inline std::string toString(const wasm::InputGlobal *G) {
  return (toString(G->File) + ":(" + G->getName() + ")").str();
}

} // namespace lld

#endif // LLD_WASM_INPUT_GLOBAL_H
