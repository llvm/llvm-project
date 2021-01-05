//===- InputTable.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_TABLE_H
#define LLD_WASM_INPUT_TABLE_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

// Represents a single Wasm Table within an input file. These are combined to
// form the final TABLES section.
class InputTable {
public:
  InputTable(const WasmTable &t, ObjFile *f)
      : file(f), table(t), live(!config->gcSections) {}

  StringRef getName() const { return table.SymbolName; }
  const WasmTableType &getType() const { return table.Type; }

  // Somewhat confusingly, we generally use the term "table index" to refer to
  // the offset of a function in the well-known indirect function table.  We
  // refer to different tables instead by "table numbers".
  uint32_t getTableNumber() const { return tableNumber.getValue(); }
  bool hasTableNumber() const { return tableNumber.hasValue(); }
  void setTableNumber(uint32_t n) {
    assert(!hasTableNumber());
    tableNumber = n;
  }

  void setLimits(const WasmLimits &limits) { table.Type.Limits = limits; }

  ObjFile *file;
  WasmTable table;

  bool live = false;

protected:
  llvm::Optional<uint32_t> tableNumber;
};

} // namespace wasm

inline std::string toString(const wasm::InputTable *t) {
  return (toString(t->file) + ":(" + t->getName() + ")").str();
}

} // namespace lld

#endif // LLD_WASM_INPUT_TABLE_H
