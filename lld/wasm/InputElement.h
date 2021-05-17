//===- InputElement.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_ELEMENT_H
#define LLD_WASM_INPUT_ELEMENT_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/LLVM.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

// Represents a single element (Global, Event, Table, etc) within an input
// file.
class InputElement {
protected:
  InputElement(StringRef name, ObjFile *f)
      : file(f), live(!config->gcSections), name(name) {}

public:
  StringRef getName() const { return name; }
  uint32_t getAssignedIndex() const { return assignedIndex.getValue(); }
  bool hasAssignedIndex() const { return assignedIndex.hasValue(); }
  void assignIndex(uint32_t index) {
    assert(!hasAssignedIndex());
    assignedIndex = index;
  }

  ObjFile *file;
  bool live = false;

protected:
  StringRef name;
  llvm::Optional<uint32_t> assignedIndex;
};

class InputGlobal : public InputElement {
public:
  InputGlobal(const WasmGlobal &g, ObjFile *f)
      : InputElement(g.SymbolName, f), type(g.Type), initExpr(g.InitExpr) {}

  const WasmGlobalType &getType() const { return type; }
  const WasmInitExpr &getInitExpr() const { return initExpr; }

  void setPointerValue(uint64_t value) {
    if (config->is64.getValueOr(false)) {
      assert(initExpr.Opcode == llvm::wasm::WASM_OPCODE_I64_CONST);
      initExpr.Value.Int64 = value;
    } else {
      assert(initExpr.Opcode == llvm::wasm::WASM_OPCODE_I32_CONST);
      initExpr.Value.Int32 = value;
    }
  }

private:
  WasmGlobalType type;
  WasmInitExpr initExpr;
};

class InputEvent : public InputElement {
public:
  InputEvent(const WasmSignature &s, const WasmEvent &e, ObjFile *f)
      : InputElement(e.SymbolName, f), signature(s), type(e.Type) {}

  const WasmEventType &getType() const { return type; }

  const WasmSignature &signature;

private:
  WasmEventType type;
};

class InputTable : public InputElement {
public:
  InputTable(const WasmTable &t, ObjFile *f)
      : InputElement(t.SymbolName, f), type(t.Type) {}

  const WasmTableType &getType() const { return type; }
  void setLimits(const WasmLimits &limits) { type.Limits = limits; }

private:
  WasmTableType type;
};

} // namespace wasm

inline std::string toString(const wasm::InputElement *d) {
  return (toString(d->file) + ":(" + d->getName() + ")").str();
}

} // namespace lld

#endif // LLD_WASM_INPUT_ELEMENT_H
