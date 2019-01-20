//===- InputEvent.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wasm events are features that suspend the current execution and transfer the
// control flow to a corresponding handler. Currently the only supported event
// kind is exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_INPUT_EVENT_H
#define LLD_WASM_INPUT_EVENT_H

#include "Config.h"
#include "InputFiles.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/Wasm.h"

namespace lld {
namespace wasm {

// Represents a single Wasm Event within an input file. These are combined to
// form the final EVENTS section.
class InputEvent {
public:
  InputEvent(const WasmSignature &S, const WasmEvent &E, ObjFile *F)
      : File(F), Event(E), Signature(S), Live(!Config->GcSections) {}

  StringRef getName() const { return Event.SymbolName; }
  const WasmEventType &getType() const { return Event.Type; }

  uint32_t getEventIndex() const { return EventIndex.getValue(); }
  bool hasEventIndex() const { return EventIndex.hasValue(); }
  void setEventIndex(uint32_t Index) {
    assert(!hasEventIndex());
    EventIndex = Index;
  }

  ObjFile *File;
  WasmEvent Event;
  const WasmSignature &Signature;

  bool Live = false;

protected:
  llvm::Optional<uint32_t> EventIndex;
};

} // namespace wasm

inline std::string toString(const wasm::InputEvent *E) {
  return (toString(E->File) + ":(" + E->getName() + ")").str();
}

} // namespace lld

#endif // LLD_WASM_INPUT_EVENT_H
