//===- Config.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_CONFIG_H
#define LLD_WASM_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Wasm.h"

#include "Symbols.h"

using llvm::wasm::WasmGlobal;

#include <set>

namespace lld {
namespace wasm {

struct Configuration {
  bool AllowUndefined = false;
  bool Demangle = true;
  bool EmitRelocs = false;
  bool ImportMemory = false;
  bool Relocatable = false;
  bool StripDebug = false;
  bool StripAll = false;
  uint32_t ZStackSize = 0;
  uint32_t MaxMemory = 0;
  uint32_t GlobalBase = 0;
  uint32_t InitialMemory = 0;
  llvm::StringRef Entry;
  llvm::StringRef Sysroot;
  llvm::StringRef OutputFile;

  std::vector<llvm::StringRef> SearchPaths;
  std::set<llvm::StringRef> AllowUndefinedSymbols;
  std::vector<std::pair<Symbol *, WasmGlobal>> SyntheticGlobals;
};

// The only instance of Configuration struct.
extern Configuration *Config;

} // namespace wasm
} // namespace lld

#endif
