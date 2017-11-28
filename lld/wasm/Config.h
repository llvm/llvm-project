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
  bool AllowUndefined;
  bool Demangle;
  bool EmitRelocs;
  bool ImportMemory;
  bool Relocatable;
  bool StripAll;
  bool StripDebug;
  uint32_t GlobalBase;
  uint32_t InitialMemory;
  uint32_t MaxMemory;
  uint32_t ZStackSize;
  llvm::StringRef Entry;
  llvm::StringRef OutputFile;
  llvm::StringRef Sysroot;

  std::set<llvm::StringRef> AllowUndefinedSymbols;
  std::vector<llvm::StringRef> SearchPaths;
  std::vector<std::pair<Symbol *, WasmGlobal>> SyntheticGlobals;
};

// The only instance of Configuration struct.
extern Configuration *Config;

} // namespace wasm
} // namespace lld

#endif
