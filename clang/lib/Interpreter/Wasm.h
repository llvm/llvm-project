//===------------------ Wasm.h - Wasm Interpreter ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements interpreter support for code execution in WebAssembly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_WASM_H
#define LLVM_CLANG_LIB_INTERPRETER_WASM_H

#ifndef __EMSCRIPTEN__
#error "This requires emscripten."
#endif // __EMSCRIPTEN__

#include "clang/Interpreter/IncrementalExecutor.h"
#include "llvm/ADT/SmallString.h"

namespace clang {

class WasmIncrementalExecutor : public IncrementalExecutor {
public:
  WasmIncrementalExecutor(llvm::Error &Err);
  ~WasmIncrementalExecutor() override;

  llvm::Error addModule(PartialTranslationUnit &PTU) override;
  llvm::Error removeModule(PartialTranslationUnit &PTU) override;
  llvm::Error runCtors() const override;
  llvm::Error cleanUp() override;
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef Name,
                   SymbolNameKind NameKind) const override;
  llvm::Error LoadDynamicLibrary(const char *name) override;

private:
  llvm::SmallString<256> TempDir;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_WASM_H
