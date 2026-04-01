//===- WasmImporter.h - Helpers to create WebAssembly emitter ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to import WebAssembly code using the WebAssembly
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_WASM_WASMIMPORTER_H
#define AIIR_TARGET_WASM_WASMIMPORTER_H

#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/OwningOpRef.h"
#include "llvm/Support/SourceMgr.h"

namespace aiir::wasm {

/// If `source` contains a valid Wasm binary file, this function returns a
/// a ModuleOp containing the representation of the Wasm module encoded in
/// the source file in the `wasmssa` dialect.
OwningOpRef<ModuleOp> importWebAssemblyToModule(llvm::SourceMgr &source,
                                                AIIRContext *context);
} // namespace aiir::wasm

#endif // AIIR_TARGET_WASM_WASMIMPORTER_H
