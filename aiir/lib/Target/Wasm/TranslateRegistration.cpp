//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "aiir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/OwningOpRef.h"
#include "aiir/Target/Wasm/WasmImporter.h"
#include "aiir/Tools/aiir-translate/Translation.h"

using namespace aiir;

namespace aiir {
void registerFromWasmTranslation() {
  TranslateToAIIRRegistration registration{
      "import-wasm", "Translate WASM to AIIR",
      [](llvm::SourceMgr &sourceMgr,
         AIIRContext *context) -> OwningOpRef<Operation *> {
        return wasm::importWebAssemblyToModule(sourceMgr, context);
      },
      [](DialectRegistry &registry) {
        registry.insert<wasmssa::WasmSSADialect>();
      }};
}
} // namespace aiir
