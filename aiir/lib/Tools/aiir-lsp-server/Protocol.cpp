//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the AIIR specific LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"

//===----------------------------------------------------------------------===//
// AIIRConvertBytecodeParams
//===----------------------------------------------------------------------===//

bool llvm::lsp::fromJSON(const llvm::json::Value &value,
                         AIIRConvertBytecodeParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("uri", result.uri);
}

//===----------------------------------------------------------------------===//
// AIIRConvertBytecodeResult
//===----------------------------------------------------------------------===//

llvm::json::Value llvm::lsp::toJSON(const AIIRConvertBytecodeResult &value) {
  return llvm::json::Object{{"output", value.output}};
}
