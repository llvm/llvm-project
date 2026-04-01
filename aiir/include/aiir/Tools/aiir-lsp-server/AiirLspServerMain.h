//===- AiirLspServerMain.h - AIIR Language Server main ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-lsp-server for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPSERVERMAIN_H
#define AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPSERVERMAIN_H
#include "aiir/Tools/aiir-lsp-server/AiirLspRegistryFunction.h"

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace aiir {

/// Implementation for tools like `aiir-lsp-server`.
/// - registry should contain all the dialects that can be parsed in source IR
///   passed to the server.
llvm::LogicalResult AiirLspServerMain(int argc, char **argv,
                                      DialectRegistry &registry);

/// Implementation for tools like `aiir-lsp-server`.
/// - registry should contain all the dialects that can be parsed in source IR
///   passed to the server and may register different dialects depending on the
///   input URI.
llvm::LogicalResult AiirLspServerMain(int argc, char **argv,
                                      lsp::DialectRegistryFn registry_fn);

} // namespace aiir

#endif // AIIR_TOOLS_AIIR_LSP_SERVER_AIIRLSPSERVERMAIN_H
