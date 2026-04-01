//===- AiirPdllLspServerMain.h - AIIR PDLL Language Server main -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-pdll-lsp-server for when built as standalone
// binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIR_PDLL_LSP_SERVER_AIIRPDLLLSPSERVERMAIN_H
#define AIIR_TOOLS_AIIR_PDLL_LSP_SERVER_AIIRPDLLLSPSERVERMAIN_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace aiir {
/// Implementation for tools like `aiir-pdll-lsp-server`.
llvm::LogicalResult AiirPdllLspServerMain(int argc, char **argv);

} // namespace aiir

#endif // AIIR_TOOLS_AIIR_PDLL_LSP_SERVER_AIIRPDLLLSPSERVERMAIN_H
