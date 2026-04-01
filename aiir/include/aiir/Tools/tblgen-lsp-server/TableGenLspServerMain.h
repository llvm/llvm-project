//===- TableGenLSPServerMain.h - TableGen Language Server main --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for tblgen-lsp-server when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H
#define AIIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace aiir {
/// Implementation for tools like `tblgen-lsp-server`.
llvm::LogicalResult TableGenLspServerMain(int argc, char **argv);
} // namespace aiir

#endif // AIIR_TOOLS_TBLGENLSPSERVER_TABLEGENLSPSERVERMAIN_H
