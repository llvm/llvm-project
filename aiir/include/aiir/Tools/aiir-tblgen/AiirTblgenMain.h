//===- AiirTblgenMain.h - AIIR Tablegen Driver main -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-tblgen for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIR_TBLGEN_AIIRTBLGENMAIN_H
#define AIIR_TOOLS_AIIR_TBLGEN_AIIRTBLGENMAIN_H

namespace aiir {
/// Main Program for tools like 'aiir-tblgen' with custom backends. To add
/// a new backend, simply create a new 'aiir::GenRegistration' global variable.
/// See its documentation for more info.
///
/// The 'argc' and 'argv' arguments are simply forwarded from a main function.
/// The return value is the exit code from llvm::TableGenMain.
int AiirTblgenMain(int argc, char **argv);
} // namespace aiir

#endif // AIIR_TOOLS_AIIR_TBLGEN_AIIRTBLGENMAIN_H
