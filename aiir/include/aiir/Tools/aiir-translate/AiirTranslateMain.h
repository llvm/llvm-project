//===- AiirTranslateMain.h - AIIR Translation Driver main -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for aiir-translate for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIRTRANSLATE_AIIRTRANSLATEMAIN_H
#define AIIR_TOOLS_AIIRTRANSLATE_AIIRTRANSLATEMAIN_H

#include "aiir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace aiir {
/// Translate to/from an AIIR module from/to an external representation (e.g.
/// LLVM IR, SPIRV binary, ...). This is the entry point for the implementation
/// of tools like `aiir-translate`. The translation to perform is parsed from
/// the command line. The `toolName` argument is used for the header displayed
/// by `--help`.
LogicalResult aiirTranslateMain(int argc, char **argv, StringRef toolName);
} // namespace aiir

#endif // AIIR_TOOLS_AIIRTRANSLATE_AIIRTRANSLATEMAIN_H
