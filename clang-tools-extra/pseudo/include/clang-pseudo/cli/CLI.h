//===--- CLI.h - Get grammar from variant sources ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides the Grammar, LRTable etc for a language specified by the `--grammar`
// flags. It is by design to be used by pseudoparser-based CLI tools.
//
// The CLI library defines a `--grammar` CLI flag, which supports 1) using a
// grammar from a file (--grammar=/path/to/lang.bnf) or using the prebuilt cxx
// language (--grammar=cxx).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_CLI_CLI_H
#define CLANG_PSEUDO_CLI_CLI_H

#include "clang-pseudo/Language.h"

namespace clang {
namespace pseudo {

// Returns the corresponding Language from the '--grammar' command-line flag.
//
// !! If the grammar flag is invalid (e.g. unexisting file), this function will
// exit the program immediately.
const Language &getLanguageFromFlags();

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_CLI_CLI_H
