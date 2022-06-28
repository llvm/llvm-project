//===-- Fuzzer.cpp - Fuzz the pseudoparser --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Token.h"
#include "clang-pseudo/cli/CLI.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace clang {
namespace pseudo {
namespace {

class Fuzzer {
  clang::LangOptions LangOpts = clang::pseudo::genericLangOpts();
  bool Print;

public:
  Fuzzer(bool Print) : Print(Print) {}

  void operator()(llvm::StringRef Code) {
    std::string CodeStr = Code.str(); // Must be null-terminated.
    auto RawStream = lex(CodeStr, LangOpts);
    auto DirectiveStructure = DirectiveTree::parse(RawStream);
    clang::pseudo::chooseConditionalBranches(DirectiveStructure, RawStream);
    // FIXME: strip preprocessor directives
    auto ParseableStream =
        clang::pseudo::stripComments(cook(RawStream, LangOpts));

    clang::pseudo::ForestArena Arena;
    clang::pseudo::GSS GSS;
    const Language &Lang = getLanguageFromFlags();
    auto &Root =
        glrParse(ParseableStream,
                 clang::pseudo::ParseParams{Lang.G, Lang.Table, Arena, GSS},
                 *Lang.G.findNonterminal("translation-unit"));
    if (Print)
      llvm::outs() << Root.dumpRecursive(Lang.G);
  }
};

Fuzzer *Fuzz = nullptr;

} // namespace
} // namespace pseudo
} // namespace clang

extern "C" {

// Set up the fuzzer from command line flags:
//  -print                     - used for testing the fuzzer
int LLVMFuzzerInitialize(int *Argc, char ***Argv) {
  bool PrintForest = false;
  auto ConsumeArg = [&](llvm::StringRef Arg) -> bool {
    if (Arg == "-print") {
      PrintForest = true;
      return true;
    }
    return false;
  };
  *Argc = std::remove_if(*Argv + 1, *Argv + *Argc, ConsumeArg) - *Argv;

  clang::pseudo::Fuzz = new clang::pseudo::Fuzzer(PrintForest);
  return 0;
}

int LLVMFuzzerTestOneInput(uint8_t *Data, size_t Size) {
  (*clang::pseudo::Fuzz)(llvm::StringRef(reinterpret_cast<char *>(Data), Size));
  return 0;
}
}
