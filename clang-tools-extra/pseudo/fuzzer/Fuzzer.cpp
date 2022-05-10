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
#include "clang-pseudo/Grammar.h"
#include "clang-pseudo/LRTable.h"
#include "clang-pseudo/Token.h"
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
  std::unique_ptr<Grammar> G;
  LRTable T;
  bool Print;

public:
  Fuzzer(llvm::StringRef GrammarPath, bool Print) : Print(Print) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> GrammarText =
        llvm::MemoryBuffer::getFile(GrammarPath);
    if (std::error_code EC = GrammarText.getError()) {
      llvm::errs() << "Error: can't read grammar file '" << GrammarPath
                   << "': " << EC.message() << "\n";
      std::exit(1);
    }
    std::vector<std::string> Diags;
    G = Grammar::parseBNF(GrammarText->get()->getBuffer(), Diags);
    if (!Diags.empty()) {
      for (const auto &Diag : Diags)
        llvm::errs() << Diag << "\n";
      std::exit(1);
    }
    T = LRTable::buildSLR(*G);
  }

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
    auto &Root = glrParse(ParseableStream,
                          clang::pseudo::ParseParams{*G, T, Arena, GSS});
    if (Print)
      llvm::outs() << Root.dumpRecursive(*G);
  }
};

Fuzzer *Fuzz = nullptr;

} // namespace
} // namespace pseudo
} // namespace clang

extern "C" {

// Set up the fuzzer from command line flags:
//  -grammar=<file> (required) - path to cxx.bnf
//  -print                     - used for testing the fuzzer
int LLVMFuzzerInitialize(int *Argc, char ***Argv) {
  llvm::StringRef GrammarFile;
  bool PrintForest = false;
  auto ConsumeArg = [&](llvm::StringRef Arg) -> bool {
    if (Arg.consume_front("-grammar=")) {
      GrammarFile = Arg;
      return true;
    } else if (Arg == "-print") {
      PrintForest = true;
      return true;
    }
    return false;
  };
  *Argc = std::remove_if(*Argv + 1, *Argv + *Argc, ConsumeArg) - *Argv;

  if (GrammarFile.empty()) {
    fprintf(stderr, "Fuzzer needs -grammar=/path/to/cxx.bnf\n");
    exit(1);
  }
  clang::pseudo::Fuzz = new clang::pseudo::Fuzzer(GrammarFile, PrintForest);
  return 0;
}

int LLVMFuzzerTestOneInput(uint8_t *Data, size_t Size) {
  (*clang::pseudo::Fuzz)(llvm::StringRef(reinterpret_cast<char *>(Data), Size));
  return 0;
}
}
