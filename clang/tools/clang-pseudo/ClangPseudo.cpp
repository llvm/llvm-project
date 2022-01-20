//===-- ClangPseudo.cpp - Clang pseudo parser tool ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using clang::syntax::pseudo::Grammar;
using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;

static opt<std::string>
    CheckGrammar("check-grammar", desc("Parse and check a BNF grammar file."),
                 init(""));

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "");

  if (CheckGrammar.getNumOccurrences()) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        llvm::MemoryBuffer::getFile(CheckGrammar);
    if (std::error_code EC = Text.getError()) {
      llvm::errs() << "Error: can't read grammar file '" << CheckGrammar
                   << "': " << EC.message() << "\n";
      return 1;
    }
    std::vector<std::string> Diags;
    auto RSpecs = Grammar::parseBNF(Text.get()->getBuffer(), Diags);

    if (!Diags.empty()) {
      llvm::errs() << llvm::join(Diags, "\n");
      return 2;
    }
    llvm::errs() << llvm::formatv("grammar file {0} is parsed successfully\n",
                                  CheckGrammar);
    return 0;
  }
  return 0;
}
