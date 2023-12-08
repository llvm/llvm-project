//===--- CLI.cpp -  ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cli/CLI.h"
#include "clang-pseudo/cxx/CXX.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

static llvm::cl::opt<std::string> Grammar(
    "grammar",
    llvm::cl::desc(
        "Specify a BNF grammar file path, or a builtin language (cxx)."),
    llvm::cl::init("cxx"));

namespace clang {
namespace pseudo {

const Language &getLanguageFromFlags() {
  if (::Grammar == "cxx")
    return cxx::getLanguage();

  static Language *Lang = []() {
    // Read from a bnf grammar file.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> GrammarText =
        llvm::MemoryBuffer::getFile(::Grammar);
    if (std::error_code EC = GrammarText.getError()) {
      llvm::errs() << "Error: can't read grammar file '" << ::Grammar
                   << "': " << EC.message() << "\n";
      std::exit(1);
    }
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(GrammarText->get()->getBuffer(), Diags);
    for (const auto &Diag : Diags)
      llvm::errs() << Diag << "\n";
    auto Table = LRTable::buildSLR(G);
    return new Language{
        std::move(G),
        std::move(Table),
        llvm::DenseMap<ExtensionID, RuleGuard>(),
        llvm::DenseMap<ExtensionID, RecoveryStrategy>(),
    };
  }();
  return *Lang;
}

} // namespace pseudo
} // namespace clang
