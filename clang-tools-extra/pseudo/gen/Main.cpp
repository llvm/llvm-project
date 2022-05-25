//===--- Main.cpp - Compile BNF grammar -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a tool to compile a BNF grammar, it is used by the build system to
// generate a necessary data bits to statically construct core pieces (Grammar,
// LRTable etc) of the LR parser.
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Grammar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>

using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;
using llvm::cl::values;

namespace {
enum EmitType {
  EmitSymbolList,
  EmitGrammarContent,
};

opt<std::string> Grammar("grammar", desc("Parse a BNF grammar file."),
                         init(""));
opt<EmitType>
    Emit(desc("which information to emit:"),
         values(clEnumValN(EmitSymbolList, "emit-symbol-list",
                           "Print nonterminal symbols (default)"),
                clEnumValN(EmitGrammarContent, "emit-grammar-content",
                           "Print the BNF grammar content as a string")));
std::string readOrDie(llvm::StringRef Path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFile(Path);
  if (std::error_code EC = Text.getError()) {
    llvm::errs() << "Error: can't read grammar file '" << Path
                 << "': " << EC.message() << "\n";
    ::exit(1);
  }
  return Text.get()->getBuffer().str();
}
} // namespace

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "");
  if (!Grammar.getNumOccurrences()) {
    llvm::errs() << "Grammar file must be provided!\n";
    return 1;
  }

  std::string GrammarText = readOrDie(Grammar);
  std::vector<std::string> Diags;
  auto G = clang::pseudo::Grammar::parseBNF(GrammarText, Diags);

  if (!Diags.empty()) {
    llvm::errs() << llvm::join(Diags, "\n");
    return 1;
  }
  switch (Emit) {

  case EmitSymbolList:
    for (clang::pseudo::SymbolID ID = 0; ID < G->table().Nonterminals.size();
         ++ID) {
      std::string Name = G->symbolName(ID).str();
      // translation-unit -> translation_unit
      std::replace(Name.begin(), Name.end(), '-', '_');
      llvm::outs() << (llvm::formatv("NONTERMINAL({0}, {1})\n", Name, ID));
    }
    break;
  case EmitGrammarContent:
    for (llvm::StringRef Line : llvm::split(GrammarText, '\n')) {
      llvm::outs() << '"';
      llvm::outs().write_escaped((Line + "\n").str());
      llvm::outs() << "\"\n";
    }
    break;
  }

  return 0;
}
