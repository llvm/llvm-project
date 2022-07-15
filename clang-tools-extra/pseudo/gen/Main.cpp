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

#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>

using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;
using llvm::cl::Required;
using llvm::cl::value_desc;
using llvm::cl::values;

namespace {
enum EmitType {
  EmitSymbolList,
  EmitGrammarContent,
};

opt<std::string> Grammar("grammar", desc("Parse a BNF grammar file."),
                         Required);
opt<EmitType>
    Emit(desc("which information to emit:"),
         values(clEnumValN(EmitSymbolList, "emit-symbol-list",
                           "Print nonterminal symbols (default)"),
                clEnumValN(EmitGrammarContent, "emit-grammar-content",
                           "Print the BNF grammar content as a string")));

opt<std::string> OutputFilename("o", init("-"), desc("Output"),
                                value_desc("file"));

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

  std::string GrammarText = readOrDie(Grammar);
  std::vector<std::string> Diags;
  auto G = clang::pseudo::Grammar::parseBNF(GrammarText, Diags);

  if (!Diags.empty()) {
    llvm::errs() << llvm::join(Diags, "\n");
    return 1;
  }

  std::error_code EC;
  llvm::ToolOutputFile Out{OutputFilename, EC, llvm::sys::fs::OF_None};
  if (EC) {
    llvm::errs() << EC.message() << '\n';
    return 1;
  }

  switch (Emit) {
  case EmitSymbolList:
    Out.os() << R"cpp(
#ifndef NONTERMINAL
#define NONTERMINAL(X, Y)
#endif
#ifndef RULE
#define RULE(X, Y)
#endif
#ifndef EXTENSION
#define EXTENSION(X, Y)
#endif
)cpp";
    for (clang::pseudo::SymbolID ID = 0; ID < G.table().Nonterminals.size();
         ++ID)
      Out.os() << llvm::formatv("NONTERMINAL({0}, {1})\n", G.mangleSymbol(ID),
                                ID);
    for (clang::pseudo::RuleID RID = 0; RID < G.table().Rules.size(); ++RID)
      Out.os() << llvm::formatv("RULE({0}, {1})\n", G.mangleRule(RID), RID);
    for (clang::pseudo::ExtensionID EID = 1 /*skip the sentinel 0 value*/;
         EID < G.table().AttributeValues.size(); ++EID) {
      llvm::StringRef Name = G.table().AttributeValues[EID];
      assert(!Name.empty());
      Out.os() << llvm::formatv("EXTENSION({0}, {1})\n", Name, EID);
    }
    Out.os() << R"cpp(
#undef NONTERMINAL
#undef RULE
#undef EXTENSION
)cpp";
    break;
  case EmitGrammarContent:
    for (llvm::StringRef Line : llvm::split(GrammarText, '\n')) {
      Out.os() << '"';
      Out.os().write_escaped((Line + "\n").str());
      Out.os() << "\"\n";
    }
    break;
  }

  Out.keep();

  return 0;
}
