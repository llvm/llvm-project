//===-- ClangPseudo.cpp - Clang pseudoparser tool -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Grammar.h"
#include "clang-pseudo/LRGraph.h"
#include "clang-pseudo/LRTable.h"
#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using clang::pseudo::Grammar;
using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;

static opt<std::string>
    Grammar("grammar", desc("Parse and check a BNF grammar file."), init(""));
static opt<bool> PrintGrammar("print-grammar", desc("Print the grammar."));
static opt<bool> PrintGraph("print-graph",
                            desc("Print the LR graph for the grammar"));
static opt<bool> PrintTable("print-table",
                            desc("Print the LR table for the grammar"));
static opt<std::string> Source("source", desc("Source file"));
static opt<bool> PrintSource("print-source", desc("Print token stream"));
static opt<bool> PrintTokens("print-tokens", desc("Print detailed token info"));
static opt<bool>
    PrintDirectiveTree("print-directive-tree",
                      desc("Print directive structure of source code"));

static std::string readOrDie(llvm::StringRef Path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFile(Path);
  if (std::error_code EC = Text.getError()) {
    llvm::errs() << "Error: can't read grammar file '" << Path
                 << "': " << EC.message() << "\n";
    ::exit(1);
  }
  return Text.get()->getBuffer().str();
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "");

  if (Grammar.getNumOccurrences()) {
    std::string Text = readOrDie(Grammar);
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(Text, Diags);

    if (!Diags.empty()) {
      llvm::errs() << llvm::join(Diags, "\n");
      return 2;
    }
    llvm::outs() << llvm::formatv("grammar file {0} is parsed successfully\n",
                                  Grammar);
    if (PrintGrammar)
      llvm::outs() << G->dump();
    if (PrintGraph)
      llvm::outs() << clang::pseudo::LRGraph::buildLR0(*G).dumpForTests(*G);
    if (PrintTable)
      llvm::outs() << clang::pseudo::LRTable::buildSLR(*G).dumpForTests(*G);
    return 0;
  }

  if (Source.getNumOccurrences()) {
    std::string Text = readOrDie(Source);
    clang::LangOptions LangOpts; // FIXME: use real options.
    auto Stream = clang::pseudo::lex(Text, LangOpts);
    auto Structure = clang::pseudo::DirectiveTree::parse(Stream);
    clang::pseudo::chooseConditionalBranches(Structure, Stream);

    if (PrintDirectiveTree)
      llvm::outs() << Structure;
    if (PrintSource)
      Stream.print(llvm::outs());
    if (PrintTokens)
      llvm::outs() << Stream;
  }

  return 0;
}
