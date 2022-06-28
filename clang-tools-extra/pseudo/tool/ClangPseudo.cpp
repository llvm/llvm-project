//===-- ClangPseudo.cpp - Clang pseudoparser tool -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Token.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRGraph.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

using clang::pseudo::Grammar;
using clang::pseudo::TokenStream;
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
static opt<bool>
    StripDirectives("strip-directives",
                    desc("Strip directives and select conditional sections"));
static opt<bool> PrintStatistics("print-statistics", desc("Print GLR parser statistics"));
static opt<bool> PrintForest("print-forest", desc("Print parse forest"));
static opt<std::string> StartSymbol("start-symbol",
                                    desc("specify the start symbol to parse"),
                                    init("translation-unit"));

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
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  clang::LangOptions LangOpts = clang::pseudo::genericLangOpts();
  std::string SourceText;
  llvm::Optional<clang::pseudo::TokenStream> RawStream;
  llvm::Optional<TokenStream> PreprocessedStream;
  llvm::Optional<clang::pseudo::TokenStream> ParseableStream;
  if (Source.getNumOccurrences()) {
    SourceText = readOrDie(Source);
    RawStream = clang::pseudo::lex(SourceText, LangOpts);
    TokenStream *Stream = RawStream.getPointer();

    auto DirectiveStructure = clang::pseudo::DirectiveTree::parse(*RawStream);
    clang::pseudo::chooseConditionalBranches(DirectiveStructure, *RawStream);

    llvm::Optional<TokenStream> Preprocessed;
    if (StripDirectives) {
      Preprocessed = DirectiveStructure.stripDirectives(*Stream);
      Stream = Preprocessed.getPointer();
    }

    if (PrintSource)
      Stream->print(llvm::outs());
    if (PrintTokens)
      llvm::outs() << *Stream;
    if (PrintDirectiveTree)
      llvm::outs() << DirectiveStructure;

    ParseableStream = clang::pseudo::stripComments(cook(*Stream, LangOpts));
    pairBrackets(*ParseableStream);
  }

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
      llvm::outs() << G.dump();
    if (PrintGraph)
      llvm::outs() << clang::pseudo::LRGraph::buildLR0(G).dumpForTests(G);
    auto LRTable = clang::pseudo::LRTable::buildSLR(G);
    if (PrintTable)
      llvm::outs() << LRTable.dumpForTests(G);
    if (PrintStatistics)
      llvm::outs() << LRTable.dumpStatistics();

    if (ParseableStream) {
      clang::pseudo::ForestArena Arena;
      clang::pseudo::GSS GSS;
      llvm::Optional<clang::pseudo::SymbolID> StartSymID =
          G.findNonterminal(StartSymbol);
      if (!StartSymID) {
        llvm::errs() << llvm::formatv(
            "The start symbol {0} doesn't exit in the grammar!\n", Grammar);
        return 2;
      }
      auto &Root = glrParse(*ParseableStream,
                            clang::pseudo::ParseParams{G, LRTable, Arena, GSS},
                            *StartSymID);
      if (PrintForest)
        llvm::outs() << Root.dumpRecursive(G, /*Abbreviated=*/true);

      if (PrintStatistics) {
        llvm::outs() << "Forest bytes: " << Arena.bytes()
                     << " nodes: " << Arena.nodeCount() << "\n";
        llvm::outs() << "GSS bytes: " << GSS.bytes()
                     << " nodes: " << GSS.nodesCreated() << "\n";
      }
    }
  }

  return 0;
}
