//===-- ClangPseudo.cpp - Clang pseudoparser tool -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Disambiguate.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Language.h"
#include "clang-pseudo/Token.h"
#include "clang-pseudo/cli/CLI.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRGraph.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

using clang::pseudo::ForestNode;
using clang::pseudo::Token;
using clang::pseudo::TokenStream;
using llvm::cl::desc;
using llvm::cl::init;
using llvm::cl::opt;

static opt<bool> PrintGrammar("print-grammar", desc("Print the grammar"));
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
static opt<bool> Disambiguate("disambiguate",
                              desc("Choose best tree from parse forest"));
static opt<bool> PrintStatistics("print-statistics", desc("Print GLR parser statistics"));
static opt<bool> PrintForest("print-forest", desc("Print parse forest"));
static opt<bool> ForestAbbrev("forest-abbrev", desc("Abbreviate parse forest"),
                              init(true));
static opt<std::string> HTMLForest("html-forest",
                                   desc("output file for HTML forest"));
static opt<std::string> StartSymbol("start-symbol",
                                    desc("Specify the start symbol to parse"),
                                    init("translation-unit"));

static std::string readOrDie(llvm::StringRef Path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFile(Path);
  if (std::error_code EC = Text.getError()) {
    llvm::errs() << "Error: can't read file '" << Path
                 << "': " << EC.message() << "\n";
    ::exit(1);
  }
  return Text.get()->getBuffer().str();
}

namespace clang {
namespace pseudo {
// Defined in HTMLForest.cpp
void writeHTMLForest(llvm::raw_ostream &OS, const Grammar &,
                     const ForestNode &Root, const Disambiguation &,
                     const TokenStream &);
namespace {

struct NodeStats {
  unsigned Total = 0;
  std::vector<std::pair<SymbolID, unsigned>> BySymbol;

  NodeStats(const ForestNode &Root,
            llvm::function_ref<bool(const ForestNode &)> Filter) {
    llvm::DenseMap<SymbolID, unsigned> Map;
    for (const ForestNode &N : Root.descendants())
      if (Filter(N)) {
        ++Total;
        ++Map[N.symbol()];
      }
    BySymbol = {Map.begin(), Map.end()};
    // Sort by count descending, then symbol ascending.
    llvm::sort(BySymbol, [](const auto &L, const auto &R) {
      return std::tie(R.second, L.first) < std::tie(L.second, R.first);
    });
  }
};

} // namespace
} // namespace pseudo
} // namespace clang

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

  const auto &Lang = clang::pseudo::getLanguageFromFlags();
  if (PrintGrammar)
    llvm::outs() << Lang.G.dump();
  if (PrintGraph)
    llvm::outs() << clang::pseudo::LRGraph::buildLR0(Lang.G).dumpForTests(
        Lang.G);

  if (PrintTable)
    llvm::outs() << Lang.Table.dumpForTests(Lang.G);
  if (PrintStatistics)
    llvm::outs() << Lang.Table.dumpStatistics();

  if (ParseableStream) {
    clang::pseudo::ForestArena Arena;
    clang::pseudo::GSS GSS;
    llvm::Optional<clang::pseudo::SymbolID> StartSymID =
        Lang.G.findNonterminal(StartSymbol);
    if (!StartSymID) {
      llvm::errs() << llvm::formatv(
          "The start symbol {0} doesn't exit in the grammar!\n", StartSymbol);
      return 2;
    }
    auto &Root =
        glrParse(clang::pseudo::ParseParams{*ParseableStream, Arena, GSS},
                 *StartSymID, Lang);
    // If we're disambiguating, we'll print at the end instead.
    if (PrintForest && !Disambiguate)
      llvm::outs() << Root.dumpRecursive(Lang.G, /*Abbreviated=*/ForestAbbrev);
    clang::pseudo::Disambiguation Disambig;
    if (Disambiguate)
      Disambig = clang::pseudo::disambiguate(&Root, {});

    if (HTMLForest.getNumOccurrences()) {
      std::error_code EC;
      llvm::raw_fd_ostream HTMLOut(HTMLForest, EC);
      if (EC) {
        llvm::errs() << "Couldn't write " << HTMLForest << ": " << EC.message()
                     << "\n";
        return 2;
      }
      clang::pseudo::writeHTMLForest(HTMLOut, Lang.G, Root, Disambig,
                                     *ParseableStream);
    }

    if (PrintStatistics) {
      llvm::outs() << "Forest bytes: " << Arena.bytes()
                   << " nodes: " << Arena.nodeCount() << "\n";
      llvm::outs() << "GSS bytes: " << GSS.bytes()
                   << " nodes: " << GSS.nodesCreated() << "\n";

      for (auto &P : {std::make_pair("Ambiguous", ForestNode::Ambiguous),
                      std::make_pair("Opaque", ForestNode::Opaque)}) {
        clang::pseudo::NodeStats Stats(
            Root, [&](const auto &N) { return N.kind() == P.second; });
        llvm::outs() << "\n" << Stats.Total << " " << P.first << " nodes:\n";
        for (const auto &S : Stats.BySymbol)
          llvm::outs() << llvm::formatv("  {0,3} {1}\n", S.second,
                                        Lang.G.symbolName(S.first));
      }

      // Metrics for how imprecise parsing was.
      // These are rough but aim to be:
      // - linear: if we eliminate half the errors the metric should halve
      // - length-independent
      unsigned UnparsedTokens = 0; // Tokens covered by Opaque. (not unique)
      unsigned Misparses = 0;      // Sum of alternatives-1
      llvm::DenseSet<const ForestNode *> Visited;
      auto DFS = [&](const ForestNode &N, Token::Index End, auto &DFS) -> void {
        if (N.kind() == ForestNode::Opaque) {
          UnparsedTokens += End - N.startTokenIndex();
        } else if (N.kind() == ForestNode::Ambiguous) {
          Misparses += N.alternatives().size() - 1;
          for (const auto *C : N.alternatives())
            if (Visited.insert(C).second)
              DFS(*C, End, DFS);
        } else if (N.kind() == ForestNode::Sequence) {
          for (unsigned I = 0, E = N.children().size(); I < E; ++I)
            if (Visited.insert(N.children()[I]).second)
              DFS(*N.children()[I],
                  I + 1 == N.children().size()
                      ? End
                      : N.children()[I + 1]->startTokenIndex(),
                  DFS);
        }
      };
      unsigned Len = ParseableStream->tokens().size();
      DFS(Root, Len, DFS);
      llvm::outs() << "\n";
      llvm::outs() << llvm::formatv("Ambiguity: {0} misparses/token\n",
                                    double(Misparses) / Len);
      llvm::outs() << llvm::formatv("Unparsed: {0}%\n",
                                    100.0 * UnparsedTokens / Len);
    }

    if (Disambiguate && PrintForest) {
      ForestNode *DisambigRoot = &Root;
      removeAmbiguities(DisambigRoot, Disambig);
      llvm::outs() << "Disambiguated tree:\n";
      llvm::outs() << DisambigRoot->dumpRecursive(Lang.G,
                                                  /*Abbreviated=*/ForestAbbrev);
    }
  }

  return 0;
}
