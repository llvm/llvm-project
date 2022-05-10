//===--- Benchmark.cpp -  clang pseudoparser benchmarks ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Benchmark for the overall pseudoparser performance, it also includes other
// important pieces of the pseudoparser (grammar compliation, LR table build
// etc).
//
// Note: make sure to build the benchmark in Release mode.
//
// Usage:
//   tools/clang/tools/extra/pseudo/benchmarks/ClangPseudoBenchmark \
//      --grammar=../clang-tools-extra/pseudo/lib/cxx.bnf \
//      --source=../clang/lib/Sema/SemaDecl.cpp
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Grammar.h"
#include "clang-pseudo/LRTable.h"
#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using llvm::cl::desc;
using llvm::cl::opt;
using llvm::cl::Required;

static opt<std::string> GrammarFile("grammar",
                                    desc("Parse and check a BNF grammar file."),
                                    Required);
static opt<std::string> Source("source", desc("Source file"), Required);

namespace clang {
namespace pseudo {
namespace bench {
namespace {

const std::string *GrammarText = nullptr;
const std::string *SourceText = nullptr;
const Grammar *G = nullptr;

void setupGrammarAndSource() {
  auto ReadFile = [](llvm::StringRef FilePath) -> std::string {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> GrammarText =
        llvm::MemoryBuffer::getFile(FilePath);
    if (std::error_code EC = GrammarText.getError()) {
      llvm::errs() << "Error: can't read file '" << FilePath
                   << "': " << EC.message() << "\n";
      std::exit(1);
    }
    return GrammarText.get()->getBuffer().str();
  };
  GrammarText = new std::string(ReadFile(GrammarFile));
  SourceText = new std::string(ReadFile(Source));
  std::vector<std::string> Diags;
  G = Grammar::parseBNF(*GrammarText, Diags).release();
}

static void parseBNF(benchmark::State &State) {
  std::vector<std::string> Diags;
  for (auto _ : State)
    Grammar::parseBNF(*GrammarText, Diags);
}
BENCHMARK(parseBNF);

static void buildSLR(benchmark::State &State) {
  for (auto _ : State)
    LRTable::buildSLR(*G);
}
BENCHMARK(buildSLR);

TokenStream lexAndPreprocess() {
  clang::LangOptions LangOpts = genericLangOpts();
  TokenStream RawStream = pseudo::lex(*SourceText, LangOpts);
  auto DirectiveStructure = DirectiveTree::parse(RawStream);
  chooseConditionalBranches(DirectiveStructure, RawStream);
  TokenStream Cook =
      cook(DirectiveStructure.stripDirectives(RawStream), LangOpts);
  return stripComments(Cook);
}

static void lex(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  for (auto _ : State)
    clang::pseudo::lex(*SourceText, LangOpts);
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(lex);

static void preprocess(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  TokenStream RawStream = clang::pseudo::lex(*SourceText, LangOpts);
  for (auto _ : State) {
    auto DirectiveStructure = DirectiveTree::parse(RawStream);
    chooseConditionalBranches(DirectiveStructure, RawStream);
    stripComments(
        cook(DirectiveStructure.stripDirectives(RawStream), LangOpts));
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(preprocess);

static void glrParse(benchmark::State &State) {
  LRTable Table = clang::pseudo::LRTable::buildSLR(*G);
  SymbolID StartSymbol = *G->findNonterminal("translation-unit");
  TokenStream Stream = lexAndPreprocess();
  for (auto _ : State) {
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    pseudo::glrParse(Stream, ParseParams{*G, Table, Forest, GSS}, StartSymbol);
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(glrParse);

static void full(benchmark::State &State) {
  LRTable Table = clang::pseudo::LRTable::buildSLR(*G);
  SymbolID StartSymbol = *G->findNonterminal("translation-unit");
  for (auto _ : State) {
    TokenStream Stream = lexAndPreprocess();
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    pseudo::glrParse(lexAndPreprocess(), ParseParams{*G, Table, Forest, GSS},
                     StartSymbol);
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(full);

} // namespace
} // namespace bench
} // namespace pseudo
} // namespace clang

int main(int argc, char *argv[]) {
  benchmark::Initialize(&argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  clang::pseudo::bench::setupGrammarAndSource();
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
