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
#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/DirectiveTree.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Token.h"
#include "clang-pseudo/cli/CLI.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
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

static opt<std::string> Source("source", desc("Source file"), Required);

namespace clang {
namespace pseudo {
namespace bench {
namespace {

const std::string *SourceText = nullptr;
const Language *Lang = nullptr;

void setup() {
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
  SourceText = new std::string(ReadFile(Source));
  Lang = &getLanguageFromFlags();
}

static void buildSLR(benchmark::State &State) {
  for (auto _ : State)
    LRTable::buildSLR(Lang->G);
}
BENCHMARK(buildSLR);

TokenStream lexAndPreprocess() {
  clang::LangOptions LangOpts = genericLangOpts();
  TokenStream RawStream = pseudo::lex(*SourceText, LangOpts);
  auto DirectiveStructure = DirectiveTree::parse(RawStream);
  chooseConditionalBranches(DirectiveStructure, RawStream);
  TokenStream Cook =
      cook(DirectiveStructure.stripDirectives(RawStream), LangOpts);
  auto Stream = stripComments(Cook);
  pairBrackets(Stream);
  return Stream;
}

static void lex(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  for (auto _ : State)
    clang::pseudo::lex(*SourceText, LangOpts);
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(lex);

static void pairBrackets(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  auto Stream = clang::pseudo::lex(*SourceText, LangOpts);
  for (auto _ : State)
    pairBrackets(Stream);
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(pairBrackets);

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
  SymbolID StartSymbol = *Lang->G->findNonterminal("translation-unit");
  TokenStream Stream = lexAndPreprocess();
  for (auto _ : State) {
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    pseudo::glrParse(Stream, ParseParams{*Lang->G, Lang->Table, Forest, GSS},
                     StartSymbol);
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(glrParse);

static void full(benchmark::State &State) {
  SymbolID StartSymbol = *Lang->G.findNonterminal("translation-unit");
  for (auto _ : State) {
    TokenStream Stream = lexAndPreprocess();
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    pseudo::glrParse(lexAndPreprocess(),
                     ParseParams{Lang->G, Lang->Table, Forest, GSS},
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
  clang::pseudo::bench::setup();
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
