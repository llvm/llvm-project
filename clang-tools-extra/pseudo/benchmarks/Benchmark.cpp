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
// Note: make sure we build it in Relase mode.
//
// Usage:
//   tools/clang/tools/extra/pseudo/benchmarks/ClangPseudoBenchmark \
//      --grammar=/path/to/cxx.bnf --source=/patch/to/source-to-parse.cpp \
//      --benchmark_filter=runParseOverall
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
using llvm::cl::init;
using llvm::cl::opt;

static opt<std::string> GrammarFile("grammar",
                                    desc("Parse and check a BNF grammar file."),
                                    init(""));
static opt<std::string> Source("source", desc("Source file"));

namespace clang {
namespace pseudo {
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

static void runParseBNFGrammar(benchmark::State &State) {
  std::vector<std::string> Diags;
  for (auto _ : State)
    Grammar::parseBNF(*GrammarText, Diags);
}
BENCHMARK(runParseBNFGrammar);

static void runBuildLR(benchmark::State &State) {
  for (auto _ : State)
    clang::pseudo::LRTable::buildSLR(*G);
}
BENCHMARK(runBuildLR);

TokenStream parseableTokenStream() {
  clang::LangOptions LangOpts = genericLangOpts();
  TokenStream RawStream = clang::pseudo::lex(*SourceText, LangOpts);
  auto DirectiveStructure = DirectiveTree::parse(RawStream);
  clang::pseudo::chooseConditionalBranches(DirectiveStructure, RawStream);
  TokenStream Cook =
      cook(DirectiveStructure.stripDirectives(RawStream), LangOpts);
  return clang::pseudo::stripComments(Cook);
}

static void runPreprocessTokens(benchmark::State &State) {
  for (auto _ : State)
    parseableTokenStream();
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(runPreprocessTokens);

static void runGLRParse(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  LRTable Table = clang::pseudo::LRTable::buildSLR(*G);
  TokenStream ParseableStream = parseableTokenStream();
  for (auto _ : State) {
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    glrParse(ParseableStream, ParseParams{*G, Table, Forest, GSS});
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(runGLRParse);

static void runParseOverall(benchmark::State &State) {
  clang::LangOptions LangOpts = genericLangOpts();
  LRTable Table = clang::pseudo::LRTable::buildSLR(*G);
  for (auto _ : State) {
    pseudo::ForestArena Forest;
    pseudo::GSS GSS;
    glrParse(parseableTokenStream(), ParseParams{*G, Table, Forest, GSS});
  }
  State.SetBytesProcessed(static_cast<uint64_t>(State.iterations()) *
                          SourceText->size());
}
BENCHMARK(runParseOverall);

} // namespace
} // namespace pseudo
} // namespace clang

int main(int argc, char *argv[]) {
  benchmark::Initialize(&argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  clang::pseudo::setupGrammarAndSource();
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
