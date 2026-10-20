//===--- IndexBenchmark.cpp - Clangd index benchmarks -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../index/Serialization.h"
#include "../index/dex/Dex.h"
#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <string>

const char *IndexFilename;
const char *RequestsFilename;

namespace clang {
namespace clangd {
namespace {

std::unique_ptr<SymbolIndex> buildMem() {
  return loadIndex(IndexFilename, clang::clangd::SymbolOrigin::Static,
                   /*UseDex=*/false, /*SupportContainedRefs=*/true);
}

std::unique_ptr<SymbolIndex> buildDex() {
  return loadIndex(IndexFilename, clang::clangd::SymbolOrigin::Static,
                   /*UseDex=*/true, /*SupportContainedRefs=*/true);
}

// Reads JSON array of serialized FuzzyFindRequest's from user-provided file.
std::vector<FuzzyFindRequest> extractQueriesFromLogs() {

  auto Buffer = llvm::MemoryBuffer::getFile(RequestsFilename);
  if (!Buffer) {
    llvm::errs() << "Error cannot open JSON request file:" << RequestsFilename
                 << ": " << Buffer.getError().message() << "\n";
    exit(1);
  }

  StringRef Log = Buffer.get()->getBuffer();

  std::vector<FuzzyFindRequest> Requests;
  auto JSONArray = llvm::json::parse(Log);

  // Panic if the provided file couldn't be parsed.
  if (!JSONArray) {
    llvm::errs() << "Error when parsing JSON requests file: "
                 << llvm::toString(JSONArray.takeError());
    exit(1);
  }
  if (!JSONArray->getAsArray()) {
    llvm::errs() << "Error: top-level value is not a JSON array: " << Log
                 << '\n';
    exit(1);
  }

  for (const auto &Item : *JSONArray->getAsArray()) {
    FuzzyFindRequest Request;
    // Panic if the provided file couldn't be parsed.
    llvm::json::Path::Root Root("FuzzyFindRequest");
    if (!fromJSON(Item, Request, Root)) {
      llvm::errs() << llvm::toString(Root.getError()) << "\n";
      Root.printErrorContext(Item, llvm::errs());
      exit(1);
    }
    Requests.push_back(Request);
  }
  return Requests;
}

static void memQueries(benchmark::State &State) {
  const auto Mem = buildMem();
  const auto Requests = extractQueriesFromLogs();
  for (auto _ : State)
    for (const auto &Request : Requests)
      Mem->fuzzyFind(Request, [](const Symbol &S) {});
}
BENCHMARK(memQueries);

static void dexQueries(benchmark::State &State) {
  const auto Dex = buildDex();
  const auto Requests = extractQueriesFromLogs();
  for (auto _ : State)
    for (const auto &Request : Requests)
      Dex->fuzzyFind(Request, [](const Symbol &S) {});
}
BENCHMARK(dexQueries);

static void dexBuild(benchmark::State &State) {
  for (auto _ : State)
    buildDex();
}
BENCHMARK(dexBuild);

} // namespace
} // namespace clangd
} // namespace clang

// FIXME(kbobyrev): Add index building time benchmarks.
// FIXME(kbobyrev): Add memory consumption "benchmarks" by manually measuring
// in-memory index size and reporting it as time.
// FIXME(kbobyrev): Create a logger wrapper to suppress debugging info printer.
int main(int argc, char *argv[]) {
  if (argc < 3) {
    llvm::errs() << "Usage: " << argv[0]
                 << " global-symbol-index.dex requests.json "
                    "BENCHMARK_OPTIONS...\n";
    return -1;
  }
  IndexFilename = argv[1];
  RequestsFilename = argv[2];
  // Trim first two arguments of the benchmark invocation and pretend no
  // arguments were passed in the first place.
  argv[2] = argv[0];
  argv += 2;
  argc -= 2;
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
