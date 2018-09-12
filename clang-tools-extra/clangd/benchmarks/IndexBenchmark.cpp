//===--- IndexBenchmark.cpp - Clangd index benchmarks -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../index/SymbolYAML.h"
#include "../index/dex/Dex.h"
#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <fstream>
#include <streambuf>
#include <string>

const char *IndexFilename;
const char *LogFilename;

namespace clang {
namespace clangd {
namespace {

std::unique_ptr<clang::clangd::SymbolIndex> buildMem() {
  return clang::clangd::loadIndex(IndexFilename, {}, false);
}

std::unique_ptr<clang::clangd::SymbolIndex> buildDex() {
  return clang::clangd::loadIndex(IndexFilename, {}, true);
}

// This function processes user-provided Log file with fuzzy find requests in
// the following format:
//
// fuzzyFind("UnqualifiedName", scopes=["clang::", "clang::clangd::"])
//
// It constructs vector of FuzzyFindRequests which is later used for the
// benchmarks.
std::vector<clang::clangd::FuzzyFindRequest> extractQueriesFromLogs() {
  llvm::Regex RequestMatcher("fuzzyFind\\(\"([a-zA-Z]*)\", scopes=\\[(.*)\\]");
  llvm::SmallVector<llvm::StringRef, 200> Matches;
  std::ifstream InputStream(LogFilename);
  std::string Log((std::istreambuf_iterator<char>(InputStream)),
                  std::istreambuf_iterator<char>());
  llvm::StringRef Temporary(Log);
  llvm::SmallVector<llvm::StringRef, 200> Strings;
  Temporary.split(Strings, '\n');

  clang::clangd::FuzzyFindRequest R;
  R.MaxCandidateCount = 100;

  llvm::SmallVector<llvm::StringRef, 200> CommaSeparatedValues;

  std::vector<clang::clangd::FuzzyFindRequest> RealRequests;
  for (auto Line : Strings) {
    if (RequestMatcher.match(Line, &Matches)) {
      R.Query = Matches[1];
      CommaSeparatedValues.clear();
      Line.split(CommaSeparatedValues, ',');
      R.Scopes.clear();
      for (auto C : CommaSeparatedValues) {
        R.Scopes.push_back(C);
      }
      RealRequests.push_back(R);
    }
  }
  return RealRequests;
}

static void MemQueries(benchmark::State &State) {
  const auto Mem = buildMem();
  const auto Requests = extractQueriesFromLogs();
  for (auto _ : State)
    for (const auto &Request : Requests)
      Mem->fuzzyFind(Request, [](const Symbol &S) {});
}
BENCHMARK(MemQueries);

static void DexQueries(benchmark::State &State) {
  const auto Dex = buildDex();
  const auto Requests = extractQueriesFromLogs();
  for (auto _ : State)
    for (const auto &Request : Requests)
      Dex->fuzzyFind(Request, [](const Symbol &S) {});
}
BENCHMARK(DexQueries);

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
                 << " global-symbol-index.yaml fuzzy-find-requests.log "
                    "BENCHMARK_OPTIONS...\n";
    return -1;
  }
  IndexFilename = argv[1];
  LogFilename = argv[2];
  // Trim first two arguments of the benchmark invocation.
  argv += 3;
  argc -= 3;
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
