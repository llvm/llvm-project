//===--- DecisionForestBenchmark.cpp ------------*- C++ -*-===//
//
// Benchmark for code completion ranking latency.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Usage:
//    ninja DecisionForestBenchmark && \
//    tools/clang/tools/extra/clangd/benchmarks/CompletionModel/DecisionForestBenchmark
//===----------------------------------------------------------------------===//

#include "CompletionModel.h"
#include "benchmark/benchmark.h"
#include "llvm/ADT/StringRef.h"

#include <random>

namespace clang {
namespace clangd {
namespace {
std::vector<Example> generateRandomDataset(int NumExamples) {
  auto FlipCoin = [&](float Probability) {
    return rand() % 1000 <= Probability * 1000;
  };
  auto RandInt = [&](int Max) { return rand() % Max; };
  auto RandFloat = [&](float Max = 1.0) {
    return rand() % 1000 / 1000.0 * Max;
  };

  std::vector<Example> Examples;
  Examples.reserve(NumExamples);
  for (int I = 0; I < NumExamples; ++I) {
    Example E;
    E.setIsDeprecated(FlipCoin(0.1));           // Boolean.
    E.setIsReservedName(FlipCoin(0.1));         // Boolean.
    E.setIsImplementationDetail(FlipCoin(0.3)); // Boolean.
    E.setNumReferences(RandInt(10000));         // Can be large integer.
    E.setSymbolCategory(RandInt(10));           // 10 Symbol Category.

    E.setIsNameInContext(FlipCoin(0.5)); // Boolean.
    E.setIsForbidden(FlipCoin(0.1));     // Boolean.
    E.setIsInBaseClass(FlipCoin(0.3));   // Boolean.
    E.setFileProximityDistance(
        FlipCoin(0.1) ? 999999 // Sometimes file distance is not available.
                      : RandInt(20));
    E.setSemaFileProximityScore(RandFloat(1)); // Float in range [0,1].
    E.setSymbolScopeDistance(
        FlipCoin(0.1) ? 999999 // Sometimes scope distance is not available.
                      : RandInt(20));
    E.setSemaSaysInScope(FlipCoin(0.5));      // Boolean.
    E.setScope(RandInt(4));                   // 4 Scopes.
    E.setContextKind(RandInt(32));            // 32 Context kinds.
    E.setIsInstanceMember(FlipCoin(0.5));     // Boolean.
    E.setHadContextType(FlipCoin(0.6));       // Boolean.
    E.setHadSymbolType(FlipCoin(0.6));        // Boolean.
    E.setTypeMatchesPreferred(FlipCoin(0.5)); // Boolean.
    E.setFilterLength(RandInt(15));
    Examples.push_back(E);
  }
  return Examples;
}

void runDecisionForestPrediciton(const std::vector<Example> Examples) {
  for (const Example &E : Examples)
    Evaluate(E);
}

static void decisionForestPredict(benchmark::State &State) {
  srand(0);
  for (auto _ : State) {
    State.PauseTiming();
    const std::vector<Example> Examples = generateRandomDataset(1000000);
    State.ResumeTiming();
    runDecisionForestPrediciton(Examples);
  }
}
BENCHMARK(decisionForestPredict);

} // namespace
} // namespace clangd
} // namespace clang

BENCHMARK_MAIN();
