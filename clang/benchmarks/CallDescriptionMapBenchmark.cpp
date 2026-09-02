//===- CallDescriptionMapBenchmark.cpp - CallDescriptionMap benchmarks ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace clang::ento;

namespace {

class CallFinder : public RecursiveASTVisitor<CallFinder> {
public:
  bool VisitCallExpr(CallExpr *Call) {
    if (!Found)
      Found = Call;
    return true;
  }

  const CallExpr *get() const { return Found; }

private:
  const CallExpr *Found = nullptr;
};

struct ParsedCall {
  std::unique_ptr<ASTUnit> AST;
  const CallExpr *Call;
};

enum class TargetPosition { None, First, Last };

ParsedCall parseCall() {
  auto AST =
      tooling::buildASTFromCode("void target(); void caller() { target(); }");
  assert(AST);

  CallFinder Finder;
  Finder.TraverseDecl(AST->getASTContext().getTranslationUnitDecl());
  assert(Finder.get());
  return {std::move(AST), Finder.get()};
}

std::vector<std::pair<CallDescription, unsigned>>
makeDescriptions(unsigned Size, TargetPosition Position) {
  std::vector<std::pair<CallDescription, unsigned>> Descriptions;
  Descriptions.reserve(Size);
  for (unsigned I = 0; I < Size; ++I) {
    const bool IsTarget = (Position == TargetPosition::First && I == 0) ||
                          (Position == TargetPosition::Last && I + 1 == Size);
    std::string Name = IsTarget ? "target" : "unrelated_" + std::to_string(I);
    Descriptions.emplace_back(
        CallDescription(CDM::SimpleFunc, {StringRef(Name)}), I);
  }
  return Descriptions;
}

static void BM_Construct(benchmark::State &State) {
  const unsigned Size = State.range(0);
  auto Descriptions = makeDescriptions(Size, TargetPosition::Last);

  for (auto _ : State) {
    CallDescriptionMap<unsigned> Map(Descriptions.begin(), Descriptions.end());
    benchmark::DoNotOptimize(Map);
  }
  State.SetItemsProcessed(State.iterations() * Size);
}

static void runLookup(benchmark::State &State, TargetPosition Position) {
  ParsedCall Parsed = parseCall();
  auto Descriptions = makeDescriptions(State.range(0), Position);
  CallDescriptionMap<unsigned> Map(Descriptions.begin(), Descriptions.end());

  for (auto _ : State)
    benchmark::DoNotOptimize(Map.lookupAsWritten(*Parsed.Call));
}

static void BM_LookupAsWrittenHitFirst(benchmark::State &State) {
  runLookup(State, TargetPosition::First);
}

static void BM_LookupAsWrittenHitLast(benchmark::State &State) {
  runLookup(State, TargetPosition::Last);
}

static void BM_LookupAsWrittenMiss(benchmark::State &State) {
  runLookup(State, TargetPosition::None);
}

#define MAP_SIZES                                                              \
  Args({1})->Args({4})->Args({7})->Args({8})->Args({16})->Args({32})->Args({64})

BENCHMARK(BM_Construct)->MAP_SIZES;
BENCHMARK(BM_LookupAsWrittenHitFirst)->MAP_SIZES;
BENCHMARK(BM_LookupAsWrittenHitLast)->MAP_SIZES;
BENCHMARK(BM_LookupAsWrittenMiss)->MAP_SIZES;

#undef MAP_SIZES

} // namespace

BENCHMARK_MAIN();
