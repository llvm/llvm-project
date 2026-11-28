//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains basic benchmarks for clang-doc's implementation and
/// library components.
///
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "BitcodeWriter.h"
#include "ClangDoc.h"
#include "Generators.h"
#include "Representation.h"
#include "Serialize.h"
#include "benchmark/benchmark.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace clang {
namespace doc {

class BenchmarkVisitor : public RecursiveASTVisitor<BenchmarkVisitor> {
public:
  explicit BenchmarkVisitor(const FunctionDecl *&Func) : Func(Func) {}

  bool VisitFunctionDecl(const FunctionDecl *D) {
    if (D->getName() == "f") {
      Func = D;
      return false;
    }
    return true;
  }

private:
  const FunctionDecl *&Func;
};

// --- Mapper Benchmarks ---

static void BM_EmitInfoFunction(benchmark::State &State) {
  std::string Code = "void f() {}";
  std::unique_ptr<clang::ASTUnit> AST = clang::tooling::buildASTFromCode(Code);
  const FunctionDecl *Func = nullptr;
  BenchmarkVisitor Visitor(Func);
  Visitor.TraverseDecl(AST->getASTContext().getTranslationUnitDecl());
  assert(Func);

  clang::comments::FullComment *FC = nullptr;
  Location Loc;

  for (auto _ : State) {
    serialize::Serializer Serializer;
    auto Result = Serializer.emitInfo(Func, FC, Loc, /*PublicOnly=*/false);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_EmitInfoFunction);

static void BM_Mapper_Scale(benchmark::State &State) {
  std::string Code;
  for (int i = 0; i < State.range(0); ++i) {
    Code += "void f" + std::to_string(i) + "() {}\n";
  }

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagID, DiagOpts, new IgnoringDiagConsumer());

  for (auto _ : State) {
    tooling::InMemoryToolResults Results;
    tooling::ExecutionContext ECtx(&Results);
    ClangDocContext CDCtx(&ECtx, "test-project", false, "", "", "", "", "", {},
                          Diags, OutputFormatTy::json, false);
    auto ActionFactory = doc::newMapperActionFactory(CDCtx);
    std::unique_ptr<FrontendAction> Action = ActionFactory->create();
    tooling::runToolOnCode(std::move(Action), Code, "test.cpp");
  }
}
BENCHMARK(BM_Mapper_Scale)->Range(10, 10000);

// --- Reducer Benchmarks ---

static void BM_SerializeFunctionInfo(benchmark::State &State) {
  auto I = allocateTransient<FunctionInfo>();
  I->Name = "f";
  I->DefLoc = Location(0, 0, "test.cpp");
  I->ReturnType = TypeInfo("void");
  I->IT = InfoType::IT_function;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagID, DiagOpts, new IgnoringDiagConsumer());

  Info *InfoPtr = I;

  for (auto _ : State) {
    auto Result = serialize::serialize(*InfoPtr, Diags);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_SerializeFunctionInfo);

static void BM_MergeInfos_Scale(benchmark::State &State) {
  SymbolID USR = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

  for (auto _ : State) {
    State.PauseTiming();
    SmallVector<Info *> Input;
    Input.reserve(State.range(0));
    for (int Idx = 0; Idx < State.range(0); ++Idx) {
      auto *I = allocateTransient<FunctionInfo>();
      I->Name = "f";
      I->USR = USR;
      I->DefLoc = Location(10, Idx, "test.cpp");
      Input.push_back(std::move(I));
    }
    State.ResumeTiming();

    auto Result = doc::mergeInfos(Input);
    if (!Result) {
      State.SkipWithError("mergeInfos failed");
      llvm::consumeError(Result.takeError());
    }
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_MergeInfos_Scale)->Range(2, 10000);

static void BM_BitcodeReader_Scale(benchmark::State &State) {
  int NumRecords = State.range(0);

  SmallString<0> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagID, DiagOpts, new IgnoringDiagConsumer());

  ClangDocBitcodeWriter Writer(Stream, Diags);
  for (int i = 0; i < NumRecords; ++i) {
    RecordInfo RI;
    RI.Name = internString("Record" + std::to_string(i));
    RI.USR = {(uint8_t)(i & 0xFF)};
    Writer.emitBlock(RI);
  }

  std::string BitcodeData = Buffer.str().str();

  for (auto _ : State) {
    llvm::BitstreamCursor Cursor(llvm::ArrayRef<uint8_t>(
        (const uint8_t *)BitcodeData.data(), BitcodeData.size()));
    ClangDocBitcodeReader Reader(Cursor, Diags);
    auto Result = Reader.readBitcode();
    if (!Result) {
      State.SkipWithError("readBitcode failed");
      llvm::consumeError(Result.takeError());
    }
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_BitcodeReader_Scale)->Range(10, 10000);

// --- Generator Benchmarks ---

static void BM_JSONGenerator_Scale(benchmark::State &State) {
  auto G = doc::findGeneratorByName("json");
  if (!G) {
    State.SkipWithError("JSON Generator not found");
    llvm::consumeError(G.takeError());
    return;
  }
  int NumRecords = State.range(0);
  auto *NI = allocateTransient<NamespaceInfo>();
  NI->Name = "GlobalNamespace";
  for (int i = 0; i < NumRecords; ++i) {
    NI->Children.Records.push_back(*allocateListNodeTransient<Reference>(
        SymbolID{(uint8_t)(i & 0xFF)}, "Record" + std::to_string(i),
        InfoType::IT_record));
  }

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagID, DiagOpts, new IgnoringDiagConsumer());
  ClangDocContext CDCtx(nullptr, "test-project", false, "", "", "", "", "", {},
                        Diags, OutputFormatTy::json, false);

  std::string Output;
  llvm::raw_string_ostream OS(Output);

  for (auto _ : State) {
    Output.clear();
    auto Err = (*G)->generateDocForInfo(NI, OS, CDCtx);
    if (Err) {
      State.SkipWithError("generateDocForInfo failed");
      llvm::consumeError(std::move(Err));
    }
    benchmark::DoNotOptimize(Output);
  }
}
BENCHMARK(BM_JSONGenerator_Scale)->Range(10, 10000);

// --- Index Benchmarks ---

static void BM_Index_Insertion(benchmark::State &State) {
  for (auto _ : State) {
    Index Idx;
    for (int i = 0; i < State.range(0); ++i) {
      RecordInfo I;
      I.Name = internString("Record" + std::to_string(i));
      // Vary USR to ensure unique entries
      I.USR = {(uint8_t)(i & 0xFF), (uint8_t)((i >> 8) & 0xFF)};
      I.Path = internString("path/to/record");
      Generator::addInfoToIndex(Idx, &I);
    }
    benchmark::DoNotOptimize(Idx);
  }
}
BENCHMARK(BM_Index_Insertion)->Range(10, 10000);

// --- String Pool Benchmarks ---

static std::unique_ptr<ConcurrentStringPool> &getBenchmarkStringPool() {
  static std::unique_ptr<ConcurrentStringPool> Pool;
  return Pool;
}

static const std::vector<llvm::StringRef> &getBenchmarkTrace() {
  static const std::vector<std::string> Corpus = [] {
    constexpr size_t NumUniqueStrings = 1 << 18;
    std::vector<std::string> V(NumUniqueStrings);
    for (size_t I = 0; I < V.size(); ++I)
      V[I] = "/usr/local/include/clang/AST/Decl_" + std::to_string(I) + ".h";
    return V;
  }();

  static const std::vector<llvm::StringRef> Trace = [] {
    constexpr size_t NumHotStrings = 64;
    constexpr size_t TraceSize = 1 << 20;
    std::minstd_rand Rng;
    std::bernoulli_distribution IsHot(0.8);
    std::uniform_int_distribution<size_t> HotDist(0, NumHotStrings - 1);
    std::uniform_int_distribution<size_t> ColdDist(0, Corpus.size() - 1);

    std::vector<llvm::StringRef> T(TraceSize);
    for (llvm::StringRef &S : T)
      S = Corpus[IsHot(Rng) ? HotDist(Rng) : ColdDist(Rng)];
    return T;
  }();
  return Trace;
}

static void BM_StringPool_ConcurrentStringPool(benchmark::State &State) {
  auto &SharedPool = *getBenchmarkStringPool();
  const auto &Trace = getBenchmarkTrace();
  const size_t Mask = Trace.size() - 1;
  size_t Idx = State.thread_index() * (Trace.size() / State.threads());

  for (auto _ : State) {
    llvm::StringRef Res = SharedPool.save(Trace[Idx & Mask]);
    benchmark::DoNotOptimize(Res);
    ++Idx;
  }
}

BENCHMARK(BM_StringPool_ConcurrentStringPool)
    ->ThreadRange(1, 128)
    ->UseRealTime()
    ->Setup([](const benchmark::State &) {
      getBenchmarkStringPool() = std::make_unique<ConcurrentStringPool>();
    })
    ->Teardown([](const benchmark::State &) {
      getBenchmarkStringPool().reset();
    });

} // namespace doc
} // namespace clang

BENCHMARK_MAIN();
