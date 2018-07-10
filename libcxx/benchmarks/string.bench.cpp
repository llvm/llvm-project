#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark.h"
#include "GenerateInput.hpp"

constexpr std::size_t MAX_STRING_LEN = 8 << 14;

// Benchmark when there is no match.
static void BM_StringFindNoMatch(benchmark::State &state) {
  std::string s1(state.range(0), '-');
  std::string s2(8, '*');
  while (state.KeepRunning())
    benchmark::DoNotOptimize(s1.find(s2));
}
BENCHMARK(BM_StringFindNoMatch)->Range(10, MAX_STRING_LEN);

// Benchmark when the string matches first time.
static void BM_StringFindAllMatch(benchmark::State &state) {
  std::string s1(MAX_STRING_LEN, '-');
  std::string s2(state.range(0), '-');
  while (state.KeepRunning())
    benchmark::DoNotOptimize(s1.find(s2));
}
BENCHMARK(BM_StringFindAllMatch)->Range(1, MAX_STRING_LEN);

// Benchmark when the string matches somewhere in the end.
static void BM_StringFindMatch1(benchmark::State &state) {
  std::string s1(MAX_STRING_LEN / 2, '*');
  s1 += std::string(state.range(0), '-');
  std::string s2(state.range(0), '-');
  while (state.KeepRunning())
    benchmark::DoNotOptimize(s1.find(s2));
}
BENCHMARK(BM_StringFindMatch1)->Range(1, MAX_STRING_LEN / 4);

// Benchmark when the string matches somewhere from middle to the end.
static void BM_StringFindMatch2(benchmark::State &state) {
  std::string s1(MAX_STRING_LEN / 2, '*');
  s1 += std::string(state.range(0), '-');
  s1 += std::string(state.range(0), '*');
  std::string s2(state.range(0), '-');
  while (state.KeepRunning())
    benchmark::DoNotOptimize(s1.find(s2));
}
BENCHMARK(BM_StringFindMatch2)->Range(1, MAX_STRING_LEN / 4);

static void BM_StringCtorDefault(benchmark::State &state) {
  while (state.KeepRunning()) {
    for (unsigned I=0; I < 1000; ++I) {
      std::string Default;
      benchmark::DoNotOptimize(Default.c_str());
    }
  }
}
BENCHMARK(BM_StringCtorDefault);

static void BM_StringCtorCStr(benchmark::State &state) {
  std::string Input = getRandomString(state.range(0));
  const char *Str = Input.c_str();
  benchmark::DoNotOptimize(Str);
  while (state.KeepRunning()) {
    std::string Tmp(Str);
    benchmark::DoNotOptimize(Tmp.c_str());
  }
}
BENCHMARK(BM_StringCtorCStr)->Arg(1)->Arg(8)->Range(16, MAX_STRING_LEN / 4);

BENCHMARK_MAIN();
