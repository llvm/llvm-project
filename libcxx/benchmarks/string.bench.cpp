#include <unordered_set>
#include <vector>
#include <cstdint>

#include "benchmark/benchmark_api.h"
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

BENCHMARK_MAIN()
