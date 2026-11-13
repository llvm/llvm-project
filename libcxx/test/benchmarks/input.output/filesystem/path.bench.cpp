//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <filesystem>
#include <ranges>

#include "../../GenerateInput.h"
#include "benchmark/benchmark.h"
#include "test_iterators.h"
#include "test_macros.h"

namespace fs = std::filesystem;

constexpr size_t short_dir_len = 4;
constexpr size_t long_dir_len  = 20;

void RunWithRandomPaths(benchmark::Benchmark* bm) {
  bm->Args({short_dir_len, /*directory_count=*/5})
      ->Args({short_dir_len, /*directory_count=*/20})
      ->Args({long_dir_len, /*directory_count=*/5})
      ->Args({long_dir_len, /*directory_count=*/20})
      ->ArgNames({"avg_dir_name_len", "dir_count"});
}

fs::path getRandomPath(benchmark::State& state) {
  size_t avg_dir_name_len = state.range(0);
  auto directory_count    = state.range(1);
  std::string path_str    = getRandomString(avg_dir_name_len * directory_count);

  for (size_t i = 0; i < path_str.size();
       i += getRandomInteger(avg_dir_name_len / 2, avg_dir_name_len / 2 + avg_dir_name_len)) {
    path_str[i] = '/';
  }
  return path_str;
}

TEST_ALIGN_BENCHMARK static void BM_PathIteratorRepeatedIteration(benchmark::State& state) {
  fs::path path = getRandomPath(state);
  for (auto _ : state) {
    benchmark::DoNotOptimize(path);
    for (const auto& dir : path)
      benchmark::DoNotOptimize(dir.native().data());
  }
}

BENCHMARK(BM_PathIteratorRepeatedIteration)
    ->Apply(RunWithRandomPaths)
    ->Name("filesystem::path::iterator (same path repeatedly)");

TEST_ALIGN_BENCHMARK static void BM_PathIteratorDifferentPaths(benchmark::State& state) {
  fs::path path = getRandomPath(state);
  for (auto _ : state) {
    benchmark::DoNotOptimize(path);
    auto path_copy = path;
    for (const auto& dir : path)
      benchmark::DoNotOptimize(dir.native().data());
  }
}

BENCHMARK(BM_PathIteratorDifferentPaths)
    ->Apply(RunWithRandomPaths)
    ->Name("filesystem::path::iterator (different paths)");

TEST_ALIGN_BENCHMARK static void BM_PathIteratorBackward(benchmark::State& state) {
  fs::path path = getRandomPath(state);
  for (auto _ : state) {
    benchmark::DoNotOptimize(path);
    auto path_copy = path;
    for (const auto& dir : path | std::views::reverse)
      benchmark::DoNotOptimize(dir.native().data());
  }
}

BENCHMARK(BM_PathIteratorBackward)
    ->Apply(RunWithRandomPaths)
    ->Name("filesystem::path::iterator (backward, different paths)");

TEST_ALIGN_BENCHMARK static void BM_PathConstructString(benchmark::State& state) {
  std::string path_str = getRandomPath(state).string();
  for (auto _ : state) {
    benchmark::DoNotOptimize(path_str);
    fs::path path(path_str);
    benchmark::DoNotOptimize(path);
  }
}

BENCHMARK(BM_PathConstructString)->Apply(RunWithRandomPaths)->Name("filesystem::path::ctor(const std::string&)");

TEST_ALIGN_BENCHMARK static void BM_PathConstructCString(benchmark::State& state) {
  std::string path_str = getRandomPath(state).string();
  for (auto _ : state) {
    benchmark::DoNotOptimize(path_str);
    fs::path path(path_str.data());
    benchmark::DoNotOptimize(path);
  }
}

BENCHMARK(BM_PathConstructCString)->Apply(RunWithRandomPaths)->Name("filesystem::path::ctor(const char*)");

template <template <class...> class Iter>
TEST_ALIGN_BENCHMARK static void BM_PathConstructIter(benchmark::State& state) {
  std::string path_str = getRandomPath(state).string();
  for (auto _ : state) {
    benchmark::DoNotOptimize(path_str);
    fs::path path(path_str.data());
    benchmark::DoNotOptimize(path);
  }
}

BENCHMARK(BM_PathConstructIter<cpp17_input_iterator>)
    ->Apply(RunWithRandomPaths)
    ->Name("filesystem::path::ctor(input-iter, input-iter)");
BENCHMARK(BM_PathConstructIter<forward_iterator>)
    ->Apply(RunWithRandomPaths)
    ->Name("filesystem::path::ctor(forward-iter, forward-iter)");

TEST_ALIGN_BENCHMARK static void BM_LexicallyNormal(benchmark::State& state) {
  fs::path path = getRandomPath(state);
  for (auto _ : state) {
    benchmark::DoNotOptimize(path);
    benchmark::DoNotOptimize(path.lexically_normal());
  }
}

BENCHMARK(BM_LexicallyNormal)->Apply(RunWithRandomPaths)->Name("filesystem::path::lexically_normal()");

TEST_ALIGN_BENCHMARK static void BM_LexicallyRelative(benchmark::State& state) {
  fs::path base_path   = getRandomPath(state);
  fs::path target_path = getRandomPath(state);
  for (auto _ : state) {
    benchmark::DoNotOptimize(base_path);
    benchmark::DoNotOptimize(base_path.lexically_relative(target_path));
  }
}

BENCHMARK(BM_LexicallyRelative)->Apply(RunWithRandomPaths)->Name("filesystem::path::lexically_normal()");

BENCHMARK_MAIN();
