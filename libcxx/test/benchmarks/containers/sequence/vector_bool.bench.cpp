//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <memory_resource>
#include <vector>

#include "test_macros.h"

static void BM_vector_bool_copy_ctor(benchmark::State& state) {
  std::vector<bool> vec(100, true);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    std::vector<bool> vec2(vec);
    benchmark::DoNotOptimize(vec2);
  }
}
BENCHMARK(BM_vector_bool_copy_ctor)->Name("vector<bool>(const vector<bool>&)");

static void BM_vector_bool_move_ctor_alloc_equal(benchmark::State& state) {
  std::vector<bool> vec(100, true);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    std::vector<bool> vec2(std::move(vec), std::allocator<bool>());
    benchmark::DoNotOptimize(vec2);
    swap(vec, vec2);
  }
}
BENCHMARK(BM_vector_bool_move_ctor_alloc_equal)
    ->Name("vector<bool>(vector<bool>&&, const allocator_type&) (equal allocators)");

#if TEST_STD_VER >= 17
static void BM_vector_bool_move_ctor_alloc_different(benchmark::State& state) {
  std::pmr::monotonic_buffer_resource resource;
  std::pmr::vector<bool> vec(100, true, &resource);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    std::pmr::vector<bool> vec2(std::move(vec), std::pmr::new_delete_resource());
    benchmark::DoNotOptimize(vec2);
  }
}
BENCHMARK(BM_vector_bool_move_ctor_alloc_different)
    ->Name("vector<bool>(vector<bool>&&, const allocator_type&) (different allocators)");
#endif

static void BM_vector_bool_size_ctor(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<bool> vec(100, true);
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(BM_vector_bool_size_ctor)->Name("vector<bool>(size_type, const value_type&)");

static void BM_vector_bool_reserve(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<bool> vec;
    vec.reserve(100);
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(BM_vector_bool_reserve)->Name("vector<bool>::reserve()");

BENCHMARK_MAIN();
