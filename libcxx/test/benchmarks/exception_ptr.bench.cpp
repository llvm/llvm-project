//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <benchmark/benchmark.h>
#include <exception>

void bm_make_exception_ptr(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::make_exception_ptr(42));
  }
}
BENCHMARK(bm_make_exception_ptr)->ThreadRange(1, 8);

static bool exception_ptr_moves_copies_swap(std::exception_ptr p1) {
  // Taken from https://llvm.org/PR45547
  std::exception_ptr p2(p1);            // Copy constructor
  std::exception_ptr p3(std::move(p2)); // Move constructor
  p2 = std::move(p1);                   // Move assignment
  p1 = p2;                              // Copy assignment
  swap(p1, p2);                         // Swap
  // Comparisons against nullptr. The overhead from creating temporary `exception_ptr`
  // instances should be optimized out.
  bool is_null  = p1 == nullptr && nullptr == p2;
  bool is_equal = p1 == p2; // Comparison
  return is_null && is_equal;
}

// Benchmark copies, moves and comparisons of a non-null exception_ptr.
void bm_nonnull_exception_ptr(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(exception_ptr_moves_copies_swap(excptr));
  }
}
BENCHMARK(bm_nonnull_exception_ptr);

// Benchmark copies, moves and comparisons of a nullptr exception_ptr
// where the compiler cannot prove that the exception_ptr is always
// a nullptr and needs to emit runtime checks.
void bm_null_exception_ptr(benchmark::State& state) {
  std::exception_ptr excptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(exception_ptr_moves_copies_swap(excptr));
  }
}
BENCHMARK(bm_null_exception_ptr);

// Benchmark copies, moves and comparisons of a nullptr exception_ptr
// where the compiler can proof that the exception_ptr is always a nullptr.
void bm_optimized_null_exception_ptr(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(exception_ptr_moves_copies_swap(std::exception_ptr{nullptr}));
  }
}
BENCHMARK(bm_optimized_null_exception_ptr);

BENCHMARK_MAIN();
