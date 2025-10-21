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

void bm_exception_ptr_copy_ctor_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(std::exception_ptr(excptr));
  }
}
BENCHMARK(bm_exception_ptr_copy_ctor_nonnull);

void bm_exception_ptr_copy_ctor_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(std::exception_ptr(excptr));
  }
}
BENCHMARK(bm_exception_ptr_copy_ctor_null);


void bm_exception_ptr_move_ctor_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    // Need to copy, such that the `excptr` is not moved from and
    // empty after the first loop iteration.
    std::exception_ptr excptr_copy(excptr);
    benchmark::DoNotOptimize(excptr_copy);
    benchmark::DoNotOptimize(std::exception_ptr(std::move(excptr_copy)));
  }
}
BENCHMARK(bm_exception_ptr_move_ctor_nonnull);

void bm_exception_ptr_move_ctor_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(std::exception_ptr(std::move(excptr)));
  }
}
BENCHMARK(bm_exception_ptr_move_ctor_null);

void bm_exception_ptr_copy_assign_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    std::exception_ptr new_excptr;
    new_excptr = excptr;
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_copy_assign_nonnull);

void bm_exception_ptr_copy_assign_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    std::exception_ptr new_excptr;
    new_excptr = excptr;
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_copy_assign_null);


void bm_exception_ptr_move_assign_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    // Need to copy, such that the `excptr` is not moved from and
    // empty after the first loop iteration.
    std::exception_ptr excptr_copy(excptr);
    benchmark::DoNotOptimize(excptr_copy);
    std::exception_ptr new_excptr;
    new_excptr = std::move(excptr_copy);
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_move_assign_nonnull);

void bm_exception_ptr_move_assign_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    std::exception_ptr new_excptr;
    new_excptr = std::move(excptr);
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_move_assign_null);

void bm_exception_ptr_swap_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(41);
  std::exception_ptr excptr2 = std::make_exception_ptr(42);
  for (auto _ : state) {
    swap(excptr, excptr2);
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(excptr2);
  }
}
BENCHMARK(bm_exception_ptr_swap_nonnull);

void bm_exception_ptr_swap_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  std::exception_ptr excptr2 = nullptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    swap(excptr, excptr2);
    benchmark::DoNotOptimize(excptr2);
  }
}
BENCHMARK(bm_exception_ptr_swap_null);

// A chain of moves, copies and swaps.
// In contrast to the previous benchmarks of individual operations,
// this benchmark performs a chain of operations. It thereby
// specifically stresses the information available to the compiler
// for optimizations from the header itself.
static bool exception_ptr_move_copy_swap(std::exception_ptr p1) {
  // Taken from https://llvm.org/PR45547
  std::exception_ptr p2(p1);
  std::exception_ptr p3(std::move(p2));
  p2 = std::move(p1);
  p1 = p2;
  swap(p1, p2);
  // Comparisons against nullptr. The overhead from creating temporary `exception_ptr`
  // instances should be optimized out.
  bool is_null  = p1 == nullptr && nullptr == p2;
  bool is_equal = p1 == p2;
  return is_null && is_equal;
}

// Benchmark copies, moves and comparisons of a non-null exception_ptr.
void bm_exception_ptr_move_copy_swap_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(exception_ptr_move_copy_swap(excptr));
  }
}
BENCHMARK(bm_exception_ptr_move_copy_swap_nonnull);

// Benchmark copies, moves and comparisons of a nullptr exception_ptr
// where the compiler cannot prove that the exception_ptr is always
// a nullptr and needs to emit runtime checks.
void bm_exception_ptr_move_copy_swap_null(benchmark::State& state) {
  std::exception_ptr excptr;
  for (auto _ : state) {
    benchmark::DoNotOptimize(excptr);
    benchmark::DoNotOptimize(exception_ptr_move_copy_swap(excptr));
  }
}
BENCHMARK(bm_exception_ptr_move_copy_swap_null);

// Benchmark copies, moves and comparisons of a nullptr exception_ptr
// where the compiler can proof that the exception_ptr is always a nullptr.
void bm_exception_ptr_move_copy_swap_null_optimized(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(exception_ptr_move_copy_swap(std::exception_ptr{nullptr}));
  }
}
BENCHMARK(bm_exception_ptr_move_copy_swap_null_optimized);

BENCHMARK_MAIN();
