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
    benchmark::DoNotOptimize(std::exception_ptr(excptr));
  }
}
BENCHMARK(bm_exception_ptr_copy_ctor_nonnull);

void bm_exception_ptr_copy_ctor_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    std::exception_ptr excptr_copy(excptr);
    // The compiler should be able to constant-fold the comparison
    benchmark::DoNotOptimize(excptr_copy == nullptr);
    benchmark::DoNotOptimize(excptr_copy);
  }
}
BENCHMARK(bm_exception_ptr_copy_ctor_null);

void bm_exception_ptr_move_ctor_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    // Need to copy, such that the `excptr` is not moved from and
    // empty after the first loop iteration.
    std::exception_ptr excptr_copy(excptr);
    benchmark::DoNotOptimize(std::exception_ptr(std::move(excptr_copy)));
  }
}
BENCHMARK(bm_exception_ptr_move_ctor_nonnull);

void bm_exception_ptr_move_ctor_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    std::exception_ptr new_excptr(std::move(excptr));
    // The compiler should be able to constant-fold the comparison
    benchmark::DoNotOptimize(new_excptr == nullptr);
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_move_ctor_null);

void bm_exception_ptr_copy_assign_nonnull(benchmark::State& state) {
  std::exception_ptr excptr = std::make_exception_ptr(42);
  for (auto _ : state) {
    std::exception_ptr new_excptr;
    new_excptr = excptr;
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_copy_assign_nonnull);

void bm_exception_ptr_copy_assign_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    std::exception_ptr new_excptr;
    new_excptr = excptr;
    // The compiler should be able to constant-fold the comparison
    benchmark::DoNotOptimize(new_excptr == nullptr);
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
    std::exception_ptr new_excptr;
    new_excptr = std::move(excptr_copy);
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_move_assign_nonnull);

void bm_exception_ptr_move_assign_null(benchmark::State& state) {
  std::exception_ptr excptr = nullptr;
  for (auto _ : state) {
    std::exception_ptr new_excptr;
    new_excptr = std::move(excptr);
    // The compiler should be able to constant-fold the comparison
    benchmark::DoNotOptimize(new_excptr == nullptr);
    benchmark::DoNotOptimize(new_excptr);
  }
}
BENCHMARK(bm_exception_ptr_move_assign_null);

void bm_exception_ptr_swap_nonnull(benchmark::State& state) {
  std::exception_ptr excptr1 = std::make_exception_ptr(41);
  std::exception_ptr excptr2 = std::make_exception_ptr(42);
  for (auto _ : state) {
    swap(excptr1, excptr2);
    benchmark::DoNotOptimize(excptr1);
    benchmark::DoNotOptimize(excptr2);
  }
}
BENCHMARK(bm_exception_ptr_swap_nonnull);

void bm_exception_ptr_swap_null(benchmark::State& state) {
  std::exception_ptr excptr1 = nullptr;
  std::exception_ptr excptr2 = nullptr;
  for (auto _ : state) {
    swap(excptr1, excptr2);
    // The compiler should be able to constant-fold those comparisons
    benchmark::DoNotOptimize(excptr1 == nullptr);
    benchmark::DoNotOptimize(excptr2 == nullptr);
    benchmark::DoNotOptimize(excptr1 == excptr2);
  }
}
BENCHMARK(bm_exception_ptr_swap_null);

BENCHMARK_MAIN();
