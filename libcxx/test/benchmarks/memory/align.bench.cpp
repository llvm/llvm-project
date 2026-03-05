//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <memory>
#include <iostream>

#include "benchmark/benchmark.h"

struct Input {
  std::size_t align;
  std::size_t size;
  void* ptr;
  std::size_t buffer_size;
};

static void BM_align(benchmark::State& state) {
  char buffer[1024];
  Input input{};
  void* ptr               = buffer + 123;
  std::size_t buffer_size = sizeof(buffer) - 123;
  input.align             = state.range();
  input.size              = state.range();
  for (auto _ : state) {
    input.ptr         = ptr;
    input.buffer_size = buffer_size;
    benchmark::DoNotOptimize(input);
    benchmark::DoNotOptimize(std::align(input.align, input.size, input.ptr, input.buffer_size));
  }
}
BENCHMARK(BM_align)->Range(1, 256);

BENCHMARK_MAIN();
