//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cstdint>
#include <cstddef>
#include <functional>

#include "benchmark/benchmark.h"

#include "GenerateInput.h"
#include "test_macros.h"

constexpr std::size_t TestNumInputs = 1024;

template <class HashFn, class GenInputs>
void BM_Hash(benchmark::State& st, HashFn fn, GenInputs gen) {
  auto in               = gen(st.range(0));
  const auto end        = in.data() + in.size();
  std::size_t last_hash = 0;
  benchmark::DoNotOptimize(&last_hash);
  while (st.KeepRunning()) {
    for (auto it = in.data(); it != end; ++it) {
      benchmark::DoNotOptimize(last_hash += fn(*it));
    }
    benchmark::ClobberMemory();
  }
}

BENCHMARK_CAPTURE(BM_Hash, uint32_random_std_hash, std::hash<uint32_t>{}, getRandomIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Hash, uint32_top_std_hash, std::hash<uint32_t>{}, getSortedTopBitsIntegerInputs<uint32_t>)
    ->Arg(TestNumInputs);

BENCHMARK_MAIN();
