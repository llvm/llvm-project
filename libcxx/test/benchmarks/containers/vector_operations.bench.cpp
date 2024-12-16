//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "ContainerBenchmarks.h"
#include "../GenerateInput.h"

using namespace ContainerBenchmarks;

constexpr std::size_t TestNumInputs = 1024;

BENCHMARK_CAPTURE(BM_ConstructSize, vector_byte, std::vector<unsigned char>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_CopyConstruct, vector_int, std::vector<int>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_Assignment, vector_int, std::vector<int>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructSizeValue, vector_byte, std::vector<unsigned char>{}, 0)->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_char, std::vector<char>{}, getRandomIntegerInputs<char>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_size_t, std::vector<size_t>{}, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_char, std::vector<char>{}, getRandomIntegerInputs<char>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_size_t, std::vector<size_t>{}, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_Pushback_no_grow, vector_int, std::vector<int>{})->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_erase_iter_in_middle, vector_int, std::vector<int>{}, getRandomIntegerInputs<int>)
    ->Range(TestNumInputs, TestNumInputs * 10);
BENCHMARK_CAPTURE(BM_erase_iter_in_middle, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Range(TestNumInputs, TestNumInputs * 10);

BENCHMARK_CAPTURE(BM_erase_iter_at_start, vector_int, std::vector<int>{}, getRandomIntegerInputs<int>)
    ->Range(TestNumInputs, TestNumInputs * 10);
BENCHMARK_CAPTURE(BM_erase_iter_at_start, vector_string, std::vector<std::string>{}, getRandomStringInputs)
    ->Range(TestNumInputs, TestNumInputs * 10);

template <class T>
void bm_grow(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<T> vec;
    benchmark::DoNotOptimize(vec);
    for (size_t i = 0; i != 2048; ++i)
      vec.emplace_back();
    benchmark::DoNotOptimize(vec);
  }
}
BENCHMARK(bm_grow<int>);
BENCHMARK(bm_grow<std::string>);
BENCHMARK(bm_grow<std::unique_ptr<int>>);
BENCHMARK(bm_grow<std::deque<int>>);

BENCHMARK_CAPTURE(BM_AssignInputIterIter, vector_int, std::vector<int>{}, getRandomIntegerInputs<int>)
    ->Args({TestNumInputs, TestNumInputs});

BENCHMARK_CAPTURE(
    BM_AssignInputIterIter<32>, vector_string, std::vector<std::string>{}, getRandomStringInputsWithLength)
    ->Args({TestNumInputs, TestNumInputs});

BENCHMARK_CAPTURE(BM_AssignInputIterIter<100>,
                  vector_vector_int,
                  std::vector<std::vector<int>>{},
                  getRandomIntegerInputsWithLength<int>)
    ->Args({TestNumInputs, TestNumInputs});

BENCHMARK_MAIN();
