//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <deque>
#include <string>

#include "benchmark/benchmark.h"

#include "ContainerBenchmarks.h"
#include "../GenerateInput.h"

using namespace ContainerBenchmarks;

constexpr std::size_t TestNumInputs = 1024;

BENCHMARK_CAPTURE(BM_ConstructSize, deque_byte, std::deque<unsigned char>{})->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructSizeValue, deque_byte, std::deque<unsigned char>{}, 0)->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructIterIter<std::deque<char>>, deque_char, getRandomIntegerInputs<char>)->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter<std::deque<size_t>>, deque_size_t, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructIterIter<std::deque<std::string>>, deque_string, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange<std::deque<char>>, deque_char, getRandomIntegerInputs<char>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange<std::deque<size_t>>, deque_size_t, getRandomIntegerInputs<size_t>)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_ConstructFromRange<std::deque<std::string>>, deque_string, getRandomStringInputs)
    ->Arg(TestNumInputs);

BENCHMARK_CAPTURE(BM_erase_iter_in_middle, deque_int, std::deque<int>{}, getRandomIntegerInputs<int>)
    ->Range(TestNumInputs, TestNumInputs * 10);
BENCHMARK_CAPTURE(BM_erase_iter_in_middle, deque_string, std::deque<std::string>{}, getRandomStringInputs)
    ->Range(TestNumInputs, TestNumInputs * 10);

BENCHMARK_CAPTURE(BM_erase_iter_at_start, deque_int, std::deque<int>{}, getRandomIntegerInputs<int>)
    ->Range(TestNumInputs, TestNumInputs * 10);
BENCHMARK_CAPTURE(BM_erase_iter_at_start, deque_string, std::deque<std::string>{}, getRandomStringInputs)
    ->Range(TestNumInputs, TestNumInputs * 10);

BENCHMARK_MAIN();
