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
#include "../../std/containers/from_range_helpers.h"
#include "../GenerateInput.h"
#include "test_iterators.h"

using namespace ContainerBenchmarks;

using vb_iter = std::vector<bool>::iterator;

// Benchmarks for forward_iterator or forward_range

BENCHMARK_CAPTURE(BM_ConstructIterIter<std::vector<bool>>,
                  forward_iterator,
                  getRandomIntegerInputs<bool>,
                  forward_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructFromRange<std::vector<bool>>,
                  forward_range,
                  getRandomIntegerInputs<bool>,
                  forward_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(
    BM_AssignIterIter<std::vector<bool>>, forward_iterator, getRandomIntegerInputs<bool>, forward_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(
    BM_AssignRange<std::vector<bool>>, forward_range, getRandomIntegerInputs<bool>, forward_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_InsertIterIterIter<std::vector<bool>>,
                  forward_iterator,
                  getRandomIntegerInputs<bool>,
                  forward_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(
    BM_InsertRange<std::vector<bool>>, forward_range, getRandomIntegerInputs<bool>, forward_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(
    BM_AppendRange<std::vector<bool>>, forward_range, getRandomIntegerInputs<bool>, forward_range_wrapper<vb_iter>())
    ->Arg(5140480);

// Benchmarks for random_access_iterator or random_access_range

BENCHMARK_CAPTURE(BM_ConstructIterIter<std::vector<bool>>,
                  random_access_iterator,
                  getRandomIntegerInputs<bool>,
                  random_access_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_ConstructFromRange<std::vector<bool>>,
                  random_access_range,
                  getRandomIntegerInputs<bool>,
                  random_access_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_AssignIterIter<std::vector<bool>>,
                  random_access_iterator,
                  getRandomIntegerInputs<bool>,
                  random_access_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_AssignRange<std::vector<bool>>,
                  random_access_range,
                  getRandomIntegerInputs<bool>,
                  random_access_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_InsertIterIterIter<std::vector<bool>>,
                  random_access_iterator,
                  getRandomIntegerInputs<bool>,
                  random_access_iterator<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_InsertRange<std::vector<bool>>,
                  random_access_range,
                  getRandomIntegerInputs<bool>,
                  random_access_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_AppendRange<std::vector<bool>>,
                  random_access_range,
                  getRandomIntegerInputs<bool>,
                  random_access_range_wrapper<vb_iter>())
    ->Arg(5140480);

BENCHMARK_MAIN();
