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
#include "sized_allocator.h"
#include "test_allocator.h"

using namespace ContainerBenchmarks;

BENCHMARK_CAPTURE(BM_Move_Assignment,
                  vector_bool_uint32_t,
                  std::vector<bool, sized_allocator<bool, std::uint32_t, std::int32_t>>{},
                  sized_allocator<bool, std::uint32_t, std::int32_t>{})
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_Move_Assignment,
                  vector_bool_uint64_t,
                  std::vector<bool, sized_allocator<bool, std::uint64_t, std::int64_t>>{},
                  sized_allocator<bool, std::uint64_t, std::int64_t>{})
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_Move_Assignment,
                  vector_bool_size_t,
                  std::vector<bool, sized_allocator<bool, std::size_t, std::ptrdiff_t>>{},
                  sized_allocator<bool, std::size_t, std::ptrdiff_t>{})
    ->Arg(5140480);

BENCHMARK_MAIN();
