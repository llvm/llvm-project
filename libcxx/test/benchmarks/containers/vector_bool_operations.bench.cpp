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
#include "test_allocator.h"

using namespace ContainerBenchmarks;

BENCHMARK_CAPTURE(BM_CopyConstruct, vector_bool, std::vector<bool>{})->Arg(5140480);
BENCHMARK_CAPTURE(BM_MoveConstruct, vector_bool, std::vector<bool>{})->Arg(5140480);
BENCHMARK_CAPTURE(
    BM_CopyConstruct_Alloc, vector_bool, std::vector<bool, test_allocator<bool>>(), test_allocator<bool>(3))
    ->Arg(5140480);
BENCHMARK_CAPTURE(
    BM_MoveConstruct_Alloc, vector_bool, std::vector<bool, test_allocator<bool>>(), test_allocator<bool>(3))
    ->Arg(5140480);

BENCHMARK_MAIN();