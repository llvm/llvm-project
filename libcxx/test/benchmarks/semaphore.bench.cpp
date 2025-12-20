//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -O3

#include <cstdint>
#include <optional>
#include <semaphore>
#include <thread>

#include "benchmark/benchmark.h"
#include "make_test_thread.h"



void BM_semaphore_timed_acquire(benchmark::State& state) {
}