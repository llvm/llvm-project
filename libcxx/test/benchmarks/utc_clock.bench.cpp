//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

#include <chrono>

#include "benchmark/benchmark.h"

// Benchmarks the performance of the UTC <-> system time conversions. These
// operations determine the sum of leap second insertions at a specific time.

static void BM_from_sys(benchmark::State& state) {
  std::chrono::sys_days date{std::chrono::July / 1 / state.range(0)};
  for (auto _ : state)
    benchmark::DoNotOptimize(std::chrono::utc_clock::from_sys(date));
}

BENCHMARK(BM_from_sys)
    ->Arg(1970)  // before the first leap seconds
    ->Arg(1979)  // in the first half of inserted leap seconds
    ->Arg(1993)  // in the second half of inserted leap seconds
    ->Arg(2100); // after the last leap second

BENCHMARK(BM_from_sys)->Arg(1970)->Arg(1979)->Arg(1993)->Arg(2100)->Threads(4);
BENCHMARK(BM_from_sys)->Arg(1970)->Arg(1979)->Arg(1993)->Arg(2100)->Threads(16);

static void BM_to_sys(benchmark::State& state) {
  // 59 sec offset means we pass th UTC offset for the leap second; assuming
  // there won't be more than 59 leap seconds ever.
  std::chrono::utc_seconds date{
      std::chrono::sys_days{std::chrono::July / 1 / state.range(0)}.time_since_epoch() + std::chrono::seconds{59}};
  for (auto _ : state)
    benchmark::DoNotOptimize(std::chrono::utc_clock::to_sys(date));
}

BENCHMARK(BM_to_sys)
    ->Arg(1970)  // before the first leap seconds
    ->Arg(1979)  // in the first half of inserted leap seconds
    ->Arg(1993)  // in the second half of inserted leap seconds
    ->Arg(2100); // after the last leap second

BENCHMARK(BM_to_sys)->Arg(1970)->Arg(1979)->Arg(1993)->Arg(2100)->Threads(4);
BENCHMARK(BM_to_sys)->Arg(1970)->Arg(1979)->Arg(1993)->Arg(2100)->Threads(16);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}
