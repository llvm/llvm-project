
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <istream>
#include <sstream>

#include <benchmark/benchmark.h>

void BM_getline_string(benchmark::State& state) {
  std::istringstream iss;

  std::string str;
  str.reserve(128);
  iss.str("A long string to let getline do some more work, making sure that longer strings are parsed fast enough");

  for (auto _ : state) {
    benchmark::DoNotOptimize(iss);

    std::getline(iss, str);
    benchmark::DoNotOptimize(str);
    iss.seekg(0);
  }
}

BENCHMARK(BM_getline_string);

BENCHMARK_MAIN();
