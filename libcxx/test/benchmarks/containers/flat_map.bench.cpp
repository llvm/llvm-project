//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

#include <flat_map>
#include <string>

#include "associative_container_benchmarks.h"
#include "benchmark/benchmark.h"

int main(int argc, char** argv) {
  support::associative_container_benchmarks<std::flat_map<int, int>>("std::flat_map<int, int>");
  support::associative_container_benchmarks<std::flat_map<std::string, int>>("std::flat_map<std::string, int>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
