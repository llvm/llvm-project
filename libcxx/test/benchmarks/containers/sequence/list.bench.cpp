//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <list>
#include <string>

#include "sequence_container_benchmarks.h"
#include "benchmark/benchmark.h"

int main(int argc, char** argv) {
  support::sequence_container_benchmarks<std::list<int>>("std::list<int>");
  support::sequence_container_benchmarks<std::list<std::string>>("std::list<std::string>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
