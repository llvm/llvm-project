//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <string>
#include <vector>

#include "sequence_container_benchmarks.h"
#include "benchmark/benchmark.h"

struct NothrowMoveConstructible {
  std::string s;
};

template <>
struct Generate<NothrowMoveConstructible> {
  static NothrowMoveConstructible arbitrary() { return {Generate<std::string>::arbitrary()}; }
  static NothrowMoveConstructible cheap() { return {Generate<std::string>::cheap()}; }
  static NothrowMoveConstructible expensive() { return {Generate<std::string>::expensive()}; }
  static NothrowMoveConstructible random() { return {Generate<std::string>::random()}; }
};

int main(int argc, char** argv) {
  support::sequence_container_benchmarks<std::vector<int>>("std::vector<int>");
  support::sequence_container_benchmarks<std::vector<std::string>>("std::vector<std::string>");
  support::sequence_container_benchmarks<std::vector<NothrowMoveConstructible>>(
      "std::vector<NothrowMoveConstructible>");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
