//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <regex>

#include "benchmark/benchmark.h"
#include "GenerateInput.h"

template <std::regex_constants::syntax_option_type Arg>
static void BM_regex_construct(benchmark::State& state) {
  static std::string_view regexes[] = {".*[abcdefghijklmnopqrtuvwxyz]{10,100}", "This is technically a regex."};
  while (state.KeepRunningBatch(std::size(regexes))) {
    for (auto& reg : regexes)
      std::regex r(reg.data(), Arg);
  }
}
BENCHMARK(BM_regex_construct<std::regex::basic>);
BENCHMARK(BM_regex_construct<std::regex::extended>);
BENCHMARK(BM_regex_construct<std::regex::awk>);
BENCHMARK(BM_regex_construct<std::regex::ECMAScript>);

static void BM_regex_run_bad_match(benchmark::State& state) {
  std::regex r("This is technically a regex.");
  std::string input = getRandomString(1 << 16);

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_bad_match);

static void BM_regex_run_almost_match(benchmark::State& state) {
  std::regex r("This is technically a regex.");
  std::string input;
  for (size_t i = 0; i != 2500; ++i)
    input += "This is technically a rege";

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_almost_match);

static void BM_regex_run_any_matcher(benchmark::State& state) {
  std::regex r(".*");
  std::string input;
  input.append(1 << 16, 'a');

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_any_matcher);

BENCHMARK_MAIN();
