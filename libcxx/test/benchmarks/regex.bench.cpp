//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <regex>

#include "benchmark/benchmark.h"
#include "GenerateInput.h"
#include "test_macros.h"

template <std::regex_constants::syntax_option_type Arg>
TEST_ALIGN_BENCHMARK static void BM_regex_construct(benchmark::State& state) {
  static std::string_view regexes[] = {".*[abcdefghijklmnopqrtuvwxyz]{10,100}", "This is technically a regex."};
  while (state.KeepRunningBatch(std::size(regexes))) {
    for (auto& reg : regexes)
      std::regex r(reg.data(), Arg);
  }
}
BENCHMARK(BM_regex_construct<std::regex::basic>)->Name("std::regex(std::regex::basic)");
BENCHMARK(BM_regex_construct<std::regex::extended>)->Name("std::regex(std::regex::extended)");
BENCHMARK(BM_regex_construct<std::regex::awk>)->Name("std::regex(std::regex::awk)");
BENCHMARK(BM_regex_construct<std::regex::ECMAScript>)->Name("std::regex(std::regex::ECMAScript)");

TEST_ALIGN_BENCHMARK static void BM_regex_run_bad_match(benchmark::State& state) {
  std::regex r("This is technically a regex.");
  std::string input = getRandomString(1 << 16);

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_bad_match)->Name("std::regex_search(\"This is technically a regex.\") (random data)");

TEST_ALIGN_BENCHMARK static void BM_regex_run_almost_match(benchmark::State& state) {
  std::regex r("This is technically a regex.");
  std::string input;
  for (size_t i = 0; i != 2500; ++i)
    input += "This is technically a rege";

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_almost_match)->Name("std::regex_search(\"This is technically a regex.\") (almost matches)");

TEST_ALIGN_BENCHMARK static void BM_regex_run_any_matcher(benchmark::State& state) {
  std::regex r(".*");
  std::string input;
  input.append(1 << 16, 'a');

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_any_matcher)->Name("std::regex_search(\".*\")");

TEST_ALIGN_BENCHMARK static void BM_regex_run_alphabet_matcher(benchmark::State& state) {
  std::regex r("[a-zA-Z]*");
  std::string input;
  input.append(1 << 16, 'a');

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_alphabet_matcher)->Name("std::regex_search(\"[a-zA-Z]*\")");

TEST_ALIGN_BENCHMARK static void BM_regex_run_alternation_loop_matcher(benchmark::State& state) {
  std::regex r("(a|b)*c");
  std::string input;
  input.append(1 << 8, 'a');

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_alternation_loop_matcher)->Name("std::regex_search(\"(a|b)*c\")");

TEST_ALIGN_BENCHMARK static void BM_regex_run_mail_matcher(benchmark::State& state) {
  std::regex r(
      R"regex((?:(?:[^<>()\[\].,;:\s@"]+(?:\.[^<>()\[\].,;:\s@"]+)*)|".+")@(?:(?:[^<>()\[\].,;:\s@"]+\.)+[^<>()\[\].,;:\s@"]{2,}))regex");
  std::string input;
  input.append(1 << 8, 'a');

  for (auto _ : state) {
    std::regex_search(input, r);
  }
}
BENCHMARK(BM_regex_run_mail_matcher)->Name("std::regex_search(E-Mail matcher)");

BENCHMARK_MAIN();
