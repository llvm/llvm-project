//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "test_macros.h"

// Each input is an exact token: [first, last) holds the digits and nothing else, as when parsing a field that has
// already been split out of a line. The number of digits is drawn uniformly from [MinDigits, MaxDigits], so the
// loop exit is not predictable when MinDigits != MaxDigits.
struct Inputs {
  std::string text;
  std::vector<std::size_t> offsets; // offsets[i] .. offsets[i + 1] is token i
};

template <int Base>
static Inputs make_inputs(int min_digits, int max_digits) {
  std::mt19937_64 generator;
  std::uniform_int_distribution<int> digits(min_digits, max_digits);
  Inputs result;
  result.offsets.push_back(0);
  char buffer[64];
  for (int i = 0; i != 10000; ++i) {
    int n = digits(generator);
    // Smallest and largest value with exactly n digits, capped at the largest uint64_t.
    std::uint64_t low = 1;
    for (int j = 1; j != n; ++j)
      low *= Base;
    std::uint64_t high =
        low <= std::numeric_limits<std::uint64_t>::max() / Base
            ? low * Base - 1
            : std::numeric_limits<std::uint64_t>::max();
    std::uint64_t value = std::uniform_int_distribution<std::uint64_t>(low, high)(generator);
    char* end           = std::to_chars(buffer, buffer + sizeof(buffer), value, Base).ptr;
    result.text.append(buffer, end);
    result.offsets.push_back(result.text.size());
  }
  return result;
}

template <int Base, int MinDigits, int MaxDigits>
static TEST_ALIGN_BENCHMARK void BM_from_chars(benchmark::State& state) {
  static const Inputs inputs = make_inputs<Base>(MinDigits, MaxDigits);
  const char* text           = inputs.text.data();
  std::size_t count          = inputs.offsets.size() - 1;
  while (state.KeepRunningBatch(count))
    for (std::size_t i = 0; i != count; ++i) {
      std::uint64_t value;
      benchmark::DoNotOptimize(std::from_chars(text + inputs.offsets[i], text + inputs.offsets[i + 1], value, Base));
      benchmark::DoNotOptimize(value);
    }
}
BENCHMARK_TEMPLATE(BM_from_chars, 10, 1, 1);   // base 10, one digit
BENCHMARK_TEMPLATE(BM_from_chars, 10, 1, 6);   // base 10, short
BENCHMARK_TEMPLATE(BM_from_chars, 10, 1, 20);  // base 10, mixed
BENCHMARK_TEMPLATE(BM_from_chars, 10, 8, 8);   // base 10, eight digits
BENCHMARK_TEMPLATE(BM_from_chars, 10, 19, 20); // base 10, long
BENCHMARK_TEMPLATE(BM_from_chars, 16, 1, 16);  // base 16, mixed
BENCHMARK_TEMPLATE(BM_from_chars, 16, 16, 16); // base 16, long

BENCHMARK_MAIN();
