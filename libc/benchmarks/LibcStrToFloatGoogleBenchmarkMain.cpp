//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Benchmarks for string to floating point conversion.
///
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "src/__support/str_to_float.h"

#include <stddef.h>
#include <stdint.h>

namespace {

// Normals.
constexpr const char *NORMAL_SHORT[] = {
    "0", "1", "1.5", "-3.75", "0.1", "12.5", "3.14159", "100", "1e10", "1e-10",
};

// Normals with enough digits or exponent to reach the fallback.
constexpr const char *NORMAL_LONG[] = {
    "1.00000000000000000001",         "0.99999999999999999999",
    "123456789012345678901234567890", "1.7976931348623157e308",
    "2.2250738585072014e-308",
};

// Subnormals.
constexpr const char *SUBNORMAL_SHORT[] = {
    "5e-324", "1e-320", "2.5e-320", "1e-315", "3.33e-318", "7e-309",
};

// Subnormals with many digits.
constexpr const char *SUBNORMAL_LONG[] = {
    "4.940656458412465441765687928682213723651e-324",
    "1.797693134862315708145274237317043567981e-320",
};

// Subnormal as a float; the doubles above underflow to zero here.
constexpr const char *FLOAT_SUBNORMAL[] = {
    "1e-38", "5e-39", "1e-40", "1e-42", "1e-44", "1.4012984643e-45",
};

template <typename T, size_t N>
void run(benchmark::State &state, const char *const (&inputs)[N]) {
  for (auto _ : state) {
    for (const char *input : inputs)
      benchmark::DoNotOptimize(
          LIBC_NAMESPACE::internal::strtofloatingpoint<T>(input));
  }
  state.SetItemsProcessed(state.iterations() * N);
}

void BM_StrToDoubleNormalShort(benchmark::State &state) {
  run<double>(state, NORMAL_SHORT);
}
void BM_StrToDoubleNormalLong(benchmark::State &state) {
  run<double>(state, NORMAL_LONG);
}
void BM_StrToDoubleSubnormalShort(benchmark::State &state) {
  run<double>(state, SUBNORMAL_SHORT);
}
void BM_StrToDoubleSubnormalLong(benchmark::State &state) {
  run<double>(state, SUBNORMAL_LONG);
}

void BM_StrToFloatNormalShort(benchmark::State &state) {
  run<float>(state, NORMAL_SHORT);
}
void BM_StrToFloatSubnormalShort(benchmark::State &state) {
  run<float>(state, SUBNORMAL_SHORT);
}
void BM_StrToFloatSubnormal(benchmark::State &state) {
  run<float>(state, FLOAT_SUBNORMAL);
}

} // namespace

BENCHMARK(BM_StrToDoubleNormalShort);
BENCHMARK(BM_StrToDoubleNormalLong);
BENCHMARK(BM_StrToDoubleSubnormalShort);
BENCHMARK(BM_StrToDoubleSubnormalLong);
BENCHMARK(BM_StrToFloatNormalShort);
BENCHMARK(BM_StrToFloatSubnormalShort);
BENCHMARK(BM_StrToFloatSubnormal);
