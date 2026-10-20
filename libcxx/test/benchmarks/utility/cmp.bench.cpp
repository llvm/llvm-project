//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <utility>
#include "../CartesianBenchmarks.h"
#include "benchmark/benchmark.h"

namespace {

enum ValueType : size_t {
  SChar,
  UChar,
  Short,
  UShort,
  Int,
  UInt,
  Long,
  ULong,
  LongLong,
  ULongLong,
#ifndef TEST_HAS_NO_INT128
  Int128,
  UInt128,
#endif
};

struct AllValueTypes : EnumValuesAsTuple<AllValueTypes, ValueType, 6> {
  static constexpr const char* Names[] = {
      "schar",
      "uchar",
      "short",
      "ushort",
      "int",
      "uint",
      "long",
      "ulong",
      "longlong",
      "ulonglong",
#ifndef TEST_HAS_NO_INT128
      "int128",
      "uint128"
#endif
  };
};

using TestType =
    std::tuple< signed char,
                unsigned char,
                short,
                unsigned short,
                int,
                unsigned int,
                long,
                unsigned long,
                long long,
                unsigned long long
#ifndef TEST_HAS_NO_INT128
                ,
                __int128_t,
                __uint128_t
#endif
                >;

template <typename TType, typename UType>
struct CmpEqual {
  static void run(benchmark::State& state) {
    using T = std::tuple_element_t<TType::value, TestType>;
    using U = std::tuple_element_t<UType::value, TestType>;

    T x1 = T{127}, x2 = T{111};
    U y1 = U{123}, y2 = U{1};
    for (auto _ : state) {
      benchmark::DoNotOptimize(x1);
      benchmark::DoNotOptimize(x2);
      benchmark::DoNotOptimize(y1);
      benchmark::DoNotOptimize(y2);
      benchmark::DoNotOptimize(std::cmp_equal(x1, y1));
      benchmark::DoNotOptimize(std::cmp_equal(y1, x1));
      benchmark::DoNotOptimize(std::cmp_equal(x1, x1));
      benchmark::DoNotOptimize(std::cmp_equal(y1, y1));

      benchmark::DoNotOptimize(std::cmp_equal(x2, y2));
      benchmark::DoNotOptimize(std::cmp_equal(y2, x2));
      benchmark::DoNotOptimize(std::cmp_equal(x2, x2));
      benchmark::DoNotOptimize(std::cmp_equal(y2, y2));
    }
  }

  static std::string name() { return "BM_CmpEqual" + TType::name() + UType::name(); }
};

template <typename TType, typename UType>
struct CmpLess {
  static void run(benchmark::State& state) {
    using T = std::tuple_element_t<TType::value, TestType>;
    using U = std::tuple_element_t<UType::value, TestType>;

    T x1 = T{127}, x2 = T{111};
    U y1 = U{123}, y2 = U{1};
    for (auto _ : state) {
      benchmark::DoNotOptimize(x1);
      benchmark::DoNotOptimize(x2);
      benchmark::DoNotOptimize(y1);
      benchmark::DoNotOptimize(y2);
      benchmark::DoNotOptimize(std::cmp_less(x1, y1));
      benchmark::DoNotOptimize(std::cmp_less(y1, x1));
      benchmark::DoNotOptimize(std::cmp_less(x1, x1));
      benchmark::DoNotOptimize(std::cmp_less(y1, y1));

      benchmark::DoNotOptimize(std::cmp_less(x2, y2));
      benchmark::DoNotOptimize(std::cmp_less(y2, x2));
      benchmark::DoNotOptimize(std::cmp_less(x2, x2));
      benchmark::DoNotOptimize(std::cmp_less(y2, y2));
    }
  }

  static std::string name() { return "BM_CmpLess" + TType::name() + UType::name(); }
};

} // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  makeCartesianProductBenchmark<CmpEqual, AllValueTypes, AllValueTypes>();
  makeCartesianProductBenchmark<CmpLess, AllValueTypes, AllValueTypes>();
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
