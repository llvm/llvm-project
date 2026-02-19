//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <array>
#include <format>
#include <random>

#include "benchmark/benchmark.h"
#include "test_macros.h"

// Tests the full range of the value.
template <class T>
static std::array<T, 1000> generate(std::uniform_int_distribution<T> distribution = std::uniform_int_distribution<T>{
                                        std::numeric_limits<T>::min(), std::numeric_limits<T>::max()}) {
  std::mt19937 generator;
  std::array<T, 1000> result;
  std::generate_n(result.begin(), result.size(), [&] { return distribution(generator); });
  return result;
}

template <class T>
static void BM_Basic(benchmark::State& state) {
  std::array data{generate<T>()};
  std::array<char, 100> output;

  while (state.KeepRunningBatch(data.size()))
    for (auto value : data)
      benchmark::DoNotOptimize(std::format_to(output.begin(), "{}", value));
}
BENCHMARK(BM_Basic<uint32_t>);
BENCHMARK(BM_Basic<int32_t>);
BENCHMARK(BM_Basic<uint64_t>);
BENCHMARK(BM_Basic<int64_t>);

// Ideally the low values of a 128-bit value are all dispatched to a 64-bit routine.
#ifndef TEST_HAS_NO_INT128
template <class T>
static void BM_BasicLow(benchmark::State& state) {
  using U = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
  std::array data{
      generate<T>(std::uniform_int_distribution<T>{std::numeric_limits<U>::min(), std::numeric_limits<U>::max()})};
  std::array<char, 100> output;

  while (state.KeepRunningBatch(data.size()))
    for (auto value : data)
      benchmark::DoNotOptimize(std::format_to(output.begin(), "{}", value));
}
BENCHMARK(BM_BasicLow<__uint128_t>);
BENCHMARK(BM_BasicLow<__int128_t>);

BENCHMARK(BM_Basic<__uint128_t>);
BENCHMARK(BM_Basic<__int128_t>);
#endif

template <class>
inline constexpr std::string to_string = "";

template <>
inline constexpr std::string to_string<int64_t> = "int64_t";

template <>
inline constexpr std::string to_string<uint64_t> = "uint64_t";

int main(int argc, char** argv) {
  auto bm = []<class IntT>(std::type_identity<IntT>, std::string fmt) {
    benchmark::RegisterBenchmark(
        "std::format(" + to_string<IntT> + ") (fmt: " + fmt + ")", [fmt](benchmark::State& state) {
          std::array data = generate<IntT>();
          std::array<char, 512> output;

          while (state.KeepRunningBatch(data.size()))
            for (auto value : data)
              benchmark::DoNotOptimize(std::vformat_to(output.begin(), fmt, std::make_format_args(value)));
        });
  };

  bm(std::type_identity<int64_t>(), "{:b}");
  bm(std::type_identity<int64_t>(), "{:0<512b}");
  bm(std::type_identity<int64_t>(), "{:0^512b}");
  bm(std::type_identity<int64_t>(), "{:0>512b}");
  bm(std::type_identity<int64_t>(), "{:0512b}");

  bm(std::type_identity<int64_t>(), "{:Lb}");
  bm(std::type_identity<int64_t>(), "{:0<512Lb}");
  bm(std::type_identity<int64_t>(), "{:0^512Lb}");
  bm(std::type_identity<int64_t>(), "{:0>512Lb}");
  bm(std::type_identity<int64_t>(), "{:0512Lb}");

  bm(std::type_identity<int64_t>(), "{:o}");
  bm(std::type_identity<int64_t>(), "{:0<512o}");
  bm(std::type_identity<int64_t>(), "{:0^512o}");
  bm(std::type_identity<int64_t>(), "{:0>512o}");
  bm(std::type_identity<int64_t>(), "{:0512o}");

  bm(std::type_identity<int64_t>(), "{:Lo}");
  bm(std::type_identity<int64_t>(), "{:0<512Lo}");
  bm(std::type_identity<int64_t>(), "{:0^512Lo}");
  bm(std::type_identity<int64_t>(), "{:0>512Lo}");
  bm(std::type_identity<int64_t>(), "{:0512Lo}");

  bm(std::type_identity<int64_t>(), "{:d}");
  bm(std::type_identity<int64_t>(), "{:0<512d}");
  bm(std::type_identity<int64_t>(), "{:0^512d}");
  bm(std::type_identity<int64_t>(), "{:0>512d}");
  bm(std::type_identity<int64_t>(), "{:0512d}");

  bm(std::type_identity<int64_t>(), "{:Ld}");
  bm(std::type_identity<int64_t>(), "{:0<512Ld}");
  bm(std::type_identity<int64_t>(), "{:0^512Ld}");
  bm(std::type_identity<int64_t>(), "{:0>512Ld}");
  bm(std::type_identity<int64_t>(), "{:0512Ld}");

  bm(std::type_identity<int64_t>(), "{:x}");
  bm(std::type_identity<int64_t>(), "{:0<512x}");
  bm(std::type_identity<int64_t>(), "{:0^512x}");
  bm(std::type_identity<int64_t>(), "{:0>512x}");
  bm(std::type_identity<int64_t>(), "{:0512x}");

  bm(std::type_identity<int64_t>(), "{:Lx}");
  bm(std::type_identity<int64_t>(), "{:0<512Lx}");
  bm(std::type_identity<int64_t>(), "{:0^512Lx}");
  bm(std::type_identity<int64_t>(), "{:0>512Lx}");
  bm(std::type_identity<int64_t>(), "{:0512Lx}");

  bm(std::type_identity<int64_t>(), "{:X}");
  bm(std::type_identity<int64_t>(), "{:0<512X}");
  bm(std::type_identity<int64_t>(), "{:0^512X}");
  bm(std::type_identity<int64_t>(), "{:0>512X}");
  bm(std::type_identity<int64_t>(), "{:0512X}");

  bm(std::type_identity<int64_t>(), "{:LX}");
  bm(std::type_identity<int64_t>(), "{:0<512LX}");
  bm(std::type_identity<int64_t>(), "{:0^512LX}");
  bm(std::type_identity<int64_t>(), "{:0>512LX}");
  bm(std::type_identity<int64_t>(), "{:0512LX}");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
