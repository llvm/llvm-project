//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <format>

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <string>

#include "benchmark/benchmark.h"

template <class>
inline constexpr std::string to_string = "";

template <>
inline constexpr std::string to_string<float> = "float";

template <>
inline constexpr std::string to_string<double> = "double";

enum class ValueE { Inf, Random };

int main(int argc, char** argv) {
  auto bm = []<class FloatingPoint>(std::type_identity<FloatingPoint>, ValueE v, std::string fmt) {
    benchmark::RegisterBenchmark(
        "std::format(" + to_string<FloatingPoint> + ") (fmt: " + fmt + ")", [fmt, v](benchmark::State& state) {
          std::array<FloatingPoint, 1000> data = [&] {
            std::array<FloatingPoint, 1000> result;
            if (v == ValueE::Inf) {
              std::fill(result.begin(), result.end(), -std::numeric_limits<FloatingPoint>::infinity());
            } else {
              std::mt19937 generator(123456);
              std::uniform_int_distribution<
                  std::conditional_t<sizeof(FloatingPoint) == sizeof(uint32_t), uint32_t, uint64_t>>
                  distribution;

              std::generate(result.begin(), result.end(), [&] {
                while (true) {
                  auto val = std::bit_cast<FloatingPoint>(distribution(generator));
                  if (std::isfinite(val))
                    return val;
                }
              });
            }
            return result;
          }();
          std::array<char, 20'000> output;

          while (state.KeepRunningBatch(1000))
            for (auto value : data)
              benchmark::DoNotOptimize(std::vformat_to(output.begin(), fmt, std::make_format_args(value)));
        });
  };

  bm(std::type_identity<float>(), ValueE::Inf, "{:.0}");
  bm(std::type_identity<float>(), ValueE::Random, "{:.0}");
  bm(std::type_identity<float>(), ValueE::Inf, "{:0<17500.0}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0}");
  bm(std::type_identity<float>(), ValueE::Inf, "{:0^17500.0}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0}");
  bm(std::type_identity<float>(), ValueE::Inf, "{:0>17500.0}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0}");
  bm(std::type_identity<float>(), ValueE::Inf, "{:017500.0}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.17000}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.17000}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.17000}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.17000}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.17000}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0L}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10L}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.17000L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.17000L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.17000L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.17000L}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.17000L}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0a}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10a}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10a}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0La}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10La}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10La}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0e}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10e}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10e}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0Le}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10Le}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10Le}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0f}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10f}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10f}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0Lf}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10Lf}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10Lf}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0g}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10g}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10g}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.0Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.0Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.0Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.0Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.0Lg}");

  bm(std::type_identity<float>(), ValueE::Random, "{:.10Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0<17500.10Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0^17500.10Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:0>17500.10Lg}");
  bm(std::type_identity<float>(), ValueE::Random, "{:017500.10Lg}");

  bm(std::type_identity<double>(), ValueE::Inf, "{:.0}");
  bm(std::type_identity<double>(), ValueE::Random, "{:.0}");
  bm(std::type_identity<double>(), ValueE::Inf, "{:0<17500.0}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0}");
  bm(std::type_identity<double>(), ValueE::Inf, "{:0^17500.0}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0}");
  bm(std::type_identity<double>(), ValueE::Inf, "{:0>17500.0}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0}");
  bm(std::type_identity<double>(), ValueE::Inf, "{:017500.0}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.17000}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.17000}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.17000}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.17000}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.17000}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0L}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10L}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.17000L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.17000L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.17000L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.17000L}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.17000L}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0a}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10a}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10a}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0La}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10La}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10La}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0e}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10e}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10e}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0Le}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10Le}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10Le}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0f}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10f}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10f}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0Lf}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10Lf}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10Lf}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0g}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10g}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10g}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.0Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.0Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.0Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.0Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.0Lg}");

  bm(std::type_identity<double>(), ValueE::Random, "{:.10Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0<17500.10Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0^17500.10Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:0>17500.10Lg}");
  bm(std::type_identity<double>(), ValueE::Random, "{:017500.10Lg}");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
