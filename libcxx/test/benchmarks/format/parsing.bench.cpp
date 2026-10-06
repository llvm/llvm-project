//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This benchmark does formatting but the main goal is to test the parsing.
// Calling std::format without formatting args may me somewhat unrealistc,
// however it is realistic for std::print. Unfortunately std::print is hard
// to benchmark since the writing to the terminal is not what is inteded to
// be measured.

#include <cstdio>
#include <format>
#include <array>
#include <string>
#include <string_view>

#include "benchmark/benchmark.h"

// No prefix

static void BM_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_empty);

static void BM_curly_open(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("{{");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_curly_open);

static void BM_curly_close(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("}}");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_curly_close);

static void BM_pipe(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("||");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_pipe);

static void BM_int(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("{}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_int);

static void BM_int_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_int_formatted);

static void BM_3_ints(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("{},{},{}", 42, 0, 99);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_3_ints);

static void BM_3_ints_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("{0:-^#6x},{0:-^#6x},{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_3_ints_formatted);

// Prefix 5

static void BM_prefix_5_and_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_empty);

static void BM_prefix_5_and_curly_open(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'{{");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_curly_open);

static void BM_prefix_5_and_curly_close(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'}}");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_curly_close);

static void BM_prefix_5_and_pipe(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'||");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_pipe);

static void BM_prefix_5_and_int(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'{}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_int);

static void BM_prefix_5_and_int_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_int_formatted);

static void BM_prefix_5_and_3_ints(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'{},{},{}", 42, 0, 99);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_3_ints);

static void BM_prefix_5_and_3_ints_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'{0:-^#6x},{0:-^#6x},{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_5_and_3_ints_formatted);

// Prefix 10

static void BM_prefix_10_and_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_empty);

static void BM_prefix_10_and_curly_open(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'{{");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_curly_open);

static void BM_prefix_10_and_curly_close(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'}}");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_curly_close);

static void BM_prefix_10_and_pipe(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'||");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_pipe);

static void BM_prefix_10_and_int(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'{}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_int);

static void BM_prefix_10_and_int_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_int_formatted);

static void BM_prefix_10_and_3_ints(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'{},{},{}", 42, 0, 99);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_3_ints);

static void BM_prefix_10_and_3_ints_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'{0:-^#6x},{0:-^#6x},{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_10_and_3_ints_formatted);

// Prefix 20

static void BM_prefix_20_and_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_empty);

static void BM_prefix_20_and_curly_open(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'{{");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_curly_open);

static void BM_prefix_20_and_curly_close(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'}}");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_curly_close);

static void BM_prefix_20_and_pipe(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'||");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_pipe);

static void BM_prefix_20_and_int(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'{}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_int);

static void BM_prefix_20_and_int_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_int_formatted);

static void BM_prefix_20_and_3_ints(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'{},{},{}", 42, 0, 99);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_3_ints);

static void BM_prefix_20_and_3_ints_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'{0:-^#6x},{0:-^#6x},{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_20_and_3_ints_formatted);

// Prefix 40

static void BM_prefix_40_and_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_empty);

static void BM_prefix_40_and_curly_open(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'{{");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_curly_open);

static void BM_prefix_40_and_curly_close(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'}}");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_curly_close);

static void BM_prefix_40_and_pipe(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'||");
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_pipe);

static void BM_prefix_40_and_int(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'{}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_int);

static void BM_prefix_40_and_int_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_int_formatted);

static void BM_prefix_40_and_3_ints(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'{},{},{}", 42, 0, 99);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_3_ints);

static void BM_prefix_40_and_3_ints_formatted(benchmark::State& state) {
  for (auto _ : state) {
    auto s = std::format("aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'{0:-^#6x},{0:-^#6x},{0:-^#6x}", 42);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_prefix_40_and_3_ints_formatted);

BENCHMARK_MAIN();
