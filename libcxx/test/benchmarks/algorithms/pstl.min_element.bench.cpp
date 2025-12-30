//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

struct Seq {
  static constexpr const auto& policy() { return std::execution::seq; }
};
struct Unseq {
  static constexpr const auto& policy() { return std::execution::unseq; }
};
struct Par {
  static constexpr const auto& policy() { return std::execution::par; }
};
struct ParUnseq {
  static constexpr const auto& policy() { return std::execution::par_unseq; }
};

struct MinFirst {
  static size_t pos(size_t) { return 0; }
};
struct MinMiddle {
  static size_t pos(size_t size) { return size / 2; }
};
struct MinLast {
  static size_t pos(size_t size) { return size - 1; }
};

void run_sizes(auto benchmark) { benchmark->Arg(64)->Arg(512)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(262144); }

template <class T, class Policy, class Position>
void BM_min_element(benchmark::State& state) {
  std::vector<T> vec(state.range(), 3);
  vec[Position::pos(vec.size())] = 1;

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    benchmark::DoNotOptimize(std::min_element(Policy::policy(), vec.begin(), vec.end()));
  }
}

struct Point3D {
  double x, y, z;
  Point3D(double v = 0.0) : x(v), y(v), z(v) {}
};

struct ExpensiveComparator {
  bool operator()(const Point3D& a, const Point3D& b) const {
    double dist_a = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    double dist_b = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    return dist_a < dist_b;
  }
};

template <class Policy, class Position>
void BM_min_element_expensive(benchmark::State& state) {
  std::vector<Point3D> vec(state.range(), Point3D(3.0));
  vec[Position::pos(vec.size())] = Point3D(0.5);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    benchmark::DoNotOptimize(std::min_element(Policy::policy(), vec.begin(), vec.end(), ExpensiveComparator{}));
  }
}

template <class Policy, class Position>
void BM_min_element_non_trivially_copyable(benchmark::State& state) {
  std::vector<std::string> vec(state.range(), "3");
  vec[Position::pos(vec.size())] = "1";

  for (auto _ : state) {
    benchmark::DoNotOptimize(vec);
    benchmark::DoNotOptimize(std::min_element(Policy::policy(), vec.begin(), vec.end()));
  }
}

#define BENCH_ALL_POLICIES(BenchFunc, Pos)                                                                             \
  BENCHMARK(BenchFunc<Seq, Pos>)->Apply(run_sizes);                                                                    \
  BENCHMARK(BenchFunc<Unseq, Pos>)->Apply(run_sizes);                                                                  \
  BENCHMARK(BenchFunc<Par, Pos>)->Apply(run_sizes);                                                                    \
  BENCHMARK(BenchFunc<ParUnseq, Pos>)->Apply(run_sizes)

#define BENCH_ALL_POLICIES_T(BenchFunc, Type, Pos)                                                                     \
  BENCHMARK(BenchFunc<Type, Seq, Pos>)->Apply(run_sizes);                                                              \
  BENCHMARK(BenchFunc<Type, Unseq, Pos>)->Apply(run_sizes);                                                            \
  BENCHMARK(BenchFunc<Type, Par, Pos>)->Apply(run_sizes);                                                              \
  BENCHMARK(BenchFunc<Type, ParUnseq, Pos>)->Apply(run_sizes)

BENCH_ALL_POLICIES_T(BM_min_element, int, MinFirst);
BENCH_ALL_POLICIES_T(BM_min_element, int, MinMiddle);
BENCH_ALL_POLICIES_T(BM_min_element, int, MinLast);

BENCH_ALL_POLICIES(BM_min_element_expensive, MinFirst);
BENCH_ALL_POLICIES(BM_min_element_expensive, MinMiddle);
BENCH_ALL_POLICIES(BM_min_element_expensive, MinLast);

BENCH_ALL_POLICIES(BM_min_element_non_trivially_copyable, MinFirst);
BENCH_ALL_POLICIES(BM_min_element_non_trivially_copyable, MinMiddle);
BENCH_ALL_POLICIES(BM_min_element_non_trivially_copyable, MinLast);

BENCHMARK_MAIN();
