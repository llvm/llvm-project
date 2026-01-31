//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <array>
#include <variant>

#include "benchmark/benchmark.h"
#include "GenerateInput.h"

template <std::size_t I>
struct S {
  static constexpr size_t v = I;
};

template <std::size_t N, std::size_t... Is>
static auto genVariants(std::index_sequence<Is...>) {
  using V                 = std::variant<S<Is>...>;
  using F                 = V (*)();
  static constexpr F fs[] = {[] { return V(std::in_place_index<Is>); }...};

  std::array<V, N> result = {};
  for (auto& v : result) {
    v = fs[getRandomInteger(std::size_t(0), sizeof...(Is) - 1)]();
  }

  return result;
}

template <std::size_t N, std::size_t Alts>
static void BM_Visit(benchmark::State& state) {
  auto args = genVariants<N>(std::make_index_sequence<Alts>{});
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        std::apply([](auto... vs) { return std::visit([](auto... is) { return (is.v + ... + 0); }, vs...); }, args));
  }
}

BENCHMARK(BM_Visit<1, 1>)->Name("std::variant<1-alt>::visit() (1 variant)");
BENCHMARK(BM_Visit<1, 8>)->Name("std::variant<8-alts>::visit() (1 variant)");
BENCHMARK(BM_Visit<1, 100>)->Name("std::variant<100-alts>::visit() (1 variant)");
BENCHMARK(BM_Visit<2, 1>)->Name("std::variant<1-alt>::visit() (2 variants)");
BENCHMARK(BM_Visit<2, 8>)->Name("std::variant<8-alts>::visit() (2 variants)");
BENCHMARK(BM_Visit<2, 50>)->Name("std::variant<50-alts>::visit() (2 variants)");
BENCHMARK(BM_Visit<3, 1>)->Name("std::variant<1-alt>::visit() (3 variants)");
BENCHMARK(BM_Visit<3, 8>)->Name("std::variant<8-alts>::visit() (3 variants)");
BENCHMARK(BM_Visit<3, 20>)->Name("std::variant<20-alts>::visit() (3 variants)");

BENCHMARK_MAIN();
