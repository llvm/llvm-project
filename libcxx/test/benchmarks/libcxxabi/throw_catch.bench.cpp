//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cstddef>

#include "benchmark/benchmark.h"
#include "test_macros.h"

template <std::size_t Depth>
struct Chain : Chain<Depth - 1> {};

template <>
struct Chain<0> {
  Chain()                        = default;
  Chain(const Chain&)            = default;
  Chain& operator=(const Chain&) = default;
  virtual ~Chain() noexcept      = default;
};

template <std::size_t Index, std::size_t Depth>
struct Dag : Dag<Index, Depth - 1>, Dag<Index + 1, Depth - 1> {};

template <std::size_t Index>
struct Dag<Index, 0> {
  Dag()                      = default;
  Dag(const Dag&)            = default;
  Dag& operator=(const Dag&) = default;
  virtual ~Dag() noexcept    = default;
};

template <std::size_t Depth>
struct VChain : virtual VChain<Depth - 1> {};

template <>
struct VChain<0> {
  VChain()                         = default;
  VChain(const VChain&)            = default;
  VChain& operator=(const VChain&) = default;
  virtual ~VChain() noexcept       = default;
};

template <std::size_t Index, std::size_t Depth>
struct VDag : virtual VDag<Index, Depth - 1>, virtual VDag<Index + 1, Depth - 1> {};

template <std::size_t Index>
struct VDag<Index, 0> {
  VDag()                       = default;
  VDag(const VDag&)            = default;
  VDag& operator=(const VDag&) = default;
  virtual ~VDag() noexcept     = default;
};

template <class ThrowT, class CatchT>
static TEST_ALIGN_BENCHMARK void ThrowCatch(benchmark::State& state) {
  for (auto _ : state) {
    try {
      throw ThrowT();
    } catch (CatchT& v) {
      benchmark::DoNotOptimize(v);
    }
  }
}

// Downcast along a chain from base to the most derived type
BENCHMARK(ThrowCatch<Chain<1>, Chain<0>>)->Name("throw/catch (Chain, 1 level)");
BENCHMARK(ThrowCatch<Chain<2>, Chain<0>>)->Name("throw/catch (Chain, 2 levels)");
BENCHMARK(ThrowCatch<Chain<3>, Chain<0>>)->Name("throw/catch (Chain, 3 levels)");
BENCHMARK(ThrowCatch<Chain<9>, Chain<0>>)->Name("throw/catch (Chain, 9 levels)");

// Downcast along a virtual inheritance chain from base to the most derived type
BENCHMARK(ThrowCatch<VChain<1>, VChain<0>>)->Name("throw/catch (VChain, 1 level)");
BENCHMARK(ThrowCatch<VChain<2>, VChain<0>>)->Name("throw/catch (VChain, 2 levels)");
BENCHMARK(ThrowCatch<VChain<5>, VChain<0>>)->Name("throw/catch (VChain, 5 levels)");

// Downcast along a DAG from base to the most derived type
BENCHMARK(ThrowCatch<Dag<0, 5>, Dag<5, 0>>)->Name("throw/catch (DAG rightmost, 5 levels)");
BENCHMARK(ThrowCatch<Dag<0, 5>, Dag<0, 0>>)->Name("throw/catch (DAG leftmost, 5 levels)");

// Downcast along a virtual inheritance DAG from base to the most derived type
BENCHMARK(ThrowCatch<VDag<0, 5>, VDag<5, 0>>)->Name("throw/catch (VDAG rightmost, 5 levels)");
BENCHMARK(ThrowCatch<VDag<0, 5>, VDag<0, 0>>)->Name("throw/catch (VDAG leftmost, 5 levels)");

[[gnu::noinline, noreturn]] void throw_exception() { throw std::exception(); }
[[gnu::noinline, noreturn]] void in_between_func() {
  int i;
  struct S {
    S(int* i) : i(i) {}
    int* i;
    [[gnu::noinline]] ~S() {
      *i = 0;
    }
  } s(&i);

  throw_exception();
}

static void multi_level_throw(benchmark::State& state) {
  for (auto _ : state) {
    try {
      in_between_func();
    } catch (const std::exception&) {
    }
  }
}

BENCHMARK(multi_level_throw);

BENCHMARK_MAIN();
