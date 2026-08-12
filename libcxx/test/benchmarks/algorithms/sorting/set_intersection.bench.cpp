//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <cstddef>
#include <deque>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "test_macros.h"
#include "../../GenerateInput.h"

// Generate two ranges where the larger range holds `n` elements and the smaller one holds every `ratio`th
// element of it, so the whole smaller range matches and the matches are spread evenly through the larger
// one.
template <class T>
std::pair<std::vector<T>, std::vector<T>> spread(std::size_t n, std::size_t ratio) {
  std::vector<T> large(n);
  std::generate_n(large.begin(), n, [] { return Generate<T>::random(); });
  std::sort(large.begin(), large.end());

  std::vector<T> small;
  small.reserve(n / ratio + 1);
  for (std::size_t i = 0; i < n; i += ratio)
    small.push_back(large[i]);
  return std::make_pair(std::move(large), std::move(small));
}

int main(int argc, char** argv) {
  auto bm = []<class Container>(std::string name, auto generate) {
    return benchmark::RegisterBenchmark(name, [generate](auto& st) TEST_ALIGN_BENCHMARK {
      using ValueType        = typename Container::value_type;
      std::size_t const size = st.range(0);

      // Generate the values, then shuffle them so we don't insert into the containers in sorted order.
      // Otherwise, node-based containers end up with an artificially good memory layout.
      auto [values1, values2] = generate(std::type_identity<ValueType>{}, size);
      std::shuffle(values1.begin(), values1.end(), getRandomEngine());
      std::shuffle(values2.begin(), values2.end(), getRandomEngine());
      Container c1(values1.begin(), values1.end());
      Container c2(values2.begin(), values2.end());

      // For containers that are not already sorted by construction, sort since that's a pre-requisite
      // for std::set_intersection.
      static constexpr bool is_ordered_container = requires(Container c, ValueType v) { c.lower_bound(v); };
      if constexpr (!is_ordered_container) {
        std::sort(c1.begin(), c1.end());
        std::sort(c2.begin(), c2.end());
      }

      std::vector<ValueType> out(std::min(c1.size(), c2.size()));

      for (auto _ : st) {
        benchmark::DoNotOptimize(c1);
        benchmark::DoNotOptimize(c2);
        benchmark::DoNotOptimize(out);
        auto result = std::set_intersection(c1.begin(), c1.end(), c2.begin(), c2.end(), out.begin());
        benchmark::DoNotOptimize(result);
      }
    });
  };

  // Each scenario below produces the two sorted ranges to intersect. `n` is the size of the larger range, and
  // when the two sizes differ, the size of the smaller one is `n` divided by a ratio. That ratio exercises
  // implementation choices that make a difference when the two ranges have very different sizes.
  //
  // Every scenario is registered with 32, which is small enough that constant factors dominate, 8192, which
  // fits in cache, and 1 << 18, which does not. The std::string rows stop at 8192, because those elements are
  // large and expensive to compare and to copy, so beyond that the benchmark spends most of its time in its
  // own setup. Most benchmarks on std::set stop at 8192, because they are too noisy above that.

  {
    // The two ranges have no element in common, and every element of the first range compares less
    // than every element of the second one. An implementation that searches ahead can dismiss the
    // whole first range almost immediately.
    auto disjoint = []<class T>(std::type_identity<T>, std::size_t n) {
      std::vector<T> values(2 * n);
      std::generate_n(values.begin(), 2 * n, [] { return Generate<T>::random(); });
      std::sort(values.begin(), values.end());
      std::vector<T> first(values.begin(), values.begin() + n);
      std::vector<T> second(values.begin() + n, values.end());
      return std::make_pair(std::move(first), std::move(second));
    };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (disjoint)", disjoint)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (disjoint)", disjoint)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (disjoint)", disjoint)->Arg(32)->Arg(8192);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (disjoint)", disjoint)->Arg(32)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (disjoint)", disjoint)->Arg(32)->Arg(8192);
    // clang-format on
  }

  {
    // The two ranges have no element in common and their elements strictly alternate:
    // a[0] < b[0] < a[1] < b[1] < ... An implementation that searches ahead in one range
    // can never advance by more than a single element.
    auto interleaved = []<class T>(std::type_identity<T>, std::size_t n) {
      std::vector<T> values(2 * n);
      std::generate_n(values.begin(), 2 * n, [] { return Generate<T>::random(); });
      std::sort(values.begin(), values.end());
      std::vector<T> first;
      std::vector<T> second;
      first.reserve(n);
      second.reserve(n);
      for (std::size_t i = 0; i != values.size(); ++i)
        (i % 2 == 0 ? first : second).push_back(values[i]);
      return std::make_pair(std::move(first), std::move(second));
    };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (interleaved)", interleaved)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (interleaved)", interleaved)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (interleaved)", interleaved)->Arg(32)->Arg(8192);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (interleaved)", interleaved)->Arg(32)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (interleaved)", interleaved)->Arg(32)->Arg(8192);
    // clang-format on
  }

  {
    // Both ranges are identical, which is the worst case for the algorithm in terms of output size.
    auto identical = []<class T>(std::type_identity<T>, std::size_t n) { return spread<T>(n, 1); };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (identical)", identical)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (identical)", identical)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (identical)", identical)->Arg(32)->Arg(8192);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (identical)", identical)->Arg(32)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (identical)", identical)->Arg(32)->Arg(8192);
    // clang-format on
  }

  {
    auto spread_1_8 = []<class T>(std::type_identity<T>, std::size_t n) { return spread<T>(n, 8); };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (spread 1:8)", spread_1_8)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (spread 1:8)", spread_1_8)->Arg(32)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (spread 1:8)", spread_1_8)->Arg(32)->Arg(8192);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (spread 1:8)", spread_1_8)->Arg(32)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (spread 1:8)", spread_1_8)->Arg(32)->Arg(8192);
    // clang-format on
  }

  {
    auto spread_1_1024 = []<class T>(std::type_identity<T>, std::size_t n) { return spread<T>(n, 1024); };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (spread 1:1024)", spread_1_1024)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (spread 1:1024)", spread_1_1024)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (spread 1:1024)", spread_1_1024)->Arg(8192);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (spread 1:1024)", spread_1_1024)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (spread 1:1024)", spread_1_1024)->Arg(8192);
    // clang-format on
  }

  {
    // The larger range holds `n` elements and the smaller one is its first `n / ratio` elements, so all the
    // matches sit at the very front. An implementation stops as soon as it has exhausted the smaller range,
    // so the cost should be proportional to the size of the smaller range, regardless of the contents of the
    // larger range.
    //
    // Here we bother testing std::set with a large range since we don't actually go through the full set.
    auto prefix = []<class T>(std::type_identity<T>, std::size_t n, std::size_t ratio) {
      std::vector<T> large(n);
      std::generate_n(large.begin(), n, [] { return Generate<T>::random(); });
      std::sort(large.begin(), large.end());

      std::vector<T> small(large.begin(), large.begin() + (n / ratio));
      return std::make_pair(std::move(large), std::move(small));
    };
    auto prefix_1_1024 = [prefix](auto type, std::size_t n) { return prefix(type, n, 1024); };

    // clang-format off
    bm.operator()<std::vector<int>>("std::set_intersection(vector<int>) (prefix 1:1024)", prefix_1_1024)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::deque<int>>("std::set_intersection(deque<int>) (prefix 1:1024)", prefix_1_1024)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::set<int>>("std::set_intersection(set<int>) (prefix 1:1024)", prefix_1_1024)->Arg(8192)->Arg(1 << 18);
    bm.operator()<std::vector<std::string>>("std::set_intersection(vector<std::string>) (prefix 1:1024)", prefix_1_1024)->Arg(8192);
    bm.operator()<std::set<std::string>>("std::set_intersection(set<std::string>) (prefix 1:1024)", prefix_1_1024)->Arg(8192);
    // clang-format on
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
