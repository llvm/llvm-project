//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that `ranges::to` can be used to convert between arbitrary standard containers.

#include <ranges>

#include <algorithm>
#include <cassert>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_iterators.h"
#include "test_range.h"
#include "type_algorithms.h"
#include "unwrap_container_adaptor.h"

std::vector<std::vector<int>> ints = {
  {5, 1, 3, 4, 2},
  {3},
  {}
};

std::vector<std::vector<char>> chars = {
  {'a', 'b', 'c'},
  {'a'},
  {}
};

std::vector<std::vector<std::pair<const int, int>>> pairs = {
  {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 0}},
  {{1, 2}},
  {}
};

template <class From, class To>
void test_is_equal(std::vector<std::vector<typename From::value_type>> inputs) {
  for (const auto& in : inputs) {
    From from(in.begin(), in.end());
    std::same_as<To> decltype(auto) result = std::ranges::to<To>(from);
    assert(std::ranges::equal(in, result));
  }
}

template <class From, class To>
void test_is_permutation(std::vector<std::vector<typename From::value_type>> inputs) {
  for (const auto& in : inputs) {
    From from(in.begin(), in.end());
    std::same_as<To> decltype(auto) result = std::ranges::to<To>(in);
    assert(std::ranges::is_permutation(in, result));
  }
}

template <class From, class To>
void test_is_equal_for_adaptors(std::vector<std::vector<typename From::value_type>> inputs) {
  for (const auto& in : inputs) {
    From from(in.begin(), in.end());
    std::same_as<To> decltype(auto) result = std::ranges::to<To>(in);

    UnwrapAdaptor<From> unwrap_from(std::move(from));
    UnwrapAdaptor<To> unwrap_to(std::move(result));
    assert(std::ranges::is_permutation(unwrap_from.get_container(), unwrap_to.get_container()));
  }
}

template <class T>
using sequence_containers = types::type_list<
    std::vector<T>,
    std::deque<T>,
    std::list<T>,
    std::forward_list<T>
>;

template <class T>
using associative_sets = types::type_list<
    std::set<T>,
    std::multiset<T>
>;

template <class K, class V>
using associative_maps = types::type_list<
    std::map<K, V>,
    std::multimap<K, V>
>;

template <class T>
using unordered_sets = types::type_list<
    std::unordered_set<T>,
    std::unordered_multiset<T>
>;

template <class K, class V>
using unordered_maps = types::type_list<
    std::unordered_map<K, V>,
    std::unordered_multimap<K, V>
>;

template <class T>
using container_adaptors = types::type_list<
    std::stack<T>,
    std::queue<T>,
    std::priority_queue<T>
>;

template <class T>
using sequences_and_sets = types::concatenate_t<sequence_containers<T>, associative_sets<T>, unordered_sets<T>>;

template <class K, class V>
using all_containers = types::concatenate_t<
    sequence_containers<std::pair<const K, V>>,
    associative_sets<std::pair<const K, V>>,
    associative_maps<K, V>,
    unordered_sets<std::pair<const K, V>>,
    unordered_maps<K, V>>;

// This is necessary to be able to use `pair`s with unordered sets.
template <class K, class V>
struct std::hash<std::pair<const K, V>> {
  std::size_t operator()(const std::pair<const K, V>& p) const {
    std::size_t h1 = std::hash<K>{}(p.first);
    std::size_t h2 = std::hash<V>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

void test() {
  { // Conversions always preserving equality.
    { // sequences <-> sequences
      types::for_each(sequence_containers<int>{}, []<class From>() {
        types::for_each(sequence_containers<int>{}, []<class To>() {
          test_is_equal<From, To>(ints);
        });
      });

      types::for_each(sequence_containers<int>{}, []<class From>() {
        types::for_each(sequence_containers<double>{}, []<class To>() {
          test_is_equal<From, To>(ints);
        });
      });
    }

    { // sequences <-> string
      types::for_each(sequence_containers<char>{}, []<class Seq>() {
        test_is_equal<Seq, std::basic_string<char>>(chars);
        test_is_equal<std::basic_string<char>, Seq>(chars);
      });
    }
  }

  { // sequences/sets <-> sequences/sets
    types::for_each(sequences_and_sets<int>{}, []<class From>() {
      types::for_each(sequences_and_sets<int>{}, []<class To>() {
        test_is_permutation<From, To>(ints);
      });
    });

    types::for_each(sequences_and_sets<int>{}, []<class From>() {
      types::for_each(sequences_and_sets<double>{}, []<class To>() {
        test_is_permutation<From, To>(ints);
      });
    });
  }

  { // sequences/sets/maps <-> sequences/sets/maps. Uses `pair` for non-map containers to allow mutual conversion with
    // map types.
    types::for_each(all_containers<int, int>{}, []<class From>() {
      types::for_each(all_containers<int, int>{}, []<class To>() {
        test_is_permutation<From, To>(pairs);
      });
    });

    types::for_each(all_containers<int, int>{}, []<class From>() {
      types::for_each(all_containers<long, double>{}, []<class To>() {
        test_is_permutation<From, To>(pairs);
      });
    });
  }

  { // adaptors <-> adaptors
    types::for_each(container_adaptors<int>{}, []<class From>() {
      types::for_each(container_adaptors<int>{}, []<class To>() {
        test_is_equal_for_adaptors<From, To>(ints);
      });
    });

    types::for_each(container_adaptors<int>{}, []<class From>() {
      types::for_each(container_adaptors<double>{}, []<class To>() {
        test_is_equal_for_adaptors<From, To>(ints);
      });
    });
  }
}

int main(int, char**) {
  test();

  return 0;
}
