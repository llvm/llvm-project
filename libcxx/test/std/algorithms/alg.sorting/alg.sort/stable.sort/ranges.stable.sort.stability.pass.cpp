//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//         class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   constexpr I                                                                             // constexpr since C++26
//     ranges::stable_sort(I first, S last, Comp comp = {}, Proj proj = {});                 // since C++20
//
// template<random_access_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   constexpr borrowed_iterator_t<R>                                                        // constexpr since C++26
//     ranges::stable_sort(R&& r, Comp comp = {}, Proj proj = {});                           // since C++20
//
// Equal elements keep their relative order in ranges that are large enough for the algorithm to merge, including when
// the elements are projected, only reachable through proxies, or move-only.

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ranges>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Element>
TEST_CONSTEXPR_CXX26 std::vector<support::KeyAndIndex> sorted_by_projected_key(const std::vector<int>& keys) {
  std::vector<Element> elements = support::make_elements<Element>(keys);
  std::ranges::stable_sort(elements, std::ranges::less(), &Element::key);
  return support::keys_and_indices(elements);
}

template <class Element>
TEST_CONSTEXPR_CXX26 std::vector<support::KeyAndIndex> sorted_up_to_sentinel(const std::vector<int>& keys) {
  std::vector<Element> elements = support::make_elements<Element>(keys);
  std::ranges::stable_sort(
      elements.begin(), sentinel_wrapper<typename std::vector<Element>::iterator>(elements.end()), support::KeyLess());
  return support::keys_and_indices(elements);
}

struct ProxyKeyLess {
  template <class T, class U>
  constexpr bool operator()(const Proxy<T>& a, const Proxy<U>& b) const {
    return key_of(a.getData()) < key_of(b.getData());
  }
};

template <class Element>
TEST_CONSTEXPR_CXX26 std::vector<support::KeyAndIndex> sorted_through_proxies(const std::vector<int>& keys) {
  std::vector<Element> elements = support::make_elements<Element>(keys);
  ProxyRange proxies{elements};
  std::ranges::stable_sort(proxies, ProxyKeyLess());
  return support::keys_and_indices(elements);
}

template <class Element>
void test_trivially_copyable_elements(int size) {
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    assert(sorted_by_projected_key<Element>(keys) == support::stable_order(keys));
    assert(sorted_up_to_sentinel<Element>(keys) == support::stable_order(keys));
    assert(sorted_through_proxies<Element>(keys) == support::stable_order(keys));
  }
}

template <std::size_t Padding>
void test_move_only_elements(int size) {
  typedef support::Probe<Padding> Probe;
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    {
      std::vector<Probe> probes = support::make_elements<Probe>(keys);
      ProxyRange proxies{probes};
      std::ranges::stable_sort(proxies, ProxyKeyLess());
      assert(support::keys_and_indices(probes) == support::stable_order(keys));
      assert(support::ProbeState::alive == size);
    }
    assert(support::ProbeState::alive == 0);
  }
}

void test_returns_the_end_of_the_range() {
  typedef support::Element<0> Element;
  std::vector<Element> elements = support::make_elements<Element>(support::make_keys(support::random_keys, 1000));
  assert(std::ranges::stable_sort(elements, support::KeyLess()) == elements.end());
  assert(std::ranges::stable_sort(elements.begin(), elements.end(), support::KeyLess()) == elements.end());
}

#if TEST_STD_VER >= 26
static_assert(sorted_by_projected_key<support::Element<0>>(support::make_keys(support::random_keys, 1100)) ==
              support::stable_order(support::make_keys(support::random_keys, 1100)));
static_assert(sorted_up_to_sentinel<support::Element<0>>(support::make_keys(support::root_keys, 300)) ==
              support::stable_order(support::make_keys(support::root_keys, 300)));
static_assert(sorted_through_proxies<support::Element<0>>(support::make_keys(support::two_keys, 300)) ==
              support::stable_order(support::make_keys(support::two_keys, 300)));
static_assert(sorted_through_proxies<support::Element<1016>>(support::make_keys(support::random_keys, 40)) ==
              support::stable_order(support::make_keys(support::random_keys, 40)));
#endif

int main(int, char**) {
  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 16, 33, 64, 65, 129, 257, 513, 1000, 1025, 2049, 4097, 10007};
    for (int size : sizes) {
      test_trivially_copyable_elements<support::Element<0>>(size);
      test_move_only_elements<0>(size);
    }
  }
  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 16, 17, 33, 64, 65, 129, 257, 513, 1025};
    for (int size : sizes) {
      test_trivially_copyable_elements<support::Element<248>>(size);
      test_move_only_elements<1008>(size);
    }
  }
  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 16, 17, 33, 64, 65, 129, 257};
    for (int size : sizes) {
      test_trivially_copyable_elements<support::Element<4100>>(size);
      test_move_only_elements<4096>(size);
    }
  }
  test_returns_the_end_of_the_range();

  return 0;
}
