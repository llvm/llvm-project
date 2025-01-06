//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K> mapped_type&       at(const K& x);
// template<class K> const mapped_type& at(const K& x) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <stdexcept>

#include "../helpers.h"
#include "min_allocator.h"
#include "MinSequenceContainer.h"
#include "test_macros.h"

// Constraints: The qualified-id Compare::is_transparent is valid and denotes a type.
template <class M>
concept CanAt           = requires(M m, Transparent<int> k) { m.at(k); };
using TransparentMap    = std::flat_map<int, double, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, double, NonTransparentComparator>;
static_assert(CanAt<TransparentMap>);
static_assert(CanAt<const TransparentMap>);
static_assert(!CanAt<NonTransparentMap>);
static_assert(!CanAt<const NonTransparentMap>);

template <class KeyContainer, class ValueContainer>
void test() {
  using P = std::pair<int, double>;
  P ar[]  = {
      P(1, 1.5),
      P(2, 2.5),
      P(3, 3.5),
      P(4, 4.5),
      P(5, 5.5),
      P(7, 7.5),
      P(8, 8.5),
  };
  const Transparent<int> one{1};
  {
    std::flat_map<int, double, TransparentComparator, KeyContainer, ValueContainer> m(
        ar, ar + sizeof(ar) / sizeof(ar[0]));
    ASSERT_SAME_TYPE(decltype(m.at(one)), double&);
    assert(m.size() == 7);
    assert(m.at(one) == 1.5);
    m.at(one) = -1.5;
    assert(m.at(Transparent<int>{1}) == -1.5);
    assert(m.at(Transparent<int>{2}) == 2.5);
    assert(m.at(Transparent<int>{3}) == 3.5);
    assert(m.at(Transparent<int>{4}) == 4.5);
    assert(m.at(Transparent<int>{5}) == 5.5);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      TEST_IGNORE_NODISCARD m.at(Transparent<int>{6});
      assert(false);
    } catch (std::out_of_range&) {
    }
#endif
    assert(m.at(Transparent<int>{7}) == 7.5);
    assert(m.at(Transparent<int>{8}) == 8.5);
    assert(m.size() == 7);
  }
  {
    const std::flat_map<int, double, TransparentComparator, KeyContainer, ValueContainer> m(
        ar, ar + sizeof(ar) / sizeof(ar[0]));
    ASSERT_SAME_TYPE(decltype(m.at(one)), const double&);
    assert(m.size() == 7);
    assert(m.at(Transparent<int>{1}) == 1.5);
    assert(m.at(Transparent<int>{2}) == 2.5);
    assert(m.at(Transparent<int>{3}) == 3.5);
    assert(m.at(Transparent<int>{4}) == 4.5);
    assert(m.at(Transparent<int>{5}) == 5.5);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      TEST_IGNORE_NODISCARD m.at(Transparent<int>{6});
      assert(false);
    } catch (std::out_of_range&) {
    }
#endif
    assert(m.at(Transparent<int>{7}) == 7.5);
    assert(m.at(Transparent<int>{8}) == 8.5);
    assert(m.size() == 7);
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    m.at(Transparent<int>{3});
    assert(transparent_used);
  }

  return 0;
}
