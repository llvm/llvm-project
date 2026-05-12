//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class T, size_t N>
//   constexpr synth-three-way-result<T>
//     operator<=>(const inplace_vector<T, N>& x, const inplace_vector<T, N>& y);

#include <cassert>
#include <compare>
#include <concepts>
#include <inplace_vector>
#include <limits>
#include <type_traits>

#include "common.h"
#include "test_comparisons.h"

static_assert(std::three_way_comparable<std::inplace_vector<int, 8>>);
static_assert(!std::three_way_comparable<std::inplace_vector<NonComparable, 8>>);

template <class Elem, class Order>
constexpr void test_with_type() {
  {
    std::inplace_vector<Elem, 8> c1;
    std::inplace_vector<Elem, 8> c2;
    std::same_as<Order> decltype(auto) result = c1 <=> c2;
    assert(result == Order::equivalent);
  }
  {
    std::inplace_vector<Elem, 8> c1{Elem(1), Elem(1)};
    std::inplace_vector<Elem, 8> c2{Elem(1), Elem(1)};
    assert(testOrder(c1, c2, Order::equivalent));
  }
  {
    std::inplace_vector<Elem, 8> c1{Elem(1), Elem(1)};
    std::inplace_vector<Elem, 8> c2{Elem(1), Elem(2)};
    assert(testOrder(c1, c2, Order::less));
  }
  {
    std::inplace_vector<Elem, 8> c1{Elem(1), Elem(3)};
    std::inplace_vector<Elem, 8> c2{Elem(1), Elem(2)};
    assert(testOrder(c1, c2, Order::greater));
  }
  {
    std::inplace_vector<Elem, 8> c1{Elem(1)};
    std::inplace_vector<Elem, 8> c2{Elem(1), Elem(2)};
    assert(testOrder(c1, c2, Order::less));
  }
  {
    std::inplace_vector<Elem, 8> c1{Elem(1), Elem(2)};
    std::inplace_vector<Elem, 8> c2{Elem(1)};
    assert(testOrder(c1, c2, Order::greater));
  }

  if constexpr (std::is_same_v<Elem, PartialOrder>) {
    std::inplace_vector<Elem, 8> c1{Elem(1), Elem(std::numeric_limits<int>::min())};
    std::inplace_vector<Elem, 8> c2{Elem(1), Elem(2)};
    assert(testOrder(c1, c2, Order::unordered));
  }
}

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c1{1, 2, 3};
    std::inplace_vector<int, 8> c2{1, 2, 3};
    std::inplace_vector<int, 8> c3{1, 2, 4};
    std::inplace_vector<int, 8> c4{1, 2};

    std::same_as<std::strong_ordering> decltype(auto) result = c1 <=> c2;
    assert(result == std::strong_ordering::equal);
    assert((c1 <=> c3) == std::strong_ordering::less);
    assert((c3 <=> c1) == std::strong_ordering::greater);
    assert((c4 <=> c1) == std::strong_ordering::less);
    assert((c1 <=> c4) == std::strong_ordering::greater);
  }

  {
    std::inplace_vector<int, 0> c1;
    std::inplace_vector<int, 0> c2;
    std::same_as<std::strong_ordering> decltype(auto) result = c1 <=> c2;
    assert(result == std::strong_ordering::equal);
  }

  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    test_with_type<StrongOrder, std::strong_ordering>();
    test_with_type<WeakOrder, std::weak_ordering>();
    test_with_type<PartialOrder, std::partial_ordering>();
    test_with_type<LessAndEqComp, std::weak_ordering>();
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
