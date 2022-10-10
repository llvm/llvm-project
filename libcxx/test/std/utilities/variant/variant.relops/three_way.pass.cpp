//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <variant>

// template <class... Types> class variant;

// template<class... Types>
//   constexpr std::common_comparison_category_t<
//     std::compare_three_way_result_t<Types>...>
//   operator<=>(const variant<Types...>& t, const variant<Types...>& u);

#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"
#include "test_comparisons.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
// MakeEmptyT throws in operator=(&&), so we can move to it to create valueless-by-exception variants.
struct MakeEmptyT {
  MakeEmptyT() = default;
  MakeEmptyT(MakeEmptyT&&) { throw 42; }
  MakeEmptyT& operator=(MakeEmptyT&&) { throw 42; }
};
inline bool operator==(const MakeEmptyT&, const MakeEmptyT&) {
  assert(false);
  return false;
}
inline std::weak_ordering operator<=>(const MakeEmptyT&, const MakeEmptyT&) {
  assert(false);
  return std::weak_ordering::equivalent;
}

template <class Variant>
void makeEmpty(Variant& v) {
  Variant v2(std::in_place_type<MakeEmptyT>);
  try {
    v = std::move(v2);
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}

void test_empty() {
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    V v2;
    makeEmpty(v2);
    assert(testOrder(v1, v2, std::weak_ordering::greater));
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    assert(testOrder(v1, v2, std::weak_ordering::less));
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(testOrder(v1, v2, std::weak_ordering::equivalent));
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

template <class T1, class T2, class Order>
constexpr bool test_with_types() {
  using V = std::variant<T1, T2>;
  AssertOrderReturn<Order, V>();
  { // same index, same value
    constexpr V v1(std::in_place_index<0>, T1{1});
    constexpr V v2(std::in_place_index<0>, T1{1});
    assert(testOrder(v1, v2, Order::equivalent));
  }
  { // same index, value < other_value
    constexpr V v1(std::in_place_index<0>, T1{0});
    constexpr V v2(std::in_place_index<0>, T1{1});
    assert(testOrder(v1, v2, Order::less));
  }
  { // same index, value > other_value
    constexpr V v1(std::in_place_index<0>, T1{1});
    constexpr V v2(std::in_place_index<0>, T1{0});
    assert(testOrder(v1, v2, Order::greater));
  }
  { // LHS.index() < RHS.index()
    constexpr V v1(std::in_place_index<0>, T1{0});
    constexpr V v2(std::in_place_index<1>, T2{0});
    assert(testOrder(v1, v2, Order::less));
  }
  { // LHS.index() > RHS.index()
    constexpr V v1(std::in_place_index<1>, T2{0});
    constexpr V v2(std::in_place_index<0>, T1{0});
    assert(testOrder(v1, v2, Order::greater));
  }

  return true;
}

constexpr bool test_three_way() {
  assert((test_with_types<int, double, std::partial_ordering>()));
  assert((test_with_types<int, long, std::strong_ordering>()));

  {
    using V              = std::variant<int, double>;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    {
      constexpr V v1(std::in_place_type<int>, 1);
      constexpr V v2(std::in_place_type<double>, nan);
      assert(testOrder(v1, v2, std::partial_ordering::less));
    }
    {
      constexpr V v1(std::in_place_type<double>, nan);
      constexpr V v2(std::in_place_type<int>, 2);
      assert(testOrder(v1, v2, std::partial_ordering::greater));
    }
    {
      constexpr V v1(std::in_place_type<double>, nan);
      constexpr V v2(std::in_place_type<double>, nan);
      assert(testOrder(v1, v2, std::partial_ordering::unordered));
    }
  }

  return true;
}

// SFINAE tests
template <class T, class U = T>
concept has_three_way_op = requires (T& t, U& u) { t <=> u; };

// std::three_way_comparable is a more stringent requirement that demands
// operator== and a few other things.
using std::three_way_comparable;

struct HasSimpleOrdering {
  constexpr bool operator==(const HasSimpleOrdering&) const;
  constexpr bool operator<(const HasSimpleOrdering&) const;
};

struct HasOnlySpaceship {
  constexpr bool operator==(const HasOnlySpaceship&) const = delete;
  constexpr std::weak_ordering operator<=>(const HasOnlySpaceship&) const;
};

struct HasFullOrdering {
  constexpr bool operator==(const HasFullOrdering&) const;
  constexpr std::weak_ordering operator<=>(const HasFullOrdering&) const;
};

// operator<=> must resolve the return types of all its union types'
// operator<=>s to determine its own return type, so it is detectable by SFINAE
static_assert(!has_three_way_op<HasSimpleOrdering>);
static_assert(!has_three_way_op<std::variant<int, HasSimpleOrdering>>);

static_assert(!three_way_comparable<HasSimpleOrdering>);
static_assert(!three_way_comparable<std::variant<int, HasSimpleOrdering>>);

static_assert( has_three_way_op<HasOnlySpaceship>);
static_assert( has_three_way_op<std::variant<int, HasOnlySpaceship>>);

// variants containing types with unavailable operator== still exist but will
// generate a compilation error if their operator== is invoked, so the variant
// type here participates when asked for operator== and operator<=> even though
// it would actually fail.
static_assert(!three_way_comparable<HasOnlySpaceship>);
static_assert( three_way_comparable<std::variant<int, HasOnlySpaceship>>);

static_assert( has_three_way_op<HasFullOrdering>);
static_assert( has_three_way_op<std::variant<int, HasFullOrdering>>);

static_assert( three_way_comparable<HasFullOrdering>);
static_assert( three_way_comparable<std::variant<int, HasFullOrdering>>);

int main(int, char**) {
  test_three_way();
  static_assert(test_three_way());

#ifndef TEST_HAS_NO_EXCEPTIONS
  test_empty();
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
