//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ranges>

//  template<class T> struct tuple_size;
//  template<size_t I, class T> struct tuple_element;

#include <ranges>
// Note: make sure to not include `<utility>` (or any other header including `<utility>`) because it also makes some
// tuple specializations available, thus obscuring whether the `<ranges>` includes work correctly.

using Iterator = int*;

class SizedSentinel {
public:
    constexpr bool operator==(int*) const;
    friend constexpr std::ptrdiff_t operator-(const SizedSentinel&, int*);
    friend constexpr std::ptrdiff_t operator-(int*, const SizedSentinel&);
};

static_assert(std::sized_sentinel_for<SizedSentinel, Iterator>);
using SizedRange = std::ranges::subrange<Iterator, SizedSentinel>;

using UnsizedSentinel = std::unreachable_sentinel_t;
static_assert(!std::sized_sentinel_for<UnsizedSentinel, Iterator>);
using UnsizedRange = std::ranges::subrange<Iterator, UnsizedSentinel>;

// Because the sentinel is unsized while the subrange is sized, an additional integer member will be used to store the
// size -- make sure it doesn't affect the value of `tuple_size`.
using ThreeElementRange = std::ranges::subrange<Iterator, UnsizedSentinel, std::ranges::subrange_kind::sized>;
static_assert(std::ranges::sized_range<ThreeElementRange>);

static_assert(std::tuple_size<SizedRange>::value == 2);
static_assert(std::tuple_size<UnsizedRange>::value == 2);
static_assert(std::tuple_size<ThreeElementRange>::value == 2);

template <int I, class Range, class Expected>
constexpr bool test_tuple_element() {
  static_assert(std::same_as<typename std::tuple_element<I, Range>::type, Expected>);
  static_assert(std::same_as<typename std::tuple_element<I, const Range>::type, Expected>);
  // Note: the Standard does not mandate a specialization of `tuple_element` for volatile, so trying a `volatile Range`
  // would fail to compile.

  return true;
}

int main(int, char**) {
  static_assert(test_tuple_element<0, SizedRange, Iterator>());
  static_assert(test_tuple_element<1, SizedRange, SizedSentinel>());
  static_assert(test_tuple_element<0, UnsizedRange, Iterator>());
  static_assert(test_tuple_element<1, UnsizedRange, UnsizedSentinel>());

  return 0;
}
