//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Iterator traits and member typedefs in adjacent_view::iterator.

#include <array>
#include <iterator>
#include <ranges>
#include <tuple>
#include <utility>

#include "test_iterators.h"

#include "../../range_adaptor_types.h"

template <class T>
struct DiffTypeIter : random_access_iterator<int*> {
  using value_type      = int;
  using difference_type = T;

  int operator*() const;
  DiffTypeIter& operator++();
  DiffTypeIter operator++(int);
  friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
};

template <class T>
struct DiffTypeView : std::ranges::view_base {
  DiffTypeIter<T> begin() const;
  DiffTypeIter<T> end() const;
};

struct Foo {};

struct ConstVeryDifferentRange {
  int* begin();
  int* end();

  forward_iterator<double*> begin() const;
  forward_iterator<double*> end() const;
};

struct OverloadComma {
  friend int operator,(auto&&, OverloadComma);
};

template <std::size_t N>
void test() {
  int buffer[] = {1, 2, 3, 4};

  const auto expectedTupleType = [] {
    if constexpr (N == 1)
      return std::tuple<int>{};
    else if constexpr (N == 2)
      return std::tuple<int, int>{};
    else if constexpr (N == 3)
      return std::tuple<int, int, int>{};
    else if constexpr (N == 4)
      return std::tuple<int, int, int, int>{};
    else if constexpr (N == 5)
      return std::tuple<int, int, int, int, int>{};
  };

  using expected_value_type = decltype(expectedTupleType());

  {
    // Base contiguous range
    std::ranges::adjacent_view<std::views::all_t<decltype((buffer))>, N> v(buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, expected_value_type>);
  }

  {
    // Base random access
    std::ranges::adjacent_view<SizedRandomAccessView, N> v(SizedRandomAccessView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, expected_value_type>);
  }

  {
    // Base bidirectional
    std::ranges::adjacent_view<BidiCommonView, N> v(BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, expected_value_type>);
  }

  {
    // Base forward
    std::ranges::adjacent_view<ForwardSizedView, N> v(ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, expected_value_type>);
  }

  {
    // difference_type
    std::ranges::adjacent_view<DiffTypeView<std::intptr_t>, N> v{DiffTypeView<std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<typename Iter::difference_type, std::intptr_t>);
  }

  {
    // value_type
    const std::array foos{Foo{}};
    auto v     = std::views::adjacent<2>(foos);
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<typename Iter::value_type, std::tuple<Foo, Foo>>);
  }

  {
    // const-iterator different from iterator
    auto v          = ConstVeryDifferentRange{} | std::views::adjacent<2>;
    using Iter      = decltype(v.begin());
    using ConstIter = decltype(std::as_const(v).begin());

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, std::tuple<int, int>>);

    static_assert(std::is_same_v<typename ConstIter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<typename ConstIter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename ConstIter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename ConstIter::value_type, std::tuple<double, double>>);
  }
}

void test() {
  test<1>();
  test<2>();
  test<3>();
  test<5>();

  {
    // overload comma operator
    OverloadComma buffer[] = {OverloadComma{}, OverloadComma{}, OverloadComma{}};
    auto v                 = std::views::adjacent<2>(buffer);
    using Iter             = decltype(v.begin());
    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, std::tuple<OverloadComma, OverloadComma>>);
  }
}
