//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Iterator traits and member typedefs in adjacent_transform_view::iterator.

#include <array>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>

#include "test_iterators.h"

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <class T>
struct DiffTypeIter : random_access_iterator<int*> {
  using value_type      = int;
  using difference_type = T;

  int& operator*() const;
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

template <class Fn, std::size_t N, class ExpectedValueType, bool IterCatShouldBeInput>
void test() {
  int buffer[] = {1, 2, 3, 4};

  {
    // Base contiguous range
    std::ranges::adjacent_transform_view<std::views::all_t<decltype((buffer))>, Fn, N> v(buffer, Fn{});
    using Iter = decltype(v.begin());
    using Cat  = std::conditional_t<IterCatShouldBeInput, std::input_iterator_tag, std::random_access_iterator_tag>;

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, Cat>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, ExpectedValueType>);
  }

  {
    // Base random access
    std::ranges::adjacent_transform_view<SizedRandomAccessView, Fn, N> v(SizedRandomAccessView{buffer}, Fn{});
    using Iter = decltype(v.begin());
    using Cat  = std::conditional_t<IterCatShouldBeInput, std::input_iterator_tag, std::random_access_iterator_tag>;

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, Cat>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, ExpectedValueType>);
  }

  {
    // Base bidirectional
    std::ranges::adjacent_transform_view<BidiCommonView, Fn, N> v(BidiCommonView{buffer}, Fn{});
    using Iter = decltype(v.begin());
    using Cat  = std::conditional_t<IterCatShouldBeInput, std::input_iterator_tag, std::bidirectional_iterator_tag>;

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, Cat>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, ExpectedValueType>);
  }

  {
    // Base forward
    std::ranges::adjacent_transform_view<ForwardSizedView, Fn, N> v(ForwardSizedView{buffer}, Fn{});
    using Iter = decltype(v.begin());
    using Cat  = std::conditional_t<IterCatShouldBeInput, std::input_iterator_tag, std::forward_iterator_tag>;

    static_assert(std::is_same_v<typename Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<typename Iter::iterator_category, Cat>);
    static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<typename Iter::value_type, ExpectedValueType>);
  }

  {
    // difference_type
    std::ranges::adjacent_transform_view<DiffTypeView<std::intptr_t>, Fn, N> v{DiffTypeView<std::intptr_t>{}, Fn{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<typename Iter::difference_type, std::intptr_t>);
  }

  {
    // const-iterator different from iterator
    auto v          = ConstVeryDifferentRange{} | std::views::adjacent_transform<2>(MakeTuple{});
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
  test<MakeTuple, 1, std::tuple<int>, true>();
  test<MakeTuple, 2, std::tuple<int, int>, true>();
  test<MakeTuple, 3, std::tuple<int, int, int>, true>();

  test<Tie, 1, std::tuple<int&>, true>();
  test<Tie, 2, std::tuple<int&, int&>, true>();
  test<Tie, 3, std::tuple<int&, int&, int&>, true>();

  test<GetFirst, 1, int, false>();
  test<GetFirst, 2, int, false>();
  test<GetFirst, 3, int, false>();

  test<Multiply, 1, int, true>();
  test<Multiply, 2, int, true>();
  test<Multiply, 3, int, true>();
}
