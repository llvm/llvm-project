//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Iterator traits and member typedefs in zip_transform_view::iterator.

#include <array>
#include <ranges>

#include "test_iterators.h"

#include "../types.h"

template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };

template <class T>
struct DiffTypeIter {
  using iterator_category = std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  int operator*() const;
  DiffTypeIter& operator++();
  void operator++(int);
  friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
};

template <class T>
struct DiffTypeRange {
  DiffTypeIter<T> begin() const;
  DiffTypeIter<T> end() const;
};

struct Foo {};
struct Bar {};

struct RValueRefFn {
  int&& operator()(auto&&...) const;
};

void test() {
  int buffer[] = {1, 2, 3, 4};
  {
    // C++20 random_access C++17 random_access
    std::ranges::zip_transform_view v(GetFirst{}, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 random_access C++17 input
    std::ranges::zip_transform_view v(MakeTuple{}, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 bidirectional C++17 bidirectional
    std::ranges::zip_transform_view v(GetFirst{}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 bidirectional C++17 input
    std::ranges::zip_transform_view v(MakeTuple{}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 forward C++17 bidirectional
    std::ranges::zip_transform_view v(GetFirst{}, ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 forward C++17 input
    std::ranges::zip_transform_view v(MakeTuple{}, ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 input C++17 not a range
    std::ranges::zip_transform_view v(GetFirst{}, InputCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(!HasIterCategory<Iter>);
  }

  {
    // difference_type of one view
    std::ranges::zip_transform_view v{MakeTuple{}, DiffTypeRange<std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::intptr_t>);
  }

  {
    // difference_type of multiple views should be the common type
    std::ranges::zip_transform_view v{MakeTuple{}, DiffTypeRange<std::intptr_t>{}, DiffTypeRange<std::ptrdiff_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::common_type_t<std::intptr_t, std::ptrdiff_t>>);
  }

  const std::array foos{Foo{}};
  std::array bars{Bar{}, Bar{}};
  {
    // value_type of one view
    std::ranges::zip_transform_view v{MakeTuple{}, foos};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, std::tuple<Foo>>);
  }

  {
    // value_type of multiple views with different value_type
    std::ranges::zip_transform_view v{MakeTuple{}, foos, bars};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, std::tuple<Foo, Bar>>);
  }

  // LWG3798 Rvalue reference and iterator_category
  {
    std::ranges::zip_transform_view v(RValueRefFn{}, buffer);
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>);
  }
}
