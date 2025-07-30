//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Iterator traits and member typedefs in zip_view::<iterator>.

#include <array>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

#include "../../range_adaptor_types.h"

template <class T>
struct ForwardView : std::ranges::view_base {
  forward_iterator<T*> begin() const;
  sentinel_wrapper<forward_iterator<T*>> end() const;
};

template <class T>
struct InputView : std::ranges::view_base {
  cpp17_input_iterator<T*> begin() const;
  sentinel_wrapper<cpp17_input_iterator<T*>> end() const;
};

template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };

template <class T>
struct DiffTypeIter {
  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using difference_type = T;

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

struct ConstVeryDifferentRange {
  int* begin();
  int* end();

  forward_iterator<double*> begin() const;
  forward_iterator<double*> end() const;
};

void test() {
  int buffer[] = {1, 2, 3, 4};
  {
    // 2 views should have 2-tuple value_type
    // random_access_iterator_tag
    std::ranges::zip_view v(buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // !=2 views should have tuple value_type
    std::ranges::zip_view v(buffer, buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, int, int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // bidirectional_iterator_tag
    std::ranges::zip_view v(BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
  }

  {
    // forward_iterator_tag
    using Iter = std::ranges::iterator_t<std::ranges::zip_view<ForwardView<int>>>;

    static_assert(std::is_same_v<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // nested zip_view
    std::ranges::zip_view v(buffer, buffer);
    std::ranges::zip_view v2(buffer, v);
    using Iter = decltype(v2.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, std::tuple<int, int>>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // input_iterator_tag
    using Iter = std::ranges::iterator_t<std::ranges::zip_view<InputView<int>>>;

    static_assert(std::is_same_v<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(!HasIterCategory<Iter>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);
  }

  {
    // difference_type of single view
    std::ranges::zip_view v{DiffTypeRange<std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::intptr_t>);
  }

  {
    // difference_type of multiple views should be the common type
    std::ranges::zip_view v{DiffTypeRange<std::intptr_t>{}, DiffTypeRange<std::ptrdiff_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::common_type_t<std::intptr_t, std::ptrdiff_t>>);
  }

  const std::array foos{Foo{}};
  std::array bars{Bar{}, Bar{}};
  {
    // value_type of single view
    std::ranges::zip_view v{foos};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, std::tuple<Foo>>);
  }

  {
    // value_type of multiple views with different value_type
    std::ranges::zip_view v{foos, bars};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, std::tuple<Foo, Bar>>);
  }

  {
    // const-iterator different from iterator
    std::ranges::zip_view v{ConstVeryDifferentRange{}};
    using Iter = decltype(v.begin());
    using ConstIter = decltype(std::as_const(v).begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int>>);

    static_assert(std::is_same_v<ConstIter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<ConstIter::value_type, std::tuple<double>>);
  }

}
