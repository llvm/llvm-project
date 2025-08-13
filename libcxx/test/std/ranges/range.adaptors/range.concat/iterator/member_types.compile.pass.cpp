//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

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

constexpr bool test() {
  int buffer[] = {1, 2, 3, 4};
  {
    // random_access_iterator_tag
    std::ranges::concat_view v(buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }


  {
    // 3 views
    std::ranges::concat_view v(buffer, buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }


  {
    // bidirectional_iterator_tag
    std::ranges::concat_view v(BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
  }

  {
    // forward_iterator_tag
    using Iter = std::ranges::iterator_t<std::ranges::concat_view<ForwardView<int>>>;

    static_assert(std::is_same_v<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // input_iterator_tag
    using Iter = std::ranges::iterator_t<std::ranges::concat_view<InputView<int>>>;

    static_assert(std::is_same_v<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(!HasIterCategory<Iter>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);
  }

  {
    // difference_type of single view
    std::ranges::concat_view v{DiffTypeRange<std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::intptr_t>);
  }

  {
    // difference_type of multiple views should be the common type
    std::ranges::concat_view v{DiffTypeRange<std::intptr_t>{}, DiffTypeRange<std::ptrdiff_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::common_type_t<std::intptr_t, std::ptrdiff_t>>);
  }

  const std::array foos{Foo{}};
  {
    // value_type of user-defined type
    std::ranges::concat_view v{foos};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, Foo>);
  }

  {
    // const-iterator different from iterator
    std::ranges::concat_view v{ConstVeryDifferentRange{}};
    using Iter = decltype(v.begin());
    using ConstIter = decltype(std::as_const(v).begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, int>);

    static_assert(std::is_same_v<ConstIter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<ConstIter::value_type, double>);
  }

  return true;
}


int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
