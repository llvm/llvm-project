//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Iterator member typedefs in cartesian_product_view::<iterator>:
//   using iterator_category = input_iterator_tag;          // always
//   using iterator_concept  = ...;                         // RA / bidi / forward / input
//   using value_type        = tuple<range_value_t<...>>;
//   using reference         = tuple<range_reference_t<...>>;
//   using difference_type   = common_type_t<range_difference_t<...>>;

#include <array>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

#include "../../range_adaptor_types.h"

template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };

template <class T>
struct DiffTypeIter {
  using iterator_category = std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  int operator*() const;
  DiffTypeIter& operator++();
  DiffTypeIter operator++(int);
  friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
};

template <class T>
struct DiffTypeRange : std::ranges::view_base {
  DiffTypeIter<T> begin() const;
  DiffTypeIter<T> end() const;
};

// Input-only first range (cartesian_product_view requires the first range to be input_range).
template <class T>
struct InputView : std::ranges::view_base {
  cpp20_input_iterator<T*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<T*>> end() const;
};

void test() {
  int buffer[] = {1, 2, 3, 4};

  { // 2 random_access ranges -> random_access_iterator_tag
    std::ranges::cartesian_product_view v(buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, int>>);
    static_assert(std::is_same_v<Iter::reference, std::tuple<int&, int&>>);
    static_assert(HasIterCategory<Iter>);
  }

  { // 3 random_access ranges
    std::ranges::cartesian_product_view v(buffer, buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, int, int>>);
    static_assert(std::is_same_v<Iter::reference, std::tuple<int&, int&, int&>>);
  }

  { // bidirectional: a bidi-common range with a bidi-common range
    std::ranges::cartesian_product_view v(BidiCommonView{buffer}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
  }

  { // forward: a forward-sized range with a forward-sized range
    std::ranges::cartesian_product_view v(ForwardSizedView{buffer}, ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
  }

  { // input: first range is input-only (no forward)
    using View = std::ranges::cartesian_product_view<InputView<int>, ForwardSizedView>;
    using Iter = std::ranges::iterator_t<View>;

    static_assert(std::is_same_v<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(std::is_same_v<Iter::iterator_category, std::input_iterator_tag>);
  }

  { // mixed: random-access first, bidi-common second -> bidirectional (not RA: second is not sized)
    std::ranges::cartesian_product_view v(SizedRandomAccessView{buffer}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(std::is_same_v<Iter::iterator_concept, std::bidirectional_iterator_tag>);
  }

  { // value_type with heterogeneous reference types
    std::array a{1, 2};
    std::array b{3.0, 4.0};
    std::ranges::cartesian_product_view v(a, b);
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::value_type, std::tuple<int, double>>);
    static_assert(std::is_same_v<Iter::reference, std::tuple<int&, double&>>);
  }

  { // difference_type -- single range
    std::ranges::cartesian_product_view v{DiffTypeRange<std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::intptr_t>);
  }

  { // difference_type -- multiple ranges with differing range_difference_t pick the common type
    std::ranges::cartesian_product_view v{DiffTypeRange<std::intptr_t>{}, DiffTypeRange<std::ptrdiff_t>{}};
    using Iter = decltype(v.begin());
    static_assert(std::is_same_v<Iter::difference_type, std::common_type_t<std::intptr_t, std::ptrdiff_t>>);
  }
}
