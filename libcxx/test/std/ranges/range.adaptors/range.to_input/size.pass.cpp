//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

//    constexpr auto size() requires sized_range<V>;
//    constexpr auto size() const requires sized_range<const V>;

#include <array>
#include <cassert>
#include <ranges>
#include <utility>

#include "test_iterators.h"

template <class T>
concept HasSize = requires(T t) { t.size(); };

constexpr bool test() {
  {
    struct SubtractableIteratorsView : std::ranges::view_base {
      forward_iterator<int*> begin();
      sized_sentinel<forward_iterator<int*>> end();
    };

    using ToInputViewT = std::ranges::to_input_view<SubtractableIteratorsView>;

    static_assert(std::ranges::sized_range<ToInputViewT&>);
    static_assert(!std::ranges::range<const ToInputViewT&>); // no begin()/end()

    static_assert(HasSize<ToInputViewT&>);
    static_assert(HasSize<ToInputViewT&&>);
    static_assert(!HasSize<const ToInputViewT&>);
    static_assert(!HasSize<const ToInputViewT&&>);
  }
  {
    struct NonSizedView : std::ranges::view_base {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();
    };

    using ToInputViewT = std::ranges::to_input_view<NonSizedView>;

    static_assert(!HasSize<ToInputViewT&>);
    static_assert(!HasSize<ToInputViewT&&>);
    static_assert(!HasSize<const ToInputViewT&>);
    static_assert(!HasSize<const ToInputViewT&&>);
  }
  {
    struct SizedView : std::ranges::view_base {
      bidirectional_iterator<int*> begin();
      bidirectional_iterator<int*> end();

      int size() const;
    };

    using ToInputViewT = std::ranges::to_input_view<SizedView>;

    static_assert(std::ranges::sized_range<ToInputViewT&>);
    static_assert(!std::ranges::range<const ToInputViewT&>); // no begin()/end()

    static_assert(HasSize<ToInputViewT&>);
    static_assert(HasSize<ToInputViewT&&>);
    static_assert(!HasSize<const ToInputViewT&>); // not a view, therefore no size()
    static_assert(!HasSize<const ToInputViewT&&>);
  }
  {
    // Test an empty view.
    int arr[] = {94};
    auto view = std::ranges::to_input_view(std::ranges::subrange(arr, arr));

    assert(view.size() == 0);
    assert(std::as_const(view).size() == 0);
  }
  {
    // Test a non-empty view.
    int arr[] = {94};
    auto view = std::ranges::to_input_view(std::ranges::subrange(arr, arr + 1));

    assert(view.size() == 1);
    assert(std::as_const(view).size() == 1);
  }
  {
    // Test a non-view.
    std::array<int, 2> arr = {94, 82};
    auto view              = std::ranges::to_input_view(std::move(arr));

    assert(view.size() == 2);
    assert(std::as_const(view).size() == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
