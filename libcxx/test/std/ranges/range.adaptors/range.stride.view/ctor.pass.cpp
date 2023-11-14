//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "test.h"
#include "test_convertible.h"
#include "test_iterators.h"
#include <type_traits>

constexpr bool test_no_default_ctor() {
  // There is no default ctor for stride_view.
  using View = InputView<cpp17_input_iterator<int*>>;
  static_assert(!std::is_default_constructible_v<std::ranges::stride_view<View>>);
  return true;
}

constexpr bool test_no_implicit_ctor() {
  using View = InputView<cpp17_input_iterator<int*>>;
  // Test that the stride_view can only be explicitly constructed.
  static_assert(!test_convertible<std::ranges::stride_view<View>, View, int>());
  return true;
}

constexpr bool test_move_ctor() {
  int arr[] = {1, 2, 3};
  // Test that the stride_view ctor properly moves from the base (and works with a move-only type).
  static_assert(!std::is_copy_constructible_v<MovedOnlyTrackedBasicView<int>>);
  static_assert(std::is_move_constructible_v<MovedOnlyTrackedBasicView<int>>);

  bool moved(false), copied(false);
  MovedOnlyTrackedBasicView<int> mov(arr, arr + 3, &moved, &copied);
  std::ranges::stride_view<MovedOnlyTrackedBasicView<int>> mosv(std::move(mov), 2);
  assert(moved);
  assert(!copied);
  return true;
}

int main(int, char**) {
  test_no_implicit_ctor();
  static_assert(test_no_implicit_ctor());
  test_no_default_ctor();
  static_assert(test_no_default_ctor());
  test_move_ctor();
  static_assert(test_move_ctor());
  return 0;
}
