//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit stride_view(_View, range_difference_t<_View>)

#include <type_traits>

#include "test_convertible.h"
#include "test_iterators.h"
#include "types.h"

// There is no default ctor for stride_view.
using View = BasicTestView<cpp17_input_iterator<int*>>;
static_assert(!std::is_default_constructible_v<std::ranges::stride_view<View>>);

// Test that the stride_view can only be explicitly constructed.
static_assert(!test_convertible<std::ranges::stride_view<View>, View, int>());

constexpr bool test() {
  {
    int arr[] = {1, 2, 3};
    // Test that what we will stride over is move only.
    static_assert(!std::is_copy_constructible_v<MoveOnlyView<cpp17_input_iterator<int*>>>);
    static_assert(std::is_move_constructible_v<MoveOnlyView<cpp17_input_iterator<int*>>>);

    MoveOnlyView<cpp17_input_iterator<int*>> mov(cpp17_input_iterator<int*>(arr), cpp17_input_iterator<int*>(arr + 3));
    // Because MoveOnlyView is, well, move only, we can test that it is moved
    // from when the stride view is constructed.
    std::ranges::stride_view<MoveOnlyView<cpp17_input_iterator<int*>>> mosv(std::move(mov), 1);

    // While we are here, make sure that the ctor captured the proper things
    assert(mosv.stride() == 1);

    auto mosv_i = mosv.begin();
    assert(*mosv_i == 1);

    mosv_i++;
    assert(*mosv_i == 2);

    mosv_i++;
    assert(*mosv_i == 3);

    mosv_i++;
    assert(mosv_i == mosv.end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
