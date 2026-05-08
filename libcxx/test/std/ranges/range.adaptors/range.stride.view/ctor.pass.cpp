//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit stride_view(V base, range_difference_t<V> stride)

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_convertible.h"
#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  {
    // There is no default ctor for stride_view.
    using View = BasicTestView<cpp17_input_iterator<int*>>;
    static_assert(!std::is_default_constructible_v<std::ranges::stride_view<View>>);

    // Test that the stride_view can only be explicitly constructed.
    static_assert(!test_convertible<std::ranges::stride_view<View>, View, int>());
  }

  {
    int arr[] = {1, 2, 3};
    // Test that what we will stride over is move only.
    using View = MoveOnlyView<cpp17_input_iterator<int*>>;
    static_assert(!std::is_copy_constructible_v<View>);
    static_assert(std::is_move_constructible_v<View>);

    View mov(cpp17_input_iterator<int*>(arr), cpp17_input_iterator<int*>(arr + 3));
    // Because MoveOnlyView is, well, move only, we can test that it is moved
    // from when the stride view is constructed.
    std::ranges::stride_view<View> strided(std::move(mov), 1);

    // While we are here, make sure that the ctor captured the stride.
    assert(strided.stride() == 1);
  }
  {
    // Verify salient properties after construction.
    int arr[]    = {10, 20, 30, 40, 50};
    using Base   = BasicTestView<int*, int*>;
    auto strided = std::ranges::stride_view(Base(arr, arr + 5), 2);

    assert(strided.stride() == 2);
    assert(*strided.begin() == 10);

    auto it = strided.begin();
    ++it;
    assert(*it == 30);
    ++it;
    assert(*it == 50);
    ++it;
    assert(it == strided.end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
