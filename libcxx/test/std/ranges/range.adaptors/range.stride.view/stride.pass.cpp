//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// constexpr range_difference_t<V> stride() const noexcept;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  using View = BasicTestView<cpp17_input_iterator<int*>>;
  int arr[]  = {1, 2, 3};
  View view(cpp17_input_iterator<int*>(arr), cpp17_input_iterator<int*>(arr + 3));

  const std::ranges::stride_view<View> strided(view, 2);
  static_assert(noexcept(strided.stride()));
  ASSERT_SAME_TYPE(std::ranges::range_difference_t<View>, decltype(strided.stride()));
  assert(strided.stride() == 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
