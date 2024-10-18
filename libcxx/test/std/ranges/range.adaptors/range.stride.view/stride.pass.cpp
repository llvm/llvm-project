//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr range_difference_t<_View> stride() const noexcept;

#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  using View = BasicTestView<cpp17_input_iterator<int*>>;
  int arr[]{1, 2, 3};
  auto arrv(View(cpp17_input_iterator<int*>(arr), cpp17_input_iterator<int*>(arr + 3)));
  // Mark str const so that we confirm that stride is a const member function.
  const std::ranges::stride_view<View> str(arrv, 1);

  // Make sure that stride member function is noexcept.
  static_assert(noexcept(str.stride()));
  // Make sure that the type returned by stride matches what is expected.
  ASSERT_SAME_TYPE(std::ranges::range_difference_t<View>, decltype(str.stride()));
  // Make sure that we get back a stride equal to the one that we gave in the ctor.
  assert(str.stride() == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
