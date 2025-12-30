//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// REQUIRES: libcpp-hardening-mode={{fast|extensive|debug}}
// XFAIL:libcpp-hardening-mode=debug && availability-verbose_abort-missing

// constexpr stride_view::<iterator>& operator++()
// constexpr __iterator& operator++()
// constexpr void operator++(int) {
// constexpr __iterator operator++(int)

#include <ranges>

#include "check_assertion.h"
#include "test_iterators.h"

#include "../../../../../std/ranges/range.adaptors/range.stride.view/types.h"

int main(int, char**) {
  {
    int range[] = {1, 2, 3};
    using Base  = BasicTestView<cpp17_input_iterator<int*>>;
    auto view   = std::ranges::views::stride(Base(cpp17_input_iterator(range), cpp17_input_iterator(range + 3)), 3);
    auto it     = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(it++, "Cannot increment an iterator already at the end.");
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Cannot increment an iterator already at the end.");
  }
  {
    int range[] = {1, 2, 3};
    using Base  = BasicTestView<forward_iterator<int*>, forward_iterator<int*>>;
    auto view   = std::ranges::views::stride(Base(forward_iterator(range), forward_iterator(range + 3)), 3);
    auto it     = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(it++, "Cannot increment an iterator already at the end.");
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Cannot increment an iterator already at the end.");
  }
  return 0;
}
