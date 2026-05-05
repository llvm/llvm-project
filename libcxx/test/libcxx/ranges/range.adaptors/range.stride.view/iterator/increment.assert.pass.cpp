//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// REQUIRES: libcpp-hardening-mode={{fast|extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// constexpr __iterator& operator++()
// constexpr void operator++(int)
// constexpr __iterator operator++(int)

#include <ranges>

#include "check_assertion.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
struct MinimalView : std::ranges::view_base {
  Iter begin_;
  Sent end_;

  constexpr MinimalView(Iter b, Sent e) : begin_(b), end_(e) {}
  MinimalView(MinimalView&&)            = default;
  MinimalView& operator=(MinimalView&&) = default;

  constexpr Iter begin() const { return begin_; }
  constexpr Sent end() const { return end_; }
};

int main(int, char**) {
  {
    int range[] = {1, 2, 3};
    using View  = MinimalView<cpp17_input_iterator<int*>>;
    auto view   = std::ranges::views::stride(
        View(cpp17_input_iterator(range), sentinel_wrapper(cpp17_input_iterator(range + 3))), 3);
    auto it = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(it++, "Cannot increment an iterator already at the end.");
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Cannot increment an iterator already at the end.");
  }
  {
    int range[] = {1, 2, 3};
    using View  = MinimalView<forward_iterator<int*>, forward_iterator<int*>>;
    auto view   = std::ranges::views::stride(View(forward_iterator(range), forward_iterator(range + 3)), 3);
    auto it     = view.begin();
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(it++, "Cannot increment an iterator already at the end.");
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Cannot increment an iterator already at the end.");
  }
  {
    int range[] = {1, 2, 3};
    using View  = MinimalView<forward_iterator<int*>, forward_iterator<int*>>;
    auto view   = std::ranges::views::stride(View(forward_iterator(range), forward_iterator(range + 3)), 3);
    auto it     = view.end();
    TEST_LIBCPP_ASSERT_FAILURE(it++, "Cannot increment an iterator already at the end.");
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Cannot increment an iterator already at the end.");
  }
  return 0;
}
