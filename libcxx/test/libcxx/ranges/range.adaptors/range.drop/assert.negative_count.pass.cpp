//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <ranges>

// Make sure views::drop catches a negative count.

#include <ranges>

#include <span>
#include <string_view>
#include <vector>

#include "check_assertion.h"
#include "test_iterators.h"

struct SizedViewWithUnsizedSentinel : std::ranges::view_base {
  using iterator = random_access_iterator<int*>;
  using sentinel = sentinel_wrapper<random_access_iterator<int*>>;

  int* begin_ = nullptr;
  int* end_   = nullptr;
  SizedViewWithUnsizedSentinel(int* begin, int* end) : begin_(begin), end_(end) {}

  iterator begin() const { return iterator(begin_); }
  sentinel end() const { return sentinel(iterator(end_)); }
  std::size_t size() const { return end_ - begin_; }
};

int main(int, char**) {
  int a[] = {1, 2, 3, 4};

  { // Goes through drop_view.
    std::vector<int> v(a, a + 4);
    TEST_LIBCPP_ASSERT_FAILURE(std::views::drop(v, -1), "count must be greater than or equal to zero.");
  }

  { // The overloads that return the same range type skip drop_view.
    TEST_LIBCPP_ASSERT_FAILURE(
        std::views::drop(std::span<int>(a), -1), "views::drop(range, count) called with a negative count");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::views::drop(std::string_view("abcd"), -1), "views::drop(range, count) called with a negative count");
    TEST_LIBCPP_ASSERT_FAILURE(std::views::drop(std::ranges::subrange(a, a + 4), -1),
                               "views::drop(range, count) called with a negative count");

    using View     = SizedViewWithUnsizedSentinel;
    using Subrange = std::ranges::subrange<View::iterator, View::sentinel, std::ranges::subrange_kind::sized>;
    View view(a, a + 4);
    Subrange sized_subrange(view.begin(), view.end(), 4);
    TEST_LIBCPP_ASSERT_FAILURE(
        std::views::drop(sized_subrange, -1), "views::drop(range, count) called with a negative count");
  }

  return 0;
}
