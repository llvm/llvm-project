//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// `check_assertion.h` requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: no-localization
// UNSUPPORTED: no-exceptions

// <flat_map>

// void swap(flat_map& y) noexcept;
// friend void swap(flat_map& x, flat_map& y) noexcept

// Test that std::terminate is called if any exception is thrown during swap

#include <flat_map>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "../helpers.h"
#include "check_assertion.h"

template <class F>
void test_swap_exception_guarantee([[maybe_unused]] F&& swap_function) {
  {
    // key swap throws
    using KeyContainer   = ThrowOnMoveContainer<int>;
    using ValueContainer = std::vector<int>;
    using M              = std::flat_map<int, int, TransparentComparator, KeyContainer, ValueContainer>;

    M m1, m2;
    m1.emplace(1, 1);
    m1.emplace(2, 2);
    m2.emplace(3, 3);
    m2.emplace(4, 4);
    // swap is noexcept
    EXPECT_STD_TERMINATE([&] { swap_function(m1, m2); });
  }

  {
    // value swap throws
    using KeyContainer   = std::vector<int>;
    using ValueContainer = ThrowOnMoveContainer<int>;
    using M              = std::flat_map<int, int, TransparentComparator, KeyContainer, ValueContainer>;

    M m1, m2;
    m1.emplace(1, 1);
    m1.emplace(2, 2);
    m2.emplace(3, 3);
    m2.emplace(4, 4);

    // swap is noexcept
    EXPECT_STD_TERMINATE([&] { swap_function(m1, m2); });
  }
}

int main(int, char**) {
  {
    auto swap_func = [](auto& m1, auto& m2) { swap(m1, m2); };
    test_swap_exception_guarantee(swap_func);
  }

  {
    auto swap_func = [](auto& m1, auto& m2) { m1.swap(m2); };
    test_swap_exception_guarantee(swap_func);
  }

  return 0;
}
