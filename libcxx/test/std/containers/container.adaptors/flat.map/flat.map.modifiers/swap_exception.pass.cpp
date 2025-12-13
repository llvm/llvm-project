//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-localization
// UNSUPPORTED: no-exceptions

// <flat_map>

// void swap(flat_map& y) noexcept;
// friend void swap(flat_map& x, flat_map& y) noexcept

// Test that the invariants are maintained if any exception is thrown during swap

#include <flat_map>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>

#include "test_macros.h"
#include "../helpers.h"

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
    try {
      swap_function(m1, m2);
      assert(false);
    } catch (int) {
      check_invariant(m1);
      check_invariant(m2);
      LIBCPP_ASSERT(m1.size() == 0);
      LIBCPP_ASSERT(m2.size() == 0);
    }
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

    try {
      swap_function(m1, m2);
      assert(false);
    } catch (int) {
      check_invariant(m1);
      check_invariant(m2);
      LIBCPP_ASSERT(m1.size() == 0);
      LIBCPP_ASSERT(m2.size() == 0);
    }
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
