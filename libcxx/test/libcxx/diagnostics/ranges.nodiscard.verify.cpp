//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that ranges are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: c++03, c++11, c++14, c++17


#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

// clang-format off

void test() {
  std::vector<int> range;
  std::ranges::less_equal pred;

  std::views::drop(pred);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
  auto rvalue_view = std::views::as_rvalue(range);
  std::views::as_rvalue(range);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::as_rvalue(rvalue_view); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::chunk_by(pred);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk_by(range, pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::drop(std::views::repeat(1)); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // std::ranges::enumerate
  {
    // test_range<cpp17_input_iterator<std::tuple<int, int*>>> r;
    // test_sized_range<cpp20_random_access_iterator> r;
    std::views::enumerate(range);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // test_sized_view<cpp17_input_iterator> v;
    // class enumerate_view<>
    auto ev = std::ranges::subrange(range.begin(), range.end() - 1) | std::views::enumerate;
    ev.begin();                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(ev).begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ev.end();                    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(ev).end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ev.size();                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(ev).size();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ev.base();                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(ev).base();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // class enumerate_view<>::__iterator
    auto it = ev.begin();
    auto const_it = std::as_const(ev).begin();
    it.base();                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(it).base();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::as_const(it).index();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    *std::as_const(it);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[2];                       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it == it;                    // expected-warning {{equality comparison result unused}}
    it <=> it;                   // expected-warning {{three-way comparison result unused}}
    it + 1;                      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;                      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;                     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(it);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // class enumerate_view<>::__sentinel
    // auto ev2 = std::ranges::subrange(range.begin(), range.end() - 1); | std::views::enumerate;
    auto st = ev.end();
    auto const_st = std::as_const(ev).end();
    st.base();                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it == st;                    // expected-warning {{equality comparison result unused}}
    it == const_st;              // expected-warning {{equality comparison result unused}}
    const_it == st;              // expected-warning {{equality comparison result unused}}
    const_it == const_st;        // expected-warning {{equality comparison result unused}}
    st == it;                    // expected-warning {{equality comparison result unused}}
    st == const_it;              // expected-warning {{equality comparison result unused}}
    const_st == it;              // expected-warning {{equality comparison result unused}}
    const_st == const_it;        // expected-warning {{equality comparison result unused}}
    it - st;                     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - it;                     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  std::views::repeat(1, std::unreachable_sentinel); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_STD_VER >= 23

  std::views::split(range, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::split(1);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::take(range, 3);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(1);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
  std::views::take(std::views::repeat(3), 3);                            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::take(std::views::repeat(3, std::unreachable_sentinel), 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::allocator<int> alloc;

  std::ranges::to<std::vector<int>>(range);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector<int>>(range, alloc);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(range);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(range, alloc);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector<int>>();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector<int>>(alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector>();           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::ranges::to<std::vector>(alloc);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif // TEST_STD_VER >= 23
}
