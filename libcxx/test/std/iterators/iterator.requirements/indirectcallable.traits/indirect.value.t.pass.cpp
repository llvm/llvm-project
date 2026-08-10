//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<indirectly_readable T>
// using indirect-value-t = see below; // exposition only

#include <cassert>

#include <algorithm>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

TEST_CONSTEXPR_CXX23 void test() {
  auto ints             = std::views::iota(0, 5);
  auto unique_ptr_maker = []<std::movable T>(T v) { return std::make_unique<T>(std::move(v)); };

  using iota_iter             = std::ranges::iterator_t<decltype(ints)>;
  using unique_ptr_projection = decltype(unique_ptr_maker);
  using projected_iter        = std::projected<iota_iter, unique_ptr_projection>;

  { // Check std::indirectly_unary_invocable
    auto consume = [](auto) {};
    static_assert(std::indirectly_unary_invocable<decltype(consume), projected_iter>);

    std::ranges::for_each(ints, consume, unique_ptr_maker);
    std::ranges::for_each(ints.begin(), ints.end(), consume, unique_ptr_maker);
  }

  { // Check std::indirectly_regular_unary_invocable
    static_assert(std::indirectly_regular_unary_invocable<decltype([](auto) {}), projected_iter>);
    using check_wellformedness [[maybe_unused]] = std::projected<projected_iter, unique_ptr_projection>;
  }

  { // Check std::indirect_unary_predicate
    auto unary_pred = [](auto) { return false; };
    static_assert(std::indirect_unary_predicate<decltype(unary_pred), projected_iter>);

    assert(std::ranges::find_if(ints, unary_pred, unique_ptr_maker) == ints.end());
    assert(std::ranges::find_if(ints.begin(), ints.end(), unary_pred, unique_ptr_maker) == ints.end());
    assert(std::ranges::count_if(ints, unary_pred, unique_ptr_maker) == 0);
    assert(std::ranges::count_if(ints.begin(), ints.end(), unary_pred, unique_ptr_maker) == 0);
  }

  { // Check std::indirect_binary_predicate
    auto binary_pred = [](auto, auto) { return false; };
    static_assert(std::indirect_binary_predicate<decltype(binary_pred), projected_iter, projected_iter>);

    assert(std::ranges::adjacent_find(ints, binary_pred, unique_ptr_maker) == ints.end());
    assert(std::ranges::adjacent_find(ints.begin(), ints.end(), binary_pred, unique_ptr_maker) == ints.end());
  }

  { // Check std::indirect_equivalence_relation
    auto rel = [](auto, auto) { return false; };
    static_assert(std::indirect_equivalence_relation<decltype(rel), projected_iter>);

    std::vector<int> out;
    (void)std::ranges::unique_copy(ints, std::back_inserter(out), rel, unique_ptr_maker);
    (void)std::ranges::unique_copy(ints.begin(), ints.end(), std::back_inserter(out), rel, unique_ptr_maker);
  }

  { // Check std::indirect_strict_weak_order
    auto rel = [](auto x, auto y) { return *x < *y; };
    static_assert(std::indirect_strict_weak_order<decltype(rel), projected_iter>);

    assert(std::ranges::is_sorted_until(ints, rel, unique_ptr_maker) == ints.end());
    assert(std::ranges::is_sorted_until(ints.begin(), ints.end(), rel, unique_ptr_maker) == ints.end());
  }
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert((test(), true));
#endif
  return 0;
}
