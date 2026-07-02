//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(...)
//   requires (indirectly_swappable<iterator_t<maybe-const<Const, First>>>
//             && (indirectly_swappable<iterator_t<maybe-const<Const, Vs>>> && ...));

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>

#include "../../range_adaptor_types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
  ThrowingMove& operator=(ThrowingMove&&) { return *this; }
};

// A view whose iterator is not indirectly_swappable (no rvalue assignment).
struct ImmovableElement {
  int v                        = 0;
  constexpr ImmovableElement() = default;
  constexpr ImmovableElement(int x) : v(x) {}
  ImmovableElement(const ImmovableElement&)            = delete;
  ImmovableElement& operator=(const ImmovableElement&) = delete;
};

constexpr bool test() {
  { // basic: swap two random-access iterators
    std::array a{1, 2, 3, 4};
    std::array b{0.1, 0.2, 0.3};

    std::ranges::cartesian_product_view v(a, b);
    auto it1 = v.begin();
    auto it2 = ++v.begin();

    std::ranges::iter_swap(it1, it2);

    // it1 was (a[0], b[0]), it2 was (a[0], b[1]).
    // Both the int& and double& slots are swapped.
    assert(b[0] == 0.2);
    assert(b[1] == 0.1);
    // The first range's iterators both pointed at a[0], so swap(a[0], a[0]) is a no-op.
    assert(a[0] == 1);

    static_assert(noexcept(std::ranges::iter_swap(it1, it2)));
  }

  { // throwing move underneath -> cartesian iter_swap is not noexcept
    std::array<ThrowingMove, 2> data{};
    std::ranges::cartesian_product_view v(data);
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_swap(it, it)));
  }

  { // ADL-customised iter_swap on the underlying iterators is invoked, once per range
    adltest::IterMoveSwapRange r1{}, r2{};
    assert(r1.iter_swap_called_times == 0);
    assert(r2.iter_swap_called_times == 0);

    std::ranges::cartesian_product_view v{r1, r2};
    auto it1 = v.begin();
    auto it2 = std::ranges::next(it1, 3);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 2);
    assert(r2.iter_swap_called_times == 2);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 4);
    assert(r2.iter_swap_called_times == 4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
