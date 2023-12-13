//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  friend constexpr void iter_swap(__iterator const& __x, __iterator const& __y)
//  noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
//  requires indirectly_swappable<iterator_t<_Base>>

#include <ranges>

#include "../test.h"

template <typename T>
concept swappable = requires(T&& t, T&& u) { std::ranges::iter_swap(t, u); };

template <bool Swappable = true, bool NoExcept = true>
struct MaybeExceptIterSwapIterator : InputIterBase<MaybeExceptIterSwapIterator<Swappable, NoExcept>> {
  int* counter_{nullptr};
  constexpr MaybeExceptIterSwapIterator()                                                    = default;
  constexpr MaybeExceptIterSwapIterator(const MaybeExceptIterSwapIterator&)                  = default;
  constexpr MaybeExceptIterSwapIterator(MaybeExceptIterSwapIterator&&)                       = default;
  constexpr MaybeExceptIterSwapIterator& operator=(const MaybeExceptIterSwapIterator& other) = default;
  constexpr MaybeExceptIterSwapIterator& operator=(MaybeExceptIterSwapIterator&& other)      = default;

  constexpr explicit MaybeExceptIterSwapIterator(int* counter) : counter_(counter) {}

  friend constexpr void iter_swap(const MaybeExceptIterSwapIterator& t, const MaybeExceptIterSwapIterator& u) noexcept
    requires Swappable && NoExcept
  {
    (*t.counter_)++;
    (*u.counter_)++;
  }

  friend constexpr void iter_swap(const MaybeExceptIterSwapIterator& t, const MaybeExceptIterSwapIterator& u)
    requires Swappable && (!NoExcept)
  {
    (*t.counter_)++;
    (*u.counter_)++;
  }
  constexpr int operator*() const { return 5; }
};

template <bool Swappable = true, bool NoExcept = true>
struct IterSwapRange : std::ranges::view_base {
  MaybeExceptIterSwapIterator<Swappable, NoExcept> begin_;
  MaybeExceptIterSwapIterator<Swappable, NoExcept> end_;
  constexpr IterSwapRange(int* counter)
      : begin_(MaybeExceptIterSwapIterator<Swappable, NoExcept>(counter)),
        end_(MaybeExceptIterSwapIterator<Swappable, NoExcept>(counter)) {}
  constexpr MaybeExceptIterSwapIterator<Swappable, NoExcept> begin() const { return begin_; }
  constexpr MaybeExceptIterSwapIterator<Swappable, NoExcept> end() const { return end_; }
};

constexpr bool test() {
  {
    int iter_move_counter_one(0);
    int iter_move_counter_two(0);
    using View       = IterSwapRange<true, true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svba        = StrideView(View(&iter_move_counter_one), 1).begin();
    auto svbb        = StrideView(View(&iter_move_counter_two), 1).begin();

    static_assert(swappable<std::ranges::iterator_t<StrideView>>);
    static_assert(noexcept(std::ranges::iter_swap(svba, svbb)));

    std::ranges::iter_swap(svba, svbb);
    assert(iter_move_counter_one == 1);
    assert(iter_move_counter_two == 1);
  }

  {
    int iter_move_counter_one(0);
    int iter_move_counter_two(0);
    using View       = IterSwapRange<true, false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svba        = StrideView(View(&iter_move_counter_one), 1).begin();
    auto svbb        = StrideView(View(&iter_move_counter_two), 1).begin();

    static_assert(swappable<std::ranges::iterator_t<StrideView>>);
    static_assert(!noexcept(std::ranges::iter_swap(svba, svbb)));

    std::ranges::iter_swap(svba, svbb);
    assert(iter_move_counter_one == 1);
    assert(iter_move_counter_two == 1);
  }

  {
    using View       = IterSwapRange<false, false>;
    using StrideView = std::ranges::stride_view<View>;

    static_assert(!swappable<std::ranges::iterator_t<StrideView>>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
