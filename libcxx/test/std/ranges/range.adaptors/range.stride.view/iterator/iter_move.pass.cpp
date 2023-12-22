//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  friend constexpr range_rvalue_reference_t<_Base> iter_move(__iterator const& __it)
//         noexcept(noexcept(ranges::iter_move(__it.__current_)))

#include <ranges>

#include "../types.h"

template <typename T>
concept iter_moveable = requires(T&& t) { std::ranges::iter_move(t); };

template <bool NoExcept = true>
struct MaybeExceptIterSwapIterator : InputIterBase<MaybeExceptIterSwapIterator<NoExcept>> {
  int* counter_{nullptr};
  constexpr MaybeExceptIterSwapIterator()                                                    = default;
  constexpr MaybeExceptIterSwapIterator(const MaybeExceptIterSwapIterator&)                  = default;
  constexpr MaybeExceptIterSwapIterator(MaybeExceptIterSwapIterator&&)                       = default;
  constexpr MaybeExceptIterSwapIterator& operator=(const MaybeExceptIterSwapIterator& other) = default;
  constexpr MaybeExceptIterSwapIterator& operator=(MaybeExceptIterSwapIterator&& other)      = default;

  constexpr explicit MaybeExceptIterSwapIterator(int* counter) : counter_(counter) {}

  friend constexpr int iter_move(const MaybeExceptIterSwapIterator& t) {
    (*t.counter_)++;
    return 5;
  }
  friend constexpr int iter_move(const MaybeExceptIterSwapIterator& t) noexcept
    requires NoExcept
  {
    (*t.counter_)++;
    return 5;
  }

  constexpr int operator*() const { return 5; }
};

template <bool NoExcept = true>
struct IterSwapRange : std::ranges::view_base {
  MaybeExceptIterSwapIterator<NoExcept> begin_;
  MaybeExceptIterSwapIterator<NoExcept> end_;
  constexpr IterSwapRange(int* counter)
      : begin_(MaybeExceptIterSwapIterator<NoExcept>(counter)), end_(MaybeExceptIterSwapIterator<NoExcept>(counter)) {}
  constexpr MaybeExceptIterSwapIterator<NoExcept> begin() const { return begin_; }
  constexpr MaybeExceptIterSwapIterator<NoExcept> end() const { return end_; }
};

constexpr bool test() {
  {
    int iter_move_counter(0);
    using View       = IterSwapRange<true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(&iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(noexcept(std::ranges::iter_move(svb)));

    [[maybe_unused]] auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
  }

  {
    int iter_move_counter(0);
    using View       = IterSwapRange<false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(&iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(!noexcept(std::ranges::iter_move(svb)));

    [[maybe_unused]] auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
