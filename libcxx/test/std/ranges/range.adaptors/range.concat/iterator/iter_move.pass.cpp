//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// friend constexpr decltype(auto) iter_move(const __iterator& __it) noexcept(
//       ((is_nothrow_invocable_v< decltype(ranges::iter_move), const iterator_t<__maybe_const<_Const, _Views>>& > &&
//         is_nothrow_convertible_v< range_rvalue_reference_t<__maybe_const<_Const, _Views>>,
//                                   __concat_rvalue_reference_t<__maybe_const<_Const, _Views>...> >) &&
//        ...))

// REQUIRES: std-at-least-c++26

#include <array>
#include <cassert>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <bool NoThrow>
struct ThowingIter {
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = int;

  int* p = nullptr;

  constexpr int& operator*() const noexcept { return *p; }
  constexpr ThowingIter& operator++() noexcept {
    ++p;
    return *this;
  }
  constexpr ThowingIter operator++(int) noexcept {
    ThowingIter tmp = *this;
    ++*this;
    return tmp;
  }
  friend constexpr bool operator==(ThowingIter, ThowingIter) = default;

  friend constexpr int&& iter_move(const ThowingIter& it) noexcept(NoThrow) { return std::move(*it.p); }
};

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct ThrowingValue {
  int v{};
  ThrowingValue() = default;
  explicit ThrowingValue(int x) noexcept(false) : v(x) {}
  ThrowingValue(const ThrowingValue&) noexcept(false) = default;
  ThrowingValue(ThrowingValue&&) noexcept(false)      = default;
};

template <bool DerefNoThrow>
struct PValIter {
  using iterator_concept  = std::input_iterator_tag;
  using iterator_category = std::input_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = std::conditional_t<DerefNoThrow, int, ThrowingValue>;

  int* p = nullptr;

  decltype(auto) operator*() const noexcept(DerefNoThrow) {
    if constexpr (DerefNoThrow)
      return *p; // int (noexcept)
    else
      return ThrowingValue{*p}; // not noexcept
  }
  PValIter& operator++() noexcept {
    ++p;
    return *this;
  }
  void operator++(int) noexcept { ++p; }
  friend bool operator==(PValIter, PValIter) = default;
};

static_assert(std::input_iterator<LRefIter<true>>);
static_assert(std::input_iterator<LRefIter<false>>);
static_assert(std::input_iterator<PValIter<true>>);
static_assert(std::input_iterator<PValIter<false>>);

template <class Iter>
struct MiniView : std::ranges::view_base {
  Iter b{}, e{};
  constexpr MiniView() = default;
  constexpr MiniView(Iter first, Iter last) : b(first), e(last) {}
  constexpr Iter begin() const noexcept { return b; }
  constexpr Iter end() const noexcept { return e; }
};

constexpr bool test() {
  int buf1[] = {1, 2, 3, 4};
  int buf2[] = {5, 6, 7};
  {
    // All underlying iter_move are noexcept -> concat iterator's iter_move is noexcept
    using I1 = ThowingIter<true>;
    using S1 = sentinel_wrapper<I1>;
    using V1 = MiniView<I1>;
    V1 v1(I1(buf1), I1(buf1 + 4));
    V1 v2(I1(buf2), I1(buf2 + 3));

    auto cv     = std::views::concat(v1, v2);
    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(noexcept(std::ranges::iter_move(std::declval<Iter&>())));
    static_assert(noexcept(std::ranges::iter_move(std::declval<CIter&>())));

    auto it = cv.begin();
    (void)std::ranges::iter_move(it);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
