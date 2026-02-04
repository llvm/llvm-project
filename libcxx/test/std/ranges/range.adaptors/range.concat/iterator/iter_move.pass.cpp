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

#include "../types.h"

struct ConvMayThrow {
  int v;
  operator int() { return v; }
};

struct ConvNoThrow {
  int v;
  operator int() noexcept { return v; }
};

template <typename T, bool NoThrow>
struct ThowingIter {
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;

  T* p = nullptr;

  constexpr T& operator*() const noexcept { return *p; }
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

  friend constexpr decltype(auto) iter_move(const ThowingIter& it) noexcept(NoThrow) { return std::move(*it.p); }
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

template <class Iter, class Sentinel>
struct MiniView : std::ranges::view_base {
  Iter b{};
  Sentinel e{};
  constexpr MiniView() = default;
  constexpr MiniView(Iter first, Sentinel last) : b(first), e(last) {}
  constexpr Iter begin() const noexcept { return b; }
  constexpr Sentinel end() const noexcept { return e; }
};

constexpr bool test() {
  int arr[2]                   = {1, 2};
  ConvMayThrow maythrow_buf[2] = {{3}, {4}};
  ConvNoThrow nothrow_buf[2]   = {{5}, {6}};
  {
    // All underlying iter_move are noexcept
    // => concat iter_move has noexcept(true)
    using Iter_NoThrow     = ThowingIter<int, true>;
    using Sentinel_NoThrow = sentinel_wrapper<Iter_NoThrow>;
    using View_NoThrow     = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    View_NoThrow v1(Iter_NoThrow(arr), Sentinel_NoThrow(Iter_NoThrow(arr + 1)));
    View_NoThrow v2(Iter_NoThrow(arr), Sentinel_NoThrow(Iter_NoThrow(arr + 1)));

    auto cv     = std::views::concat(v1, v2);
    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  {
    // All underlying iter_move are noexcept
    // underlying ranges have different
    // => concat iter_move has noexcept(true)
    using Iter_NoThrow     = ThowingIter<int, true>;
    using Sentinel_NoThrow = sentinel_wrapper<Iter_NoThrow>;
    using View_NoThrow     = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    View_NoThrow v1(Iter_NoThrow(arr), Sentinel_NoThrow(Iter_NoThrow(arr + 1)));
    View_NoThrow v2(Iter_NoThrow(arr), Sentinel_NoThrow(Iter_NoThrow(arr + 1)));

    auto cv     = std::views::concat(v1, v2);
    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  {
    // One underlying may throw
    // concat iter_move has noexcept(false)
    using Iter_NoThrow     = ThowingIter<int, true>;
    using Iter_Throw       = ThowingIter<int, false>;
    using Sentinel_NoThrow = sentinel_wrapper<Iter_NoThrow>;
    using Sentinel_Throw   = sentinel_wrapper<Iter_Throw>;
    using View_NoThrow     = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    using View_Throw       = MiniView<Iter_Throw, Sentinel_Throw>;

    auto cv = std::views::concat(View_NoThrow{Iter_NoThrow{arr}, Sentinel_NoThrow{Iter_NoThrow{arr + 1}}},
                                 View_Throw{Iter_Throw{arr}, Sentinel_Throw{Iter_Throw{arr + 1}}});

    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(!noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(!noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  {
    // one underlying iter_move may throw, convert ConvNoThrow to int has noexcept
    // => iter_move has noexcept(false)
    using Iter_NoThrow         = ThowingIter<int, true>;
    using IterConv_NoThrow     = ThowingIter<ConvNoThrow, false>;
    using Sentinel_NoThrow     = sentinel_wrapper<Iter_NoThrow>;
    using SentinelConv_NoThrow = sentinel_wrapper<IterConv_NoThrow>;
    using View_NoThrow         = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    using ViewHasConv_NoThrow  = MiniView<IterConv_NoThrow, SentinelConv_NoThrow>;

    auto cv = std::views::concat(
        View_NoThrow{Iter_NoThrow{arr}, Sentinel_NoThrow{Iter_NoThrow{arr + 1}}},
        ViewHasConv_NoThrow{IterConv_NoThrow{nothrow_buf}, SentinelConv_NoThrow{IterConv_NoThrow{nothrow_buf + 1}}});

    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(!noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(!noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  {
    // all underlying iter_move has noexcept, and convert ConvNoThrow to int has noexcept
    // => concat iter_move has noexcept(true)
    using Iter_NoThrow         = ThowingIter<int, true>;
    using IterConv_NoThrow     = ThowingIter<ConvNoThrow, true>;
    using Sentinel_NoThrow     = sentinel_wrapper<Iter_NoThrow>;
    using SentinelConv_NoThrow = sentinel_wrapper<IterConv_NoThrow>;
    using View_NoThrow         = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    using ViewHasConv_NoThrow  = MiniView<IterConv_NoThrow, SentinelConv_NoThrow>;

    auto cv = std::views::concat(
        View_NoThrow{Iter_NoThrow{arr}, Sentinel_NoThrow{Iter_NoThrow{arr + 1}}},
        ViewHasConv_NoThrow{IterConv_NoThrow{nothrow_buf}, SentinelConv_NoThrow{IterConv_NoThrow{nothrow_buf + 1}}});

    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  {
    // underlying iter_move has noexcept, but convert ConvMayThrow to int is noexcept(false)
    // => concat iter_move has noexcept(false)
    using Iter_NoThrow          = ThowingIter<int, true>;
    using IterConv_MayThrow     = ThowingIter<ConvMayThrow, true>;
    using Sentinel_NoThrow      = sentinel_wrapper<Iter_NoThrow>;
    using SentinelConv_MayThrow = sentinel_wrapper<IterConv_MayThrow>;
    using View_NoThrow          = MiniView<Iter_NoThrow, Sentinel_NoThrow>;
    using ViewHasConv_MayThrow  = MiniView<IterConv_MayThrow, SentinelConv_MayThrow>;

    auto cv = std::views::concat(
        View_NoThrow{Iter_NoThrow{arr}, Sentinel_NoThrow{Iter_NoThrow{arr + 1}}},
        ViewHasConv_MayThrow{
            IterConv_MayThrow{maythrow_buf}, SentinelConv_MayThrow{IterConv_MayThrow{maythrow_buf + 1}}});

    using Iter  = decltype(cv.begin());
    using CIter = decltype(std::as_const(cv).begin());

    static_assert(!noexcept(std::ranges::iter_move(std::declval<Iter>())));
    static_assert(!noexcept(std::ranges::iter_move(std::declval<CIter>())));

    auto it = cv.begin();
    auto x  = std::ranges::iter_move(it);
    static_assert(std::is_same_v<decltype(x), int>);
    assert(x == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
