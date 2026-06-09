//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// friend constexpr void iter_swap(const __iterator& __x, const __iterator& __y)
//     noexcept((noexcept(ranges::swap(*__x, *__y))) &&
//               (noexcept(ranges::iter_swap(std::declval<const iterator_t<__maybe_const<_Const, _Views>>>(),
//                                           std::declval<const iterator_t<__maybe_const<_Const, _Views>>>())) &&
//               ...))

// REQUIRES: std-at-least-c++26

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"
#include "../../range_adaptor_types.h"

template <class It>
concept has_iter_swap = requires(It it) { std::ranges::iter_swap(it, it); };

struct Elem {
  int v;
  friend constexpr void swap(Elem& a, Elem& b) noexcept { std::ranges::swap(a.v, b.v); }
};

template <typename T>
struct SwapIter {
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;

  T* p = nullptr;

  constexpr T& operator*() const noexcept { return *p; }
  constexpr SwapIter& operator++() noexcept {
    ++p;
    return *this;
  }
  constexpr SwapIter operator++(int) noexcept {
    SwapIter tmp = *this;
    ++*this;
    return tmp;
  }
  friend constexpr bool operator==(SwapIter, SwapIter) = default;

  friend constexpr void iter_swap(const SwapIter& it1, const SwapIter& it2) { std::ranges::swap(*it1, *it2); }
};

template <typename T>
struct SwapIterNoCustom {
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;

  T* p = nullptr;

  constexpr T& operator*() const noexcept { return *p; }
  constexpr SwapIterNoCustom& operator++() noexcept {
    ++p;
    return *this;
  }
  constexpr SwapIterNoCustom operator++(int) noexcept {
    SwapIterNoCustom tmp = *this;
    ++*this;
    return tmp;
  }
  friend constexpr bool operator==(SwapIterNoCustom, SwapIterNoCustom) = default;
};

template <typename T>
struct SwapIterTracked {
  using iterator_concept  = std::forward_iterator_tag;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;

  T* p                        = nullptr;
  int* iter_swap_called_times = nullptr;

  constexpr T& operator*() const noexcept { return *p; }
  constexpr SwapIterTracked& operator++() noexcept {
    ++p;
    return *this;
  }
  constexpr SwapIterTracked operator++(int) noexcept {
    SwapIterTracked tmp = *this;
    ++*this;
    return tmp;
  }
  friend constexpr bool operator==(const SwapIterTracked& x, const SwapIterTracked& y) { return x.p == y.p; }

  friend constexpr void iter_swap(const SwapIterTracked& x, const SwapIterTracked& y) {
    if (x.iter_swap_called_times)
      ++*x.iter_swap_called_times;
    std::ranges::swap(*x, *y);
  }
};

template <typename Iter, class Sentinel>
struct MiniView : std::ranges::view_base {
  Iter b{};
  Sentinel e{};
  constexpr MiniView() = default;
  constexpr MiniView(Iter first, Sentinel last) : b(first), e(last) {}
  constexpr Iter begin() const noexcept { return b; }
  constexpr Sentinel end() const noexcept { return e; }
};

constexpr bool test() {
  using IteratorA = SwapIter<Elem>;
  using SentinelA = sentinel_wrapper<IteratorA>;
  using IteratorB = SwapIterNoCustom<Elem>;
  using SentinelB = sentinel_wrapper<IteratorB>;
  using ViewA     = MiniView<IteratorA, SentinelA>;
  using ViewB     = MiniView<IteratorB, SentinelB>;

  {
    Elem a1[2]{{1}, {2}};
    Elem a2[2]{{3}, {4}};

    ViewA v1{IteratorA(a1), SentinelA(IteratorA(a1 + 2))};
    ViewB v2{IteratorB(a2), SentinelB(IteratorB(a2 + 2))};

    std::ranges::concat_view cv(v1, v2);

    auto it1 = cv.begin();
    auto it2 = ++cv.begin();
    it2++;

    // always false: https://cplusplus.github.io/LWG/lwg-active.html#4489
    static_assert(noexcept(std::ranges::iter_swap(it1, it2)) == false);

    std::ranges::iter_swap(it1, it2);

    // iter_swap
    assert(a1[0].v == 3 && a2[0].v == 1);
  }

  // Test that the underlying iterator's iter_swap specialization is called
  // when both iterators point to the same underlying range
  {
    int iter_swap_call_count = 0;
    Elem a1[3]{{10}, {20}, {30}};
    Elem a2[2]{{40}, {50}};

    using TrackedIter     = SwapIterTracked<Elem>;
    using TrackedSentinel = sentinel_wrapper<TrackedIter>;
    using TrackedView     = MiniView<TrackedIter, TrackedSentinel>;

    TrackedView tv{TrackedIter{a1, &iter_swap_call_count}, TrackedSentinel{TrackedIter{a1 + 3, &iter_swap_call_count}}};
    ViewB vb{IteratorB{a2}, SentinelB{IteratorB{a2 + 2}}};

    std::ranges::concat_view cv(tv, vb);

    auto it1 = cv.begin();
    auto it2 = ++cv.begin();

    assert(iter_swap_call_count == 0);
    std::ranges::iter_swap(it1, it2);
    // The underlying iter_swap specialization should have been called
    assert(iter_swap_call_count == 1);
    assert(a1[0].v == 20 && a1[1].v == 10);

    // Call again to verify the count keeps incrementing
    std::ranges::iter_swap(it1, it2);
    assert(iter_swap_call_count == 2);
    assert(a1[0].v == 10 && a1[1].v == 20);
  }

  // Test that iter_swap requires the underlying iterator to be iter_swappable
  {
    using Iterator       = int const*;
    using View           = minimal_view<Iterator, Iterator>;
    using ConcatView     = std::ranges::concat_view<View>;
    using ConcatIterator = std::ranges::iterator_t<ConcatView>;
    static_assert(!std::indirectly_swappable<Iterator>);
    static_assert(!has_iter_swap<ConcatIterator>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
