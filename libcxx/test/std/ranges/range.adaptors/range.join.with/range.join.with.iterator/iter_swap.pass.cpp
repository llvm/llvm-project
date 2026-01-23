//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// friend constexpr void iter_swap(const iterator& x, const iterator& y);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

template <class I>
concept CanIterSwap = requires(I i) { iter_swap(i); };

enum class SwapKind { no_swap, with_same_type, with_different_type };
enum class IterKind { inner_view, pattern };

template <std::forward_iterator Iter, IterKind Kind>
class IterSwapTrackingIterator {
public:
  using value_type      = std::iter_value_t<Iter>;
  using difference_type = std::iter_difference_t<Iter>;

  constexpr Iter get_iter() const { return iter_; }

  constexpr SwapKind* get_flag() const { return flag_; }

  IterSwapTrackingIterator() = default;
  constexpr explicit IterSwapTrackingIterator(Iter iter, SwapKind* flag = nullptr)
      : iter_(std::move(iter)), flag_(flag) {}

  constexpr IterSwapTrackingIterator& operator++() {
    ++iter_;
    return *this;
  }

  constexpr IterSwapTrackingIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  constexpr decltype(auto) operator*() const { return *iter_; }

  constexpr bool operator==(const IterSwapTrackingIterator& other) const { return iter_ == other.iter_; }

  friend constexpr decltype(auto) iter_swap(const IterSwapTrackingIterator& lhs, const IterSwapTrackingIterator& rhs) {
    assert(lhs.flag_ != nullptr && rhs.flag_ != nullptr);
    *lhs.flag_ = *rhs.flag_ = SwapKind::with_same_type;
    return std::ranges::iter_swap(lhs.iter_, rhs.iter_);
  }

  template <std::indirectly_swappable<Iter> OtherIter, IterKind OtherKind>
  friend constexpr decltype(auto)
  iter_swap(const IterSwapTrackingIterator& lhs, const IterSwapTrackingIterator<OtherIter, OtherKind>& rhs) {
    assert(lhs.flag_ != nullptr && rhs.get_flag() != nullptr);
    *lhs.flag_ = *rhs.get_flag() = SwapKind::with_different_type;
    return std::ranges::iter_swap(lhs.iter_, rhs.get_iter());
  }

private:
  Iter iter_      = Iter();
  SwapKind* flag_ = nullptr;
};

static_assert(std::forward_iterator<IterSwapTrackingIterator<int*, IterKind::inner_view>> &&
              !std::bidirectional_iterator<IterSwapTrackingIterator<int*, IterKind::inner_view>>);

constexpr bool test() {
  { // Test common usage
    using V       = std::vector<std::string>;
    using Pattern = std::string;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;
    using namespace std::string_view_literals;

    JWV jwv(V{"std", "ranges", "views", "join_with_view"}, Pattern{":: "});
    assert(std::ranges::equal(jwv, "std:: ranges:: views:: join_with_view"sv));

    auto it = jwv.begin();
    iter_swap(it, std::ranges::next(it, 2)); // Swap elements of the same inner range.
    assert(std::ranges::equal(jwv, "dts:: ranges:: views:: join_with_view"sv));

    std::ranges::advance(it, 3);
    iter_swap(std::as_const(it), std::ranges::next(it, 2)); // Swap elements of the pattern.
    assert(std::ranges::equal(jwv, "dts ::ranges ::views ::join_with_view"sv));

    std::ranges::advance(it, 3);
    const auto it2 = jwv.begin();
    iter_swap(std::as_const(it), it2); // Swap elements of different inner ranges.
    assert(std::ranges::equal(jwv, "rts ::danges ::views ::join_with_view"sv));

    std::ranges::advance(it, 6);
    iter_swap(std::as_const(it), it2); // Swap element from inner range with element from the pattern.
    assert(std::ranges::equal(jwv, " tsr::dangesr::viewsr::join_with_view"sv));

    static_assert(std::is_void_v<decltype(iter_swap(it, it))>);
    static_assert(std::is_void_v<decltype(iter_swap(it2, it2))>);
    static_assert(!CanIterSwap<std::ranges::iterator_t<const JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<const JWV>>);
  }

  { // Make sure `iter_swap` calls underlying's iterator `iter_swap` (not `ranges::swap(*i1, *i2)`).
    using Inner               = std::vector<int>;
    using InnerTrackingIter   = IterSwapTrackingIterator<Inner::iterator, IterKind::inner_view>;
    using TrackingInner       = std::ranges::subrange<InnerTrackingIter>;
    using Pattern             = std::array<int, 2>;
    using PatternTrackingIter = IterSwapTrackingIterator<Pattern::iterator, IterKind::pattern>;
    using TrackingPattern     = std::ranges::subrange<PatternTrackingIter>;
    using JWV                 = std::ranges::join_with_view<std::span<TrackingInner>, TrackingPattern>;

    std::array<Inner, 3> v{{{1, 2, 3}, {4, 5}}};
    Pattern pat{-1, -2};

    SwapKind v_swap_kind = SwapKind::no_swap;
    std::array<TrackingInner, 2> tracking_v{
        TrackingInner(InnerTrackingIter(v[0].begin(), &v_swap_kind), InnerTrackingIter(v[0].end())),
        TrackingInner(InnerTrackingIter(v[1].begin(), &v_swap_kind), InnerTrackingIter(v[1].end()))};

    SwapKind pat_swap_kind = SwapKind::no_swap;
    TrackingPattern tracking_pat(PatternTrackingIter(pat.begin(), &pat_swap_kind), PatternTrackingIter(pat.end()));

    JWV jwv(tracking_v, tracking_pat);
    auto it1 = jwv.begin();
    auto it2 = std::ranges::next(it1);

    // Test calling `iter_swap` when both `it1` and `it2` point to elements of `v`.
    assert(v_swap_kind == SwapKind::no_swap);
    iter_swap(it1, it2);
    assert(*it1 == 2 && *it2 == 1);
    assert(v_swap_kind == SwapKind::with_same_type && pat_swap_kind == SwapKind::no_swap);

    // Test calling `iter_swap` when `it1` points to element of `v` and `it2` points to element of `pat`.
    std::ranges::advance(it2, 2);
    v_swap_kind = SwapKind::no_swap;
    assert(pat_swap_kind == SwapKind::no_swap);
    iter_swap(it1, it2);
    assert(*it1 == -1 && *it2 == 2);
    assert(v_swap_kind == SwapKind::with_different_type && pat_swap_kind == SwapKind::with_different_type);

    // Test calling `iter_swap` when `it1` and `it2` point to elements of `pat`.
    std::ranges::advance(it1, 4);
    v_swap_kind = pat_swap_kind = SwapKind::no_swap;
    iter_swap(it1, it2);
    assert(*it1 == 2 && *it2 == -2);
    assert(v_swap_kind == SwapKind::no_swap && pat_swap_kind == SwapKind::with_same_type);

    // Test calling `iter_swap` when `it1` points to element of `pat` and `it2` points to element of `v`.
    std::ranges::advance(it2, 3);
    v_swap_kind = pat_swap_kind = SwapKind::no_swap;
    iter_swap(it1, it2);
    assert(*it1 == 5 && *it2 == 2);
    assert(v_swap_kind == SwapKind::with_different_type && pat_swap_kind == SwapKind::with_different_type);
  }

  { // InnerIter and PatternIter don't model indirectly swappable
    using JWV = std::ranges::join_with_view<std::span<std::string>, std::string_view>;
    static_assert(!CanIterSwap<std::ranges::iterator_t<JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<JWV>>);
    static_assert(!CanIterSwap<std::ranges::iterator_t<const JWV>>);
    static_assert(!CanIterSwap<const std::ranges::iterator_t<const JWV>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
