//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// friend constexpr decltype(auto) iter_move(const iterator& x);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "../types.h"

class MoveOnlyInt {
public:
  enum Status { constructed, move_constructed, moved_from_this };

  MoveOnlyInt() = default;
  constexpr MoveOnlyInt(int val) : val_(val) {}

  constexpr MoveOnlyInt(MoveOnlyInt&& other) noexcept : val_(other.val_), status_(move_constructed) {
    other.val_    = -1;
    other.status_ = moved_from_this;
  }

  constexpr MoveOnlyInt(const MoveOnlyInt&& other) noexcept : val_(other.val_), status_(move_constructed) {
    other.val_    = -1;
    other.status_ = moved_from_this;
  }

  MoveOnlyInt(const MoveOnlyInt&) { assert(false); } // Should never be called in this test.

  MoveOnlyInt& operator=(MoveOnlyInt&&) { // Should never be called in this test.
    assert(false);
    return *this;
  }

  constexpr bool was_normally_constructed() const { return status_ == constructed; }
  constexpr bool was_move_constructed() const { return status_ == move_constructed; }
  constexpr bool was_moved_from() const { return status_ == moved_from_this; }

  friend constexpr bool operator==(const MoveOnlyInt& left, int right) { return left.val_ == right; }
  friend constexpr bool operator==(const MoveOnlyInt& left, const MoveOnlyInt& right) {
    return left.val_ == right.val_;
  }

private:
  mutable int val_       = -1;
  mutable Status status_ = constructed;
};

static_assert(std::movable<MoveOnlyInt>);

struct ProxyRvalueRef {
  MoveOnlyInt&& val;
};

class CommonProxyRvalueRef {
public:
  constexpr CommonProxyRvalueRef(ProxyRvalueRef i) : val_(std::move(i.val)) {}
  constexpr CommonProxyRvalueRef(MoveOnlyInt i) : val_(std::move(i)) {}

  constexpr MoveOnlyInt&& get() { return std::move(val_); }

private:
  MoveOnlyInt val_;
};

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<ProxyRvalueRef, MoveOnlyInt, TQual, UQual> {
  using type = CommonProxyRvalueRef;
};

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<MoveOnlyInt, ProxyRvalueRef, TQual, UQual> {
  using type = CommonProxyRvalueRef;
};

static_assert(std::common_reference_with<MoveOnlyInt&&, ProxyRvalueRef>);
static_assert(std::common_reference_with<MoveOnlyInt&&, CommonProxyRvalueRef>);

class ProxyIter {
public:
  using value_type      = MoveOnlyInt;
  using difference_type = std::ptrdiff_t;

  constexpr ProxyIter() : ptr_(nullptr) {}
  constexpr explicit ProxyIter(MoveOnlyInt* it) : ptr_(std::move(it)) {}

  constexpr decltype(auto) operator*() const { return *ptr_; }

  constexpr ProxyIter& operator++() {
    ++ptr_;
    return *this;
  }

  constexpr ProxyIter operator++(int) {
    ProxyIter copy = *this;
    ++ptr_;
    return copy;
  }

  constexpr ProxyIter& operator--() {
    --ptr_;
    return *this;
  }

  constexpr ProxyIter operator--(int) {
    ProxyIter copy = *this;
    --ptr_;
    return copy;
  }

  friend bool operator==(const ProxyIter&, const ProxyIter&) = default;

  friend constexpr ProxyRvalueRef iter_move(const ProxyIter iter) {
    return ProxyRvalueRef{std::ranges::iter_move(iter.ptr_)};
  }

private:
  MoveOnlyInt* ptr_;
};

static_assert(std::forward_iterator<ProxyIter>);

template <std::forward_iterator Iter>
class IterMoveTrackingIterator {
public:
  using value_type      = std::iter_value_t<Iter>;
  using difference_type = std::iter_difference_t<Iter>;

  IterMoveTrackingIterator() = default;
  constexpr explicit IterMoveTrackingIterator(Iter iter, bool* flag = nullptr) : iter_(std::move(iter)), flag_(flag) {}

  constexpr IterMoveTrackingIterator& operator++() {
    ++iter_;
    return *this;
  }

  constexpr IterMoveTrackingIterator operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  constexpr decltype(auto) operator*() const { return *iter_; }

  constexpr bool operator==(const IterMoveTrackingIterator& other) const { return iter_ == other.iter_; }

  friend constexpr decltype(auto) iter_move(const IterMoveTrackingIterator& iter) {
    assert(iter.flag_ != nullptr);
    *iter.flag_ = true;
    return std::ranges::iter_move(iter.iter_);
  }

private:
  Iter iter_  = Iter();
  bool* flag_ = nullptr;
};

static_assert(std::forward_iterator<IterMoveTrackingIterator<int*>> &&
              !std::bidirectional_iterator<IterMoveTrackingIterator<int*>>);

constexpr bool test() {
  { // Test `iter_move` when result is true rvalue reference. Test return types.
    using V       = std::array<std::array<char, 1>, 2>;
    using Pattern = std::array<char, 1>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv(V{{{'0'}, {'1'}}}, Pattern{','});

    {
      auto it                                     = jwv.begin();
      std::same_as<char&&> decltype(auto) v_rref1 = iter_move(it);
      std::same_as<char&&> decltype(auto) v_rref2 = iter_move(std::as_const(it));
      std::same_as<char&&> decltype(auto) v_rref3 = std::ranges::iter_move(it);
      std::same_as<char&&> decltype(auto) v_rref4 = std::ranges::iter_move(std::as_const(it));
      assert(std::ranges::equal(std::array{v_rref1, v_rref2, v_rref3, v_rref4}, std::views::repeat('0', 4)));

      ++it; // `it` points to element of `Pattern` from here
      std::same_as<char&&> decltype(auto) pattern_rref1 = iter_move(it);
      std::same_as<char&&> decltype(auto) pattern_rref2 = iter_move(std::as_const(it));
      std::same_as<char&&> decltype(auto) pattern_rref3 = std::ranges::iter_move(it);
      std::same_as<char&&> decltype(auto) pattern_rref4 = std::ranges::iter_move(std::as_const(it));
      assert(std::ranges::equal(
          std::array{pattern_rref1, pattern_rref2, pattern_rref3, pattern_rref4}, std::views::repeat(',', 4)));
    }

    {
      auto cit                                           = std::prev(std::as_const(jwv).end());
      std::same_as<const char&&> decltype(auto) cv_rref1 = iter_move(cit);
      std::same_as<const char&&> decltype(auto) cv_rref2 = iter_move(std::as_const(cit));
      std::same_as<const char&&> decltype(auto) cv_rref3 = std::ranges::iter_move(cit);
      std::same_as<const char&&> decltype(auto) cv_rref4 = std::ranges::iter_move(std::as_const(cit));
      assert(std::ranges::equal(std::array{cv_rref1, cv_rref2, cv_rref3, cv_rref4}, std::views::repeat('1', 4)));

      cit--; // `cit` points to element of `Pattern` from here
      std::same_as<const char&&> decltype(auto) cpattern_rref1 = iter_move(cit);
      std::same_as<const char&&> decltype(auto) cpattern_rref2 = iter_move(std::as_const(cit));
      std::same_as<const char&&> decltype(auto) cpattern_rref3 = std::ranges::iter_move(cit);
      std::same_as<const char&&> decltype(auto) cpattern_rref4 = std::ranges::iter_move(std::as_const(cit));
      assert(std::ranges::equal(
          std::array{cpattern_rref1, cpattern_rref2, cpattern_rref3, cpattern_rref4}, std::views::repeat(',', 4)));
    }
  }

  { // Test `iter_move` when result is true rvalue reference. Test moving.
    using Inner   = std::vector<MoveOnlyInt>;
    using V       = std::vector<Inner>;
    using Pattern = std::vector<MoveOnlyInt>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    V v;
    v.reserve(2);
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(0, 4)));
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(12, 16)));
    JWV jwv(std::move(v), std::ranges::to<Pattern>(std::views::iota(4, 12)));
    assert(std::ranges::all_of(jwv, &MoveOnlyInt::was_normally_constructed));

    {
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto it = jwv.begin();
      values.emplace_back(iter_move(it));
      ++it;
      values.emplace_back(iter_move(std::as_const(it)));
      it++;
      values.emplace_back(std::ranges::iter_move(it));
      ++it;
      values.emplace_back(std::ranges::iter_move(std::as_const(it)));
      it++; // `it` points to element of `Pattern` from here
      values.emplace_back(iter_move(it));
      ++it;
      values.emplace_back(iter_move(std::as_const(it)));
      it++;
      values.emplace_back(std::ranges::iter_move(it));
      ++it;
      values.emplace_back(std::ranges::iter_move(std::as_const(it)));

      assert(std::ranges::equal(values, std::views::iota(0, 8)));
      assert(std::ranges::all_of(values, &MoveOnlyInt::was_move_constructed));
    }

    {
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto cit = std::prev(std::as_const(jwv).end());
      values.emplace_back(iter_move(cit));
      cit--;
      values.emplace_back(iter_move(std::as_const(cit)));
      --cit;
      values.emplace_back(std::ranges::iter_move(cit));
      cit--;
      values.emplace_back(std::ranges::iter_move(std::as_const(cit)));
      --cit; // `it` points to element of `Pattern` from here
      values.emplace_back(iter_move(cit));
      cit--;
      values.emplace_back(iter_move(std::as_const(cit)));
      --cit;
      values.emplace_back(std::ranges::iter_move(cit));
      cit--;
      values.emplace_back(std::ranges::iter_move(std::as_const(cit)));

      assert(std::ranges::equal(std::views::reverse(values), std::views::iota(8, 16)));
      assert(std::ranges::all_of(values, &MoveOnlyInt::was_move_constructed));
    }

    assert(std::ranges::all_of(jwv, &MoveOnlyInt::was_moved_from));
  }

  { // Test `iter_move` when result is proxy rvalue reference type, which is different from
    // range_rvalue_reference_t<InnerRng> and range_rvalue_reference_t<Pattern>.
    using Inner   = std::vector<MoveOnlyInt>;
    using V       = std::vector<Inner>;
    using Pattern = std::ranges::subrange<ProxyIter, ProxyIter>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, Pattern>;

    static_assert(!std::same_as<std::ranges::range_rvalue_reference_t<V>, std::ranges::range_rvalue_reference_t<JWV>>);
    static_assert(
        !std::same_as<std::ranges::range_rvalue_reference_t<Pattern>, std::ranges::range_rvalue_reference_t<JWV>>);
    static_assert(std::same_as<CommonProxyRvalueRef, std::ranges::range_rvalue_reference_t<JWV>>);

    V v;
    v.reserve(2);
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(0, 4)));
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(12, 16)));

    auto pattern = std::ranges::to<std::vector<MoveOnlyInt>>(std::views::iota(4, 12));
    Pattern pattern_as_subrange(ProxyIter{pattern.data()}, ProxyIter{pattern.data() + pattern.size()});

    JWV jwv(std::move(v), pattern_as_subrange);
    assert(std::ranges::all_of(jwv, &MoveOnlyInt::was_normally_constructed));

    {
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto it                                                 = jwv.begin();
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref1 = iter_move(it);
      values.emplace_back(rref1.get());
      ++it;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref2 = iter_move(std::as_const(it));
      values.emplace_back(rref2.get());
      it++;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref3 = std::ranges::iter_move(it);
      values.emplace_back(rref3.get());
      ++it;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref4 = std::ranges::iter_move(std::as_const(it));
      values.emplace_back(rref4.get());
      it++; // `it` points to element of `Pattern` from here
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref5 = iter_move(it);
      values.emplace_back(rref5.get());
      ++it;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref6 = iter_move(std::as_const(it));
      values.emplace_back(rref6.get());
      it++;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref7 = std::ranges::iter_move(it);
      values.emplace_back(rref7.get());
      ++it;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref8 = std::ranges::iter_move(std::as_const(it));
      values.emplace_back(rref8.get());

      assert(std::ranges::equal(values, std::views::iota(0, 8)));
      assert(std::ranges::all_of(values, &MoveOnlyInt::was_move_constructed));
    }

    {
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto cit                                                = std::prev(std::as_const(jwv).end());
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref1 = iter_move(cit);
      values.emplace_back(rref1.get());
      cit--;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref2 = iter_move(std::as_const(cit));
      values.emplace_back(rref2.get());
      --cit;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref3 = std::ranges::iter_move(cit);
      values.emplace_back(rref3.get());
      cit--;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref4 = std::ranges::iter_move(std::as_const(cit));
      values.emplace_back(rref4.get());
      --cit; // `it` points to element of `Pattern` from here
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref5 = iter_move(cit);
      values.emplace_back(rref5.get());
      cit--;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref6 = iter_move(std::as_const(cit));
      values.emplace_back(rref6.get());
      --cit;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref7 = std::ranges::iter_move(cit);
      values.emplace_back(rref7.get());
      cit--;
      std::same_as<CommonProxyRvalueRef> decltype(auto) rref8 = std::ranges::iter_move(std::as_const(cit));
      values.emplace_back(rref8.get());

      assert(std::ranges::equal(std::views::reverse(values), std::views::iota(8, 16)));
      assert(std::ranges::all_of(values, &MoveOnlyInt::was_move_constructed));
    }

    assert(std::ranges::all_of(jwv, &MoveOnlyInt::was_moved_from));
  }

  { // Make sure `iter_move` calls underlying's iterator `iter_move` (not `std::move(*i)`).
    using Inner               = std::vector<int>;
    using InnerTrackingIter   = IterMoveTrackingIterator<Inner::iterator>;
    using TrackingInner       = std::ranges::subrange<InnerTrackingIter>;
    using Pattern             = std::array<int, 1>;
    using PatternTrackingIter = IterMoveTrackingIterator<Pattern::iterator>;
    using TrackingPattern     = std::ranges::subrange<PatternTrackingIter>;
    using JWV                 = std::ranges::join_with_view<std::span<TrackingInner>, TrackingPattern>;

    std::array<Inner, 2> v{{{1}, {2}}};
    Pattern pat{-1};

    bool v_moved = false;
    std::array<TrackingInner, 2> tracking_v{
        TrackingInner(InnerTrackingIter(v[0].begin(), &v_moved), InnerTrackingIter(v[0].end())),
        TrackingInner(InnerTrackingIter(v[1].begin()), InnerTrackingIter(v[1].end()))};

    bool pat_moved = false;
    TrackingPattern tracking_pat(PatternTrackingIter(pat.begin(), &pat_moved), PatternTrackingIter(pat.end()));

    JWV jwv(tracking_v, tracking_pat);
    auto it = jwv.begin();

    // Test calling `iter_move` when `it` points to element of `v`
    assert(!v_moved);
    assert(iter_move(it) == 1);
    assert(v_moved);

    // Test calling `iter_move` when `it` points to element of `pat`
    ++it;
    assert(!pat_moved);
    assert(iter_move(it) == -1);
    assert(pat_moved);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
