//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// friend constexpr decltype(auto) iter_move(const iterator& x) {
//   using rvalue_reference = common_reference_t<
//     iter_rvalue_reference_t<InnerIter>,
//     iter_rvalue_reference_t<PatternIter>>;
//   return visit<rvalue_reference>(ranges::iter_move, x.inner_it_);
// }

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
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

  constexpr Status get_status() const { return status_; }

  friend constexpr bool operator==(const MoveOnlyInt& left, int right) { return left.val_ == right; }
  friend constexpr bool operator==(const MoveOnlyInt& left, const MoveOnlyInt& right) {
    return left.val_ == right.val_;
  }

private:
  mutable int val_       = -1;
  mutable Status status_ = constructed;
};

static_assert(std::movable<MoveOnlyInt>);

template <class T>
class ProxyRvalue {
  T val_;

public:
  constexpr ProxyRvalue(T val) : val_(std::move(val)) {}

  ProxyRvalue(ProxyRvalue&&)            = default;
  ProxyRvalue& operator=(ProxyRvalue&&) = default;

  constexpr explicit operator T&&() noexcept { return std::move(val_); }
};

static_assert(std::common_reference_with<ProxyRvalue<int>, int>);
static_assert(std::common_reference_with<ProxyRvalue<MoveOnlyInt>, MoveOnlyInt>);

template <std::bidirectional_iterator It>
class ProxyOnIterMoveIter {
  It it_ = It();

public:
  using value_type      = std::iter_value_t<It>;
  using difference_type = std::iter_difference_t<It>;

  ProxyOnIterMoveIter() = default;
  constexpr ProxyOnIterMoveIter(It it) : it_(std::move(it)) {}

  constexpr decltype(auto) operator*() const { return *it_; }

  constexpr ProxyOnIterMoveIter& operator++() {
    ++it_;
    return *this;
  }

  constexpr ProxyOnIterMoveIter operator++(int) {
    ProxyOnIterMoveIter copy = *this;
    ++it_;
    return copy;
  }

  constexpr ProxyOnIterMoveIter& operator--() {
    --it_;
    return *this;
  }

  constexpr ProxyOnIterMoveIter operator--(int) {
    ProxyOnIterMoveIter copy = *this;
    --it_;
    return copy;
  }

  friend bool operator==(const ProxyOnIterMoveIter&, const ProxyOnIterMoveIter&) = default;

  friend constexpr ProxyRvalue<value_type> iter_move(const ProxyOnIterMoveIter iter) {
    return ProxyRvalue<value_type>{std::ranges::iter_move(iter.it_)};
  }
};

template <class It>
ProxyOnIterMoveIter(It) -> ProxyOnIterMoveIter<It>;

static_assert(std::bidirectional_iterator<ProxyOnIterMoveIter<int*>>);

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
    assert(std::ranges::all_of(jwv, [](const MoveOnlyInt& i) { return i.get_status() == MoveOnlyInt::constructed; }));

    {
      using enum MoveOnlyInt::Status;
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
      assert(std::ranges::all_of(values, [](const MoveOnlyInt& i) { return i.get_status() == move_constructed; }));
    }

    {
      using enum MoveOnlyInt::Status;
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
      assert(std::ranges::all_of(values, [](const MoveOnlyInt& i) { return i.get_status() == move_constructed; }));
    }

    assert(
        std::ranges::all_of(jwv, [](const MoveOnlyInt& i) { return i.get_status() == MoveOnlyInt::moved_from_this; }));
  }

  { // Test `iter_move` when result is proxy rvalue reference. Test return types and moving.
    using Inner   = std::vector<MoveOnlyInt>;
    using V       = std::vector<Inner>;
    using Pattern = BasicVectorView<MoveOnlyInt, ViewProperties{}, ProxyOnIterMoveIter>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    using RRef = ProxyRvalue<MoveOnlyInt>;
    static_assert(std::same_as<RRef, std::ranges::range_rvalue_reference_t<JWV>>);

    V v;
    v.reserve(2);
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(0, 4)));
    v.emplace_back(std::ranges::to<Inner>(std::views::iota(12, 16)));
    JWV jwv(std::move(v), Pattern{std::ranges::to<std::vector<MoveOnlyInt>>(std::views::iota(4, 12))});
    assert(std::ranges::all_of(jwv, [](const MoveOnlyInt& i) { return i.get_status() == MoveOnlyInt::constructed; }));

    {
      using enum MoveOnlyInt::Status;
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto it                                 = jwv.begin();
      std::same_as<RRef> decltype(auto) rref1 = iter_move(it);
      values.emplace_back(std::move(rref1));
      ++it;
      std::same_as<RRef> decltype(auto) rref2 = iter_move(std::as_const(it));
      values.emplace_back(rref2);
      it++;
      std::same_as<RRef> decltype(auto) rref3 = std::ranges::iter_move(it);
      values.emplace_back(rref3);
      ++it;
      std::same_as<RRef> decltype(auto) rref4 = std::ranges::iter_move(std::as_const(it));
      values.emplace_back(rref4);
      it++; // `it` points to element of `Pattern` from here
      std::same_as<RRef> decltype(auto) rref5 = iter_move(it);
      values.emplace_back(rref5);
      ++it;
      std::same_as<RRef> decltype(auto) rref6 = iter_move(std::as_const(it));
      values.emplace_back(rref6);
      it++;
      std::same_as<RRef> decltype(auto) rref7 = std::ranges::iter_move(it);
      values.emplace_back(rref7);
      ++it;
      std::same_as<RRef> decltype(auto) rref8 = std::ranges::iter_move(std::as_const(it));
      values.emplace_back(rref8);

      assert(std::ranges::equal(values, std::views::iota(0, 8)));
      assert(std::ranges::all_of(values, [](const MoveOnlyInt& i) { return i.get_status() == move_constructed; }));
    }

    {
      using enum MoveOnlyInt::Status;
      std::vector<MoveOnlyInt> values;
      values.reserve(8);

      auto cit                                = std::prev(std::as_const(jwv).end());
      std::same_as<RRef> decltype(auto) rref1 = iter_move(cit);
      values.emplace_back(rref1);
      cit--;
      std::same_as<RRef> decltype(auto) rref2 = iter_move(std::as_const(cit));
      values.emplace_back(rref2);
      --cit;
      std::same_as<RRef> decltype(auto) rref3 = std::ranges::iter_move(cit);
      values.emplace_back(rref3);
      cit--;
      std::same_as<RRef> decltype(auto) rref4 = std::ranges::iter_move(std::as_const(cit));
      values.emplace_back(rref4);
      --cit; // `it` points to element of `Pattern` from here
      std::same_as<RRef> decltype(auto) rref5 = iter_move(cit);
      values.emplace_back(rref5);
      cit--;
      std::same_as<RRef> decltype(auto) rref6 = iter_move(std::as_const(cit));
      values.emplace_back(rref6);
      --cit;
      std::same_as<RRef> decltype(auto) rref7 = std::ranges::iter_move(cit);
      values.emplace_back(rref7);
      cit--;
      std::same_as<RRef> decltype(auto) rref8 = std::ranges::iter_move(std::as_const(cit));
      values.emplace_back(rref8);

      assert(std::ranges::equal(std::views::reverse(values), std::views::iota(8, 16)));
      assert(std::ranges::all_of(values, [](const MoveOnlyInt& i) { return i.get_status() == move_constructed; }));
    }

    assert(
        std::ranges::all_of(jwv, [](const MoveOnlyInt& i) { return i.get_status() == MoveOnlyInt::moved_from_this; }));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
