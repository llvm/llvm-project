//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// template<input_range R>
//   requires constructible_from<V, views::all_t<R>> &&
//   constructible_from<Pattern, single_view<range_value_t<InnerRng>>>
// constexpr explicit join_with_view(R&& r, range_value_t<InnerRng> e);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>
#include <utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_range.h"

struct MoveOnlyInt {
  MoveOnlyInt()                         = default;
  MoveOnlyInt(MoveOnlyInt&&)            = default;
  MoveOnlyInt& operator=(MoveOnlyInt&&) = default;

  constexpr MoveOnlyInt(int val) : val_(val) {}
  constexpr operator int() const { return val_; }

  int val_ = 0;
};

template <>
struct std::common_type<MoveOnlyInt, int> {
  using type = int;
};

template <>
struct std::common_type<int, MoveOnlyInt> {
  using type = int;
};

struct OutputView : std::ranges::view_base {
  using It = cpp20_output_iterator<int*>;
  It begin() const;
  sentinel_wrapper<It> end() const;
};

static_assert(std::ranges::output_range<OutputView, int>);
static_assert(std::ranges::view<OutputView>);

struct InputRange {
  using It = cpp20_input_iterator<int*>;
  It begin() const;
  sentinel_wrapper<It> end() const;
};

struct InputView : InputRange, std::ranges::view_base {};

static_assert(std::ranges::input_range<InputRange>);
static_assert(std::ranges::input_range<const InputRange>);
static_assert(std::ranges::view<InputView>);
static_assert(std::ranges::input_range<InputView>);
static_assert(std::ranges::input_range<const InputView>);

class View : public std::ranges::view_base {
  using OuterRange = std::array<std::array<MoveOnlyInt, 2>, 3>;

  static constexpr OuterRange range_on_input_view            = {{{1, 1}, {1, 1}, {1, 1}}};
  static constexpr OuterRange range_on_ref_input_range       = {{{2, 2}, {2, 2}, {2, 2}}};
  static constexpr OuterRange range_on_const_ref_input_range = {{{3, 3}, {3, 3}, {3, 3}}};
  static constexpr OuterRange range_on_owning_input_range    = {{{4, 4}, {4, 4}, {4, 4}}};

  const OuterRange* r_;

public:
  // Those functions should never be called in this test.
  View(View&&) { assert(false); }
  View(OutputView) { assert(false); }
  View& operator=(View&&) {
    assert(false);
    return *this;
  }

  constexpr explicit View(InputView) : r_(&range_on_input_view) {}
  constexpr explicit View(InputRange) = delete;
  constexpr explicit View(std::ranges::ref_view<InputRange>) : r_(&range_on_ref_input_range) {}
  constexpr explicit View(std::ranges::ref_view<const InputRange>) : r_(&range_on_const_ref_input_range) {}
  constexpr explicit View(std::ranges::owning_view<InputRange>) : r_(&range_on_owning_input_range) {}

  constexpr auto begin() const { return r_->begin(); }
  constexpr auto end() const { return r_->end(); }
};

static_assert(std::ranges::input_range<View>);
static_assert(std::ranges::input_range<const View>);

class Pattern : public std::ranges::view_base {
  int val_;

public:
  // Those functions should never be called in this test.
  Pattern(Pattern&&) { assert(false); }
  template <class T>
  Pattern(const std::ranges::single_view<T>&) {
    assert(false);
  }
  Pattern& operator=(Pattern&&) {
    assert(false);
    return *this;
  }

  template <class T>
  constexpr explicit Pattern(std::ranges::single_view<T>&& v) : val_(v[0]) {}

  constexpr const int* begin() const { return &val_; }
  constexpr const int* end() const { return &val_ + 1; }
};

static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::forward_range<const Pattern>);

constexpr void test_ctor_with_view_and_element() {
  // Check construction from `r` and `e`, when `r` models `std::ranges::view`

  { // `r` and `e` are glvalues
    InputView r;
    int e = 0;
    std::ranges::join_with_view<View, Pattern> jwv(r, e);
    assert(std::ranges::equal(jwv, std::array{1, 1, 0, 1, 1, 0, 1, 1}));
  }

  { // `r` and `e` are const glvalues
    const InputView r;
    const int e = 1;
    std::ranges::join_with_view<View, Pattern> jwv(r, e);
    assert(std::ranges::equal(jwv, std::array{1, 1, 1, 1, 1, 1, 1, 1}));
  }

  { // `r` and `e` are prvalues
    std::ranges::join_with_view<View, Pattern> jwv(InputView{}, MoveOnlyInt{2});
    assert(std::ranges::equal(jwv, std::array{1, 1, 2, 1, 1, 2, 1, 1}));
  }

  { // `r` and `e` are xvalues
    InputView r;
    MoveOnlyInt e = 3;
    std::ranges::join_with_view<View, Pattern> jwv(std::move(r), std::move(e));
    assert(std::ranges::equal(jwv, std::array{1, 1, 3, 1, 1, 3, 1, 1}));
  }

  // Check explicitness
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputView, MoveOnlyInt>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputView, int>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputView&, int&>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const InputView, const int>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const InputView&, const int&>);
}

constexpr void test_ctor_with_non_view_and_element() {
  // Check construction from `r` and `e`, when `r` does not model `std::ranges::view`

  { // `r` and `e` are glvalues
    InputRange r;
    int e = 0;
    std::ranges::join_with_view<View, Pattern> jwv(r, e);
    assert(std::ranges::equal(jwv, std::array{2, 2, 0, 2, 2, 0, 2, 2}));
  }

  { // `r` and `e` are const glvalues
    const InputRange r;
    const int e = 1;
    std::ranges::join_with_view<View, Pattern> jwv(r, e);
    assert(std::ranges::equal(jwv, std::array{3, 3, 1, 3, 3, 1, 3, 3}));
  }

  { // `r` and `e` are prvalues
    std::ranges::join_with_view<View, Pattern> jwv(InputRange{}, MoveOnlyInt{2});
    assert(std::ranges::equal(jwv, std::array{4, 4, 2, 4, 4, 2, 4, 4}));
  }

  { // `r` and `e` are xvalues
    InputRange r;
    MoveOnlyInt e = 3;
    std::ranges::join_with_view<View, Pattern> jwv(std::move(r), std::move(e));
    assert(std::ranges::equal(jwv, std::array{4, 4, 3, 4, 4, 3, 4, 4}));
  }

  // Check explicitness
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputRange, MoveOnlyInt>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputRange, int>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, InputRange&, int&>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const InputRange&, const int&>);
}

constexpr void test_constraints() {
  { // `R` is not an input range
    using R = OutputView;
    static_assert(!std::ranges::input_range<R>);
    static_assert(std::constructible_from<View, std::views::all_t<R>>);
    static_assert(std::constructible_from<Pattern, std::ranges::single_view<int>>);
    static_assert(!std::constructible_from<std::ranges::join_with_view<View, Pattern>, R, int>);
  }

  { // `V` is not constructible from `views::all_t<R>`
    using R = test_range<cpp20_input_iterator>;
    static_assert(std::ranges::input_range<R>);
    static_assert(!std::constructible_from<View, std::views::all_t<R>>);
    static_assert(std::constructible_from<Pattern, std::ranges::single_view<int>>);
    static_assert(!std::constructible_from<std::ranges::join_with_view<View, Pattern>, R, int>);
  }

  { // `Pattern` is not constructible from `single_view<range_value_t<InnerRng>>`
    using R   = InputView;
    using Pat = test_view<forward_iterator>;
    static_assert(std::ranges::input_range<R>);
    static_assert(std::constructible_from<View, std::views::all_t<R>>);
    static_assert(!std::constructible_from<Pat, std::ranges::single_view<int>>);
    static_assert(!std::constructible_from<std::ranges::join_with_view<View, Pat>, R, int>);
  }
}

constexpr bool test() {
  test_ctor_with_view_and_element();
  test_ctor_with_non_view_and_element();
  test_constraints();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
