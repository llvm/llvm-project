//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr auto end();
// constexpr auto end() const
//   requires forward_range<const V> && forward_range<const Pattern> &&
//            is_reference_v<range_reference_t<const V>> &&
//            input_range<range_reference_t<const V>>;

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=10000000

#include <ranges>

#include <algorithm>
#include <string>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

template <class V, class Pattern>
concept JoinWithViewHasConstEnd = requires(const std::ranges::join_with_view<V, Pattern> jwv) { jwv.end(); };

template <size_t Bits>
  requires(Bits < (1 << 7))
constexpr void test_end() {
  constexpr bool v_models_forward_range           = static_cast<bool>(Bits & (1 << 0));
  constexpr bool inner_range_is_reference         = static_cast<bool>(Bits & (1 << 1));
  constexpr bool inner_range_models_forward_range = static_cast<bool>(Bits & (1 << 2));
  constexpr bool v_models_common_range            = static_cast<bool>(Bits & (1 << 3));
  constexpr bool inner_range_models_common_range  = static_cast<bool>(Bits & (1 << 4));
  constexpr bool v_models_simple_range            = static_cast<bool>(Bits & (1 << 5));
  constexpr bool pattern_models_simple_range      = static_cast<bool>(Bits & (1 << 6));

  constexpr ViewProperties inner_range_props{.common = inner_range_models_common_range};
  using InnerRange =
      std::conditional_t<inner_range_models_forward_range,
                         BasicView<std::vector<int>, inner_range_props, forward_iterator>,
                         BasicView<std::vector<int>, inner_range_props, DefaultCtorInputIter>>;

  constexpr ViewProperties v_props{.simple = v_models_simple_range, .common = v_models_common_range};
  using UnderlyingV = std::conditional_t<inner_range_is_reference, std::vector<InnerRange>, RvalueVector<InnerRange>>;
  using V           = std::conditional_t<v_models_forward_range,
                                         BasicView<UnderlyingV, v_props, forward_iterator>,
                                         BasicView<UnderlyingV, v_props, DefaultCtorInputIter>>;

  using UnderlyingPattern = std::vector<int>;
  using Pattern = BasicView<UnderlyingPattern, ViewProperties{.simple = pattern_models_simple_range}, forward_iterator>;

  using JWV  = std::ranges::join_with_view<V, Pattern>;
  using Iter = std::ranges::iterator_t<JWV>;

  // Test when `JWV` models common range
  static_assert(std::same_as<Iter, std::ranges::sentinel_t<JWV>> ==
                (v_models_forward_range && inner_range_is_reference && inner_range_models_forward_range &&
                 v_models_common_range && inner_range_models_common_range));

  { // `V` and `Pattern` are empty
    V v{};
    Pattern pattern{};
    JWV jwv(std::move(v), std::move(pattern));
    Iter it                                   = jwv.begin();
    std::sentinel_for<Iter> decltype(auto) se = jwv.end();
    assert(it == se);
  }

  { // `V` is empty, `Pattern` contains some elements
    V v{};
    Pattern pattern{std::vector<int>{0}};
    JWV jwv(std::move(v), std::move(pattern));
    Iter it                                   = jwv.begin();
    std::sentinel_for<Iter> decltype(auto) se = jwv.end();
    assert(it == se);
  }

  { // `V` is not empty, `Pattern is empty`
    V v{UnderlyingV{
        std::vector<InnerRange>{InnerRange(std::vector<int>{1, 2, 3}), InnerRange(std::vector<int>{4, 5, 6})}}};
    Pattern pattern{};
    JWV jwv(std::move(v), std::move(pattern));
    Iter it                                   = jwv.begin();
    std::sentinel_for<Iter> decltype(auto) se = jwv.end();
    assert(std::ranges::next(it, 6) == se);
  }

  { // `V` and `Pattern` are not empty
    V v{UnderlyingV{std::vector<InnerRange>{
        InnerRange(std::vector<int>{6, 5}),
        InnerRange(std::vector<int>{4, 3}),
        InnerRange(std::vector<int>{2, 1, 0})}}};
    Pattern pattern{std::vector<int>{-1, -1}};
    JWV jwv(std::move(v), std::move(pattern));
    Iter it                                   = jwv.begin();
    std::sentinel_for<Iter> decltype(auto) se = jwv.end();
    assert(std::ranges::next(it, 11) == se);
  }
}

template <std::size_t Bits>
  requires(Bits < (1 << 7))
constexpr void test_const_end() {
  constexpr bool const_v_models_forward_range           = static_cast<bool>(Bits & (1 << 0));
  constexpr bool const_pattern_models_forward_range     = static_cast<bool>(Bits & (1 << 1));
  constexpr bool inner_const_range_is_reference         = static_cast<bool>(Bits & (1 << 2));
  constexpr bool inner_const_range_models_input_range   = static_cast<bool>(Bits & (1 << 3));
  constexpr bool inner_const_range_models_forward_range = static_cast<bool>(Bits & (1 << 4));
  constexpr bool const_v_models_common_range            = static_cast<bool>(Bits & (1 << 5));
  constexpr bool inner_const_range_models_common_range  = static_cast<bool>(Bits & (1 << 6));

  constexpr ViewProperties inner_range_props{.common = inner_const_range_models_common_range};
  using InnerRange =
      std::conditional_t<inner_const_range_models_forward_range,
                         BasicView<std::vector<int>, inner_range_props, forward_iterator>,
                         std::conditional_t<inner_const_range_models_input_range,
                                            BasicView<std::vector<int>, inner_range_props, DefaultCtorInputIter>,
                                            InputRangeButOutputWhenConst<int>>>;

  constexpr ViewProperties v_props{.common = const_v_models_common_range};
  using UnderlyingV =
      std::conditional_t<inner_const_range_is_reference, std::vector<InnerRange>, RvalueVector<InnerRange>>;
  using V = std::conditional_t<const_v_models_forward_range,
                               BasicView<UnderlyingV, v_props, forward_iterator>,
                               BasicView<UnderlyingV, v_props, DefaultCtorInputIter>>;
  using Pattern =
      std::conditional_t<const_pattern_models_forward_range,
                         BasicView<std::vector<int>, ViewProperties{}, forward_iterator>,
                         ForwardViewButInputWhenConst<int>>;

  using JWV = std::ranges::join_with_view<V, Pattern>;
  static_assert(JoinWithViewHasConstEnd<V, Pattern> ==
                (const_v_models_forward_range && const_pattern_models_forward_range && inner_const_range_is_reference &&
                 (inner_const_range_models_input_range || inner_const_range_models_forward_range)));
  static_assert(JoinWithViewHasConstEnd<V, Pattern> == std::ranges::range<const JWV>);

  if constexpr (std::ranges::range<const JWV>) {
    using ConstIter = std::ranges::iterator_t<const JWV>;

    // Test when `const JWV` models common range
    static_assert(std::same_as<ConstIter, std::ranges::sentinel_t<const JWV>> ==
                  (inner_const_range_models_forward_range && const_v_models_common_range &&
                   inner_const_range_models_common_range));

    { // `const V` and `const Pattern` are empty
      V v{};
      Pattern pattern{};
      const JWV jwv(std::move(v), std::move(pattern));
      ConstIter it                                   = jwv.begin();
      std::sentinel_for<ConstIter> decltype(auto) se = jwv.end();
      assert(it == se);
    }

    { // `const V` is empty, `const Pattern` contains some elements
      V v{};
      Pattern pattern{std::vector<int>{1}};
      const JWV jwv(std::move(v), std::move(pattern));
      ConstIter it                                   = jwv.begin();
      std::sentinel_for<ConstIter> decltype(auto) se = jwv.end();
      assert(it == se);
    }

    { // `const V` is not empty, `const Pattern is empty`
      V v{UnderlyingV{
          std::vector<InnerRange>{InnerRange(std::vector<int>{1, 2, 3}), InnerRange(std::vector<int>{4, 5, 6})}}};
      Pattern pattern{};
      const JWV jwv(std::move(v), std::move(pattern));
      ConstIter it                                   = jwv.begin();
      std::sentinel_for<ConstIter> decltype(auto) se = jwv.end();
      assert(std::ranges::next(it, 6) == se);
    }

    { // `const V` and `const Pattern` are not empty
      V v{UnderlyingV{std::vector<InnerRange>{
          InnerRange(std::vector<int>{1}), InnerRange(std::vector<int>{2, 2}), InnerRange(std::vector<int>{3, 3, 3})}}};
      Pattern pattern{std::vector<int>{0}};
      const JWV jwv(std::move(v), std::move(pattern));
      ConstIter it                                   = jwv.begin();
      std::sentinel_for<ConstIter> decltype(auto) se = jwv.end();
      assert(std::ranges::next(it, 8) == se);
    }
  }
}

constexpr bool test() {
  []<std::size_t... Bits>(std::index_sequence<Bits...>) {
    (test_end<Bits>(), ...);
    (test_const_end<Bits>(), ...);
  }(std::make_index_sequence<(1 << 7)>{});

  { // Check situation when iterators returned by `end()` and `end() const` are of the same type
    using V             = BasicView<std::vector<std::string>, ViewProperties{.simple = true}, forward_iterator>;
    using Pattern       = BasicView<std::string, ViewProperties{.simple = true}, forward_iterator>;
    using JWV           = std::ranges::join_with_view<V, Pattern>;
    using Sentinel      = std::ranges::sentinel_t<JWV&>;
    using ConstSentinel = std::ranges::sentinel_t<const JWV&>;
    static_assert(std::input_iterator<Sentinel>);
    static_assert(std::input_iterator<ConstSentinel>);
    static_assert(std::same_as<Sentinel, ConstSentinel>);
  }

  { // Check situation when sentinels returned by `end()` and `end() const` are of the same type
    using V = BasicView<std::vector<std::string>, ViewProperties{.simple = true, .common = false}, forward_iterator>;
    using Pattern       = BasicView<std::string, ViewProperties{.simple = true}, forward_iterator>;
    using JWV           = std::ranges::join_with_view<V, Pattern>;
    using Sentinel      = std::ranges::sentinel_t<JWV&>;
    using ConstSentinel = std::ranges::sentinel_t<const JWV&>;
    static_assert(!std::input_iterator<Sentinel>);
    static_assert(!std::input_iterator<ConstSentinel>);
    static_assert(std::same_as<Sentinel, ConstSentinel>);
  }

  // Check LWG-4074: compatible-joinable-ranges is underconstrained
  static_assert(!JoinWithViewHasConstEnd<BasicVectorView<int, ViewProperties{}, forward_iterator>,
                                         lwg4074::PatternWithProxyConstAccess>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
