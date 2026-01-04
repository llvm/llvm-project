//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// constexpr auto begin();
// constexpr auto begin() const
//   requires forward_range<const V> &&
//            forward_range<const Pattern> &&
//            is_reference_v<range_reference_t<const V>> &&
//            input_range<range_reference_t<const V>>;

#include <ranges>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "../types.h"
#include "test_iterators.h"

template <bool Simple>
using MaybeSimpleForwardView = BasicView<std::vector<std::string>, ViewProperties{.simple = Simple}, forward_iterator>;

template <bool Simple>
using MaybeSimpleForwardRvalueView =
    BasicView<RvalueVector<std::string>, ViewProperties{.simple = Simple}, forward_iterator>;

template <bool Simple>
using MaybeSimplePattern = BasicView<std::string, ViewProperties{.simple = Simple}, forward_iterator>;

template <class V, class Pattern>
concept JoinWithViewHasConstBegin = requires(const std::ranges::join_with_view<V, Pattern> jwv) {
  { jwv.begin() } -> std::input_iterator;
};

constexpr void test_begin() {
  using Str = std::string;
  using Vec = std::vector<Str>;

  { // `V` models simple-view
    // `is_reference_v<InnerRng>` is true
    // `Pattern` models simple-view
    // `V` and `Pattern` contain some elements
    using V       = MaybeSimpleForwardView<true>;
    using Pattern = MaybeSimplePattern<true>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"A", "B", "C"}), Pattern(Str{">>"}));
    auto it = jwv.begin();
    assert(std::ranges::equal(std::views::counted(it, 7), Str{"A>>B>>C"}));
  }

  { // `V` does not model simple-view
    // `is_reference_v<InnerRng>` is true
    // `Pattern` models simple-view
    // `V` and `Pattern` are empty
    using V       = MaybeSimpleForwardView<false>;
    using Pattern = MaybeSimplePattern<true>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{}), Pattern(Str{}));
    auto it = jwv.begin();
    assert(it == jwv.end());
  }

  { // `V` models simple-view
    // `is_reference_v<InnerRng>` is false
    // `Pattern` models simple-view
    // `V` contains two elements, `Pattern` is empty
    using V       = MaybeSimpleForwardRvalueView<true>;
    using Pattern = MaybeSimplePattern<true>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"1", "2"}), Pattern(Str{""}));
    auto it = jwv.begin();
    assert(*it == '1');
    assert(*++it == '2');
  }

  { // `V` models simple-view
    // `is_reference_v<InnerRng>` is true
    // `Pattern` does not model simple-view
    // `V` contains one element, `Pattern` is empty
    using V       = MaybeSimpleForwardView<true>;
    using Pattern = MaybeSimplePattern<false>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"07"}), Pattern(Str{}));
    auto it = jwv.begin();
    assert(*it++ == '0');
    assert(*it == '7');
  }

  { // `V` does not model simple-view
    // `is_reference_v<InnerRng>` is false
    // `Pattern` models simple-view
    // `V` contains three elements (2nd is empty), `Pattern` is not empty
    using V       = MaybeSimpleForwardRvalueView<false>;
    using Pattern = MaybeSimplePattern<true>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"A", "", "C"}), Pattern(Str{"--"}));
    auto it = jwv.begin();
    assert(std::ranges::equal(std::views::counted(it, 6), Str("A----C")));
  }

  { // `V` does not model simple-view
    // `is_reference_v<InnerRng>` is true
    // `Pattern` does not model simple-view
    // `V` contains some empty elements, `Pattern` is not empty
    using V       = MaybeSimpleForwardView<false>;
    using Pattern = MaybeSimplePattern<false>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"", "", ""}), Pattern(Str{"-"}));
    auto it = jwv.begin();
    assert(*it++ == '-');
    assert(*it == '-');
  }

  { // `V` models simple-view
    // `is_reference_v<InnerRng>` is false
    // `Pattern` does not model simple-view
    // `V` contains two elements, `Pattern` is not empty
    using V       = MaybeSimpleForwardRvalueView<true>;
    using Pattern = MaybeSimplePattern<false>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"X", "Z"}), Pattern(Str{"Y"}));
    auto it = jwv.begin();
    assert(*it == 'X');
    assert(*++it == 'Y');
    assert(*++it == 'Z');
  }

  { // `V` does not model simple-view
    // `is_reference_v<InnerRng>` is false
    // `Pattern` does not model simple-view
    // `V` contains two empty elements, `Pattern` is not empty
    using V       = MaybeSimpleForwardRvalueView<false>;
    using Pattern = MaybeSimplePattern<false>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"", ""}), Pattern(Str{"?"}));
    auto it = jwv.begin();
    assert(*it == '?');
    assert(++it == jwv.end());
  }

  { // `V` does not model forward range
    // `V` contains some empty elements, `Pattern` is empty
    using V       = BasicView<Vec, ViewProperties{.common = false}, cpp20_input_iterator>;
    using Pattern = MaybeSimplePattern<false>;
    std::ranges::join_with_view<V, Pattern> jwv(V(Vec{"", "", ""}), Pattern(Str{""}));
    auto it = jwv.begin();
    assert(it == jwv.end());
  }
}

constexpr void test_const_begin() {
  using Vec = std::vector<std::array<int, 2>>;
  using Pat = std::array<int, 2>;

  { // `const V` models forward range
    // `const Pattern` models forward range
    // `is_reference_v<range_reference_t<const V>>` is true
    // `range_reference_t<const V>` models input range
    using V       = BasicView<Vec, ViewProperties{}, forward_iterator>;
    using Pattern = BasicView<Pat, ViewProperties{}, forward_iterator>;

    const std::ranges::join_with_view<V, Pattern> jwv{V{Vec{std::array{1, 2}, std::array{3, 4}}}, Pattern{Pat{0, 0}}};
    auto it = jwv.begin();
    assert(std::ranges::equal(std::views::counted(it, 6), std::array{1, 2, 0, 0, 3, 4}));
  }

  // `const V` does not model forward range
  // `const Pattern` models forward range
  // `is_reference_v<range_reference_t<const V>>` is true
  // `range_reference_t<const V>` models input range
  static_assert(!JoinWithViewHasConstBegin<BasicView<Vec, ViewProperties{.common = false}, cpp20_input_iterator>,
                                           BasicView<Pat, ViewProperties{}, forward_iterator>>);

  // `const V` models forward range
  // `const Pattern` does not model forward range
  // `is_reference_v<range_reference_t<const V>>` is true
  // `range_reference_t<const V>` models input range
  static_assert(!JoinWithViewHasConstBegin<BasicView<Vec, ViewProperties{}, forward_iterator>,
                                           BasicView<Pat, ViewProperties{.common = false}, cpp20_input_iterator>>);

  // `const V` models forward range
  // `const Pattern` models forward range
  // `is_reference_v<range_reference_t<const V>>` is false
  // `range_reference_t<const V>` models input range
  static_assert(
      !JoinWithViewHasConstBegin<BasicView<RvalueVector<std::vector<int>>, ViewProperties{}, forward_iterator>,
                                 BasicView<Pat, ViewProperties{}, forward_iterator>>);

  // `const V` models forward range
  // `const Pattern` models forward range
  // `is_reference_v<range_reference_t<const V>>` is true
  // `range_reference_t<const V>` does not model input range
  static_assert(!JoinWithViewHasConstBegin<
                BasicView<std::vector<InputRangeButOutputWhenConst<int>>, ViewProperties{}, forward_iterator>,
                BasicView<Pat, ViewProperties{}, forward_iterator>>);

  // `concatable<range_reference_t<const V>, const Pattern>` is not satisfied
  // See also LWG-4074: compatible-joinable-ranges is underconstrained
  static_assert(!JoinWithViewHasConstBegin<BasicVectorView<int, ViewProperties{}, forward_iterator>,
                                           lwg4074::PatternWithProxyConstAccess>);

  // Check situation when iterators returned by `begin()` and `begin() const` are the same
  using JWV = std::ranges::join_with_view<MaybeSimpleForwardView<true>, MaybeSimplePattern<true>>;
  static_assert(std::same_as<std::ranges::iterator_t<JWV&>, std::ranges::iterator_t<const JWV&>>);
}

constexpr bool test() {
  test_begin();
  test_const_begin();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
