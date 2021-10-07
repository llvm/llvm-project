//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr sentinel<false> end();
// constexpr iterator<false> end() requires common_range<V>;
// constexpr sentinel<true> end() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;
// constexpr iterator<true> end() const
//   requires common_range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <ranges>

#include "test_macros.h"
#include "types.h"

template<class T>
concept HasConstQualifiedEnd = requires(const T& t) { t.end(); };

constexpr bool test() {
  {
    using TransformView = std::ranges::transform_view<ForwardView, PlusOneMutable>;
    static_assert(std::ranges::common_range<TransformView>);
    TransformView tv;
    auto end = tv.end();
    ASSERT_SAME_TYPE(decltype(end.base()), std::ranges::sentinel_t<ForwardView>);
    assert(base(end.base()) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOneMutable>;
    static_assert(!std::ranges::common_range<TransformView>);
    TransformView tv;
    auto end = tv.end();
    ASSERT_SAME_TYPE(decltype(end.base()), std::ranges::sentinel_t<InputView>);
    assert(base(base(end.base())) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOne>;
    static_assert(!std::ranges::common_range<TransformView>);
    TransformView tv;
    auto end = tv.end();
    ASSERT_SAME_TYPE(decltype(end.base()), std::ranges::sentinel_t<InputView>);
    assert(base(base(end.base())) == globalBuff + 8);
    auto cend = std::as_const(tv).end();
    ASSERT_SAME_TYPE(decltype(cend.base()), std::ranges::sentinel_t<const InputView>);
    assert(base(base(cend.base())) == globalBuff + 8);
  }
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    static_assert(std::ranges::common_range<TransformView>);
    TransformView tv;
    auto end = tv.end();
    ASSERT_SAME_TYPE(decltype(end.base()), std::ranges::sentinel_t<MoveOnlyView>);
    assert(end.base() == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOne>;
    static_assert(std::ranges::common_range<TransformView>);
    TransformView tv;
    auto end = tv.end();
    ASSERT_SAME_TYPE(decltype(end.base()), std::ranges::sentinel_t<MoveOnlyView>);
    assert(end.base() == globalBuff + 8);
    auto cend = std::as_const(tv).end();
    ASSERT_SAME_TYPE(decltype(cend.base()), std::ranges::sentinel_t<const MoveOnlyView>);
    assert(cend.base() == globalBuff + 8);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
