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
    auto it = tv.end();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), const forward_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), forward_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), const forward_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), const forward_iterator<int*>&);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(std::move(it).base()) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOneMutable>;
    static_assert(!std::ranges::common_range<TransformView>);
    TransformView tv;
    auto sent = tv.end();
    using Sent = decltype(sent);
    ASSERT_SAME_TYPE(decltype(static_cast<Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    assert(base(base(sent.base())) == globalBuff + 8);
    assert(base(base(std::move(sent).base())) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<InputView, PlusOne>;
    static_assert(!std::ranges::common_range<TransformView>);
    TransformView tv;
    auto sent = tv.end();
    using Sent = decltype(sent);
    ASSERT_SAME_TYPE(decltype(static_cast<Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const Sent&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const Sent&&>(sent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    assert(base(base(sent.base())) == globalBuff + 8);
    assert(base(base(std::move(sent).base())) == globalBuff + 8);

    auto csent = std::as_const(tv).end();
    using CSent = decltype(csent);
    ASSERT_SAME_TYPE(decltype(static_cast<CSent&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<CSent&&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const CSent&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(static_cast<const CSent&&>(csent).base()), sentinel_wrapper<cpp20_input_iterator<int*>>);
    assert(base(base(csent.base())) == globalBuff + 8);
    assert(base(base(std::move(csent).base())) == globalBuff + 8);
  }
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    static_assert(std::ranges::common_range<TransformView>);
    TransformView tv;
    auto it = tv.end();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), int*);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), int* const&);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(std::move(it).base()) == globalBuff + 8);
    static_assert(!HasConstQualifiedEnd<TransformView>);
  }
  {
    using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOne>;
    static_assert(std::ranges::common_range<TransformView>);
    TransformView tv;
    auto it = tv.end();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), int*);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), int* const&);
    assert(base(it.base()) == globalBuff + 8);
    assert(base(std::move(it).base()) == globalBuff + 8);

    auto csent = std::as_const(tv).end();
    using CSent = decltype(csent);
    ASSERT_SAME_TYPE(decltype(static_cast<CSent&>(csent).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<CSent&&>(csent).base()), int*);
    ASSERT_SAME_TYPE(decltype(static_cast<const CSent&>(csent).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<const CSent&&>(csent).base()), int* const&);
    assert(base(base(csent.base())) == globalBuff + 8);
    assert(base(base(std::move(csent).base())) == globalBuff + 8);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
