//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::join

#include <ranges>

#include <cassert>
#include <type_traits>

#include "types.h"

struct MoveOnlyOuter : SimpleForwardCommonOuter<ForwardCommonInner> {
  using SimpleForwardCommonOuter<ForwardCommonInner>::SimpleForwardCommonOuter;

  constexpr MoveOnlyOuter(MoveOnlyOuter&&) = default;
  constexpr MoveOnlyOuter(const MoveOnlyOuter&) = delete;

  constexpr MoveOnlyOuter& operator=(MoveOnlyOuter&&) = default;
  constexpr MoveOnlyOuter& operator=(const MoveOnlyOuter&) = delete;
};

struct Foo {
  int i;
  constexpr Foo(int ii) : i(ii) {}
};

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

constexpr bool test() {
  int buffer1[3] = {1, 2, 3};
  int buffer2[2] = {4, 5};
  int buffer3[4] = {6, 7, 8, 9};
  Foo nested[2][3][3] = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}};

  {
    // Test `views::join(v)`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result = std::ranges::join_view<std::ranges::ref_view<ForwardCommonInner[3]>>;
    std::same_as<Result> decltype(auto) v = std::views::join(inners);
    assert(std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);
  }

  {
    // Test `views::join(move-only-view)`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result = std::ranges::join_view<MoveOnlyOuter>;
    std::same_as<Result> decltype(auto) v = std::views::join(MoveOnlyOuter{inners});
    assert(std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);

    static_assert(std::invocable<decltype(std::views::join), MoveOnlyOuter>);
    static_assert(!std::invocable<decltype(std::views::join), MoveOnlyOuter&>);
  }

  {
    // LWG3474 Nesting `join_views` is broken because of CTAD
    // views::join(join_view) should join the view instead of calling copy constructor
    auto jv = std::views::join(nested);
    ASSERT_SAME_TYPE(std::ranges::range_reference_t<decltype(jv)>, Foo(&)[3]);

    auto jv2 = std::views::join(jv);
    ASSERT_SAME_TYPE(std::ranges::range_reference_t<decltype(jv2)>, Foo&);

    assert(&(*jv2.begin()) == &nested[0][0][0]);
  }

  {
    // Test `v | views::join`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};

    using Result = std::ranges::join_view<std::ranges::ref_view<ForwardCommonInner[3]>>;
    std::same_as<Result> decltype(auto) v = inners | std::views::join;
    assert(std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);
    static_assert(CanBePiped<decltype((inners)), decltype((std::views::join))>);
  }

  {
    // Test `move-only-view | views::join`
    ForwardCommonInner inners[3] = {buffer1, buffer2, buffer3};
    using Result = std::ranges::join_view<MoveOnlyOuter>;
    std::same_as<Result> decltype(auto) v = MoveOnlyOuter{inners} | std::views::join;
    assert(std::ranges::next(v.begin(), 9) == v.end());
    assert(&(*v.begin()) == buffer1);

    static_assert(CanBePiped<MoveOnlyOuter, decltype((std::views::join))>);
    static_assert(!CanBePiped<MoveOnlyOuter&, decltype((std::views::join))>);
  }

  {
    // LWG3474 Nesting `join_views` is broken because of CTAD
    // join_view | views::join should join the view instead of calling copy constructor
    auto jv = nested | std::views::join | std::views::join;
    ASSERT_SAME_TYPE(std::ranges::range_reference_t<decltype(jv)>, Foo&);

    assert(&(*jv.begin()) == &nested[0][0][0]);
    static_assert(CanBePiped<decltype((nested)), decltype((std::views::join))>);
  }

  {
    // Test `adaptor | views::join`
    auto join_twice = std::views::join | std::views::join;
    auto jv = nested | join_twice;
    ASSERT_SAME_TYPE(std::ranges::range_reference_t<decltype(jv)>, Foo&);

    assert(&(*jv.begin()) == &nested[0][0][0]);
    static_assert(CanBePiped<decltype((nested)), decltype((join_twice))>);
  }

  {
    static_assert(!CanBePiped<int, decltype((std::views::join))>);
    static_assert(!CanBePiped<Foo, decltype((std::views::join))>);
    static_assert(!CanBePiped<int(&)[2], decltype((std::views::join))>);
    static_assert(CanBePiped<int(&)[2][2], decltype((std::views::join))>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
