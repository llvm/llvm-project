//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit zip_transform_view(F, Views...)

#include <algorithm>
#include <ranges>
#include <vector>

#include "types.h"

struct Fn {
  int operator()(auto&&...) const { return 5; }
};

template <class T, class... Args>
concept IsImplicitlyConstructible = requires(T val, Args... args) { val = {std::forward<Args>(args)...}; };

// test constructor is explicit
static_assert(std::constructible_from<std::ranges::zip_transform_view<Fn, IntView>, Fn, IntView>);
static_assert(!IsImplicitlyConstructible<std::ranges::zip_transform_view<Fn, IntView>, Fn, IntView>);

static_assert(std::constructible_from<std::ranges::zip_transform_view<Fn, IntView, IntView>, Fn, IntView, IntView>);
static_assert(!IsImplicitlyConstructible<std::ranges::zip_transform_view<Fn, IntView, IntView>, Fn, IntView, IntView>);

struct MoveAwareView : std::ranges::view_base {
  int moves                 = 0;
  constexpr MoveAwareView() = default;
  constexpr MoveAwareView(MoveAwareView&& other) : moves(other.moves + 1) { other.moves = 1; }
  constexpr MoveAwareView& operator=(MoveAwareView&& other) {
    moves       = other.moves + 1;
    other.moves = 0;
    return *this;
  }
  constexpr const int* begin() const { return &moves; }
  constexpr const int* end() const { return &moves + 1; }
};

template <class View1, class View2>
constexpr void constructorTest(auto&& buffer1, auto&& buffer2) {
  std::ranges::zip_transform_view v{MakeTuple{}, View1{buffer1}, View2{buffer2}};
  auto [i, j] = *v.begin();
  assert(i == buffer1[0]);
  assert(j == buffer2[0]);
};

constexpr bool test() {
  int buffer[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[4] = {9, 8, 7, 6};

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer2});
    assert(std::ranges::equal(v, std::vector{std::tuple(9), std::tuple(8), std::tuple(7), std::tuple(6)}));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    assert(std::ranges::equal(v, std::vector{1, 2, 3, 4, 5, 6, 7, 8}));
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer2}, std::ranges::single_view(2.));
    assert(std::ranges::equal(v, std::vector{std::tuple(1, 9, 2.0)}));
  }

  {
    // single empty range
    std::ranges::zip_transform_view v(MakeTuple{}, std::ranges::empty_view<int>());
    assert(std::ranges::empty(v));
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(std::ranges::empty(v));
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(std::ranges::empty(v));
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::empty_view<int>());
    assert(std::ranges::empty(v));
  }
  {
    // constructor from views
    std::ranges::zip_transform_view v(
        MakeTuple{}, SizedRandomAccessView{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    auto [i, j, k] = *v.begin();
    assert(i == 1);
    assert(j == 0);
    assert(k == 2.0);
  }

  {
    // arguments are moved once
    MoveAwareView mv;
    std::ranges::zip_transform_view v{MakeTuple{}, std::move(mv), MoveAwareView{}};
    auto [numMoves1, numMoves2] = *v.begin();
    assert(numMoves1 == 3); // one move from the local variable to parameter, one move from parameter to member
    assert(numMoves2 == 2);
  }

  // input and forward
  {
    constructorTest<InputCommonView, ForwardSizedView>(buffer, buffer2);
  }

  // bidi and random_access
  {
    constructorTest<BidiCommonView, SizedRandomAccessView>(buffer, buffer2);
  }

  // contiguous
  {
    constructorTest<ContiguousCommonView, ContiguousCommonView>(buffer, buffer2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
