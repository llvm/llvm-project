//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class... TArgs, class... BoundArgs>
//       requires constructible_from<T, TArgs...> &&
//                constructible_from<Bound, BoundArgs...>
//     constexpr explicit repeat_view(piecewise_construct_t,
//       tuple<TArgs...> value_args, tuple<BoundArgs...> bound_args = tuple<>{});

#include <ranges>
#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

struct C {};

struct B {
  int v;
};

struct A {
  int x = 111;
  int y = 222;

  constexpr A() = default;
  constexpr A(B b) : x(b.v), y(b.v + 1) {}
  constexpr A(int _x, int _y) : x(_x), y(_y) {}
};

static_assert(std::constructible_from<std::ranges::repeat_view<A, int>,
                                      std::piecewise_construct_t,
                                      std::tuple<int, int>,
                                      std::tuple<int>>);
static_assert(std::constructible_from<std::ranges::repeat_view<A, int>,
                                      std::piecewise_construct_t,
                                      std::tuple<B>,
                                      std::tuple<int>>);
static_assert(std::constructible_from<std::ranges::repeat_view<A, int>,
                                      std::piecewise_construct_t,
                                      std::tuple<>,
                                      std::tuple<int>>);
static_assert(std::constructible_from<std::ranges::repeat_view<A>,
                                      std::piecewise_construct_t,
                                      std::tuple<int, int>,
                                      std::tuple<std::unreachable_sentinel_t>>);
static_assert(std::constructible_from<std::ranges::repeat_view<A>,
                                      std::piecewise_construct_t,
                                      std::tuple<B>,
                                      std::tuple<std::unreachable_sentinel_t>>);
static_assert(std::constructible_from<std::ranges::repeat_view<A>,
                                      std::piecewise_construct_t,
                                      std::tuple<>,
                                      std::tuple<std::unreachable_sentinel_t>>);
static_assert(!std::constructible_from<std::ranges::repeat_view<A, int>,
                                       std::piecewise_construct_t,
                                       std::tuple<C>,
                                       std::tuple<int>>);
static_assert(!std::constructible_from<std::ranges::repeat_view<A>,
                                       std::piecewise_construct_t,
                                       std::tuple<C>,
                                       std::tuple<std::unreachable_sentinel_t>>);
static_assert(!std::constructible_from<std::ranges::repeat_view<A, int>,
                                       std::piecewise_construct_t,
                                       std::tuple<int, int, int>,
                                       std::tuple<int>>);
static_assert(!std::constructible_from<std::ranges::repeat_view<A>,
                                       std::piecewise_construct_t,
                                       std::tuple<int, int, int>,
                                       std::tuple<std::unreachable_sentinel_t>>);
static_assert(
    !std::constructible_from<std::ranges::repeat_view<A>, std::piecewise_construct_t, std::tuple<B>, std::tuple<int>>);

constexpr bool test() {
  {
    std::ranges::repeat_view<A, int> rv(std::piecewise_construct, std::tuple{}, std::tuple{3});
    assert(rv.size() == 3);
    assert(rv[0].x == 111);
    assert(rv[0].y == 222);
    assert(rv.begin() + 3 == rv.end());
  }
  {
    std::ranges::repeat_view<A> rv(std::piecewise_construct, std::tuple{}, std::tuple{std::unreachable_sentinel});
    assert(rv[0].x == 111);
    assert(rv[0].y == 222);
    assert(rv.begin() + 300 != rv.end());
  }
  {
    std::ranges::repeat_view<A, int> rv(std::piecewise_construct, std::tuple{1, 2}, std::tuple{3});
    assert(rv.size() == 3);
    assert(rv[0].x == 1);
    assert(rv[0].y == 2);
    assert(rv.begin() + 3 == rv.end());
  }
  {
    std::ranges::repeat_view<A> rv(std::piecewise_construct, std::tuple{1, 2}, std::tuple{std::unreachable_sentinel});
    assert(rv[0].x == 1);
    assert(rv[0].y == 2);
    assert(rv.begin() + 300 != rv.end());
  }
  {
    std::ranges::repeat_view<A, int> rv(std::piecewise_construct, std::tuple{B{11}}, std::tuple{3});
    assert(rv.size() == 3);
    assert(rv[0].x == 11);
    assert(rv[0].y == 12);
    assert(rv.begin() + 3 == rv.end());
  }
  {
    std::ranges::repeat_view<A> rv(std::piecewise_construct, std::tuple{B{10}}, std::tuple{std::unreachable_sentinel});
    assert(rv[0].x == 10);
    assert(rv[0].y == 11);
    assert(rv.begin() + 300 != rv.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
