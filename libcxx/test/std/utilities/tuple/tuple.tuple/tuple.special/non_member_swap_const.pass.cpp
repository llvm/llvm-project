//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   void swap(const tuple<Types...>& x, const tuple<Types...>& y);

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <tuple>
#include <cassert>

struct S {
  int* calls;
  friend constexpr void swap(S& a, S& b) {
    *a.calls += 1;
    *b.calls += 1;
  }
};
struct CS {
  int* calls;
  friend constexpr void swap(const CS& a, const CS& b) {
    *a.calls += 1;
    *b.calls += 1;
  }
};

static_assert(std::is_swappable_v<std::tuple<>>);
static_assert(std::is_swappable_v<std::tuple<S>>);
static_assert(std::is_swappable_v<std::tuple<CS>>);
static_assert(std::is_swappable_v<std::tuple<S&>>);
static_assert(std::is_swappable_v<std::tuple<CS, S>>);
static_assert(std::is_swappable_v<std::tuple<CS, S&>>);
static_assert(std::is_swappable_v<const std::tuple<>>);
static_assert(!std::is_swappable_v<const std::tuple<S>>);
static_assert(std::is_swappable_v<const std::tuple<CS>>);
static_assert(std::is_swappable_v<const std::tuple<S&>>);
static_assert(!std::is_swappable_v<const std::tuple<CS, S>>);
static_assert(std::is_swappable_v<const std::tuple<CS, S&>>);

constexpr bool test() {
  int cs_calls = 0;
  int s_calls = 0;
  S s1{&s_calls};
  S s2{&s_calls};
  const std::tuple<CS, S&> t1 = {CS{&cs_calls}, s1};
  const std::tuple<CS, S&> t2 = {CS{&cs_calls}, s2};
  swap(t1, t2);
  assert(cs_calls == 2);
  assert(s_calls == 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
