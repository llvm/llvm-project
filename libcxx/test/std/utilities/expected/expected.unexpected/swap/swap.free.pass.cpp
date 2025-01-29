//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr void swap(unexpected& x, unexpected& y) noexcept(noexcept(x.swap(y)));
//
// Constraints: is_swappable_v<E> is true.
//
// Effects: Equivalent to x.swap(y).

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

// test noexcept
struct NoexceptSwap {
  friend void swap(NoexceptSwap&, NoexceptSwap&) noexcept;
};

struct MayThrowSwap {
  friend void swap(MayThrowSwap&, MayThrowSwap&);
};

template <class T>
concept ADLSwapNoexcept =
    requires(T& t1, T& t2) {
      { swap(t1, t2) } noexcept;
    };

static_assert(ADLSwapNoexcept<std::unexpected<NoexceptSwap>>);
static_assert(!ADLSwapNoexcept<std::unexpected<MayThrowSwap>>);

// test constraint
struct NonSwappable {
  NonSwappable& operator=(const NonSwappable&) = delete;
};

static_assert(std::is_swappable_v<std::unexpected<int>>);
static_assert(std::is_swappable_v<std::unexpected<MayThrowSwap>>);
static_assert(!std::is_swappable_v<std::unexpected<NonSwappable>>);

struct ADLSwap {
  constexpr ADLSwap(int ii) : i(ii) {}
  ADLSwap& operator=(const ADLSwap&) = delete;
  int i;
  constexpr friend void swap(ADLSwap& x, ADLSwap& y) { std::swap(x.i, y.i); }
};

constexpr bool test() {
  std::unexpected<ADLSwap> unex1(5);
  std::unexpected<ADLSwap> unex2(6);
  swap(unex1, unex2);
  assert(unex1.error().i == 6);
  assert(unex2.error().i == 5);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
