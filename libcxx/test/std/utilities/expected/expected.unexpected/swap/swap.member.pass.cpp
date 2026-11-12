//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr void swap(unexpected& other) noexcept(is_nothrow_swappable_v<E>);
//
// Mandates: is_swappable_v<E> is true.
//
// Effects: Equivalent to: using std::swap; swap(unex, other.unex);

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

// test noexcept
struct NoexceptSwap {
  friend void swap(NoexceptSwap&, NoexceptSwap&) noexcept;
};

struct MayThrowSwap {
  friend void swap(MayThrowSwap&, MayThrowSwap&);
};

template <class T>
concept MemberSwapNoexcept =
    requires(T& t1, T& t2) {
      { t1.swap(t2) } noexcept;
    };

static_assert(MemberSwapNoexcept<std::unexpected<NoexceptSwap>>);
static_assert(!MemberSwapNoexcept<std::unexpected<MayThrowSwap>>);

struct ADLSwap {
  constexpr ADLSwap(int ii) : i(ii) {}
  ADLSwap& operator=(const ADLSwap&) = delete;
  int i;
  constexpr friend void swap(ADLSwap& x, ADLSwap& y) { std::swap(x.i, y.i); }
};

constexpr bool test() {
  // using std::swap;
  {
    std::unexpected<int> unex1(5);
    std::unexpected<int> unex2(6);
    unex1.swap(unex2);
    assert(unex1.error() == 6);
    assert(unex2.error() == 5);
  }

  // adl swap
  {
    std::unexpected<ADLSwap> unex1(5);
    std::unexpected<ADLSwap> unex2(6);
    unex1.swap(unex2);
    assert(unex1.error().i == 6);
    assert(unex2.error().i == 5);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
