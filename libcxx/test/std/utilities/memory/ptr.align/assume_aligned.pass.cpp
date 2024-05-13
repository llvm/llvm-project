//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <memory>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <typename T>
constexpr void check(T* p) {
  ASSERT_SAME_TYPE(T*, decltype(std::assume_aligned<1>(p)));
  constexpr std::size_t alignment = alignof(T);

  if constexpr (alignment >= 1)
    assert(p == std::assume_aligned<1>(p));
  if constexpr (alignment >= 2)
    assert(p == std::assume_aligned<2>(p));
  if constexpr (alignment >= 4)
    assert(p == std::assume_aligned<4>(p));
  if constexpr (alignment >= 8)
    assert(p == std::assume_aligned<8>(p));
  if constexpr (alignment >= 16)
    assert(p == std::assume_aligned<16>(p));
  if constexpr (alignment >= 32)
    assert(p == std::assume_aligned<32>(p));
  if constexpr (alignment >= 64)
    assert(p == std::assume_aligned<64>(p));
  if constexpr (alignment >= 128)
    assert(p == std::assume_aligned<128>(p));
}

struct              S    { };
struct alignas(  4) S4   { };
struct alignas(  8) S8   { };
struct alignas( 16) S16  { };
struct alignas( 32) S32  { };
struct alignas( 64) S64  { };
struct alignas(128) S128 { };

constexpr bool tests() {
  char        c;
  int         i;
  long        l;
  double      d;
  long double ld;
  check( &c);
  check( &i);
  check( &l);
  check( &d);
  check(&ld);

  S    s;
  S4   s4;
  S8   s8;
  S16  s16;
  S32  s32;
  S64  s64;
  S128 s128;
  check(&s);
  check(&s4);
  check(&s8);
  check(&s16);
  check(&s32);
  check(&s64);
  check(&s128);

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
