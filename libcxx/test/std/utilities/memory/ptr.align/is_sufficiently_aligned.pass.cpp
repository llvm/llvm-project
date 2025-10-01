//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// #include <memory>

// template<size_t Alignment, class T>
//   bool is_sufficiently_aligned(T* ptr);

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_is_sufficiently_aligned() {
  constexpr std::size_t N = alignof(T);

  alignas(8 * N) std::remove_cv_t<T> buf[5];
  constexpr std::size_t Sz = sizeof(T);

  assert(std::is_sufficiently_aligned<N>(&buf[0]));
  assert(std::is_sufficiently_aligned<2 * N>(&buf[0]));
  assert(std::is_sufficiently_aligned<4 * N>(&buf[0]));
  assert(std::is_sufficiently_aligned<8 * N>(&buf[0]));

  assert(std::is_sufficiently_aligned<N>(&buf[1]));
  assert(std::is_sufficiently_aligned<2 * N>(&buf[1]) == (((1 * Sz) % (2 * N)) == 0));
  assert(std::is_sufficiently_aligned<4 * N>(&buf[1]) == (((1 * Sz) % (4 * N)) == 0));
  assert(std::is_sufficiently_aligned<8 * N>(&buf[1]) == (((1 * Sz) % (8 * N)) == 0));

  assert(std::is_sufficiently_aligned<N>(&buf[2]));
  assert(std::is_sufficiently_aligned<2 * N>(&buf[2]) == (((2 * Sz) % (2 * N)) == 0));
  assert(std::is_sufficiently_aligned<4 * N>(&buf[2]) == (((2 * Sz) % (4 * N)) == 0));
  assert(std::is_sufficiently_aligned<8 * N>(&buf[2]) == (((2 * Sz) % (8 * N)) == 0));

  assert(std::is_sufficiently_aligned<N>(&buf[3]));
  assert(std::is_sufficiently_aligned<2 * N>(&buf[3]) == (((3 * Sz) % (2 * N)) == 0));
  assert(std::is_sufficiently_aligned<4 * N>(&buf[3]) == (((3 * Sz) % (4 * N)) == 0));
  assert(std::is_sufficiently_aligned<8 * N>(&buf[3]) == (((3 * Sz) % (8 * N)) == 0));

  assert(std::is_sufficiently_aligned<N>(&buf[4]));
  assert(std::is_sufficiently_aligned<2 * N>(&buf[4]) == (((4 * Sz) % (2 * N)) == 0));
  assert(std::is_sufficiently_aligned<4 * N>(&buf[4]) == (((4 * Sz) % (4 * N)) == 0));
  assert(std::is_sufficiently_aligned<8 * N>(&buf[4]) == (((4 * Sz) % (8 * N)) == 0));
}

template <typename T>
void check(T* p) {
  ASSERT_SAME_TYPE(bool, decltype(std::is_sufficiently_aligned<alignof(T)>(p)));
  test_is_sufficiently_aligned<T>();
  test_is_sufficiently_aligned<const T>();
}

struct S {};
struct alignas(4) S4 {};
struct alignas(8) S8 {};
struct alignas(16) S16 {};
struct alignas(32) S32 {};
struct alignas(64) S64 {};
struct alignas(128) S128 {};

struct alignas(1) X {
  unsigned char d[2];
};
static_assert(sizeof(X) == 2 * alignof(X));

bool tests() {
  char c;
  int i;
  long l;
  double d;
  long double ld;
  check(&c);
  check(&i);
  check(&l);
  check(&d);
  check(&ld);

  S s;
  S4 s4;
  S8 s8;
  S16 s16;
  S32 s32;
  S64 s64;
  S128 s128;
  check(&s);
  check(&s4);
  check(&s8);
  check(&s16);
  check(&s32);
  check(&s64);
  check(&s128);

  X x;
  check(&x);

  return true;
}

int main(int, char**) {
  tests();

  return 0;
}
