//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// static void reserve(size_type n);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 10>;
    V v;
    for (int i = 0; i <= 10; ++i) {
      v.reserve(i);
    }
    static_assert((v.reserve(0), v.reserve(1), v.reserve(2), v.reserve(8), v.reserve(9), v.reserve(10), true));
    static_assert((V::reserve(0), V::reserve(1), V::reserve(2), V::reserve(8), V::reserve(9), V::reserve(10), true));
    if !consteval {
#ifndef TEST_HAS_NO_EXCEPTIONS
      try {
        V::reserve(11);
        assert(false);
      } catch (const std::bad_alloc&) {
      } catch (...) {
        assert(false);
      }
      try {
        V::reserve(-1);
        assert(false);
      } catch (const std::bad_alloc&) {
      } catch (...) {
        assert(false);
      }
#endif
    }
  }
  {
    using V = std::inplace_vector<int, 0>;
    V v;
    v.reserve(0);
    static_assert((v.reserve(0), true));
    static_assert((V::reserve(0), true));
    if !consteval {
#ifndef TEST_HAS_NO_EXCEPTIONS
      try {
        V::reserve(1);
        assert(false);
      } catch (const std::bad_alloc&) {
      } catch (...) {
        assert(false);
      }
      try {
        V::reserve(-1);
        assert(false);
      } catch (const std::bad_alloc&) {
      } catch (...) {
        assert(false);
      }
#endif
    }
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
