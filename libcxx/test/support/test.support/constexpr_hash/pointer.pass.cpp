//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

#include "constexpr_hash.h"

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

template <class T>
constexpr void test() {
  typedef support::constexpr_hash<T> H;

  ASSERT_NOEXCEPT(H()(T()));
  H h;

  typedef typename std::remove_pointer<T>::type type;
  type i;
  type j;
  assert(h(&i) != h(&j));
}

constexpr void test_nullptr() {
  typedef std::nullptr_t T;
  typedef support::constexpr_hash<T> H;

  ASSERT_NOEXCEPT(H()(T()));
}

constexpr bool test_with_hash() {
  test_nullptr();
  return true;
}

int main(int, char**) {
  assert(test_with_hash());

  static_assert(test_with_hash());

  return 0;
}
