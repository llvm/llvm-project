//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector() noexcept;

#include <cassert>
#include <inplace_vector>
#include <type_traits>

#include "test_macros.h"
#include "../../../NotConstructible.h"
#include "../common.h"

struct ThrowDefault {
  constexpr ThrowDefault() {}
};

struct NoDefault {
  NoDefault() = delete;
};

template <class C>
constexpr void test() {
  ASSERT_NOEXCEPT(C{});
  C c;
  assert(c.empty());
  assert(c.size() == 0);
  assert(c.capacity() == C::capacity());

  C c1 = {};
  assert(c1.empty());
}

constexpr bool tests() {
  test<std::inplace_vector<int, 0> >();
  test<std::inplace_vector<int, 8> >();
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    test<std::inplace_vector<NotConstructible, 8> >();
  }

  {
    ASSERT_NOEXCEPT(std::inplace_vector<int, 0>());
    ASSERT_NOEXCEPT(std::inplace_vector<int, 8>());
    ASSERT_NOEXCEPT(std::inplace_vector<ThrowDefault, 1>());
    ASSERT_NOEXCEPT(std::inplace_vector<ThrowDefault, 0>());
    static_assert(std::is_nothrow_default_constructible_v<std::inplace_vector<NoDefault, 8>>);
    static_assert(std::is_nothrow_default_constructible_v<std::inplace_vector<NoDefault, 0>>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
