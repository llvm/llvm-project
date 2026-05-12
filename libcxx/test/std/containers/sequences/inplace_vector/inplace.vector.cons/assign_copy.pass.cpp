//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector& operator=(const inplace_vector& other);

#include <cassert>
#include <inplace_vector>
#include <utility>

#include "../common.h"
#include "test_macros.h"

struct ThrowCopyAssign {
  ThrowCopyAssign(const ThrowCopyAssign&) = default;
  ThrowCopyAssign& operator=(const ThrowCopyAssign&) { return *this; }
};

struct ThrowCopyCtor {
  ThrowCopyCtor(const ThrowCopyCtor&) {}
  ThrowCopyCtor& operator=(const ThrowCopyCtor&) = default;
};

constexpr bool test() {
  {
    using C = std::inplace_vector<int, 8>;
    C c{1, 2, 3};
    C other{4, 5};
    ASSERT_SAME_TYPE(C&, decltype(c = other));
    C& result = (c = other);
    assert(&result == &c);
    assert_inplace_vector_equal(c, {4, 5});
    assert_inplace_vector_equal(other, {4, 5});

    C& self = c;
    c       = self;
    assert_inplace_vector_equal(c, {4, 5});
  }

  {
    using C = std::inplace_vector<ThrowCopyAssign, 4>;
    ASSERT_NOT_NOEXCEPT(std::declval<C>() = std::declval<const C&>());
  }

  {
    using C = std::inplace_vector<ThrowCopyCtor, 4>;
    ASSERT_NOT_NOEXCEPT(std::declval<C>() = std::declval<const C&>());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
