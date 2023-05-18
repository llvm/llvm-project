//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct B {};

struct A1 {
  mutable B b_;
  TEST_CONSTEXPR operator B&() const { return b_; }
};

struct A2 {
  mutable B b_;
  TEST_CONSTEXPR operator B&() const TEST_NOEXCEPT { return b_; }
};

void implicitly_convert(std::reference_wrapper<B>) TEST_NOEXCEPT;

TEST_CONSTEXPR_CXX20 bool test()
{
  {
    A1 a;
    ASSERT_NOT_NOEXCEPT(implicitly_convert(a));
    std::reference_wrapper<B> b1 = a;
    assert(&b1.get() == &a.b_);
    ASSERT_NOT_NOEXCEPT(b1 = a);
    b1 = a;
    assert(&b1.get() == &a.b_);
  }
  {
    A2 a;
    ASSERT_NOEXCEPT(implicitly_convert(a));
    std::reference_wrapper<B> b2 = a;
    assert(&b2.get() == &a.b_);
    ASSERT_NOEXCEPT(b2 = a);
    b2 = a;
    assert(&b2.get() == &a.b_);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
