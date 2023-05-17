//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test release

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_basic() {
  typedef typename std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
#if TEST_STD_VER >= 11
  {
    using U = std::unique_ptr<VT>;
    U u; ((void)u);
    ASSERT_NOEXCEPT(u.release());
  }
#endif
  {
    std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == expect_alive);
    A* ap = p.get();
    A* a = p.release();
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == expect_alive);
    assert(p.get() == nullptr);
    assert(ap == a);
    assert(a != nullptr);

    if (IsArray)
      delete[] a;
    else
      delete a;

    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_basic</*IsArray*/ false>();
  test_basic<true>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
