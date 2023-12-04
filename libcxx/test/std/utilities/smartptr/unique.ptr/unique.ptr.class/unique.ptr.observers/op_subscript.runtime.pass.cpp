//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op[](size_t)

#include <memory>
#include <cassert>

// TODO: Move TEST_IS_CONSTANT_EVALUATED into its own header
#include <type_traits>

#include "test_macros.h"

class A {
  int state_;
  static int next_;

public:
  TEST_CONSTEXPR_CXX23 A() : state_(0) {
    if (!TEST_IS_CONSTANT_EVALUATED)
      state_ = ++next_;
  }

  TEST_CONSTEXPR_CXX23 int get() const { return state_; }

  friend TEST_CONSTEXPR_CXX23 bool operator==(const A& x, int y) { return x.state_ == y; }

  TEST_CONSTEXPR_CXX23 A& operator=(int i) {
    state_ = i;
    return *this;
  }
};

int A::next_ = 0;

TEST_CONSTEXPR_CXX23 bool test() {
  std::unique_ptr<A[]> p(new A[3]);
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
  }
  p[0] = 3;
  p[1] = 2;
  p[2] = 1;
  assert(p[0] == 3);
  assert(p[1] == 2);
  assert(p[2] == 1);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
