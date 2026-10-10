//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr T* data() noexcept;
// constexpr const T* data() const noexcept;

#include <cassert>
#include <inplace_vector>
#include <utility>

#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 4> c{1, 2, 3};
  ASSERT_SAME_TYPE(decltype(c.data()), int*);
  ASSERT_SAME_TYPE(decltype(std::as_const(c).data()), const int*);

  assert(c.data() == &c.front());
  assert(std::as_const(c).data() == &c.front());
  assert(std::as_const(c).data() == &std::as_const(c).front());

  c.data()[1] = 5;
  assert(c[1] == 5);
  assert(std::as_const(c)[1] == 5);
  c.push_back(6);
  assert(c.data()[3] == 6);
  assert(std::as_const(c).data()[3] == 6);

  std::inplace_vector<int, 0> empty;
  assert(empty.data() == empty.data());
  assert(std::as_const(empty).data() == std::as_const(empty).data());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
