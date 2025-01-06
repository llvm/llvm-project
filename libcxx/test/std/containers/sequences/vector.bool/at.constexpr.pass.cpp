//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr const_reference at(size_type n) const;
// constexpr reference at(size_type n);

#include <cassert>
#include <memory>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"

template <typename T, typename Allocator>
constexpr void test() {
  std::vector<T, Allocator> v{1, 0, 1};
  assert(v.at(0) == 1);
  assert(v.at(1) == 0);
  assert(v.at(2) == 1);
}

constexpr bool tests() {
  test<bool, std::allocator<bool>>();
  test<bool, min_allocator<bool>>();
  test<bool, test_allocator<bool>>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
