//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// back_insert_iterator

// _Container* __get_container(); // constexpr in C++20

#include <iterator>
#include <vector>

#include "test_macros.h"
#include "nasty_containers.h"
#include "test_constexpr_container.h"

template <class C>
TEST_CONSTEXPR_CXX20 bool test(C c) {
  const std::back_insert_iterator<C> i(c);
  assert(i.__get_container() == std::addressof(c));
  return true;
}

int main(int, char**) {
  test(std::vector<int>());
  test(nasty_vector<int>());
#if TEST_STD_VER >= 20
  test(ConstexprFixedCapacityDeque<int, 10>());
  static_assert(test(ConstexprFixedCapacityDeque<int, 10>()));
#endif
  return 0;
}
