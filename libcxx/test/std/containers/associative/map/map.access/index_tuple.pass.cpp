//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// mapped_type& operator[](const key_type& k); // constexpr since C++26

// https://llvm.org/PR16542

#include <map>
#include <tuple>

#include <cassert>
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  std::map<std::tuple<int, int>, std::size_t> m;
  m[std::make_tuple(2, 3)] = 7;
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
