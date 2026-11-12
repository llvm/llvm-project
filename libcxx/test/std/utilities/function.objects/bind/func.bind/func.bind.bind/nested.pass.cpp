//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);    // constexpr since C++20
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);    // constexpr since C++20

// https://llvm.org/PR16343

#include <functional>
#include <cassert>

#include "test_macros.h"

struct multiply {
  template <typename T>
  TEST_CONSTEXPR_CXX20 T operator()(T a, T b) {
    return a * b;
  }
};

struct plus_one {
  template <typename T>
  TEST_CONSTEXPR_CXX20 T operator()(T a) {
    return a + 1;
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  using std::placeholders::_1;
  auto g = std::bind(multiply(), 2, _1);
  assert(g(5) == 10);
  assert(std::bind(plus_one(), g)(5) == 11);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
