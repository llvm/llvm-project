//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// class deque

// bool empty() const noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <string>

#include "test_macros.h"

bool test() {
  std::string c;
  c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
