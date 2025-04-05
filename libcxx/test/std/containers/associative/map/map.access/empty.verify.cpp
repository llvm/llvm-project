//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// bool empty() const noexcept;// constexpr since C++26

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <map>

bool test() {
  std::map<int, int> c;
  c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  return true;
}

int main() {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
}
