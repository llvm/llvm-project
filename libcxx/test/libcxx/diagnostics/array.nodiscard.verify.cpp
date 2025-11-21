//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <array> functions are marked [[nodiscard]]

#include <array>
#include <utility>

#include <test_macros.h>

template <typename ArrT>
void test_members() {
  ArrT a{};

  a.begin();   // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.end();     // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rbegin();  // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rend();    // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cbegin();  // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cend();    // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crbegin(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crend();   // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.size();     // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.max_size(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.empty();    // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a[0];    // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.at(0); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.front(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.back();  // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.data(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
}

template <typename ArrT>
void test_get() {
  ArrT a{};

  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(a);
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(a));
}

#if TEST_STD_VER >= 20
void test_to_array() {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_array("zmt");
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_array({94, 82, 49});
}
#endif

void test() {
  test_members<std::array<int, 82>>();
  test_members<const std::array<int, 82>>();
  test_members<std::array<int, 0>>();
  test_members<const std::array<int, 0>>();

  test_get<std::array<int, 82>>();
  test_get<const std::array<int, 82>>();
}
