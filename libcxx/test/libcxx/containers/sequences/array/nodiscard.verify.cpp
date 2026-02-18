//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <array> functions are marked [[nodiscard]]

#include <array>
#include <utility>

#include <test_macros.h>

template <std::size_t N>
void test_members() {
  std::array<int, N> a;
  const std::array<int, N> ca = {};

  a.begin();    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.begin();   // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.end();      // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.end();     // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rbegin();   // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rbegin();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.rend();     // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.rend();    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cbegin();   // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cbegin();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.cend();     // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.cend();    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crbegin();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crbegin(); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.crend();    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.crend();   // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.size();     // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.max_size(); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.empty();    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a[0];     // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca[0];    // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.at(0);  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.at(0); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.front();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.front(); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.back();   // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.back();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}

  a.data();  // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ca.data(); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
}

template <typename ArrT>
void test_get() {
  std::array<int, 94> a = {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(a);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
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
  test_members<0>();
  test_members<82>();
}
