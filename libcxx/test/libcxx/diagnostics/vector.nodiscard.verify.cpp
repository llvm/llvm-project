//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <vector> functions are marked [[nodiscard]]

#include <type_traits>
#include <vector>

template <class VecT>
void test() {
  VecT v;

  v.at(0);           // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.back();          // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();         // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.capacity();      // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.cbegin();        // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.cend();          // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.crbegin();       // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.crend();         // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.empty();         // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();           // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.front();         // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.get_allocator(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.max_size();      // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.rbegin();        // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.rend();          // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();          // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v[0];              // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
}

template <class VecT>
void test_non_vector_bool() {
  VecT v;

  v.data(); // expected-warning 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void instantiate() {
  test<std::vector<int> >();
  test<const std::vector<int> >();
  test<std::vector<bool> >();
  test<const std::vector<bool> >();

  test_non_vector_bool<std::vector<int> >();
  test_non_vector_bool<const std::vector<int> >();
}
