//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <functional> functions are marked [[nodiscard]]

#include <cstddef>
#include <functional>

#include "test_macros.h"

void test() {
  int i = 0;

  // Function wrappers

#if TEST_STD_VER >= 11 && !defined(TEST_HAS_NO_RTTI)
  std::function<void(int)> f;
  const std::function<void(int)> cf;

  f.target_type();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  f.target<void(int)>();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cf.target<void(int)>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  struct ZMT {
    void member_function() {};
  };
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mem_fn(&ZMT::member_function);

  // Identity

#if TEST_STD_VER >= 20
  std::identity{}(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  // Partial function application

#if TEST_STD_VER >= 23
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bind_back([](int a) { return a; }, 94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bind_front([](int a) { return a; }, 94);
#endif
#if TEST_STD_VER >= 11
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bind([](int a) { return a; }, 94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::bind<float>([](int a) { return a; }, 94);
#endif

  // Reference wrappers

  std::reference_wrapper<int> rw = i;
  rw.get(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::ref(i);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cref(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Hash specializations

  std::hash<std::nullptr_t> hash;
  hash(nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
