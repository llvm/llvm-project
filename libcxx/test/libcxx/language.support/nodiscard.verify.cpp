//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Check that functions are marked [[nodiscard]]

#include <compare>
#include <coroutine>
#include <functional>
#include <initializer_list>

#include "test_macros.h"

void test() {
#if TEST_STD_VER >= 20
  { // <compare>
    int x     = 94;
    int y     = 82;
    auto oRes = x <=> y;

    std::is_eq(oRes);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_neq(oRes);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_lt(oRes);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_lteq(oRes); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_gt(oRes);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_gteq(oRes); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

#if TEST_STD_VER >= 20
  { // <coroutine>
    struct EmptyPromise {
    } promise;

    {
      std::coroutine_handle<void> cr{};

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.address();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::coroutine_handle<void>::from_address(&promise);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.done();

      std::hash<std::coroutine_handle<void>> hash;
      hash(cr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    }
    {
      std::coroutine_handle<EmptyPromise> cr;

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::coroutine_handle<EmptyPromise>::from_promise(promise);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.address();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::coroutine_handle<EmptyPromise>::from_address(&promise);
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.done();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.promise();
    }
    {
      std::coroutine_handle<std::noop_coroutine_promise> cr = std::noop_coroutine();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.done();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.promise();
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      cr.address();

      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::noop_coroutine();
    }
  }
#endif

  { // <initializer_list>
    std::initializer_list<int> il{94, 82, 49};

    il.size();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    il.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    il.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
