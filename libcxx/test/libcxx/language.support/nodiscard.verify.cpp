//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION

// Check that functions are marked [[nodiscard]]

#include <compare>
#include <coroutine>
#include <exception>
#include <initializer_list>
#include <new>
#include <typeinfo>
#include <typeindex>

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

  { // <exception>
    {
      std::bad_exception bex;

      bex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    }
    {
      std::exception ex;

      ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    }
    {
      std::nested_exception nex;

      nex.nested_ptr(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    }

    { // Removed in C++17
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::get_unexpected();
    }

    std::get_terminate(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    { // Removed in C++20
      // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
      std::uncaught_exception();
    }

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::uncaught_exceptions();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::current_exception();
  }

#if TEST_STD_VER >= 11
  { // <initializer_list>
    std::initializer_list<int> il{94, 82, 49};

    il.size();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    il.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    il.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

#if !defined(TEST_HAS_NO_RTTI)
  { // <typeindex>
    const std::type_index ti(typeid(int));

    ti.hash_code(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ti.name();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::hash<std::type_index> hash;

    hash(ti); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

#if !defined(TEST_HAS_NO_RTTI)
  { // <typeinfo>
    const std::type_info& ti = typeid(int);

    ti.name();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ti.before(ti);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ti.hash_code(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    const std::bad_cast bc;

    bc.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    const std::bad_typeid bt;

    bc.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

  {
    std::bad_alloc ex;

    ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::bad_array_new_length ex;

    ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::nothrow);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::nothrow);
#if _LIBCPP_HAS_ALIGNED_ALLOCATION
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::align_val_t{1});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new(0, std::align_val_t{1}, std::nothrow);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::align_val_t{1});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    ::operator new[](0, std::align_val_t{1}, std::nothrow);
#endif // _LIBCPP_HAS_ALIGNED_ALLOCATION
  }

#if TEST_STD_VER >= 17
  {
    int* ptr = nullptr;

    std::launder(ptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}
