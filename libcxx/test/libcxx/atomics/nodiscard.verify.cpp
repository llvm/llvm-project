//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// Check that functions are marked [[nodiscard]]

#include <atomic>

#include "test_macros.h"

void test() {
#if TEST_STD_VER >= 20
  {
    int i = 49;
    const std::atomic_ref<int> atRef{i};

#  if TEST_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    atRef.address();
#  endif

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    atRef.is_lock_free();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    atRef.load();
  }
#endif

  {
    const volatile std::atomic<int> vat(82);
    const std::atomic<int> at(94);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    vat.load();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    at.load();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_is_lock_free(&vat);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_is_lock_free(&at);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_load(&vat);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_load(&at);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_load_explicit(&vat, std::memory_order_seq_cst);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::atomic_load_explicit(&at, std::memory_order_seq_cst);
  }
}
