//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
// struct atomic;

// This test checks that we static_assert inside std::atomic<T> when T
// is cv-qualified, however Clang will sometimes emit additional
// errors while trying to instantiate the rest of std::atomic<T>.
// We silence those to make the test more robust.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

#include <atomic>

void f() {
  std::atomic<const int> a; // expected-error@*:* {{std::atomic<T> requires that 'T' be a cv-unqualified type}}
}

