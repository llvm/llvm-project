//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// GCC doesn't have thread safety attributes
// UNSUPPORTED: gcc

// ADDITIONAL_COMPILE_FLAGS: -Wthread-safety -Wno-comment

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <mutex>

#include "test_macros.h"

std::mutex m0;
std::mutex m1;
std::mutex m2;
std::mutex m3;

void f1() {
  std::lock(m0, m1);
} // expected-warning {{mutex 'm0' is still held at the end of function}} \
     expected-warning {{mutex 'm1' is still held at the end of function}}

#if TEST_STD_VER >= 11 && TEST_CLANG_VER >= 2101
void f2() {
  std::lock(m0, m1, m2);
} // expected-warning {{mutex 'm0' is still held at the end of function}} \
     expected-warning {{mutex 'm1' is still held at the end of function}} \
     expected-warning {{mutex 'm2' is still held at the end of function}}

void f3() {
  std::lock(m0, m1, m2, m3);
} // expected-warning {{mutex 'm0' is still held at the end of function}} \
     expected-warning {{mutex 'm1' is still held at the end of function}} \
     expected-warning {{mutex 'm2' is still held at the end of function}} \
     expected-warning {{mutex 'm3' is still held at the end of function}}
#endif
