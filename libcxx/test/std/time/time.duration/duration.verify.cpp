//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// LWG 4481
// A program that instantiates duration with a cv-qualified Rep is ill-formed.

#include <chrono>

void test() {
  // expected-error-re@*:* {{static assertion failed {{.*}}A duration representation cannot be qualified}}
  (void)sizeof(std::chrono::duration<const int>);

  // expected-error-re@*:* {{static assertion failed {{.*}}A duration representation cannot be qualified}}
  (void)sizeof(std::chrono::duration<volatile int>);

  // expected-error-re@*:* {{static assertion failed {{.*}}A duration representation cannot be qualified}}
  (void)sizeof(std::chrono::duration<const volatile int>);
}
