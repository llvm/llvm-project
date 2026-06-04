//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization
// UNSUPPORTED: availability-te-environment-missing

// <text_encoding>
// In a separate file because environment() and environment_is() are guarded by availability

#include <text_encoding>

void test() {
  std::text_encoding te;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.environment();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  te.environment_is<std::text_encoding::UTF8>();
}
