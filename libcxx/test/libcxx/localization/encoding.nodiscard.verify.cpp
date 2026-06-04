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

// <locale>
// In a separate file because the implementation for locale::encoding() is guarded by availability

#include <locale>

void test() {
  std::locale l;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  l.encoding();
}
