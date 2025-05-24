//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization
// UNSUPPORTED: android
// UNSUPPORTED: availability-te-environment-missing

// <text_encoding>

// text_encoding text_encoding::environment()

// Split from the general [[nodiscard]] tests to accomodate no-localization builds.

#include <text_encoding>

int main(int, char**) {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::text_encoding::environment();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::text_encoding::environment_is<std::text_encoding::UTF8>();

  return 0;
}
