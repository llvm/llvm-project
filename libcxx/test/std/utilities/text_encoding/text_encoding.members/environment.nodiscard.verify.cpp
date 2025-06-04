//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization

// class text_encoding

// text_encoding text_encoding::environment();
// text_encoding text_encoding::environment_is<>();

// Concerns:
// 1. Verify that std::text_encoding::environmnet(), std::text_encoding::environment_is<>() are marked nodiscard

#include "test_text_encoding.h"

int main() {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::text_encoding::environment();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::text_encoding::environment_is<std::text_encoding::id::unknown>();
}
