//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23, c++26

// check that <embed> functions are marked [[nodiscard]]

#include <embed>

#depend __FILE__

#include "test_macros.h"

#define STR_PREFIX_(a, b) a##b
#define STR_PREFIX(a, b) STR_PREFIX_(a, b)

consteval bool test() {
#if __cpp_lib_embed
  // make sure to test each individual overload
  std::embed(__FILE__);
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::embed<1>(__FILE__);
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::embed(STR_PREFIX(L, __FILE__));
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::embed<1>(STR_PREFIX(L, __FILE__));
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::embed(STR_PREFIX(u8, __FILE__));
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::embed<1>(STR_PREFIX(u8, __FILE__));
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  return true;
}

#undef STR_PREFIX_
#undef STR_PREFIX

int main(int, char*[]) {
  static_assert(test());
  return 0;
}
