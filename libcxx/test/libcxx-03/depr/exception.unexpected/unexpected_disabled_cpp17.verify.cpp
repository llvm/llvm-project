//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// test unexpected

#include <exception>

void f() {}

int main(int, char**) {
  using T = std::unexpected_handler; // expected-error {{no type named 'unexpected_handler' in namespace 'std'}}
  std::unexpected(); // expected-error {{no member named 'unexpected' in namespace 'std'}}
  std::get_unexpected(); // expected-error {{no member named 'get_unexpected' in namespace 'std'}}
  std::set_unexpected(f); // expected-error {{no type named 'set_unexpected' in namespace 'std'}}

  return 0;
}
