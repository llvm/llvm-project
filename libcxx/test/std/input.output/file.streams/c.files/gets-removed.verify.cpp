//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Verify that std::gets has been removed in C++14 and later

#include <cstdio>

void f(char const* str) {
  (void)std::gets(str); // expected-error {{no member named 'gets' in namespace 'std'}}
}
