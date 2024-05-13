//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Note: libc++ supports string_view before C++17, but literals were introduced in C++14
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: !stdlib=libc++ && c++14

#include <string_view>

void f() {
  {
    using std::string_view;
    string_view foo = ""sv; // expected-error {{no matching literal operator}}
  }
  {
    std::string_view foo = ""sv; // expected-error {{no matching literal operator}}
  }
}
