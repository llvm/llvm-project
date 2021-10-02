//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

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
