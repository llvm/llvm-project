//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Requires 396145d in the built library.
// XFAIL: using-built-library-before-llvm-9

// <istream>

// basic_istream& ignore(streamsize n, char_type delim);

#include <cassert>
#include <sstream>
#include <string>

#include "test_macros.h"

void test() {
  std::istringstream in("\xF0\x9F\xA4\xA1 Clown Face");
  in.ignore(100, -1L); // expected-error {{call to member function 'ignore' is ambiguous}}
}
