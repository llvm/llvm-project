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

int main(int, char**) {
  std::istringstream in("\xF0\x9F\xA4\xA1 Clown Face");
  in.ignore(100, '\xA1'); // ignore up to '\xA1' delimiter,
                          // previously might have ignored to EOF

  assert(in.gcount() == 4); // 4 bytes were ignored
  assert(in.peek() == ' '); // next character is a space

  std::string str; // read the next word
  in >> str;
  assert(str == "Clown");

  return 0;
}
