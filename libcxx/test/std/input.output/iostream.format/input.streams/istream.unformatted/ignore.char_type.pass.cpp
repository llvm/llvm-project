//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <istream>

// basic_istream& ignore(streamsize n, char_type delim);

#include <cassert>
#include <sstream>
#include <string>

#include "test_macros.h"

int main(int, char**) {
  std::istringstream in("\xF0\x9F\xA4\xA1 Clown Face");
  in.ignore(100, '\xA1'); // Ignore up to '\xA1' delimiter,
                          // previously might have ignored to EOF.

  assert(in.gcount() == 4); // 4 bytes were ignored.
  assert(in.peek() == ' '); // Next character is a space.

  std::string str; // Read the next word.
  in >> str;
  assert(str == "Clown");

  // Parameter value "-1L" doesn't cause ambiguity with the char_type overload.
  in.ignore(100, -1L); // Ignore up to EOF, which is the default behavior.
  assert(in.eof());    // Stream should be at EOF now.
  assert(in.gcount() == 5);

  return 0;
}
