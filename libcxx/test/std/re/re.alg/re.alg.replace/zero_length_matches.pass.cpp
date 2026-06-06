//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// Test that replacing zero-length matches works correctly.

#include <cassert>
#include <regex>
#include <string>
#include "test_macros.h"

int main(int, char**) {
  // Various patterns that produce zero-length matches.
  assert(std::regex_replace("abc", std::regex(""), "!") == "!a!b!c!");
  assert(std::regex_replace("abc", std::regex("X*"), "!") == "!a!b!c!");
  assert(std::regex_replace("abc", std::regex("X{0,3}"), "!") == "!a!b!c!");

  // Replacement string has several characters.
  assert(std::regex_replace("abc", std::regex(""), "[!]") == "[!]a[!]b[!]c[!]");

  // Empty replacement string.
  assert(std::regex_replace("abc", std::regex(""), "") == "abc");

  // Empty input.
  assert(std::regex_replace("", std::regex(""), "!") == "!");

  // Not all matches are zero-length.
  assert(std::regex_replace("abCabCa", std::regex("C*"), "!") == "!a!b!!a!b!!a!");

  return 0;
}
