//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string_view>

//   constexpr bool contains(const CharT *x) const;

#include <string_view>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  using SV = std::string_view;

  const char* s = "abcde";
  SV sv0;
  SV sv1{s + 4, 1};
  SV sv3{s + 2, 3};
  SV svNot{"xyz", 3};

  assert(sv0.contains(""));
  assert(!sv0.contains("e"));

  assert(sv1.contains(""));
  assert(!sv1.contains("d"));
  assert(sv1.contains("e"));
  assert(!sv1.contains("de"));
  assert(!sv1.contains("cd"));
  assert(!sv1.contains("cde"));
  assert(!sv1.contains("bcde"));
  assert(!sv1.contains("abcde"));
  assert(!sv1.contains("xyz"));

  assert(sv3.contains(""));
  assert(sv3.contains("d"));
  assert(sv3.contains("e"));
  assert(sv3.contains("de"));
  assert(sv3.contains("cd"));
  assert(!sv3.contains("ce"));
  assert(sv3.contains("cde"));
  assert(!sv3.contains("edc"));
  assert(!sv3.contains("bcde"));
  assert(!sv3.contains("abcde"));
  assert(!sv3.contains("xyz"));

  assert(svNot.contains(""));
  assert(!svNot.contains("d"));
  assert(!svNot.contains("e"));
  assert(!svNot.contains("de"));
  assert(!svNot.contains("cd"));
  assert(!svNot.contains("cde"));
  assert(!svNot.contains("bcde"));
  assert(!svNot.contains("abcde"));
  assert(svNot.contains("xyz"));
  assert(!svNot.contains("zyx"));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
