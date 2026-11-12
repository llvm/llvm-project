//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string_view>

//   constexpr bool contains(string_view x) const noexcept;

#include <string_view>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  using SV = std::string_view;

  const char* s = "abcde";
  SV sv0;
  SV sv1{s + 1, 1};
  SV sv2{s + 1, 2};
  SV sv3{s + 1, 3};
  SV sv4{s + 1, 4};
  SV sv5{s, 5};
  SV svNot{"xyz", 3};
  SV svNot2{"bd", 2};
  SV svNot3{"dcb", 3};

  ASSERT_NOEXCEPT(sv0.contains(sv0));

  assert(sv0.contains(sv0));
  assert(!sv0.contains(sv1));

  assert(sv1.contains(sv0));
  assert(sv1.contains(sv1));
  assert(!sv1.contains(sv2));
  assert(!sv1.contains(sv3));
  assert(!sv1.contains(sv4));
  assert(!sv1.contains(sv5));
  assert(!sv1.contains(svNot));
  assert(!sv1.contains(svNot2));
  assert(!sv1.contains(svNot3));

  assert(sv3.contains(sv0));
  assert(sv3.contains(sv1));
  assert(sv3.contains(sv2));
  assert(sv3.contains(sv3));
  assert(!sv3.contains(sv4));
  assert(!sv3.contains(sv5));
  assert(!sv3.contains(svNot));
  assert(!sv3.contains(svNot2));
  assert(!sv3.contains(svNot3));

  assert(sv5.contains(sv0));
  assert(sv5.contains(sv1));
  assert(sv5.contains(sv2));
  assert(sv5.contains(sv3));
  assert(sv5.contains(sv4));
  assert(sv5.contains(sv5));
  assert(!sv5.contains(svNot));
  assert(!sv5.contains(svNot2));
  assert(!sv5.contains(svNot3));

  assert(svNot.contains(sv0));
  assert(!svNot.contains(sv1));
  assert(!svNot.contains(sv2));
  assert(!svNot.contains(sv3));
  assert(!svNot.contains(sv4));
  assert(!svNot.contains(sv5));
  assert(svNot.contains(svNot));
  assert(!svNot.contains(svNot2));
  assert(!svNot.contains(svNot3));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
