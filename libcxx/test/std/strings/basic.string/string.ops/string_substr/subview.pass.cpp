//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <string>

// constexpr basic_string_view<_CharT, _Traits> subview(size_type __pos = 0, size_type __n = npos) const;

#include <cassert>
#include <string>

constexpr bool test() {
  std::string s{"Hello cruel world!"};
  auto sub = s.subview(6);
  assert(sub == "cruel world!");
  auto subsub = sub.subview(0, 5);
  assert(subsub == "cruel");

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
