//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23, c++26

// clang-format off

#include <embed>

#depend __FILE__
#depend "a"
#depend "empty"

#include "test_macros.h"

consteval bool test() {
#if __cpp_lib_embed
  constexpr auto v0 = std::embed("a.txt");
  static_assert(v0.size() == 1);
  static_assert(v0[0] == (std::byte)u8'a');

  constexpr auto v1 = std::embed(__FILE__);
  static_assert(v1.size() == 1831);
  static_assert(v1[0] == (std::byte)'/');
  static_assert(v1[1] == (std::byte)'/');
  static_assert(v1[2] == (std::byte)'=');
  static_assert(v1[3] == (std::byte)'=');
  static_assert(v1[4] == (std::byte)'=');

  constexpr auto v2 = std::embed(__FILE__, 0x123456);
  static_assert(v2.data() == nullptr);
  static_assert(v2.size() == 0);

  constexpr auto v3 = std::embed(__FILE__, 0, 0);
  static_assert(v3.data() == nullptr);
  static_assert(v3.size() == 0);

  constexpr auto v4 = std::embed<0>(__FILE__);
  static_assert(v4.data() == nullptr);
  static_assert(v4.size() == 0);

  constexpr auto v5 = std::embed("empty");
  static_assert(v5.data() == nullptr);
  static_assert(v5.size() == 0);

  constexpr auto v6 = std::embed(__FILE__, 3, 3);
  static_assert(v6.size() == 3);
  static_assert(v6[0] == (std::byte)u8'=');
  static_assert(v6[1] == (std::byte)u8'=');
  static_assert(v6[2] == (std::byte)u8'-');
#endif
  return true;
}

int main(int, char*[]) {
  static_assert(test());
  return 0;
}
