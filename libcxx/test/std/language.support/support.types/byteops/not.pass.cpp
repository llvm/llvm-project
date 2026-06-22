//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// constexpr byte operator~(byte b) noexcept;

#include <cassert>
#include <cstddef>
#include <type_traits>

static_assert(noexcept(~std::byte{}));
static_assert(std::is_same_v<decltype(~std::byte{}), std::byte>);

constexpr bool test() {
  std::byte b1{1};
  std::byte b2{2};
  std::byte b8{8};

  assert(std::to_integer<int>(~b1) == 254);
  assert(std::to_integer<int>(~b2) == 253);
  assert(std::to_integer<int>(~b8) == 247);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
