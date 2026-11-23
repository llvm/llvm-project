//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// constexpr byte operator&(byte l, byte r) noexcept;

#include <cassert>
#include <cstddef>
#include <type_traits>

static_assert(noexcept(std::byte{} & std::byte{}));
static_assert(std::is_same_v<decltype(std::byte{} & std::byte{}), std::byte>);

constexpr bool test() {
  std::byte b1{1};
  std::byte b8{8};
  std::byte b9{9};

  assert(std::to_integer<int>(b1 & b8) == 0);
  assert(std::to_integer<int>(b1 & b9) == 1);
  assert(std::to_integer<int>(b8 & b9) == 8);

  assert(std::to_integer<int>(b8 & b1) == 0);
  assert(std::to_integer<int>(b9 & b1) == 1);
  assert(std::to_integer<int>(b9 & b8) == 8);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
