//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// constexpr byte& operator^=(byte& l, byte r) noexcept;

#include <cassert>
#include <cstddef>
#include <type_traits>

constexpr std::byte test_op(std::byte b1, std::byte b2) {
  static_assert(noexcept(b1 ^= b2));
  static_assert(std::is_same_v<decltype(b1 ^= b2), std::byte&>);

  std::byte& ret = b1 ^= b2;
  assert(&ret == &b1);
  return ret;
}

constexpr bool test() {
  std::byte b1{1};
  std::byte b8{8};
  std::byte b9{9};

  assert(std::to_integer<int>(test_op(b1, b8)) == 9);
  assert(std::to_integer<int>(test_op(b1, b9)) == 8);
  assert(std::to_integer<int>(test_op(b8, b9)) == 1);

  assert(std::to_integer<int>(test_op(b8, b1)) == 9);
  assert(std::to_integer<int>(test_op(b9, b1)) == 8);
  assert(std::to_integer<int>(test_op(b9, b8)) == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
