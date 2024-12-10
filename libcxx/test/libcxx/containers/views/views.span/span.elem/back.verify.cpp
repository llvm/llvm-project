//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>
//
// constexpr reference back() const noexcept;

// Make sure that accessing a statically-sized span out-of-bounds triggers a
// compile-time error.

#include <array>
#include <span>

int main(int, char**) {
  std::array<int, 3> array{0, 1, 2};
  {
    std::span<int, 0> const s(array.data(), 0);
    s.back(); // expected-error@span:* {{span<T, N>::back() on empty span}}
  }
  {
    std::span<int, 3> const s(array.data(), 3);
    s.back(); // nothing
  }

  return 0;
}
