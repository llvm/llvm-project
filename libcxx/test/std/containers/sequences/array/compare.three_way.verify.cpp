//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <array>

// template<class T, size_t N>
//   constexpr synth-three-way-result<T>
//     operator<=>(const array<T, N>& x, const array<T, N>& y);

// arrays with different sizes should not compare

#include <array>

int main(int, char**) {
  {
    std::array a1{1};
    std::array a2{1, 2};

    // expected-error@*:* {{invalid operands to binary expression}}
    a1 <=> a2;
  }
  {
    std::array a1{1, 2};
    std::array a2{1};

    // expected-error@*:* {{invalid operands to binary expression}}
    a1 <=> a2;
  }

  return 0;
}
