//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// std::ranges::advance

// Regression test for https://github.com/llvm/llvm-project/issues/175091
// LWG4510: https://cplusplus.github.io/LWG/issue4510

#include <ranges>
#include <cassert>

struct T {
  constexpr T(int) {}
  constexpr T(const auto&) {}
  friend constexpr bool operator==(T, T) { return true; }
};

int main() {
  {
    int buffer[10];
    int* it = buffer;
  
    std::ranges::advance(it, 3);
  
    assert(it == buffer + 3);
  }

  // LWG4510 Reproducer: Test with type T that has a conversion constructor 
  // and an operator==, which previously caused ambiguity in ranges::advance.
  {
    T t{0};
    T* it = &t;
    std::ranges::advance(it, 0); 
    assert(it == &t);
  }
}
