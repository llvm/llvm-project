//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <memory>

void f() {
  int *p = nullptr;
  std::assume_aligned<4>(p); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
