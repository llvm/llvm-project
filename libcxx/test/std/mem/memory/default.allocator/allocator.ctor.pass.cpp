//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>
//
// template <class T>
// class allocator
// {
// public: // All of these are constexpr after C++17
//  allocator() noexcept;
//  allocator(const allocator&) noexcept;
//  template<class U> allocator(const allocator<U>&) noexcept;
// ...
// };

#include <memory>
#include <cstddef>

#include "test_macros.h"

template<class T>
TEST_CONSTEXPR_CXX20 bool test() {
  typedef std::allocator<T> A1;
  typedef std::allocator<long> A2;

  A1 a1;
  A1 a1_copy = a1; (void)a1_copy;
  A2 a2 = a1; (void)a2;

  return true;
}

int main(int, char**) {
  test<char>();
  test<int>();
  test<void>();

#if TEST_STD_VER > 17
  static_assert(test<char>());
  static_assert(test<int>());
  static_assert(test<void>());
#endif
  return 0;
}
