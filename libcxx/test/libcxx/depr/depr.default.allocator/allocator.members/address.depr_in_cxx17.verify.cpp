//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// pointer address(reference x) const;
// const_pointer address(const_reference x) const;

// Deprecated in C++17

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
  int x = 0;
  std::allocator<int> a;

  int* p = a.address(x); // expected-warning {{'address' is deprecated}}
  (void)p;

  return 0;
}
