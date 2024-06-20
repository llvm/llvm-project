//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// pointer allocate(size_type n, const_void_pointer hint);

#include <scoped_allocator>

#include "allocators.h"

void f() {
  std::scoped_allocator_adaptor<A1<int>> a;
  a.allocate(10, (const void*)0);
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
}
