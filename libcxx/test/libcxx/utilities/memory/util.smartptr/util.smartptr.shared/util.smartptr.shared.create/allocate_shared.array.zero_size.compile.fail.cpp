//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Make sure that std::allocate_shared<T[0]>(...) fails at compile-time.
// While Clang and GCC appear to support T[0] as a language extension, that support is
// unreliable (for example T[0] doesn't match a T[N] partial specialization on Clang as
// of writing this). So instead, we make sure that this doesn't work at all with our
// implementation.

#include <memory>

void f() {
  auto p = std::allocate_shared<int[0]>(std::allocator<int[0]>());
  (void)p;
}
