//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// T* allocate(size_t n, const void* hint);

// Deprecated in C++17

// REQUIRES: c++17

#include <memory>

#include "test_macros.h"

void f() {
    std::allocator<int> a;
    TEST_IGNORE_NODISCARD a.allocate(3, nullptr); // expected-warning {{'allocate' is deprecated}}
}
