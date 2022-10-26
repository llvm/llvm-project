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

// FIXME: Remove 'clang-16' from UNSUPPORTED by 2022-11-05 (bugfix D136533).
// UNSUPPORTED: c++03, c++11, c++14, clang-16

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
    std::allocator<int> a;
    TEST_IGNORE_NODISCARD a.allocate(3, nullptr);
    // expected-warning@-1 {{'allocate' is deprecated}}
#if defined(TEST_CLANG_VER) && TEST_CLANG_VER >= 1600
    // expected-warning@*:* {{'pointer' is deprecated}}
    // expected-warning@*:* {{'const_pointer' is deprecated}}
#endif
    return 0;
}
