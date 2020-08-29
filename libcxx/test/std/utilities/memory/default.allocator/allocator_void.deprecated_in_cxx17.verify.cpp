//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that allocator<void> is deprecated in C++17.

// REQUIRES: c++17

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
    typedef std::allocator<void>::pointer AP;             // expected-warning {{'allocator<void>' is deprecated}}
    typedef std::allocator<void>::const_pointer ACP;      // expected-warning {{'allocator<void>' is deprecated}}
    typedef std::allocator<void>::rebind<int>::other ARO; // expected-warning {{'allocator<void>' is deprecated}}
    return 0;
}
