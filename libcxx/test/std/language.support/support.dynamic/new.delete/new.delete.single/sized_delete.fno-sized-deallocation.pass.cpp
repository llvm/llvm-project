//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that libc++ still provides the declaration of sized operator delete even
// when sized deallocation support is disabled at the language level, since it should
// still be valid to call these operators explicitly (as opposed to via a compiler
// rewrite of a delete expression).

// UNSUPPORTED: c++03, c++11

// ADDITIONAL_COMPILE_FLAGS: -fno-sized-deallocation

// Sized deallocation support was introduced in LLVM 11
// XFAIL: using-built-library-before-llvm-11

#include <new>

int main(int, char**) {
    void* p = ::operator new(10);
    ::operator delete(p, 10);
    return 0;
}
