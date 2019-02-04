//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <exception>

// typedef unspecified exception_ptr;

// exception_ptr shall satisfy the requirements of NullablePointer.

#include <exception>
#include <cassert>

int main()
{
    std::exception_ptr p;
    assert(p == nullptr);
    std::exception_ptr p2 = p;
    assert(nullptr == p);
    assert(!p);
    assert(p2 == p);
    p2 = p;
    assert(p2 == p);
    assert(p2 == nullptr);
    std::exception_ptr p3 = nullptr;
    assert(p3 == nullptr);
    p3 = nullptr;
    assert(p3 == nullptr);
}
