//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test placement new array

#include <new>
#include <cassert>

int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    const std::size_t Size = 3;
    // placement new might require additional space.
    const std::size_t ExtraSize = 64;
    char buf[Size*sizeof(A) + ExtraSize];

    A* ap = new(buf) A[Size];
    assert((char*)ap >= buf);
    assert((char*)ap < (buf + ExtraSize));
    assert(A_constructed == Size);
}
