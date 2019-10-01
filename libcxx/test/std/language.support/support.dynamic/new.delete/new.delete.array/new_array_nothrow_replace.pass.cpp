//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test operator new [] nothrow by replacing only operator new

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: libcpp-no-vcruntime

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "test_macros.h"

int new_called = 0;

void* operator new(std::size_t s) TEST_THROW_SPEC(std::bad_alloc)
{
    ++new_called;
    void* ret = std::malloc(s);
    if (!ret) std::abort(); // placate MSVC's unchecked malloc warning
    return  ret;
}

void  operator delete(void* p) TEST_NOEXCEPT
{
    --new_called;
    std::free(p);
}

int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main(int, char**)
{
    A *ap = new (std::nothrow) A[3];
    DoNotOptimize(ap);
    assert(ap);
    assert(A_constructed == 3);
    assert(new_called);
    delete [] ap;
    DoNotOptimize(ap);
    assert(A_constructed == 0);
    assert(!new_called);

  return 0;
}
