//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test operator new[] replacement by replacing only operator new

// UNSUPPORTED: sanitizer-new-delete


#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "test_macros.h"

volatile int new_called = 0;

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

A* volatile ap;

int main()
{
    ap = new A[3];
    assert(ap);
    assert(A_constructed == 3);
    assert(new_called == 1);
    delete [] ap;
    assert(A_constructed == 0);
    assert(new_called == 0);
}
