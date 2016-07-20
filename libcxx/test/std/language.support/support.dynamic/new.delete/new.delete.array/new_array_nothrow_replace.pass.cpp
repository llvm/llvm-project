//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test operator new [] nothrow by replacing only operator new

// UNSUPPORTED: sanitizer-new-delete

// TODO Investigate why UBSAN prevents new from calling our replacement.
// XFAIL: ubsan


#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

int new_called = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_called;
    void* ret = std::malloc(s);
    if (!ret) std::abort(); // placate MSVC's unchecked malloc warning
    return  ret;
}

void  operator delete(void* p) throw()
{
    --new_called;
    std::free(p);
}

volatile int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    A* ap = new (std::nothrow) A[3];
    assert(ap);
    assert(A_constructed == 3);
    assert(new_called);
    delete [] ap;
    assert(A_constructed == 0);
    assert(!new_called);
}
