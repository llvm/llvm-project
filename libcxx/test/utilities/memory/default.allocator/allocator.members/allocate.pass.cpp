//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// pointer allocate(size_type n, allocator<void>::const_pointer hint=0);

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

int new_called = 0;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    ++new_called;
    assert(s == 3 * sizeof(int));
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    --new_called;
    std::free(p);
}

int A_constructed = 0;

struct A
{
    int data;
    A() {++A_constructed;}
    A(const A&) {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    std::allocator<A> a;
    assert(new_called == 0);
    assert(A_constructed == 0);
    A* ap = a.allocate(3);
    assert(new_called == 1);
    assert(A_constructed == 0);
    a.deallocate(ap, 3);
    assert(new_called == 0);
    assert(A_constructed == 0);

    A* ap2 = a.allocate(3, (const void*)5);
    assert(new_called == 1);
    assert(A_constructed == 0);
    a.deallocate(ap2, 3);
    assert(new_called == 0);
    assert(A_constructed == 0);
}
