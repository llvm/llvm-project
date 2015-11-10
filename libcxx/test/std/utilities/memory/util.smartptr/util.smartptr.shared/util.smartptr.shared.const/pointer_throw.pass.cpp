//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <memory>

// template<class Y> explicit shared_ptr(Y* p);

// UNSUPPORTED: sanitizer-new-delete

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

bool throw_next = false;

void* operator new(std::size_t s) throw(std::bad_alloc)
{
    if (throw_next)
        throw std::bad_alloc();
    return std::malloc(s);
}

void  operator delete(void* p) throw()
{
    std::free(p);
}

int main()
{
    {
    A* ptr = new A;
    throw_next = true;
    assert(A::count == 1);
    try
    {
        std::shared_ptr<A> p(ptr);
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(A::count == 0);
    }
    }
}
