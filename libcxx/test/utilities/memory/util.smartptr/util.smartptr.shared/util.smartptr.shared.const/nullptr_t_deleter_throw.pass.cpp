//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class D> shared_ptr(nullptr_t, D d);

#include <memory>
#include <cassert>
#include <new>
#include <cstdlib>
#include "../test_deleter.h"

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
    throw_next = true;
    try
    {
        std::shared_ptr<A> p(nullptr, test_deleter<A>(3));
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(A::count == 0);
        assert(test_deleter<A>::count == 0);
        assert(test_deleter<A>::dealloc_count == 1);
    }
}
