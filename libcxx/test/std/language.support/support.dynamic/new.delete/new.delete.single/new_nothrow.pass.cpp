//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test operator new (nothrow)

// asan and msan will not call the new handler.
// UNSUPPORTED: sanitizer-new-delete

#include <new>
#include <cstddef>
#include <cassert>
#include <limits>

int new_handler_called = 0;

void new_handler()
{
    ++new_handler_called;
    std::set_new_handler(0);
}

bool A_constructed = false;

struct A
{
    A() {A_constructed = true;}
    ~A() {A_constructed = false;}
};

int main()
{
    std::set_new_handler(new_handler);
    try
    {
        void* vp = operator new (std::numeric_limits<std::size_t>::max(), std::nothrow);
        assert(new_handler_called == 1);
        assert(vp == 0);
    }
    catch (...)
    {
        assert(false);
    }
    A* ap = new(std::nothrow) A;
    assert(ap);
    assert(A_constructed);
    delete ap;
    assert(!A_constructed);
}
