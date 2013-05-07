//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

 // test operator new [] (nothrow)

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

int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    std::set_new_handler(new_handler);
    try
    {
        void*volatile vp = operator new [] (std::numeric_limits<std::size_t>::max(), std::nothrow);
        assert(new_handler_called == 1);
        assert(vp == 0);
    }
    catch (...)
    {
        assert(false);
    }
    A* ap = new(std::nothrow) A[3];
    assert(ap);
    assert(A_constructed == 3);
    delete [] ap;
    assert(A_constructed == 0);
}
