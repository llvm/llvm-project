//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test operator new[]
// NOTE: asan and msan will not call the new handler.
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
        void* volatile vp = operator new[] (std::numeric_limits<std::size_t>::max());
        ((void)vp);
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(new_handler_called == 1);
    }
    catch (...)
    {
        assert(false);
    }
    A* ap = new A[3];
    assert(ap);
    assert(A_constructed == 3);
    delete [] ap;
    assert(A_constructed == 0);
}
