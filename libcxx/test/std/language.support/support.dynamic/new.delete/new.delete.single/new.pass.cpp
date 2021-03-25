//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test operator new

// asan and msan will not call the new handler.
// UNSUPPORTED: sanitizer-new-delete
// XFAIL: LIBCXX-WINDOWS-FIXME

#include <new>
#include <cstddef>
#include <cassert>
#include <limits>

#include "test_macros.h"

int new_handler_called = 0;

void my_new_handler()
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

int main(int, char**)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    std::set_new_handler(my_new_handler);
    try
    {
        void* vp = operator new (std::numeric_limits<std::size_t>::max());
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
#endif
    A* ap = new A;
    assert(ap);
    assert(A_constructed);
    delete ap;
    assert(!A_constructed);

  return 0;
}
