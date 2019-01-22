//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// scoped_allocator_adaptor(const scoped_allocator_adaptor& other);

#include <scoped_allocator>
#include <cassert>

#include "allocators.h"

int main()
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a1(A1<int>(3));
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        A a2 = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(a2 == a1);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a1(A1<int>(4), A2<int>(5));
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        A2<int>::copy_called = false;
        A2<int>::move_called = false;
        A a2 = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(a2 == a1);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a1(A1<int>(4), A2<int>(5), A3<int>(6));
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        A2<int>::copy_called = false;
        A2<int>::move_called = false;
        A3<int>::copy_called = false;
        A3<int>::move_called = false;
        A a2 = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == true);
        assert(A3<int>::move_called == false);
        assert(a2 == a1);
    }
}
