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

// pointer allocate(size_type n, const_void_pointer hint);

#include <scoped_allocator>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a;
        A1<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)0) == (int*)10);
        assert(A1<int>::allocate_called == true);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a;
        A1<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)10) == (int*)10);
        assert(A1<int>::allocate_called == true);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a;
        A1<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)20) == (int*)10);
        assert(A1<int>::allocate_called == true);
    }

    {
        typedef std::scoped_allocator_adaptor<A2<int>> A;
        A a;
        A2<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)0) == (int*)0);
        assert(A2<int>::allocate_called == true);
    }
    {
        typedef std::scoped_allocator_adaptor<A2<int>, A2<int>> A;
        A a;
        A2<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)10) == (int*)10);
        assert(A2<int>::allocate_called == true);
    }
    {
        typedef std::scoped_allocator_adaptor<A2<int>, A2<int>, A3<int>> A;
        A a;
        A2<int>::allocate_called = false;
        assert(a.allocate(10, (const void*)20) == (int*)20);
        assert(A2<int>::allocate_called == true);
    }

  return 0;
}
