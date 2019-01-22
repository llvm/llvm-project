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

// scoped_allocator_adaptor();

#include <scoped_allocator>
#include <cassert>

#include "allocators.h"

int main()
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert(a.inner_allocator() == a);
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>());
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == false);
        assert(A2<int>::move_called == false);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert((a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>, A3<int>>()));
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == false);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == false);
        assert(A3<int>::move_called == false);
    }

}
