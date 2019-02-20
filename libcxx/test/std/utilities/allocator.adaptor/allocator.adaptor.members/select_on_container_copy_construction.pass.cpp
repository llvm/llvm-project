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

// scoped_allocator_adaptor select_on_container_copy_construction() const;

#include <scoped_allocator>
#include <cassert>

#include "allocators.h"

int main(int, char**)
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a1(A1<int>(3));
        assert(a1.outer_allocator().id() == 3);
        A a2 = std::allocator_traits<A>::select_on_container_copy_construction(a1);
        assert(a2.outer_allocator().id() == 3);
    }

    {
        typedef std::scoped_allocator_adaptor<A3<int>> A;
        A a1(A3<int>(3));
        assert(a1.outer_allocator().id() == 3);
        A a2 = std::allocator_traits<A>::select_on_container_copy_construction(a1);
        assert(a2.outer_allocator().id() == -1);
    }

    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a1(A1<int>(1), A2<int>(2), A3<int>(3));
        assert(a1.outer_allocator().id() == 1);
        assert(a1.inner_allocator().outer_allocator().id() == 2);
        assert(a1.inner_allocator().inner_allocator().outer_allocator().id() == 3);
        A a2 = std::allocator_traits<A>::select_on_container_copy_construction(a1);
        assert(a2.outer_allocator().id() == 1);
        assert(a2.inner_allocator().outer_allocator().id() == 2);
        assert(a2.inner_allocator().inner_allocator().outer_allocator().id() == -1);
    }


  return 0;
}
