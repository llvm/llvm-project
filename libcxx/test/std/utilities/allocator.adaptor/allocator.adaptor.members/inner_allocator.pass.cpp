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

// inner_allocator_type& inner_allocator();
// const inner_allocator_type& inner_allocator() const;

#include <scoped_allocator>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a(A1<int>(5));
        assert(a.inner_allocator() == a);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a(A1<int>(5), A2<int>(6));
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>(A2<int>(6)));
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a(A1<int>(5), A2<int>(6), A3<int>(8));
        assert((a.inner_allocator() ==
            std::scoped_allocator_adaptor<A2<int>, A3<int>>(A2<int>(6), A3<int>(8))));
    }


  return 0;
}
