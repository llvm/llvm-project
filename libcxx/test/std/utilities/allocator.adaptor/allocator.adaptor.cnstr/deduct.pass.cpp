//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <scoped_allocator>

// template<class _OuterAlloc, class... _InnerAllocs>
//     scoped_allocator_adaptor(_OuterAlloc, _InnerAllocs...)
//         -> scoped_allocator_adaptor<_OuterAlloc, _InnerAllocs...>;

#include <scoped_allocator>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    // Deduct from (const OuterAlloc&).
    {
        typedef A1<int> OuterAlloc;
        OuterAlloc outer(3);
        std::scoped_allocator_adaptor a(outer);
        ASSERT_SAME_TYPE(decltype(a), std::scoped_allocator_adaptor<OuterAlloc>);
    }

    // Deduct from (OuterAlloc&&).
    {
        typedef A1<int> OuterAlloc;
        std::scoped_allocator_adaptor a(OuterAlloc(3));
        ASSERT_SAME_TYPE(decltype(a), std::scoped_allocator_adaptor<OuterAlloc>);
    }

    // Deduct from (const OuterAlloc&, const InnerAlloc&).
    {
        typedef A1<int> OuterAlloc;
        typedef A2<int> InnerAlloc;
        OuterAlloc outer(3);
        InnerAlloc inner(4);

        std::scoped_allocator_adaptor a(outer, inner);
        ASSERT_SAME_TYPE(decltype(a), std::scoped_allocator_adaptor<OuterAlloc, InnerAlloc>);
    }

    // Deduct from (const OuterAlloc&, const InnerAlloc1&, InnerAlloc2&&).
    {
        typedef A1<int> OuterAlloc;
        typedef A2<int> InnerAlloc1;
        typedef A2<float> InnerAlloc2;
        OuterAlloc outer(3);
        InnerAlloc1 inner(4);

        std::scoped_allocator_adaptor a(outer, inner, InnerAlloc2(5));
        ASSERT_SAME_TYPE(
            decltype(a), std::scoped_allocator_adaptor<OuterAlloc, InnerAlloc1, InnerAlloc2>);
    }

  return 0;
}
