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

// typedef see below propagate_on_container_swap;

#include <scoped_allocator>
#include <type_traits>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::propagate_on_container_swap,
        std::false_type>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>, A2<int>>::propagate_on_container_swap,
        std::false_type>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>>::propagate_on_container_swap,
        std::true_type>::value), "");

  return 0;
}
