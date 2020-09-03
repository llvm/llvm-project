//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <experimental/memory_resource>

// template <class Alloc> class resource_adaptor_imp;

// resource_adaptor_imp<Alloc>::resource_adaptor_imp() = default;

#include <experimental/memory_resource>
#include <memory>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    {
        typedef CountingAllocator<char> AllocT; // Not default constructible
        typedef ex::resource_adaptor<AllocT> R;
        static_assert(!std::is_default_constructible<R>::value, "");
    }
    {
        typedef std::allocator<char> AllocT; // Is default constructible
        typedef ex::resource_adaptor<AllocT> R;
        static_assert(std::is_default_constructible<R>::value, "");
        R r; ((void)r);
    }

  return 0;
}
