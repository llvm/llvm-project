//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <experimental/memory_resource>

// template <class Alloc> class resource_adaptor_imp;

// resource_adaptor_imp<Alloc>::resource_adaptor_imp(Alloc &&)

#include <experimental/memory_resource>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    typedef CountingAllocator<char> AllocT;
    typedef ex::resource_adaptor<AllocT> R;
    {
        AllocController P;
        AllocT a(P);
        R const r(std::move(a));
        assert(P.copy_constructed == 0);
        assert(P.move_constructed == 1);
        assert(r.get_allocator() == a);
    }
    {
        AllocController P;
        R const r(AllocT{P});
        assert(P.copy_constructed == 0);
        assert(P.move_constructed == 1);
        assert(r.get_allocator() == AllocT{P});
    }

  return 0;
}
