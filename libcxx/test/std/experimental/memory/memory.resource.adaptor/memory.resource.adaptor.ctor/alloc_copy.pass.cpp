//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/memory_resource>

// template <class Alloc> class resource_adaptor_imp;

// resource_adaptor_imp<Alloc>::resource_adaptor_imp(Alloc const &)

#include <experimental/memory_resource>
#include <cassert>

#include "test_memory_resource.hpp"

namespace ex = std::experimental::pmr;

int main()
{
    typedef CountingAllocator<char> AllocT;
    typedef ex::resource_adaptor<AllocT> R;
    {
        AllocController P;
        AllocT const a(P);
        R const r(a);
        assert(P.copy_constructed == 1);
        assert(P.move_constructed == 0);
        assert(r.get_allocator() == a);
    }
    {
        AllocController P;
        AllocT a(P);
        R const r(a);
        assert(P.copy_constructed == 1);
        assert(P.move_constructed == 0);
        assert(r.get_allocator() == a);
    }
    {
        AllocController P;
        AllocT const a(P);
        R const r(std::move(a));
        assert(P.copy_constructed == 1);
        assert(P.move_constructed == 0);
        assert(r.get_allocator() == a);
    }
}
