//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const Types&...);

// UNSUPPORTED: c++03

#include <tuple>
#include <memory>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"
#include "../alloc_first.h"
#include "../alloc_last.h"

struct ImplicitCopy {
  explicit ImplicitCopy(int) {}
  ImplicitCopy(ImplicitCopy const&) {}
};

// Test that tuple(std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
std::tuple<ImplicitCopy> testImplicitCopy1() {
    ImplicitCopy i(42);
    return {std::allocator_arg, std::allocator<int>{}, i};
}

std::tuple<ImplicitCopy> testImplicitCopy2() {
    const ImplicitCopy i(42);
    return {std::allocator_arg, std::allocator<int>{}, i};
}

int main(int, char**)
{
    {
        // check that the literal '0' can implicitly initialize a stored pointer.
        std::tuple<int*>{std::allocator_arg, std::allocator<int>{}, 0};
    }
    {
        std::tuple<int> t(std::allocator_arg, A1<int>(), 3);
        assert(std::get<0>(t) == 3);
    }
    {
        assert(!alloc_first::allocator_constructed);
        std::tuple<alloc_first> t(std::allocator_arg, A1<int>(5), alloc_first(3));
        assert(alloc_first::allocator_constructed);
        assert(std::get<0>(t) == alloc_first(3));
    }
    {
        assert(!alloc_last::allocator_constructed);
        std::tuple<alloc_last> t(std::allocator_arg, A1<int>(5), alloc_last(3));
        assert(alloc_last::allocator_constructed);
        assert(std::get<0>(t) == alloc_last(3));
    }
    {
        alloc_first::allocator_constructed = false;
        std::tuple<int, alloc_first> t(std::allocator_arg, A1<int>(5),
                                       10, alloc_first(15));
        assert(std::get<0>(t) == 10);
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first(15));
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<int, alloc_first, alloc_last> t(std::allocator_arg,
                                                   A1<int>(5), 1, alloc_first(2),
                                                   alloc_last(3));
        assert(std::get<0>(t) == 1);
        assert(alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first(2));
        assert(alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last(3));
    }
    {
        alloc_first::allocator_constructed = false;
        alloc_last::allocator_constructed = false;
        std::tuple<int, alloc_first, alloc_last> t(std::allocator_arg,
                                                   A2<int>(5), 1, alloc_first(2),
                                                   alloc_last(3));
        assert(std::get<0>(t) == 1);
        assert(!alloc_first::allocator_constructed);
        assert(std::get<1>(t) == alloc_first(2));
        assert(!alloc_last::allocator_constructed);
        assert(std::get<2>(t) == alloc_last(3));
    }
    {
        // Test that we can use a tag derived from allocator_arg_t
        struct DerivedFromAllocatorArgT : std::allocator_arg_t { };
        DerivedFromAllocatorArgT derived;
        std::tuple<> t1(derived, A1<int>());
        std::tuple<int> t2(derived, A1<int>(), 1);
        std::tuple<int, int> t3(derived, A1<int>(), 1, 2);
    }

    return 0;
}
