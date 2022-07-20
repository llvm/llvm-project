//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// template <class U1, class U2>
// void polymorphic_allocator<T>::construct(pair<U1, U2>*)

#include <experimental/memory_resource>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cassert>
#include <cstdlib>
#include "uses_alloc_types.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int constructed = 0;

struct default_constructible
{
    default_constructible() : x(42)  { ++constructed; }
    int x{0};
};

int main(int, char**)
{
    // pair<default_constructible, default_constructible> as T()
    {
        typedef default_constructible T;
        typedef std::pair<T, T> P;
        typedef ex::polymorphic_allocator<void> A;
        P * ptr = (P*)std::malloc(sizeof(P));
        A a;
        a.construct(ptr);
        assert(constructed == 2);
        assert(ptr->first.x == 42);
        assert(ptr->second.x == 42);
        std::free(ptr);
    }

  return 0;
}
