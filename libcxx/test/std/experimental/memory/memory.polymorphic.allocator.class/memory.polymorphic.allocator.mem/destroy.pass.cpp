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

// template <class U>
// void polymorphic_allocator<T>::destroy(U * ptr);

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <type_traits>
#include <new>
#include <cassert>
#include <cstdlib>

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int count = 0;

struct destroyable
{
    destroyable() { ++count; }
    ~destroyable() { --count; }
};

int main(int, char**)
{
    typedef ex::polymorphic_allocator<double> A;
    {
        A a;
        static_assert(
            std::is_same<decltype(a.destroy((destroyable*)nullptr)), void>::value,
            "");
    }
    {
        destroyable * ptr = ::new (std::malloc(sizeof(destroyable))) destroyable();
        assert(count == 1);
        A{}.destroy(ptr);
        assert(count == 0);
        std::free(ptr);
    }

  return 0;
}
