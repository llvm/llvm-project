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

// XFAIL: availability-aligned_allocation-missing

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// polymorphic_allocator<T>::polymorphic_allocator() noexcept

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    {
        static_assert(
            std::is_nothrow_default_constructible<ex::polymorphic_allocator<void>>::value
          , "Must me nothrow default constructible"
          );
    }
    {
        // test that the allocator gets its resource from get_default_resource
        TestResource R1(42);
        ex::set_default_resource(&R1);

        typedef ex::polymorphic_allocator<void> A;
        A const a;
        assert(a.resource() == &R1);

        ex::set_default_resource(nullptr);
        A const a2;
        assert(a.resource() == &R1);
        assert(a2.resource() == ex::new_delete_resource());
    }

  return 0;
}
