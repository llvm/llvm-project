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

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// T* polymorphic_allocator<T>::deallocate(T*, size_t size)

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

namespace ex = std::experimental::pmr;

template <size_t S, size_t Align>
void testForSizeAndAlign() {
    using T = typename std::aligned_storage<S, Align>::type;

    TestResource R;
    ex::polymorphic_allocator<T> a(&R);

    for (int N = 1; N <= 5; ++N) {
        auto ret = a.allocate(N);
        assert(R.checkAlloc(ret, N * sizeof(T), alignof(T)));

        a.deallocate(ret, N);
        assert(R.checkDealloc(ret, N * sizeof(T), alignof(T)));

        R.reset();
    }
}

int main(int, char**)
{
    {
        ex::polymorphic_allocator<int> a;
        static_assert(
            std::is_same<decltype(a.deallocate(nullptr, 0)), void>::value, "");
    }
    {
        constexpr std::size_t MA = alignof(std::max_align_t);
        testForSizeAndAlign<1, 1>();
        testForSizeAndAlign<1, 2>();
        testForSizeAndAlign<1, MA>();
        testForSizeAndAlign<2, 2>();
        testForSizeAndAlign<73, alignof(void*)>();
        testForSizeAndAlign<73, MA>();
        testForSizeAndAlign<13, MA>();
    }

  return 0;
}
