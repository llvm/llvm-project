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

// T* polymorphic_allocator<T>::allocate(size_t n)

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <limits>
#include <memory>
#include <exception>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_memory_resource.h"

namespace ex = std::experimental::pmr;

template <size_t S, size_t Align>
void testForSizeAndAlign() {
    struct T { alignas(Align) char data[S]; };
    TestResource R;
    ex::polymorphic_allocator<T> a(&R);

    for (int N = 1; N <= 5; ++N) {
        auto ret = a.allocate(N);
        assert(R.checkAlloc(ret, N * sizeof(T), alignof(T)));

        a.deallocate(ret, N);
        R.reset();
    }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
template <size_t S>
void testAllocForSizeThrows() {
    struct T { char data[S]; };
    using Alloc = ex::polymorphic_allocator<T>;
    using Traits = std::allocator_traits<Alloc>;
    NullResource R;
    Alloc a(&R);

    // Test that allocating exactly the max size does not throw.
    size_t maxSize = Traits::max_size(a);
    try {
        a.allocate(maxSize);
    } catch (...) {
        assert(false);
    }

    size_t sizeTypeMax = std::numeric_limits<std::size_t>::max();
    if (maxSize != sizeTypeMax)
    {
        // Test that allocating size_t(~0) throws bad_array_new_length.
        try {
            a.allocate(sizeTypeMax);
            assert(false);
        } catch (std::bad_array_new_length const&) {
        }

        // Test that allocating even one more than the max size does throw.
        size_t overSize = maxSize + 1;
        try {
            a.allocate(overSize);
            assert(false);
        } catch (std::bad_array_new_length const&) {
        }
    }
}
#endif // TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
    {
        ex::polymorphic_allocator<int> a;
        static_assert(std::is_same<decltype(a.allocate(0)), int*>::value, "");
        static_assert(!noexcept(a.allocate(0)), "");
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
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testAllocForSizeThrows<1>();
        testAllocForSizeThrows<2>();
        testAllocForSizeThrows<4>();
        testAllocForSizeThrows<8>();
        testAllocForSizeThrows<16>();
        testAllocForSizeThrows<73>();
        testAllocForSizeThrows<13>();
    }
#endif

  return 0;
}
