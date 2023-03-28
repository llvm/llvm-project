//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(); // Deprecated in C++20.
// void reserve(size_type);

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// This test ensures that libc++ implements https://wg21.link/P0966R1 (reserve never shrinks)
// even before C++20. This is required in order to avoid ODR violations because basic_string::reserve(size)
// is compiled into the shared library. Hence, it needs to have the same definition in all Standard modes.
//
// However, note that reserve() does shrink, and it does so in all Standard modes.
//
// Reported as https://llvm.org/PR53170.

// reserve(n) used to shrink the string until https://llvm.org/D117332 was shipped.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 bool test() {
    // Test that a call to reserve() does shrink the string.
    if (!TEST_IS_CONSTANT_EVALUATED) {
        S s(1000, 'a');
        typename S::size_type old_cap = s.capacity();
        s.resize(20);
        assert(s.capacity() == old_cap);

        s.reserve();
        assert(s.capacity() < old_cap);
    }

    // Test that a call to reserve(smaller-than-capacity) never shrinks the string.
    {
        S s(1000, 'a');
        typename S::size_type old_cap = s.capacity();
        s.resize(20);
        assert(s.capacity() == old_cap);

        s.reserve(10);
        assert(s.capacity() == old_cap);
    }

    // In particular, test that reserve(0) does NOT shrink the string.
    {
        S s(1000, 'a');
        typename S::size_type old_cap = s.capacity();
        s.resize(20);
        assert(s.capacity() == old_cap);

        s.reserve(0);
        assert(s.capacity() == old_cap);
    }

    return true;
}

int main(int, char**) {
    test<std::string>();

#if TEST_STD_VER >= 11
    test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
#endif

#if TEST_STD_VER > 17
    static_assert(test<std::string>());
#endif

    return 0;
}
