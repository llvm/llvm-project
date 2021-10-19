//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(); // Deprecated in C++20.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test()
{
    // Tests that a call to reserve() on a long string is equivalent to shrink_to_fit().
    S s(1000, 'a');
    typename S::size_type old_cap = s.capacity();
    s.resize(20);
    assert(s.capacity() == old_cap);
    s.reserve();
    assert(s.capacity() < old_cap);
}

int main(int, char**)
{
    {
    typedef std::string S;
    test<S>();
    }
#if TEST_STD_VER >= 11
    {
    typedef min_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;
    test<S>();
    }
#endif

    return 0;
}
