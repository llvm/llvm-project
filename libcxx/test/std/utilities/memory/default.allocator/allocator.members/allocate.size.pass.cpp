//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <memory>

// allocator:
// constexpr T* allocate(size_t n);

#include <memory>
#include <cassert>

#include "test_macros.h"

template <typename T>
void test_max(size_t count)
{
    std::allocator<T> a;
    try {
        TEST_IGNORE_NODISCARD a.allocate(count);
        assert(false);
    } catch (const std::exception &) {
    }
}

template <typename T>
void test()
{
    // Bug 26812 -- allocating too large
    typedef std::allocator<T> A;
    typedef std::allocator_traits<A> AT;
    A a;
    test_max<T> (AT::max_size(a) + 1);             // just barely too large
    test_max<T> (AT::max_size(a) * 2);             // significantly too large
    test_max<T> (((size_t) -1) / sizeof(T) + 1);   // multiply will overflow
    test_max<T> ((size_t) -1);                     // way too large
}

int main(int, char**)
{
    test<double>();
    LIBCPP_ONLY(test<const double>());

  return 0;
}
