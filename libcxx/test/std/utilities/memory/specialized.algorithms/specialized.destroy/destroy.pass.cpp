//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// constexpr destructors are only supported starting with clang 10
// UNSUPPORTED: clang-5, clang-6, clang-7, clang-8, clang-9
// constexpr destructors are only supported starting with gcc 10
// UNSUPPORTED: gcc-8, gcc-9

// <memory>

// template <class ForwardIt>
// constexpr void destroy(ForwardIt, ForwardIt);

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct Counted {
    int* counter_;
    TEST_CONSTEXPR Counted(int* counter) : counter_(counter) { ++*counter_; }
    TEST_CONSTEXPR Counted(Counted const& other) : counter_(other.counter_) { ++*counter_; }
    TEST_CONSTEXPR_CXX20 ~Counted() { --*counter_; }
    friend void operator&(Counted) = delete;
};

TEST_CONSTEXPR_CXX20 bool test()
{
    using Alloc = std::allocator<Counted>;
    int counter = 0;
    int const N = 5;
    Alloc alloc;
    Counted* pool = std::allocator_traits<Alloc>::allocate(alloc, N);

    for (Counted* p = pool; p != pool + N; ++p)
        std::allocator_traits<Alloc>::construct(alloc, p, &counter);
    assert(counter == 5);

    std::destroy(pool, pool + 1);
    assert(counter == 4);

    std::destroy(forward_iterator<Counted*>(pool + 1), forward_iterator<Counted*>(pool + 5));
    assert(counter == 0);

    std::allocator_traits<Alloc>::deallocate(alloc, pool, N);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    return 0;
}
