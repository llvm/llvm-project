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

// <memory>

// template <class ForwardIt>
// constexpr void destroy(ForwardIt, ForwardIt);

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

struct Counted {
    int* counter_;
    TEST_CONSTEXPR Counted(int* counter) : counter_(counter) { ++*counter_; }
    TEST_CONSTEXPR Counted(Counted const& other) : counter_(other.counter_) { ++*counter_; }
    TEST_CONSTEXPR_CXX20 ~Counted() { --*counter_; }
    friend void operator&(Counted) = delete;
};

#if TEST_STD_VER > 17
constexpr bool test_arrays() {
    {
        using Array = Counted[3];
        using Alloc = std::allocator<Array>;
        int counter = 0;
        Alloc alloc;
        Array* pool = std::allocator_traits<Alloc>::allocate(alloc, 5);

        for (Array* p = pool; p != pool + 5; ++p) {
            Array& arr = *p;
            for (int i = 0; i != 3; ++i) {
                std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i]), &counter);
            }
        }
        assert(counter == 5 * 3);

        std::destroy(pool, pool + 5);
        ASSERT_SAME_TYPE(decltype(std::destroy(pool, pool + 5)), void);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, pool, 5);
    }
    {
        using Array = Counted[3][2];
        using Alloc = std::allocator<Array>;
        int counter = 0;
        Alloc alloc;
        Array* pool = std::allocator_traits<Alloc>::allocate(alloc, 5);

        for (Array* p = pool; p != pool + 5; ++p) {
            Array& arr = *p;
            for (int i = 0; i != 3; ++i) {
                for (int j = 0; j != 2; ++j) {
                    std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i][j]), &counter);
                }
            }
        }
        assert(counter == 5 * 3 * 2);

        std::destroy(pool, pool + 5);
        ASSERT_SAME_TYPE(decltype(std::destroy(pool, pool + 5)), void);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, pool, 5);
    }

    return true;
}
#endif

template <class It>
TEST_CONSTEXPR_CXX20 void test() {
    using Alloc = std::allocator<Counted>;
    int counter = 0;
    Alloc alloc;
    Counted* pool = std::allocator_traits<Alloc>::allocate(alloc, 5);

    for (Counted* p = pool; p != pool + 5; ++p)
        std::allocator_traits<Alloc>::construct(alloc, p, &counter);
    assert(counter == 5);

    std::destroy(It(pool), It(pool + 5));
    ASSERT_SAME_TYPE(decltype(std::destroy(It(pool), It(pool + 5))), void);
    assert(counter == 0);

    std::allocator_traits<Alloc>::deallocate(alloc, pool, 5);
}

TEST_CONSTEXPR_CXX20 bool tests() {
    test<Counted*>();
    test<forward_iterator<Counted*>>();
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 17
    test_arrays();
    static_assert(tests());
    // TODO: Until std::construct_at has support for arrays, it's impossible to test this
    //       in a constexpr context.
    // static_assert(test_arrays());
#endif
    return 0;
}
