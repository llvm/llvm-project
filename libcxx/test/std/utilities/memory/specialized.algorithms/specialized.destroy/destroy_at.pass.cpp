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

// template <class T>
// constexpr void destroy_at(T*);

#include <memory>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

struct Counted {
    int* counter_;
    TEST_CONSTEXPR Counted(int* counter) : counter_(counter) { ++*counter_; }
    TEST_CONSTEXPR_CXX20 ~Counted() { --*counter_; }
    friend void operator&(Counted) = delete;
};

struct VirtualCounted {
    int* counter_;
    TEST_CONSTEXPR VirtualCounted(int* counter) : counter_(counter) { ++*counter_; }
    TEST_CONSTEXPR_CXX20 virtual ~VirtualCounted() { --*counter_; }
    friend void operator&(VirtualCounted) = delete;
};

struct DerivedCounted : VirtualCounted {
    TEST_CONSTEXPR DerivedCounted(int* counter) : VirtualCounted(counter) { }
    TEST_CONSTEXPR_CXX20 ~DerivedCounted() override { }
    friend void operator&(DerivedCounted) = delete;
};

#if TEST_STD_VER > 17
constexpr bool test_arrays() {
    {
        using Array = Counted[3];
        using Alloc = std::allocator<Array>;
        Alloc alloc;
        Array* ptr = std::allocator_traits<Alloc>::allocate(alloc, 1);
        Array& arr = *ptr;

        int counter = 0;
        for (int i = 0; i != 3; ++i)
            std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i]), &counter);
        assert(counter == 3);

        std::destroy_at(ptr);
        ASSERT_SAME_TYPE(decltype(std::destroy_at(ptr)), void);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, ptr, 1);
    }
    {
        using Array = Counted[3][2];
        using Alloc = std::allocator<Array>;
        Alloc alloc;
        Array* ptr = std::allocator_traits<Alloc>::allocate(alloc, 1);
        Array& arr = *ptr;

        int counter = 0;
        for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 2; ++j)
                std::allocator_traits<Alloc>::construct(alloc, std::addressof(arr[i][j]), &counter);
        assert(counter == 3 * 2);

        std::destroy_at(ptr);
        ASSERT_SAME_TYPE(decltype(std::destroy_at(ptr)), void);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, ptr, 1);
    }
    return true;
}
#endif

TEST_CONSTEXPR_CXX20 bool test() {
    {
        using Alloc = std::allocator<Counted>;
        Alloc alloc;
        Counted* ptr1 = std::allocator_traits<Alloc>::allocate(alloc, 1);
        Counted* ptr2 = std::allocator_traits<Alloc>::allocate(alloc, 1);

        int counter = 0;
        std::allocator_traits<Alloc>::construct(alloc, ptr1, &counter);
        std::allocator_traits<Alloc>::construct(alloc, ptr2, &counter);
        assert(counter == 2);

        std::destroy_at(ptr1);
        ASSERT_SAME_TYPE(decltype(std::destroy_at(ptr1)), void);
        assert(counter == 1);

        std::destroy_at(ptr2);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, ptr1, 1);
        std::allocator_traits<Alloc>::deallocate(alloc, ptr2, 1);
    }
    {
        using Alloc = std::allocator<DerivedCounted>;
        Alloc alloc;
        DerivedCounted* ptr1 = std::allocator_traits<Alloc>::allocate(alloc, 1);
        DerivedCounted* ptr2 = std::allocator_traits<Alloc>::allocate(alloc, 1);

        int counter = 0;
        std::allocator_traits<Alloc>::construct(alloc, ptr1, &counter);
        std::allocator_traits<Alloc>::construct(alloc, ptr2, &counter);
        assert(counter == 2);

        std::destroy_at(ptr1);
        ASSERT_SAME_TYPE(decltype(std::destroy_at(ptr1)), void);
        assert(counter == 1);

        std::destroy_at(ptr2);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, ptr1, 1);
        std::allocator_traits<Alloc>::deallocate(alloc, ptr2, 1);
    }

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 17
    test_arrays();
    static_assert(test());
    // TODO: Until std::construct_at has support for arrays, it's impossible to test this
    //       in a constexpr context.
    // static_assert(test_arrays());
#endif
    return 0;
}
