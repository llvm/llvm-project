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

TEST_CONSTEXPR_CXX20 bool test()
{
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
        assert(counter == 1);

        std::destroy_at(ptr2);
        assert(counter == 0);

        std::allocator_traits<Alloc>::deallocate(alloc, ptr1, 1);
        std::allocator_traits<Alloc>::deallocate(alloc, ptr2, 1);
    }

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
