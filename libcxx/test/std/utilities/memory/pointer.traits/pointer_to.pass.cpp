//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// struct pointer_traits<T*>
// {
//     static pointer pointer_to(<details>); // constexpr in C++20
//     ...
// };

#include <memory>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test()
{
    {
        int i = 0;
        static_assert(std::is_same<decltype(std::pointer_traits<int*>::pointer_to(i)), int*>::value, "");
        assert(std::pointer_traits<int*>::pointer_to(i) == &i);
    }
    {
        int i = 0;
        static_assert(std::is_same<decltype(std::pointer_traits<const int*>::pointer_to(i)), const int*>::value, "");
        assert(std::pointer_traits<const int*>::pointer_to(i) == &i);
    }
    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif

    {
        // Check that pointer_traits<void*> is still well-formed, even though it has no pointer_to.
        static_assert(std::is_same<std::pointer_traits<void*>::element_type, void>::value, "");
        static_assert(std::is_same<std::pointer_traits<const void*>::element_type, const void>::value, "");
        static_assert(std::is_same<std::pointer_traits<volatile void*>::element_type, volatile void>::value, "");
    }

    return 0;
}
