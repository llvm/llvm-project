//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct B {} b;

struct A1 {
    operator B& () const { return b; }
};
struct A2 {
    operator B& () const noexcept { return b; }
};

int main(int, char**)
{
    {
    std::reference_wrapper<B> b1 = A1();
    assert(&b1.get() == &b);
    b1 = A1();
    assert(&b1.get() == &b);

    static_assert(std::is_convertible<A1, std::reference_wrapper<B>>::value, "");
    static_assert(!std::is_nothrow_constructible<std::reference_wrapper<B>, A1>::value, "");
#if TEST_STD_VER >= 20
    static_assert(!std::is_nothrow_convertible_v<A1, std::reference_wrapper<B>>);
#endif
    static_assert(std::is_assignable<std::reference_wrapper<B>, A1>::value, "");
    static_assert(!std::is_nothrow_assignable<std::reference_wrapper<B>, A1>::value, "");
    }

    {
    std::reference_wrapper<B> b2 = A2();
    assert(&b2.get() == &b);
    b2 = A2();
    assert(&b2.get() == &b);

    static_assert(std::is_convertible<A2, std::reference_wrapper<B>>::value, "");
    static_assert(std::is_nothrow_constructible<std::reference_wrapper<B>, A2>::value, "");
#if TEST_STD_VER >= 20
    static_assert(std::is_nothrow_convertible_v<A2, std::reference_wrapper<B>>);
#endif
    static_assert(std::is_assignable<std::reference_wrapper<B>, A2>::value, "");
    static_assert(std::is_nothrow_assignable<std::reference_wrapper<B>, A2>::value, "");
    }

    return 0;
}
