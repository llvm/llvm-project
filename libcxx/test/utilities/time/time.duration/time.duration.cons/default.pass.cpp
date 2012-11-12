//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// duration() = default;

// Rep must be default initialized, not initialized with 0

#include <chrono>
#include <cassert>

#include "../../rep.h"

template <class D>
void
test()
{
    D d;
    assert(d.count() == typename D::rep());
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    constexpr D d2;
    static_assert(d2.count() == typename D::rep(), "");
#endif
}

int main()
{
    test<std::chrono::duration<Rep> >();
}
