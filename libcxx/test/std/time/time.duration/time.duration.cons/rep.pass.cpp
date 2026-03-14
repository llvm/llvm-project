//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include <chrono>
#include <cassert>
#include <ratio>

#include "test_macros.h"
#include "../../rep.h"

#if TEST_STD_VER >= 11
struct NotValueConvertible {
    operator int() const&& = delete;
    constexpr operator int() const& { return 1; }
};
#endif

template <class D, class R>
TEST_CONSTEXPR_CXX14 void check(R r) {
    D d(r);
    assert(d.count() == r);
}

TEST_CONSTEXPR_CXX14 bool test() {
    check<std::chrono::duration<int> >(5);
    check<std::chrono::duration<int, std::ratio<3, 2> > >(5);
    check<std::chrono::duration<Rep, std::ratio<3, 2> > >(Rep(3));
    check<std::chrono::duration<double, std::ratio<2, 3> > >(5.5);

    // test for [time.duration.cons]/1
#if TEST_STD_VER >= 11
    check<std::chrono::duration<int> >(NotValueConvertible());
#endif

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 11
    static_assert(test(), "");
#endif

    // Basic test for constexpr-friendliness in C++11
#if TEST_STD_VER >= 11
    {
        constexpr std::chrono::duration<int> d(5);
        static_assert(d.count() == 5, "");
    }
#endif
    return 0;
}
