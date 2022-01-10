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

template <class D, class R>
void
test(R r)
{
    D d(r);
    assert(d.count() == r);
#if TEST_STD_VER >= 11
    constexpr D d2(R(2));
    static_assert(d2.count() == 2, "");
#endif
}

int main(int, char**)
{
    test<std::chrono::duration<int> >(5);
    test<std::chrono::duration<int, std::ratio<3, 2> > >(5);
    test<std::chrono::duration<Rep, std::ratio<3, 2> > >(Rep(3));
    test<std::chrono::duration<double, std::ratio<2, 3> > >(5.5);

  return 0;
}
