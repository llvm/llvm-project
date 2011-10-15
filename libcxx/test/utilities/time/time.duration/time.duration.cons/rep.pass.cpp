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

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include <chrono>
#include <cassert>

#include "../../rep.h"

template <class D, class R>
void
test(R r)
{
    D d(r);
    assert(d.count() == r);
}

int main()
{
    test<std::chrono::duration<int> >(5);
    test<std::chrono::duration<int, std::ratio<3, 2> > >(5);
    test<std::chrono::duration<Rep, std::ratio<3, 2> > >(Rep(3));
    test<std::chrono::duration<double, std::ratio<2, 3> > >(5.5);
}
