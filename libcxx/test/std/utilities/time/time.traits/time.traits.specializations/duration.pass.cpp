//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// template <class Rep1, class Period1, class Rep2, class Period2>
// struct common_type<chrono::duration<Rep1, Period1>, chrono::duration<Rep2, Period2>>
// {
//     typedef chrono::duration<typename common_type<Rep1, Rep2>::type, see below }> type;
// };

#include <chrono>

template <class D1, class D2, class De>
void
test()
{
    typedef typename std::common_type<D1, D2>::type Dc;
    static_assert((std::is_same<Dc, De>::value), "");
}

int main()
{
    test<std::chrono::duration<int, std::ratio<1, 100> >,
         std::chrono::duration<long, std::ratio<1, 1000> >,
         std::chrono::duration<long, std::ratio<1, 1000> > >();
    test<std::chrono::duration<long, std::ratio<1, 100> >,
         std::chrono::duration<int, std::ratio<1, 1000> >,
         std::chrono::duration<long, std::ratio<1, 1000> > >();
    test<std::chrono::duration<char, std::ratio<1, 30> >,
         std::chrono::duration<short, std::ratio<1, 1000> >,
         std::chrono::duration<int, std::ratio<1, 3000> > >();
    test<std::chrono::duration<double, std::ratio<21, 1> >,
         std::chrono::duration<short, std::ratio<15, 1> >,
         std::chrono::duration<double, std::ratio<3, 1> > >();
}
