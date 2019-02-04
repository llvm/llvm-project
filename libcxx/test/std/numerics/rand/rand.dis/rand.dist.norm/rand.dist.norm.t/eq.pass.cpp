//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class student_t_distribution

// bool operator=(const student_t_distribution& x,
//                const student_t_distribution& y);
// bool operator!(const student_t_distribution& x,
//                const student_t_distribution& y);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::student_t_distribution<> D;
        D d1(2.5);
        D d2(2.5);
        assert(d1 == d2);
    }
    {
        typedef std::student_t_distribution<> D;
        D d1(4);
        D d2(4.5);
        assert(d1 != d2);
    }
}
