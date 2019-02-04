//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class geometric_distribution

// bool operator=(const geometric_distribution& x,
//                const geometric_distribution& y);
// bool operator!(const geometric_distribution& x,
//                const geometric_distribution& y);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::geometric_distribution<> D;
        D d1(.25);
        D d2(.25);
        assert(d1 == d2);
    }
    {
        typedef std::geometric_distribution<> D;
        D d1(.28);
        D d2(.25);
        assert(d1 != d2);
    }
}
