//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Define a hexfloat literal emulator since we can't depend on being able to
//   for hexfloat literals

// 0x10.F5p-10 == hexfloat<double>(0x10, 0xF5, -10)

#ifndef HEXFLOAT_H
#define HEXFLOAT_H

#include <algorithm>
#include <cmath>
#include <climits>

template <class T>
class hexfloat
{
    T value_;
public:
    hexfloat(long long m1, unsigned long long m0, int exp)
    {
        const std::size_t n = sizeof(unsigned long long) * CHAR_BIT;
        int s = m1 < 0 ? -1 : 1;
        value_ = std::ldexp(m1 + s * std::ldexp(T(m0), -static_cast<int>(n -
                                                     std::__clz(m0)/4*4)), exp);
    }

    operator T() const {return value_;}
};

#endif
