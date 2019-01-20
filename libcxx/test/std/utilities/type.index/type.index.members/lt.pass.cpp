//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <typeindex>

// class type_index

// bool operator< (const type_index& rhs) const;
// bool operator<=(const type_index& rhs) const;
// bool operator> (const type_index& rhs) const;
// bool operator>=(const type_index& rhs) const;

#include <typeindex>
#include <cassert>

int main()
{
    std::type_index t1 = typeid(int);
    std::type_index t2 = typeid(int);
    std::type_index t3 = typeid(long);
    assert(!(t1 <  t2));
    assert( (t1 <= t2));
    assert(!(t1 >  t2));
    assert( (t1 >= t2));
    if (t1 < t3)
    {
        assert( (t1 <  t3));
        assert( (t1 <= t3));
        assert(!(t1 >  t3));
        assert(!(t1 >= t3));
    }
    else
    {
        assert(!(t1 <  t3));
        assert(!(t1 <= t3));
        assert( (t1 >  t3));
        assert( (t1 >= t3));
    }
}
