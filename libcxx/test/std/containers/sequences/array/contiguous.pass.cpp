//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// An array is a contiguous container

#include <array>
#include <cassert>

template <class C>
void test_contiguous ( const C &c )
{
    for ( size_t i = 0; i < c.size(); ++i )
        assert ( *(c.begin() + i) == *(std::addressof(*c.begin()) + i));
}

int main(int, char**)
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        test_contiguous (C());
    }

  return 0;
}
