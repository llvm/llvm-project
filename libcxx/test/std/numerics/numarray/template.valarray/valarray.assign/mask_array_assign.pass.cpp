//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray& operator=(const mask_array<value_type>& ma);

#include <valarray>
#include <cassert>

int main(int, char**)
{
    int a1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::size_t N1 = sizeof(a1)/sizeof(a1[0]);
    bool b[N1] = {true,  false, false, true,  true,  false,
                  false, true,  false, false, false, true};
    std::valarray<int> v1(a1, N1);
    std::valarray<bool> vb(b, N1);
    std::valarray<int> v2(5);
    v2 = v1[vb];
    assert(v2.size() == 5);
    assert(v2[ 0] ==  0);
    assert(v2[ 1] ==  3);
    assert(v2[ 2] ==  4);
    assert(v2[ 3] ==  7);
    assert(v2[ 4] == 11);

  return 0;
}
