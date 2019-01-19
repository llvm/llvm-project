//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>
// REQUIRES: c++98 || c++03 || c++11 || c++14

// template<RandomAccessIterator Iter, Callable<auto, Iter::difference_type> Rand>
//   requires ShuffleIterator<Iter>
//         && Convertible<Rand::result_type, Iter::difference_type>
//   void
//   random_shuffle(Iter first, Iter last, Rand&& rand);

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

struct gen
{
    std::ptrdiff_t operator()(std::ptrdiff_t n)
    {
        return n-1;
    }
};

int main()
{
    int ia[] = {1, 2, 3, 4};
    int ia1[] = {4, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    gen r;
    std::random_shuffle(ia, ia+sa, r);
    LIBCPP_ASSERT(std::equal(ia, ia+sa, ia1));
    assert(std::is_permutation(ia, ia+sa, ia1));
}
