//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class RandomAccessIterator>
//     void
//     random_shuffle(RandomAccessIterator first, RandomAccessIterator last);
//
// template <class RandomAccessIterator, class RandomNumberGenerator>
//     void
//     random_shuffle(RandomAccessIterator first, RandomAccessIterator last,
//                    RandomNumberGenerator& rand);

// UNSUPPORTED: clang-4.0
// UNSUPPORTED: c++03, c++11

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE

#include <algorithm>
#include <cstddef>

#include "test_macros.h"

struct gen
{
    std::ptrdiff_t operator()(std::ptrdiff_t n)
    {
        return n-1;
    }
};


int main(int, char**)
{
    int v[1] = {1};
    std::random_shuffle(&v[0], &v[1]); // expected-warning {{'random_shuffle<int *>' is deprecated}}
    gen r;
    std::random_shuffle(&v[0], &v[1], r); // expected-warning {{'random_shuffle<int *, gen &>' is deprecated}}

  return 0;
}
