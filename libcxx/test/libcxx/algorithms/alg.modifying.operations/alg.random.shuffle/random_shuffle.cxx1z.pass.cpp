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

//
//  In C++17, random_shuffle has been removed.
//  However, for backwards compatibility, if _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE
//  is defined before including <algorithm>, then random_shuffle will be restored.

// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE
#define _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE

#include <algorithm>
#include <vector>

struct gen
{
    std::ptrdiff_t operator()(std::ptrdiff_t n)
    {
        return n-1;
    }
};


int main(int, char**)
{
    std::vector<int> v;
    std::random_shuffle(v.begin(), v.end());
    gen r;
    std::random_shuffle(v.begin(), v.end(), r);

  return 0;
}
