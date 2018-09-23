//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE

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


int main()
{
#if TEST_STD_VER < 14
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'random_shuffle<int *>' is deprecated}}
    // expected-error@* 1 {{'random_shuffle<int *, gen &>' is deprecated}}
#endif
    int v[1] = {1};
    std::random_shuffle(&v[0], &v[1]);
    gen r;
    std::random_shuffle(&v[0], &v[1], r);
}
