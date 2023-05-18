//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void assign(size_type n, const_reference v);

#include <vector>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

TEST_CONSTEXPR bool is6(int x) { return x == 6; }

template <typename Vec>
TEST_CONSTEXPR_CXX20 void test(Vec &v)
{
    v.assign(5, 6);
    assert(v.size() == 5);
    assert(is_contiguous_container_asan_correct(v));
    assert(std::all_of(v.begin(), v.end(), is6));
}

TEST_CONSTEXPR_CXX20 bool tests() {
    {
    typedef std::vector<int> V;
    V d1;
    V d2;
    d2.reserve(10);  // no reallocation during assign.
    test(d1);
    test(d2);
    }
    {
    std::vector<int> vec;
    vec.reserve(32);
    vec.resize(16); // destruction during assign
    test(vec);
    }
#if TEST_STD_VER >= 11
    {
    typedef std::vector<int, min_allocator<int>> V;
    V d1;
    V d2;
    d2.reserve(10);  // no reallocation during assign.
    test(d1);
    test(d2);
    }
#endif

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER > 17
    static_assert(tests());
#endif
    return 0;
}
