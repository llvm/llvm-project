//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(initializer_list<value_type> il);

#include <vector>
#include <cassert>
#include "min_allocator.h"
#include "asan_testing.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
    std::vector<int> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(is_contiguous_container_asan_correct(d));
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    }
#if __cplusplus >= 201103L
    {
    std::vector<int, min_allocator<int>> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(is_contiguous_container_asan_correct(d));
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    }
#endif
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
