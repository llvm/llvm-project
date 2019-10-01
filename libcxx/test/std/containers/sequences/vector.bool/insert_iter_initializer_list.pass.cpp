//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <vector>

// iterator insert(const_iterator p, initializer_list<value_type> il);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    std::vector<bool> d(10, true);
    std::vector<bool>::iterator i = d.insert(d.cbegin() + 2, {false, true, true, false});
    assert(d.size() == 14);
    assert(i == d.begin() + 2);
    assert(d[0] == true);
    assert(d[1] == true);
    assert(d[2] == false);
    assert(d[3] == true);
    assert(d[4] == true);
    assert(d[5] == false);
    assert(d[6] == true);
    assert(d[7] == true);
    assert(d[8] == true);
    assert(d[9] == true);
    assert(d[10] == true);
    assert(d[11] == true);
    assert(d[12] == true);
    assert(d[13] == true);
    }
    {
    std::vector<bool, min_allocator<bool>> d(10, true);
    std::vector<bool, min_allocator<bool>>::iterator i = d.insert(d.cbegin() + 2, {false, true, true, false});
    assert(d.size() == 14);
    assert(i == d.begin() + 2);
    assert(d[0] == true);
    assert(d[1] == true);
    assert(d[2] == false);
    assert(d[3] == true);
    assert(d[4] == true);
    assert(d[5] == false);
    assert(d[6] == true);
    assert(d[7] == true);
    assert(d[8] == true);
    assert(d[9] == true);
    assert(d[10] == true);
    assert(d[11] == true);
    assert(d[12] == true);
    assert(d[13] == true);
    }

  return 0;
}
