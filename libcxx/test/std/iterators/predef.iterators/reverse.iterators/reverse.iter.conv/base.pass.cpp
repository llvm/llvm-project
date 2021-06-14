//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// iterator_type base() const; // constexpr since C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

TEST_CONSTEXPR_CXX17 bool test() {
    typedef bidirectional_iterator<int*> Iter;
    int i = 0;
    Iter iter(&i);
    std::reverse_iterator<Iter> const reverse(iter);
    std::reverse_iterator<Iter>::iterator_type base = reverse.base();
    assert(base == Iter(&i));
    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 14
    static_assert(test(), "");
#endif
    return 0;
}
