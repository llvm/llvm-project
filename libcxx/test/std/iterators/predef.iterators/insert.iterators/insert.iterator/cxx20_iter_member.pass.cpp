//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// insert_iterator
// C++20 and above use ranges::iterator_t<Container> instead of Container::iterator.

#include <iterator>

#include <cassert>
#include <type_traits>

#include "test_macros.h"

struct NoIteratorAlias {
    double data_[3] = {};
    using value_type = int;
    double *begin();
    constexpr double *insert(double *pos, int value) {
        *pos = value;
        return pos;
    }
};

static_assert(std::is_constructible_v<std::insert_iterator<NoIteratorAlias>, NoIteratorAlias&, double*>);
static_assert(!std::is_constructible_v<std::insert_iterator<NoIteratorAlias>, NoIteratorAlias&, int*>);

constexpr bool test() {
    NoIteratorAlias c;
    double half = 0.5;
    auto it = std::insert_iterator<NoIteratorAlias>(c, c.data_);
    ASSERT_SAME_TYPE(decltype(std::inserter(c, c.data_)), std::insert_iterator<NoIteratorAlias>);
    *it++ = 1 + half;  // test that RHS is still implicitly converted to _Container::value_type
    *it++ = 2 + half;
    assert(c.data_[0] == 1.0);
    assert(c.data_[1] == 2.0);
    assert(c.data_[2] == 0.0);
    return true;
}

int main(int, char**) {
    test();
    static_assert(test());

    return 0;
}
