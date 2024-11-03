//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class _Compare> struct __debug_less

// Make sure __debug_less asserts when the comparator is not consistent.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <algorithm>
#include <iterator>

#include "check_assertion.h"

template <int ID>
struct MyType {
    int value;
    explicit MyType(int xvalue = 0) : value(xvalue) {}
};

template <int ID1, int ID2>
bool operator<(MyType<ID1> const& LHS, MyType<ID2> const& RHS) {
    return LHS.value < RHS.value;
}

template <class ValueType>
struct BadComparator {
    bool operator()(ValueType const&, ValueType const&) const {
        return true;
    }
};

int main(int, char**) {
    typedef MyType<0> MT0;
    MT0 one(1);
    MT0 two(2);

    BadComparator<MT0> c;
    std::__debug_less<BadComparator<MT0>> d(c);

    TEST_LIBCPP_ASSERT_FAILURE(d(one, two), "Comparator does not induce a strict weak ordering");

    return 0;
}
