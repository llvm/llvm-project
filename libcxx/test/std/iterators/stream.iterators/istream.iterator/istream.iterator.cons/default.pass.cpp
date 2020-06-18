//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// constexpr istream_iterator();
// C++17 says: If is_trivially_default_constructible_v<T> is true, then this
//    constructor is a constexpr constructor.

#include <iterator>
#include <cassert>
#include <string>

#include "test_macros.h"

struct S { S(); }; // not constexpr

#if TEST_STD_VER > 14
template <typename T, bool isTrivial = std::is_trivially_default_constructible_v<T>>
struct test_trivial {
void operator ()() const {
    constexpr std::istream_iterator<T> it;
    (void)it;
    }
};

template <typename T>
struct test_trivial<T, false> {
void operator ()() const {}
};
#endif


int main(int, char**)
{
    {
    typedef std::istream_iterator<int> T;
    T it;
    assert(it == T());
#if TEST_STD_VER >= 11
    constexpr T it2;
    (void)it2;
#endif
    }

#if TEST_STD_VER > 14
    test_trivial<int>()();
    test_trivial<char>()();
    test_trivial<double>()();
    test_trivial<S>()();
    test_trivial<std::string>()();
#endif

  return 0;
}
