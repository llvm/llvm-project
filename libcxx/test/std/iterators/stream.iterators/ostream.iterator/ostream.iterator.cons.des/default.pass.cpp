//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// class ostream_iterator

// constexpr ostream_iterator() noexcept = default;

#include <iterator>
#include <string> // char_traits

#include "test_macros.h"

struct MyTraits : std::char_traits<char> {
    MyTraits();  // This should not be called.
};

constexpr bool test() {
    std::ostream_iterator<int> it;
    (void)it;
    std::ostream_iterator<int, char, MyTraits> wit;
    (void)wit;
    return true;
}

int main(int, char**) {
    ASSERT_NOEXCEPT(std::ostream_iterator<int>());
    ASSERT_NOEXCEPT(std::ostream_iterator<int, char, MyTraits>());

    test();
    static_assert(test());

    return 0;
}
