//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// class ostreambuf_iterator

// constexpr ostreambuf_iterator() noexcept = default;

#include <iterator>

#include "test_macros.h"

constexpr bool test() {
    std::ostreambuf_iterator<char> it;
    (void)it;
    std::ostreambuf_iterator<wchar_t> wit;
    (void)wit;
    return true;
}

int main(int, char**) {
    ASSERT_NOEXCEPT(std::ostreambuf_iterator<char>());
    ASSERT_NOEXCEPT(std::ostreambuf_iterator<wchar_t>());

    test();
    static_assert(test());

    return 0;
}
