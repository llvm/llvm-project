//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// class front_insert_iterator

// constexpr front_insert_iterator() noexcept = default;

#include <iterator>
#include <vector>

#include "test_macros.h"

struct T { };
using Container = std::vector<T>;

constexpr bool test() {
    std::front_insert_iterator<Container> it;
    (void)it;
    return true;
}

int main(int, char**) {
    ASSERT_NOEXCEPT(std::front_insert_iterator<Container>());

    test();
    static_assert(test());

    return 0;
}
