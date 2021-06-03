//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// class insert_iterator

// insert_iterator() = default;

#include <iterator>
#include <vector>

struct T { };
using Container = std::vector<T>;

int main(int, char**) {
    std::insert_iterator<Container> it; (void)it;
    return 0;
}
