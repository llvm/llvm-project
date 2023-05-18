//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Make sure we can use `std::unreachable()`. We can't actually execute it cause that would be
// UB, but we can make sure that it doesn't cause a linker error or something like that.

#include <cassert>
#include <type_traits>
#include <utility>

int main(int argc, char**) {
    assert(argc == 1);
    if (argc != 1) {
        std::unreachable();
    }

    static_assert(std::is_same_v<decltype(std::unreachable()), void>);

    return 0;
}
