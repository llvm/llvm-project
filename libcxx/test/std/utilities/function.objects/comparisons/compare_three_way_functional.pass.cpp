//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <functional>

// Test that std::compare_three_way is defined in <functional>,
// not only in <compare>.

#include <functional>
#include <cassert>

int main(int, char**)
{
    assert(std::compare_three_way()(1, 2) < 0);
    assert(std::compare_three_way()(1, 1) == 0);
    assert(std::compare_three_way()(2, 1) > 0);

    return 0;
}
