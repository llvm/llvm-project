//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that a std::vector containing move-only types can't be copied.

// UNSUPPORTED: c++03

#include <vector>

struct move_only
{
    move_only() = default;
    move_only(move_only&&) = default;
    move_only& operator=(move_only&&) = default;
};

int main(int, char**)
{
    std::vector<move_only> v;
    std::vector<move_only> copy = v; // expected-error-re@memory:* {{{{(no matching function for call to 'construct_at')|(call to implicitly-deleted copy constructor of 'move_only')}}}}
    return 0;
}
