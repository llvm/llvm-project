//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: has-fblocks
// ADDITIONAL_COMPILE_FLAGS: -fblocks

// <optional>

// This test makes sure that we can create a `std::optional` containing
// an Objective-C++ block.

#include <optional>
#include <cassert>

int main(int, char**)
{
    using Block = void (^)(void);
    std::optional<Block> block;
    assert(!block);

    return 0;
}
