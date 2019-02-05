//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// TODO: Remove this when filesystem gets integrated into the dylib
// REQUIRES: c++filesystem

// <chrono>

// file_clock

// rep should be signed

#include <chrono>
#include <cassert>

int main()
{
    static_assert(std::is_signed<std::chrono::file_clock::rep>::value, "");
    assert(std::chrono::file_clock::duration::min() <
           std::chrono::file_clock::duration::zero());
}
