//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>

// size_t should:

//  1. be in namespace std.
//  2. be the same sizeof as void*.
//  3. be an unsigned integral.

int main()
{
    static_assert(sizeof(std::size_t) == sizeof(void*),
                  "sizeof(std::size_t) == sizeof(void*)");
    static_assert(std::is_unsigned<std::size_t>::value,
                  "std::is_unsigned<std::size_t>::value");
    static_assert(std::is_integral<std::size_t>::value,
                  "std::is_integral<std::size_t>::value");
}
