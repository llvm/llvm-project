//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>

// ptrdiff_t should:

//  1. be in namespace std.
//  2. be the same sizeof as void*.
//  3. be a signed integral.

int main()
{
    static_assert(sizeof(std::ptrdiff_t) == sizeof(void*),
                  "sizeof(std::ptrdiff_t) == sizeof(void*)");
    static_assert(std::is_signed<std::ptrdiff_t>::value,
                  "std::is_signed<std::ptrdiff_t>::value");
    static_assert(std::is_integral<std::ptrdiff_t>::value,
                  "std::is_integral<std::ptrdiff_t>::value");
}
