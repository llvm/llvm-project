//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14

// template <class IntegerType>
//    constexpr byte operator <<(byte b, IntegerType shift) noexcept;
// These functions shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.


constexpr std::byte test(std::byte b) {
    return b <<= 2;
    }


int main () {
    constexpr std::byte b100{static_cast<std::byte>(100)};
    constexpr std::byte b115{static_cast<std::byte>(115)};

    static_assert(noexcept(b100 << 2), "" );

    static_assert(std::to_integer<int>(b100 >> 1) ==  50, "");
    static_assert(std::to_integer<int>(b100 >> 2) ==  25, "");
    static_assert(std::to_integer<int>(b115 >> 3) ==  14, "");
    static_assert(std::to_integer<int>(b115 >> 6) ==   1, "");

}
