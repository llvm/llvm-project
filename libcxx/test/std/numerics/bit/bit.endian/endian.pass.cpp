//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// enum class endian;
// <bit>

#include <bit>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "test_macros.h"

int main(int, char**) {
    static_assert(std::is_enum<std::endian>::value, "");

// Check that E is a scoped enum by checking for conversions.
    typedef std::underlying_type<std::endian>::type UT;
    static_assert(!std::is_convertible<std::endian, UT>::value, "");

// test that the enumeration values exist
    static_assert( std::endian::little == std::endian::little );
    static_assert( std::endian::big    == std::endian::big );
    static_assert( std::endian::native == std::endian::native );
    static_assert( std::endian::little != std::endian::big );

//  Technically not required, but true on all existing machines
    static_assert( std::endian::native == std::endian::little ||
                   std::endian::native == std::endian::big );

//  Try to check at runtime
    {
    uint32_t i = 0x01020304;
    char c[4];
    static_assert(sizeof(i) == sizeof(c));
    std::memcpy(c, &i, sizeof(c));

    assert ((c[0] == 1) == (std::endian::native == std::endian::big));
    }

  return 0;
}
