//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>
#include "test_macros.h"

// XFAIL: c++98, c++03, c++11, c++14

// std::byte is not an integer type, nor a character type.
// It is a distinct type for accessing the bits that ultimately make up object storage.

#if TEST_STD_VER > 17
static_assert( std::is_trivial<std::byte>::value, "" );   // P0767
#else
static_assert( std::is_pod<std::byte>::value, "" );
#endif
static_assert(!std::is_arithmetic<std::byte>::value, "" );
static_assert(!std::is_integral<std::byte>::value, "" );

static_assert(!std::is_same<std::byte,          char>::value, "" );
static_assert(!std::is_same<std::byte,   signed char>::value, "" );
static_assert(!std::is_same<std::byte, unsigned char>::value, "" );

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(std::byte) == 1, "" );

int main(int, char**) {
  return 0;
}
