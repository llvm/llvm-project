//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>
#include <test_macros.h>

// XFAIL: c++98, c++03, c++11, c++14

// std::byte is not an integer type, nor a character type. 
// It is a distinct type for accessing the bits that ultimately make up object storage.

static_assert( std::is_pod<std::byte>::value, "" );
static_assert(!std::is_arithmetic<std::byte>::value, "" );
static_assert(!std::is_integral<std::byte>::value, "" );

static_assert(!std::is_same<std::byte,          char>::value, "" );
static_assert(!std::is_same<std::byte,   signed char>::value, "" );
static_assert(!std::is_same<std::byte, unsigned char>::value, "" );

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(std::byte) == 1, "" );

int main () {}
