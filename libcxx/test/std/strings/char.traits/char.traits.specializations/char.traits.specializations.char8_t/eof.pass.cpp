//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string>

// template<> struct char_traits<char8_t>

// static constexpr int_type eof();

#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    std::char_traits<char8_t>::int_type i = std::char_traits<char8_t>::eof();
    ((void)i); // Prevent unused warning
#endif

  return 0;
}
