//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Ensure that strings which fit within the SSO size can be constant-initialized
// as both a global and local.

#include <string>

#if __SIZE_WIDTH__ == 64
#  define LONGEST_STR "0123456789012345678901"
#elif __SIZE_WIDTH__ == 32
#  define LONGEST_STR "0123456789"
#else
#  error "std::size_t has an unexpected size"
#endif

constinit std::string g_str = LONGEST_STR;
void fn() { constexpr std::string l_str = LONGEST_STR; }
