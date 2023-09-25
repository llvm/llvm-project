//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Ensure that strings which fit within the SSO size can be constant-initialized
// globals.  (this is permitted but not required to work by the standard).

#include <string>

#if __SIZE_WIDTH__ == 64
constinit std::string my_str = "0123456789012345678901";
#elif __SIZE_WIDTH__ == 32
constinit std::string my_str = "0123456789";
#else
#  error "std::size_t has an unexpected size"
#endif
