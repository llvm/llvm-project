//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <ios>

// template <> struct is_error_code_enum<io_errc> : public true_type {};

#include <ios>
#include "test_macros.h"

int main()
{
    static_assert(std::is_error_code_enum  <std::io_errc>::value, "");
#if TEST_STD_VER > 14
    static_assert(std::is_error_code_enum_v<std::io_errc>, "");
#endif
}
