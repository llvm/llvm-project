//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// template <> struct is_error_code_enum<future_errc> : public true_type {};

#include <future>
#include "test_macros.h"

int main()
{
    static_assert(std::is_error_code_enum  <std::future_errc>::value, "");
#if TEST_STD_VER > 14
    static_assert(std::is_error_code_enum_v<std::future_errc>, "");
#endif
}
