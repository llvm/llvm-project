//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// template <> struct is_error_code_enum<future_errc> : public true_type {};

#include <future>

int main()
{
    static_assert(std::is_error_code_enum<std::future_errc>::value, "");
}
