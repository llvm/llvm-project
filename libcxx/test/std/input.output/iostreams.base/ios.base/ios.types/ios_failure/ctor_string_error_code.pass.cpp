//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base::failure

// explicit failure(const string& msg, const error_code& ec = io_errc::stream);

#include <cassert>
#include <ios>
#include <string>
#include <system_error>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    // LWG2462 std::ios_base::failure is overspecified
    static_assert((std::is_base_of<std::system_error, std::ios_base::failure>::value), "");

    {
        std::string what_arg("io test message");
        std::ios_base::failure se(what_arg, make_error_code(std::errc::is_a_directory));
        assert(se.code() == std::make_error_code(std::errc::is_a_directory));
        std::string what_message(se.what());
        assert(what_message.find(what_arg) != std::string::npos);
        assert(what_message.find(std::generic_category().message(static_cast<int>
            (std::errc::is_a_directory))) != std::string::npos);
    }
    {
        std::string what_arg("io test message");
        std::ios_base::failure se(what_arg);
        assert(se.code() == std::make_error_code(std::io_errc::stream));
        std::string what_message(se.what());
        assert(what_message.find(what_arg) != std::string::npos);
        assert(what_message.find(std::iostream_category().message(static_cast<int>
            (std::io_errc::stream))) != std::string::npos);
    }

  return 0;
}
