//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class system_error

// system_error(error_code ec, const string& what_arg);

// Test is slightly non-portable

#include <system_error>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
    std::string what_arg("test message");
    std::system_error se(make_error_code(std::errc::not_a_directory), what_arg);
    assert(se.code() == std::make_error_code(std::errc::not_a_directory));
    std::string what_message(se.what());
    assert(what_message.find(what_arg) != std::string::npos);
    assert(what_message.find("Not a directory") != std::string::npos);

    return 0;
}
