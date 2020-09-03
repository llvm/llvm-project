//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class system_error

// system_error(error_code ec);

// Test is slightly non-portable

#include <system_error>
#include <string>
#include <cassert>
#include <clocale>

#include "test_macros.h"

int main(int, char**)
{
    std::setlocale (LC_ALL, "C");
    std::system_error se(static_cast<int>(std::errc::not_a_directory),
                         std::generic_category(), "some text");
    assert(se.code() == std::make_error_code(std::errc::not_a_directory));
    std::string what_message(se.what());
    assert(what_message.find("Not a directory") != std::string::npos);

  return 0;
}
