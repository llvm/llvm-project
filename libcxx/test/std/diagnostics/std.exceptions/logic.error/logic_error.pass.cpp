//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test logic_error

#include <stdexcept>
#include <type_traits>
#include <cstring>
#include <string>
#include <cassert>

int main()
{
    static_assert((std::is_base_of<std::exception, std::logic_error>::value),
                 "std::is_base_of<std::exception, std::logic_error>::value");
    static_assert(std::is_polymorphic<std::logic_error>::value,
                 "std::is_polymorphic<std::logic_error>::value");
    {
    const char* msg = "logic_error message";
    std::logic_error e(msg);
    assert(std::strcmp(e.what(), msg) == 0);
    std::logic_error e2(e);
    assert(std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(std::strcmp(e2.what(), msg) == 0);
    }
    {
    std::string msg("another logic_error message");
    std::logic_error e(msg);
    assert(e.what() == msg);
    std::logic_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
    }
}
