//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test runtime_error

#include <stdexcept>
#include <type_traits>
#include <cstring>
#include <string>
#include <cassert>

int main(int, char**)
{
    static_assert((std::is_base_of<std::exception, std::runtime_error>::value),
                 "std::is_base_of<std::exception, std::runtime_error>::value");
    static_assert(std::is_polymorphic<std::runtime_error>::value,
                 "std::is_polymorphic<std::runtime_error>::value");
    {
    const char* msg = "runtime_error message";
    std::runtime_error e(msg);
    assert(std::strcmp(e.what(), msg) == 0);
    std::runtime_error e2(e);
    assert(std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(std::strcmp(e2.what(), msg) == 0);
    }
    {
    std::string msg("another runtime_error message");
    std::runtime_error e(msg);
    assert(e.what() == msg);
    std::runtime_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
    }

  return 0;
}
