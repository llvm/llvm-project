//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bad_exception

#include <exception>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::exception, std::bad_exception>::value),
                 "std::is_base_of<std::exception, std::bad_exception>::value");
    static_assert(std::is_polymorphic<std::bad_exception>::value,
                 "std::is_polymorphic<std::bad_exception>::value");
    std::bad_exception b;
    std::bad_exception b2 = b;
    b2 = b;
    const char* w = b2.what();
    assert(w);

  return 0;
}
