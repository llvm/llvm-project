//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test exception

#include <exception>
#include <type_traits>
#include <cassert>

int main()
{
    static_assert(std::is_polymorphic<std::exception>::value,
                 "std::is_polymorphic<std::exception>::value");
    std::exception b;
    std::exception b2 = b;
    b2 = b;
    const char* w = b2.what();
    assert(w);
}
