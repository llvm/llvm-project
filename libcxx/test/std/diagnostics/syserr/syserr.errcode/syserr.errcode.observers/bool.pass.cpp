//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// explicit operator bool() const;

#include <system_error>
#include <cassert>
#include <string>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert(std::is_constructible<bool, std::error_code>::value, "");
    static_assert(!std::is_convertible<std::error_code, bool>::value, "");

    {
        const std::error_code ec(6, std::generic_category());
        assert(static_cast<bool>(ec));
    }
    {
        const std::error_code ec(0, std::generic_category());
        assert(!static_cast<bool>(ec));
    }

  return 0;
}
