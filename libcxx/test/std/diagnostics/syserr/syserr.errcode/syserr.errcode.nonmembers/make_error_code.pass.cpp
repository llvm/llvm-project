//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// error_code make_error_code(errc e);

#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::error_code ec = make_error_code(std::errc::operation_canceled);
        assert(ec.value() == static_cast<int>(std::errc::operation_canceled));
        assert(ec.category() == std::generic_category());
    }

  return 0;
}
