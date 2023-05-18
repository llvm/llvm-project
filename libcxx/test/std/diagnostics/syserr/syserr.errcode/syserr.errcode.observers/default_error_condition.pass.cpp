//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// error_condition default_error_condition() const;

#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        const std::error_code ec(6, std::generic_category());
        std::error_condition e_cond = ec.default_error_condition();
        assert(e_cond == ec);
    }
    {
        const std::error_code ec(6, std::system_category());
        std::error_condition e_cond = ec.default_error_condition();
        assert(e_cond == ec);
    }

  return 0;
}
