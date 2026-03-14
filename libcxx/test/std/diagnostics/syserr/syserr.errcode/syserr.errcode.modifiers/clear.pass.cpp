//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// void clear();

#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::error_code ec;
        ec.assign(6, std::generic_category());
        assert(ec.value() == 6);
        assert(ec.category() == std::generic_category());
        ec.clear();
        assert(ec.value() == 0);
        assert(ec.category() == std::system_category());
    }

  return 0;
}
