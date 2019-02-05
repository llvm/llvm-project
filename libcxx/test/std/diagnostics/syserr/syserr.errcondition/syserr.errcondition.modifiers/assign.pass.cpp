//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// void assign(int val, const error_category& cat);

#include <system_error>
#include <cassert>

int main(int, char**)
{
    {
        std::error_condition ec;
        ec.assign(6, std::system_category());
        assert(ec.value() == 6);
        assert(ec.category() == std::system_category());
    }
    {
        std::error_condition ec;
        ec.assign(8, std::generic_category());
        assert(ec.value() == 8);
        assert(ec.category() == std::generic_category());
    }

  return 0;
}
