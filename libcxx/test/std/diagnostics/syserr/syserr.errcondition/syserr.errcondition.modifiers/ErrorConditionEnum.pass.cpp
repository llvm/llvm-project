//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// template <ErrorConditionEnum E> error_condition& operator=(E e);

#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::error_condition ec;
        ec = std::errc::not_enough_memory;
        assert(ec.value() == static_cast<int>(std::errc::not_enough_memory));
        assert(ec.category() == std::generic_category());
    }

  return 0;
}
