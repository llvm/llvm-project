//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// const error_category& category() const;

#include <system_error>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    const std::error_code ec(6, std::generic_category());
    assert(ec.category() == std::generic_category());

  return 0;
}
