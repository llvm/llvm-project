//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test unexpected

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS

#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

void fexit()
{
    std::exit(0);
}

int main(int, char**)
{
    std::set_unexpected(fexit);
    std::unexpected();
    assert(false);

  return 0;
}
