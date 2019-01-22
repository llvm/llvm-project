//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test unexpected

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS
#define _LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS
#include <exception>
#include <cstdlib>
#include <cassert>

void fexit()
{
    std::exit(0);
}

int main()
{
    std::set_unexpected(fexit);
    std::unexpected();
    assert(false);
}
