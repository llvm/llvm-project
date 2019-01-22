//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// error_code();

#include <system_error>
#include <cassert>

int main()
{
    std::error_code ec;
    assert(ec.value() == 0);
    assert(ec.category() == std::system_category());
}
