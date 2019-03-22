//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// string message() const;

#include <system_error>
#include <string>
#include <cassert>

int main(int, char**)
{
    const std::error_code ec(6, std::generic_category());
    assert(ec.message() == std::generic_category().message(6));

  return 0;
}
