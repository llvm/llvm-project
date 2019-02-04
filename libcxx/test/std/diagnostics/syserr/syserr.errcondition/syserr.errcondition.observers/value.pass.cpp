//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// int value() const;

#include <system_error>
#include <cassert>

int main()
{
    const std::error_condition ec(6, std::system_category());
    assert(ec.value() == 6);
}
