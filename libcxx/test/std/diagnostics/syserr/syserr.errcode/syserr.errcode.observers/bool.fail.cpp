//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: c++98, c++03

// <system_error>

// class error_code

// explicit operator bool() const;

#include <system_error>

bool test_func(void)
{
    const std::error_code ec(0, std::generic_category());
    return ec;   // conversion to bool is explicit; should fail.
}

int main()
{
    return 0;
}

