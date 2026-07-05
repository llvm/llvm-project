//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// bool operator<(const error_code& lhs, const error_code& rhs);

#include <system_error>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        const std::error_code ec1(6, std::generic_category());
        const std::error_code ec2(7, std::generic_category());
        assert(ec1 < ec2);
    }

  return 0;
}
