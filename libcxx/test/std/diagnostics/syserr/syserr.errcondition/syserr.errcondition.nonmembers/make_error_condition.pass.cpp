//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// error_condition make_error_condition(errc e);

#include <system_error>
#include <cassert>

int main()
{
    {
        const std::error_condition ec1 = std::make_error_condition(std::errc::message_size);
        assert(ec1.value() == static_cast<int>(std::errc::message_size));
        assert(ec1.category() == std::generic_category());
    }
}
