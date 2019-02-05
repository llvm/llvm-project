//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// template <ErrorCodeEnum E> error_code(E e);

#include <system_error>
#include <cassert>

enum testing
{
    zero, one, two
};

namespace std
{

template <> struct is_error_code_enum<testing> : public std::true_type {};

}

std::error_code
make_error_code(testing x)
{
    return std::error_code(static_cast<int>(x), std::generic_category());
}

int main()
{
    {
        std::error_code ec(two);
        assert(ec.value() == 2);
        assert(ec.category() == std::generic_category());
    }
}
