//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class error_code

// error_code make_error_code(future_errc e);

#include <future>
#include <cassert>

int main()
{
    {
        std::error_code ec = make_error_code(std::future_errc::broken_promise);
        assert(ec.value() == static_cast<int>(std::future_errc::broken_promise));
        assert(ec.category() == std::future_category());
    }
}
