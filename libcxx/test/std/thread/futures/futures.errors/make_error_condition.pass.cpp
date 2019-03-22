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

// class error_condition

// error_condition make_error_condition(future_errc e);

#include <future>
#include <cassert>

int main(int, char**)
{
    {
        const std::error_condition ec1 =
          std::make_error_condition(std::future_errc::future_already_retrieved);
        assert(ec1.value() ==
                  static_cast<int>(std::future_errc::future_already_retrieved));
        assert(ec1.category() == std::future_category());
    }

  return 0;
}
