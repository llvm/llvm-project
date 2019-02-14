//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ios>

// error_code make_error_code(io_errc e);

#include <ios>
#include <cassert>

int main(int, char**)
{
    {
        std::error_code ec = make_error_code(std::io_errc::stream);
        assert(ec.value() == static_cast<int>(std::io_errc::stream));
        assert(ec.category() == std::iostream_category());
    }

  return 0;
}
