//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class directory_iterator

// directory_iterator::directory_iterator() noexcept


#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


int main(int, char**) {
    {
        static_assert(std::is_nothrow_default_constructible<fs::directory_iterator>::value, "");
    }
    {
        fs::directory_iterator d1;
        const fs::directory_iterator d2;
        assert(d1 == d2);
    }

  return 0;
}
