//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test terminate_handler

#include <exception>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

void f() {}

int main(int, char**)
{
    static_assert((std::is_same<std::terminate_handler, void(*)()>::value), "");
    std::terminate_handler p = f;
    assert(p == &f);

  return 0;
}
