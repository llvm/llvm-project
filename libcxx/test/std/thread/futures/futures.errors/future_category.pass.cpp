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

// const error_category& future_category();

#include <future>
#include <cstring>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    const std::error_category& ec = std::future_category();
    assert(std::strcmp(ec.name(), "future") == 0);

  return 0;
}
