//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// bool unique() const; // deprecated in C++17, removed in C++20

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX20_REMOVED_SHARED_PTR_UNIQUE

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    const std::shared_ptr<int> p(new int(32));
    assert(p.unique());
    {
    std::shared_ptr<int> p2 = p;
    assert(!p.unique());
    }
    assert(p.unique());

  return 0;
}
