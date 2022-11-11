//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// class bad_weak_ptr
//     : public std::exception
// {
// public:
//     bad_weak_ptr();
// };

#include <exception>
#include <memory>
#include <type_traits>
#include <cassert>
#include <cstring>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_base_of<std::exception, std::bad_weak_ptr>::value), "");
    std::bad_weak_ptr e;
    std::bad_weak_ptr e2 = e;
    e2 = e;
    assert(std::strcmp(e.what(), "bad_weak_ptr") == 0);

  return 0;
}
