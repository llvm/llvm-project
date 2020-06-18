//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// polymorphic_allocator operator=(polymorphic_allocator const &) = delete

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace ex = std::experimental::pmr;

int main(int, char**)
{
    typedef ex::polymorphic_allocator<void> T;
    static_assert(!std::is_copy_assignable<T>::value, "");
    static_assert(!std::is_move_assignable<T>::value, "");

  return 0;
}
