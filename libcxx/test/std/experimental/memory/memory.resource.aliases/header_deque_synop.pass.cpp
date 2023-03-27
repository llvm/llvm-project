//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: availability-aligned_allocation-missing

// <experimental/deque>

// namespace std { namespace experimental { namespace pmr {
// template <class T>
// using deque =
//     ::std::deque<T, polymorphic_allocator<T>>
//
// }}} // namespace std::experimental::pmr

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/deque>
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

int main(int, char**)
{
    using StdDeque = std::deque<int, pmr::polymorphic_allocator<int>>;
    using PmrDeque = pmr::deque<int>;
    static_assert(std::is_same<StdDeque, PmrDeque>::value, "");
    PmrDeque d;
    assert(d.get_allocator().resource() == pmr::get_default_resource());

  return 0;
}
