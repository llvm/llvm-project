// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/list>

// namespace std { namespace experimental { namespace pmr {
// template <class T>
// using list =
//     ::std::list<T, polymorphic_allocator<T>>
//
// }}} // namespace std::experimental::pmr

#include <experimental/list>
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

int main(int, char**)
{
    using StdList = std::list<int, pmr::polymorphic_allocator<int>>;
    using PmrList = pmr::list<int>;
    static_assert(std::is_same<StdList, PmrList>::value, "");
    PmrList d;
    assert(d.get_allocator().resource() == pmr::get_default_resource());

  return 0;
}
