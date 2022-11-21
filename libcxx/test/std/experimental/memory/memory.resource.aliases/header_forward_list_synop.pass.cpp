//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/forward_list>

// namespace std { namespace experimental { namespace pmr {
// template <class T>
// using forward_list =
//     ::std::forward_list<T, polymorphic_allocator<T>>
//
// }}} // namespace std::experimental::pmr

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/forward_list>
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

int main(int, char**)
{
    using StdForwardList = std::forward_list<int, pmr::polymorphic_allocator<int>>;
    using PmrForwardList = pmr::forward_list<int>;
    static_assert(std::is_same<StdForwardList, PmrForwardList>::value, "");
    PmrForwardList d;
    assert(d.get_allocator().resource() == pmr::get_default_resource());

  return 0;
}
