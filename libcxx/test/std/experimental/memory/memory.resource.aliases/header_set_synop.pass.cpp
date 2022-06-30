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

// <experimental/set>

// namespace std { namespace experimental { namespace pmr {
// template <class V, class Compare = less<V> >
// using set =
//     ::std::set<V, Compare, polymorphic_allocator<V>>
//
// template <class V, class Compare = less<V> >
// using multiset =
//     ::std::multiset<V, Compare, polymorphic_allocator<V>>
//
// }}} // namespace std::experimental::pmr

#include <experimental/set>
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

int main(int, char**)
{
    using V = char;
    using DC = std::less<V>;
    using OC = std::greater<V>;
    {
        using StdSet = std::set<V, DC, pmr::polymorphic_allocator<V>>;
        using PmrSet = pmr::set<V>;
        static_assert(std::is_same<StdSet, PmrSet>::value, "");
    }
    {
        using StdSet = std::set<V, OC, pmr::polymorphic_allocator<V>>;
        using PmrSet = pmr::set<V, OC>;
        static_assert(std::is_same<StdSet, PmrSet>::value, "");
    }
    {
        pmr::set<int> m;
        assert(m.get_allocator().resource() == pmr::get_default_resource());
    }
    {
        using StdSet = std::multiset<V, DC, pmr::polymorphic_allocator<V>>;
        using PmrSet = pmr::multiset<V>;
        static_assert(std::is_same<StdSet, PmrSet>::value, "");
    }
    {
        using StdSet = std::multiset<V, OC, pmr::polymorphic_allocator<V>>;
        using PmrSet = pmr::multiset<V, OC>;
        static_assert(std::is_same<StdSet, PmrSet>::value, "");
    }
    {
        pmr::multiset<int> m;
        assert(m.get_allocator().resource() == pmr::get_default_resource());
    }

  return 0;
}
