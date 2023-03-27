//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-pmr-missing

// <set>

// namespace std::pmr {
// template <class V, class Compare = less<V> >
// using set =
//     ::std::set<V, Compare, polymorphic_allocator<V>>
//
// template <class V, class Compare = less<V> >
// using multiset =
//     ::std::multiset<V, Compare, polymorphic_allocator<V>>
//
// } // namespace std::pmr

#include <set>
#include <memory_resource>
#include <type_traits>
#include <cassert>

int main(int, char**) {
  using V  = char;
  using DC = std::less<V>;
  using OC = std::greater<V>;
  {
    using StdSet = std::set<V, DC, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::set<V>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::set<V, OC, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::set<V, OC>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    std::pmr::set<int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }
  {
    using StdSet = std::multiset<V, DC, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::multiset<V>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::multiset<V, OC, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::multiset<V, OC>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    std::pmr::multiset<int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }

  return 0;
}
