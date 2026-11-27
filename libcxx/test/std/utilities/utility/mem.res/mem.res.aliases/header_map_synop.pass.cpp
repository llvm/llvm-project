//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <map>

// namespace std::pmr {
// template <class K, class V, class Compare = less<Key> >
// using map =
//     ::std::map<K, V, Compare, polymorphic_allocator<pair<const K, V>>>
//
// template <class K, class V, class Compare = less<Key> >
// using multimap =
//     ::std::multimap<K, V, Compare, polymorphic_allocator<pair<const K, V>>>
//
// } // namespace std::pmr

#include <map>
#include <memory_resource>
#include <type_traits>
#include <cassert>

int main(int, char**) {
  using K  = int;
  using V  = char;
  using DC = std::less<int>;
  using OC = std::greater<int>;
  using P  = std::pair<const K, V>;
  {
    using StdMap = std::map<K, V, DC, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::map<K, V>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::map<K, V, OC, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::map<K, V, OC>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    std::pmr::map<int, int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }
  {
    using StdMap = std::multimap<K, V, DC, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::multimap<K, V>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::multimap<K, V, OC, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::multimap<K, V, OC>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    std::pmr::multimap<int, int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }

  return 0;
}
