//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-pmr-missing

// <unordered_map>

// namespace std::pmr {
// template <class K, class V, class H = hash<K>, class P = equal_to<K> >
// using unordered_map =
//     ::std::unordered_map<K, V, H, P, polymorphic_allocator<pair<const K, V>>>
//
// template <class K, class V,  class H = hash<K>, class P = equal_to<K> >
// using unordered_multimap =
//     ::std::unordered_multimap<K, V, H, P, polymorphic_allocator<pair<const K, V>>>
//
// } // namespace std::pmr

#include <unordered_map>
#include <memory_resource>
#include <type_traits>
#include <cassert>

template <class T>
struct MyHash : std::hash<T> {};

template <class T>
struct MyPred : std::equal_to<T> {};

int main(int, char**) {
  using K  = int;
  using V  = char;
  using DH = std::hash<K>;
  using MH = MyHash<K>;
  using DP = std::equal_to<K>;
  using MP = MyPred<K>;
  using P  = std::pair<const K, V>;
  {
    using StdMap = std::unordered_map<K, V, DH, DP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_map<K, V>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::unordered_map<K, V, MH, DP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_map<K, V, MH>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::unordered_map<K, V, MH, MP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_map<K, V, MH, MP>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    std::pmr::unordered_map<int, int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }
  {
    using StdMap = std::unordered_multimap<K, V, DH, DP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_multimap<K, V>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::unordered_multimap<K, V, MH, DP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_multimap<K, V, MH>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    using StdMap = std::unordered_multimap<K, V, MH, MP, std::pmr::polymorphic_allocator<P>>;
    using PmrMap = std::pmr::unordered_multimap<K, V, MH, MP>;
    static_assert(std::is_same<StdMap, PmrMap>::value, "");
  }
  {
    std::pmr::unordered_multimap<int, int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }

  return 0;
}
