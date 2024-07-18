//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <unordered_set>

// namespace std::pmr {
// template <class V, class H = hash<V>, class P = equal_to<V> >
// using unordered_set =
//     ::std::unordered_set<V, H, P, polymorphic_allocator<V>>
//
// template <class V,  class H = hash<V>, class P = equal_to<V> >
// using unordered_multiset =
//     ::std::unordered_multiset<V, H, P, polymorphic_allocator<V>>
//
// } // namespace std::pmr

#include <unordered_set>
#include <memory_resource>
#include <type_traits>
#include <cassert>

template <class T>
struct MyHash : std::hash<T> {};

template <class T>
struct MyPred : std::equal_to<T> {};

int main(int, char**) {
  using V  = char;
  using DH = std::hash<V>;
  using MH = MyHash<V>;
  using DP = std::equal_to<V>;
  using MP = MyPred<V>;
  {
    using StdSet = std::unordered_set<V, DH, DP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_set<V>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::unordered_set<V, MH, DP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_set<V, MH>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::unordered_set<V, MH, MP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_set<V, MH, MP>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    std::pmr::unordered_set<int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }
  {
    using StdSet = std::unordered_multiset<V, DH, DP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_multiset<V>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::unordered_multiset<V, MH, DP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_multiset<V, MH>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    using StdSet = std::unordered_multiset<V, MH, MP, std::pmr::polymorphic_allocator<V>>;
    using PmrSet = std::pmr::unordered_multiset<V, MH, MP>;
    static_assert(std::is_same<StdSet, PmrSet>::value, "");
  }
  {
    std::pmr::unordered_multiset<int> m;
    assert(m.get_allocator().resource() == std::pmr::get_default_resource());
  }

  return 0;
}
