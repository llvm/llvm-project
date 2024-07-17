//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_map iterators should be C++20 random access iterators
// supporting `operator<=>`, even when the underlying container's
// iterators are not.

#include <compare>
#include <concepts>
#include <flat_map>
#include <functional>

#include "MinSequenceContainer.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using V = MinSequenceContainer<int, random_access_iterator<int*>, random_access_iterator<const int*>>;
    using M = std::flat_map<int, int, std::less<int>, V, V>;
    using VI = V::iterator;
    using VCI = V::const_iterator;
    using I = M::iterator;
    using CI = M::const_iterator;
    using RI = M::reverse_iterator;
    using CRI = M::const_reverse_iterator;
    static_assert(!std::three_way_comparable<VI>); // But, despite this...
    static_assert(!std::three_way_comparable<VCI>);
    static_assert(std::three_way_comparable<I>); // ...the wrapped iterators support <=>.
    static_assert(std::three_way_comparable<CI>);
    static_assert(std::three_way_comparable<RI>);
    static_assert(std::three_way_comparable<CRI>);
    static_assert(std::same_as<decltype(I() <=> I()), std::strong_ordering>);
    static_assert(std::same_as<decltype(I() <=> CI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(CI() <=> CI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(RI() <=> RI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(RI() <=> CRI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(CRI() <=> CRI()), std::strong_ordering>);
  }
  {
    using V = MinSequenceContainer<int, int*, const int*>;
    using M = std::flat_map<int, int, std::less<int>, V, V>;
    using VI = V::iterator;
    using VCI = V::const_iterator;
    using I = M::iterator;
    using CI = M::const_iterator;
    using RI = M::reverse_iterator;
    using CRI = M::const_reverse_iterator;
    static_assert(std::three_way_comparable<VI>); // And when VI does support <=>...
    static_assert(std::three_way_comparable<VCI>);
    static_assert(std::three_way_comparable<I>); // ...of course the wrapped iterators still support <=>.
    static_assert(std::three_way_comparable<CI>);
    static_assert(std::three_way_comparable<RI>);
    static_assert(std::three_way_comparable<CRI>);
    static_assert(std::same_as<decltype(I() <=> I()), std::strong_ordering>);
    static_assert(std::same_as<decltype(I() <=> CI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(CI() <=> CI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(RI() <=> RI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(RI() <=> CRI()), std::strong_ordering>);
    static_assert(std::same_as<decltype(CRI() <=> CRI()), std::strong_ordering>);
  }
  return 0;
}
