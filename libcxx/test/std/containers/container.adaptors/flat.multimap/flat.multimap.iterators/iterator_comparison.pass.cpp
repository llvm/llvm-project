//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap iterators should be C++20 random access iterators

#include <compare>
#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using KI    = typename KeyContainer::iterator;
  using I     = M::iterator;
  using CI    = M::const_iterator;
  using RI    = M::reverse_iterator;
  using CRI   = M::const_reverse_iterator;

  static_assert(std::equality_comparable<I>);
  static_assert(std::equality_comparable<CI>);
  static_assert(std::equality_comparable<RI>);
  static_assert(std::equality_comparable<CRI>);

  static_assert(std::totally_ordered<I>);
  static_assert(std::totally_ordered<CI>);
  static_assert(std::totally_ordered<RI>);
  static_assert(std::totally_ordered<CRI>);

  M m = {{1, 'a'}, {2, 'b'}, {2, 'e'}, {3, 'z'}, {3, 'y'}, {3, 'c'}, {4, 'd'}};

  I i1 = m.begin();
  I i2 = m.begin() + 1;

  assert(i1 == i1);
  assert(!(i1 != i1));
  assert(i1 != i2);
  assert(!(i1 == i2));
  assert(i1 < i2);
  assert(!(i1 < i1));
  assert(i1 <= i1);
  assert(i1 <= i2);
  assert(!(i2 <= i1));
  assert(i2 > i1);
  assert(!(i2 > i2));
  assert(i2 >= i1);
  assert(i2 >= i2);
  assert(!(i1 >= i2));

  CI ci1 = m.cbegin();
  CI ci2 = m.cbegin() + 1;
  assert(ci1 == ci1);
  assert(!(ci1 != ci1));
  assert(ci1 != ci2);
  assert(!(ci1 == ci2));
  assert(ci1 < ci2);
  assert(!(ci1 < ci1));
  assert(ci1 <= ci1);
  assert(ci1 <= ci2);
  assert(!(ci2 <= ci1));
  assert(ci2 > ci1);
  assert(!(ci2 > ci2));
  assert(ci2 >= ci1);
  assert(ci2 >= ci2);
  assert(!(ci1 >= ci2));

  RI ri1 = m.rbegin();
  RI ri2 = m.rbegin() + 1;
  assert(ri1 == ri1);
  assert(!(ri1 != ri1));
  assert(ri1 != ri2);
  assert(!(ri1 == ri2));
  assert(ri1 < ri2);
  assert(!(ri1 < ri1));
  assert(ri1 <= ri1);
  assert(ri1 <= ri2);
  assert(!(ri2 <= ri1));
  assert(ri2 > ri1);
  assert(!(ri2 > ri2));
  assert(ri2 >= ri1);
  assert(ri2 >= ri2);
  assert(!(ri1 >= ri2));

  CRI cri1 = m.crbegin();
  CRI cri2 = m.crbegin() + 1;
  assert(cri1 == cri1);
  assert(!(cri1 != cri1));
  assert(cri1 != cri2);
  assert(!(cri1 == cri2));
  assert(cri1 < cri2);
  assert(!(cri1 < cri1));
  assert(cri1 <= cri1);
  assert(cri1 <= cri2);
  assert(!(cri2 <= cri1));
  assert(cri2 > cri1);
  assert(!(cri2 > cri2));
  assert(cri2 >= cri1);
  assert(cri2 >= cri2);
  assert(!(cri1 >= cri2));

  if constexpr (std::three_way_comparable<KI>) {
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

    assert(i1 <=> i1 == std::strong_ordering::equivalent);
    assert(i1 <=> i2 == std::strong_ordering::less);
    assert(i2 <=> i1 == std::strong_ordering::greater);

    assert(ci1 <=> ci1 == std::strong_ordering::equivalent);
    assert(ci1 <=> ci2 == std::strong_ordering::less);
    assert(ci2 <=> ci1 == std::strong_ordering::greater);

    assert(ri1 <=> ri1 == std::strong_ordering::equivalent);
    assert(ri1 <=> ri2 == std::strong_ordering::less);
    assert(ri2 <=> ri1 == std::strong_ordering::greater);

    assert(cri1 <=> cri1 == std::strong_ordering::equivalent);
    assert(cri1 <=> cri2 == std::strong_ordering::less);
    assert(cri2 <=> cri1 == std::strong_ordering::greater);
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<char>>();
  test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return 0;
}
