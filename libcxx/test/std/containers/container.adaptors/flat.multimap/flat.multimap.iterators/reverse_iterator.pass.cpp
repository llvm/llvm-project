//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

//       reverse_iterator rbegin() noexcept;
// const_reverse_iterator rbegin() const noexcept;
//       reverse_iterator rend()   noexcept;
// const_reverse_iterator rend()   const noexcept;
//
// const_reverse_iterator crbegin() const noexcept;
// const_reverse_iterator crend()   const noexcept;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>

#include <iterator>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  M m         = {{1, 'a'}, {1, 'b'}, {2, 'c'}, {2, 'd'}, {3, 'e'}, {3, 'f'}, {4, 'g'}, {4, 'h'}};
  const M& cm = m;
  ASSERT_SAME_TYPE(decltype(m.rbegin()), typename M::reverse_iterator);
  ASSERT_SAME_TYPE(decltype(m.crbegin()), typename M::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(cm.rbegin()), typename M::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(m.rend()), typename M::reverse_iterator);
  ASSERT_SAME_TYPE(decltype(m.crend()), typename M::const_reverse_iterator);
  ASSERT_SAME_TYPE(decltype(cm.rend()), typename M::const_reverse_iterator);
  static_assert(noexcept(m.rbegin()));
  static_assert(noexcept(cm.rbegin()));
  static_assert(noexcept(m.crbegin()));
  static_assert(noexcept(m.rend()));
  static_assert(noexcept(cm.rend()));
  static_assert(noexcept(m.crend()));
  assert(m.size() == 8);
  assert(std::distance(m.rbegin(), m.rend()) == 8);
  assert(std::distance(cm.rbegin(), cm.rend()) == 8);
  assert(std::distance(m.crbegin(), m.crend()) == 8);
  assert(std::distance(cm.crbegin(), cm.crend()) == 8);
  typename M::reverse_iterator i; // default-construct
  ASSERT_SAME_TYPE(decltype(i->first), const int&);
  ASSERT_SAME_TYPE(decltype(i->second), char&);
  i                                    = m.rbegin(); // move-assignment
  typename M::const_reverse_iterator k = i;          // converting constructor
  assert(i == k);                                    // comparison
  for (int j = 8; j >= 1; --j, ++i) {                // pre-increment
    assert(i->first == (j + 1) / 2);                 // operator->
  }
  assert(i == m.rend());
  for (int j = 1; j <= 8; ++j) {
    --i; // pre-decrement
    assert((*i).first == (j + 1) / 2);
  }
  assert(i == m.rbegin());
}

constexpr bool test() {
  test<std::vector<int>, std::vector<char>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<char>>();
  }
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  {
    // N3644 testing
    using C = std::flat_multimap<int, char>;
    C::reverse_iterator ii1{}, ii2{};
    C::reverse_iterator ii4 = ii1;
    C::const_reverse_iterator cii{};
    assert(ii1 == ii2);
    assert(ii1 == ii4);
    assert(!(ii1 != ii2));

    assert((ii1 == cii));
    assert((cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
