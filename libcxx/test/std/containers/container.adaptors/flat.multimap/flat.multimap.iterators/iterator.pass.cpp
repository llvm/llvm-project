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

//       iterator begin()   noexcept;
// const_iterator begin()   const noexcept
//       iterator end()     noexcept;
// const_iterator end()     const noexcept;
//
// const_iterator cbegin()  const noexcept;
// const_iterator cend()    const noexcept;

#include <cassert>
#include <cstddef>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  M m         = {{1, 'a'}, {1, 'z'}, {2, 'b'}, {3, 'a'}, {3, 'b'}, {3, 'c'}, {4, 'd'}};
  const M& cm = m;
  ASSERT_SAME_TYPE(decltype(m.begin()), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(m.cbegin()), typename M::const_iterator);
  ASSERT_SAME_TYPE(decltype(cm.begin()), typename M::const_iterator);
  ASSERT_SAME_TYPE(decltype(m.end()), typename M::iterator);
  ASSERT_SAME_TYPE(decltype(m.cend()), typename M::const_iterator);
  ASSERT_SAME_TYPE(decltype(cm.end()), typename M::const_iterator);
  static_assert(noexcept(m.begin()));
  static_assert(noexcept(cm.begin()));
  static_assert(noexcept(m.cbegin()));
  static_assert(noexcept(m.end()));
  static_assert(noexcept(cm.end()));
  static_assert(noexcept(m.cend()));
  assert(m.size() == 7);
  assert(std::distance(m.begin(), m.end()) == 7);
  assert(std::distance(cm.begin(), cm.end()) == 7);
  assert(std::distance(m.cbegin(), m.cend()) == 7);
  typename M::iterator i;                   // default-construct
  i                            = m.begin(); // move-assignment
  typename M::const_iterator k = i;         // converting constructor
  assert(i == k);                           // comparison
  assert(i->first == 1);                    // operator->
  assert(i->second == 'a');                 // operator->
  ++i;                                      // pre-increment
  assert(i->first == 1);                    // operator->
  assert(i->second == 'z');                 // operator->
  i = i + 3;                                // operator+
  assert((*i).first == 3);                  // operator*
  assert((*i).second == 'b');               // operator*
  i += 3;                                   // operator+=
  assert(i == m.end());                     // operator==
  --i;                                      // pre-decrement
  assert(i->first == 4);                    // operator->
  assert(i->second == 'd');                 // operator->
  i = i - 2;                                // operator-
  assert(i->first == 3);                    // operator->
  assert(i->second == 'b');                 // operator->
  i -= 2;                                   // operator-=
  assert(i > m.begin());                    // operator>
}

constexpr bool test() {
  test<std::vector<int>, std::vector<char>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<char>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  {
    // N3644 testing
    using C = std::flat_multimap<int, char>;
    C::iterator ii1{}, ii2{};
    C::iterator ii4 = ii1;
    C::const_iterator cii{};
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
