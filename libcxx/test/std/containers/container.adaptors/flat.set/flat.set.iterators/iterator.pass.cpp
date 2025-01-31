//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

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
#include <flat_set>
#include <functional>
#include <string>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;

  M m         = {1, 2, 3, 4};
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
  assert(m.size() == 4);
  assert(std::distance(m.begin(), m.end()) == 4);
  assert(std::distance(cm.begin(), cm.end()) == 4);
  assert(std::distance(m.cbegin(), m.cend()) == 4);
  typename M::iterator i;                   // default-construct
  i                            = m.begin(); // move-assignment
  typename M::const_iterator k = i;         // converting constructor
  assert(i == k);                           // comparison
  for (int j = 1; j <= 4; ++j, ++i) {       // pre-increment
    assert(*i == j);                        // operator*
  }
  assert(i == m.end());
  for (int j = 4; j >= 1; --j) {
    --i; // pre-decrement
    assert((*i) == j);
  }
  assert(i == m.begin());
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  {
    // N3644 testing
    using C = std::flat_set<int, char>;
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

  return 0;
}
