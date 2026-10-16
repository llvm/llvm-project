//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

//       reverse_iterator rbegin() noexcept;
// const_reverse_iterator rbegin() const noexcept;
//       reverse_iterator rend()   noexcept;
// const_reverse_iterator rend()   const noexcept;
//
// const_reverse_iterator crbegin() const noexcept;
// const_reverse_iterator crend()   const noexcept;

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>

#include <iterator>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
constexpr void test_one() {
  using Key      = typename KeyContainer::value_type;
  using M        = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  M m            = {1, 1, 2, 2, 3, 4};
  int expected[] = {1, 1, 2, 2, 3, 4};
  const M& cm    = m;
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
  assert(m.size() == 6);
  assert(std::distance(m.rbegin(), m.rend()) == 6);
  assert(std::distance(cm.rbegin(), cm.rend()) == 6);
  assert(std::distance(m.crbegin(), m.crend()) == 6);
  assert(std::distance(cm.crbegin(), cm.crend()) == 6);
  typename M::reverse_iterator i; // default-construct
  ASSERT_SAME_TYPE(decltype(*i), const int&);
  i                                    = m.rbegin(); // move-assignment
  typename M::const_reverse_iterator k = i;          // converting constructor
  assert(i == k);                                    // comparison
  for (int j = 5; j >= 0; --j, ++i) {                // pre-increment
    assert(*i == expected[j]);
  }
  assert(i == m.rend());
  for (int j = 0; j <= 5; ++j) {
    --i; // pre-decrement
    assert(*i == expected[j]);
  }
  assert(i == m.rbegin());
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  {
    // N3644 testing
    using C = std::flat_multiset<int>;
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
