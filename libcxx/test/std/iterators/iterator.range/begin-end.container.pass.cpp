//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>
//
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());                   // constexpr since C++17
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());             // constexpr since C++17
// template <class C> constexpr auto end(C& c) -> decltype(c.end());                       // constexpr since C++17
// template <class C> constexpr auto end(const C& c) -> decltype(c.end());                 // constexpr since C++17
//
// template <class C> constexpr auto cbegin(const C& c) -> decltype(std::begin(c));        // C++14
// template <class C> constexpr auto cend(const C& c) -> decltype(std::end(c));            // C++14
// template <class C> constexpr auto rbegin(C& c) -> decltype(c.rbegin());                 // C++14, constexpr since C++17
// template <class C> constexpr auto rbegin(const C& c) -> decltype(c.rbegin());           // C++14, constexpr since C++17
// template <class C> constexpr auto rend(C& c) -> decltype(c.rend());                     // C++14, constexpr since C++17
// template <class C> constexpr auto rend(const C& c) -> decltype(c.rend());               // C++14, constexpr since C++17
// template <class C> constexpr auto crbegin(const C& c) -> decltype(std::rbegin(c));      // C++14, constexpr since C++17
// template <class C> constexpr auto crend(const C& c) -> decltype(std::rend(c));          // C++14, constexpr since C++17

#include <array>
#include <cassert>
#include <iterator>
#include <list>
#include <vector>

#include "test_macros.h"

template <typename C,
          typename Iterator             = typename C::iterator,
          typename ConstIterator        = typename C::const_iterator,
          typename ReverseIterator      = typename C::reverse_iterator,
          typename ConstReverseIterator = typename C::const_reverse_iterator>
TEST_CONSTEXPR_CXX17 bool test() {
  C c         = {1, 2, 3};
  const C& cc = c;

  // std::begin(C& c) / std::end(C& c)
  {
    ASSERT_SAME_TYPE(decltype(std::begin(c)), Iterator);
    assert(std::begin(c) == c.begin());

    ASSERT_SAME_TYPE(decltype(std::end(c)), Iterator);
    assert(std::end(c) == c.end());
  }

  // std::begin(C const& c) / std::end(C const& c)
  {
    ASSERT_SAME_TYPE(decltype(std::begin(cc)), ConstIterator);
    assert(std::begin(cc) == cc.begin());

    ASSERT_SAME_TYPE(decltype(std::end(cc)), ConstIterator);
    assert(std::end(cc) == cc.end());
  }

#if TEST_STD_VER >= 14
  // std::cbegin(C const&) / std::cend(C const&)
  {
    ASSERT_SAME_TYPE(decltype(std::cbegin(cc)), ConstIterator);
    static_assert(noexcept(std::cbegin(cc)) == noexcept(std::begin(cc)), "");
    assert(std::cbegin(cc) == std::begin(cc));

    ASSERT_SAME_TYPE(decltype(std::cend(cc)), ConstIterator);
    static_assert(noexcept(std::cend(cc)) == noexcept(std::end(cc)), "");
    assert(std::cend(cc) == std::end(cc));

    // kind of overkill, but whatever
    ASSERT_SAME_TYPE(decltype(std::cbegin(c)), ConstIterator);
    static_assert(noexcept(std::cbegin(c)) == noexcept(std::begin(cc)), "");
    assert(std::cbegin(c) == std::begin(cc));

    ASSERT_SAME_TYPE(decltype(std::cend(c)), ConstIterator);
    static_assert(noexcept(std::cend(c)) == noexcept(std::end(cc)), "");
    assert(std::cend(c) == std::end(cc));
  }

  // std::rbegin(C& c) / std::rend(C& c)
  {
    ASSERT_SAME_TYPE(decltype(std::rbegin(c)), ReverseIterator);
    assert(std::rbegin(c) == c.rbegin());

    ASSERT_SAME_TYPE(decltype(std::rend(c)), ReverseIterator);
    assert(std::rend(c) == c.rend());
  }

  // std::rbegin(C const&) / std::rend(C const&)
  {
    ASSERT_SAME_TYPE(decltype(std::rbegin(cc)), ConstReverseIterator);
    assert(std::rbegin(cc) == cc.rbegin());

    ASSERT_SAME_TYPE(decltype(std::rend(cc)), ConstReverseIterator);
    assert(std::rend(cc) == cc.rend());
  }

  // std::crbegin(C const&) / std::crend(C const&)
  {
    ASSERT_SAME_TYPE(decltype(std::crbegin(cc)), ConstReverseIterator);
    assert(std::crbegin(cc) == std::rbegin(cc));

    ASSERT_SAME_TYPE(decltype(std::crend(cc)), ConstReverseIterator);
    assert(std::crend(cc) == std::rend(cc));

    // kind of overkill, but whatever
    ASSERT_SAME_TYPE(decltype(std::crbegin(c)), ConstReverseIterator);
    assert(std::crbegin(c) == std::rbegin(cc));

    ASSERT_SAME_TYPE(decltype(std::crend(c)), ConstReverseIterator);
    assert(std::crend(c) == std::rend(cc));
  }
#endif // TEST_STD_VER >= 14

  return true;
}

int main(int, char**) {
  test<std::array<int, 3>>();
  test<std::list<int>>();
  test<std::vector<int>>();

#if TEST_STD_VER >= 17
  static_assert(test<std::array<int, 3>>());
#endif

  return 0;
}
