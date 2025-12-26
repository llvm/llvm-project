//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>
// template <class C> constexpr auto begin(C& c) noexcept(see-below) -> decltype(c.begin());       // constexpr since C++17
// template <class C> constexpr auto begin(const C& c) noexcept(see-below) -> decltype(c.begin()); // constexpr since C++17
// template <class C> constexpr auto end(C& c) noexcept(see-below) -> decltype(c.end());           // constexpr since C++17
// template <class C> constexpr auto end(const C& c) noexcept(see-below) -> decltype(c.end());     // constexpr since C++17
//
// template <class C> constexpr auto cbegin(const C& c) noexcept(see-below) -> decltype(std::begin(c)); // C++14
// template <class C> constexpr auto cend(const C& c) noexcept(see-below) -> decltype(std::end(c));     // C++14
// template <class C>
// constexpr auto rbegin(C& c) noexcept(see-below) -> decltype(c.rbegin());            // C++14, constexpr since C++17
// template <class C>
// constexpr auto rbegin(const C& c) noexcept(see-below) -> decltype(c.rbegin());      // C++14, constexpr since C++17
// template <class C>
// constexpr auto rend(C& c) noexcept(see-below) -> decltype(c.rend());                // C++14, constexpr since C++17
// template <class C>
// constexpr auto rend(const C& c) noexcept(see-below) -> decltype(c.rend());          // C++14, constexpr since C++17
// template <class C>
// constexpr auto crbegin(const C& c) noexcept(see-below) -> decltype(std::rbegin(c)); // C++14, constexpr since C++17
// template <class C>
// constexpr auto crend(const C& c) noexcept(see-below) -> decltype(std::rend(c));     // C++14, constexpr since C++17

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
    static_assert(noexcept(std::begin(c)) == noexcept(c.begin()), "");
    assert(std::begin(c) == c.begin());

    ASSERT_SAME_TYPE(decltype(std::end(c)), Iterator);
    static_assert(noexcept(std::end(c)) == noexcept(c.end()), "");
    assert(std::end(c) == c.end());
  }

  // std::begin(C const& c) / std::end(C const& c)
  {
    ASSERT_SAME_TYPE(decltype(std::begin(cc)), ConstIterator);
    static_assert(noexcept(std::begin(cc)) == noexcept(cc.begin()), "");
    assert(std::begin(cc) == cc.begin());

    ASSERT_SAME_TYPE(decltype(std::end(cc)), ConstIterator);
    static_assert(noexcept(std::end(cc)) == noexcept(cc.end()), "");
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
    static_assert(noexcept(std::rbegin(c)) == noexcept(c.rbegin()), "");
    assert(std::rbegin(c) == c.rbegin());

    ASSERT_SAME_TYPE(decltype(std::rend(c)), ReverseIterator);
    static_assert(noexcept(std::rend(c)) == noexcept(c.rend()), "");
    assert(std::rend(c) == c.rend());
  }

  // std::rbegin(C const&) / std::rend(C const&)
  {
    ASSERT_SAME_TYPE(decltype(std::rbegin(cc)), ConstReverseIterator);
    static_assert(noexcept(std::rbegin(cc)) == noexcept(cc.rbegin()), "");
    assert(std::rbegin(cc) == cc.rbegin());

    ASSERT_SAME_TYPE(decltype(std::rend(cc)), ConstReverseIterator);
    static_assert(noexcept(std::rend(cc)) == noexcept(cc.rend()), "");
    assert(std::rend(cc) == cc.rend());
  }

  // std::crbegin(C const&) / std::crend(C const&)
  {
    ASSERT_SAME_TYPE(decltype(std::crbegin(cc)), ConstReverseIterator);
    static_assert(noexcept(std::crbegin(cc)) == noexcept(std::rbegin(cc)), "");
    assert(std::crbegin(cc) == std::rbegin(cc));

    ASSERT_SAME_TYPE(decltype(std::crend(cc)), ConstReverseIterator);
    static_assert(noexcept(std::crbegin(cc)) == noexcept(std::rend(cc)), "");
    assert(std::crend(cc) == std::rend(cc));

    // kind of overkill, but whatever
    ASSERT_SAME_TYPE(decltype(std::crbegin(c)), ConstReverseIterator);
    static_assert(noexcept(std::crbegin(c)) == noexcept(std::rbegin(cc)), "");
    assert(std::crbegin(c) == std::rbegin(cc));

    ASSERT_SAME_TYPE(decltype(std::crend(c)), ConstReverseIterator);
    static_assert(noexcept(std::crbegin(c)) == noexcept(std::rend(cc)), "");
    assert(std::crend(c) == std::rend(cc));
  }
#endif // TEST_STD_VER >= 14

  return true;
}

TEST_CONSTEXPR_CXX17 bool test_ilist() {
  // unqualified begin, etc.
  {
    std::initializer_list<int> il = {3, 2, 1};
    ASSERT_NOEXCEPT(begin(il));
    ASSERT_NOEXCEPT(end(il));
#if TEST_STD_VER >= 14
    ASSERT_NOEXCEPT(cbegin(il));
    ASSERT_NOEXCEPT(cend(il));
    ASSERT_NOEXCEPT(rbegin(il));
    ASSERT_NOEXCEPT(rend(il));
    ASSERT_NOEXCEPT(crbegin(il));
    ASSERT_NOEXCEPT(crend(il));
#endif
    ASSERT_SAME_TYPE(decltype(begin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(end(il)), const int*);
    const int* b = begin(il);
    const int* e = end(il);
    assert(il.size() == 3);
    assert(static_cast<std::size_t>(e - b) == il.size());
    assert(*b++ == 3);
    assert(*b++ == 2);
    assert(*b++ == 1);
#if TEST_STD_VER >= 14
    ASSERT_SAME_TYPE(decltype(cbegin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(cend(il)), const int*);
    ASSERT_SAME_TYPE(decltype(rbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(rend(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(crbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(crend(il)), std::reverse_iterator<const int*>);

    assert(cbegin(il) == begin(il));
    assert(end(il) == end(il));
    assert(rbegin(il) == std::reverse_iterator<const int*>(end(il)));
    assert(rend(il) == std::reverse_iterator<const int*>(begin(il)));
    assert(crbegin(il) == std::reverse_iterator<const int*>(end(il)));
    assert(crend(il) == std::reverse_iterator<const int*>(begin(il)));

    auto rb = rbegin(il);
    auto re = rend(il);
    assert(static_cast<std::size_t>(re - rb) == il.size());
    assert(*rb++ == 1);
    assert(*rb++ == 2);
    assert(*rb++ == 3);
#endif
  }

  // qualified begin, etc.
  {
    std::initializer_list<int> il = {1, 2, 3};
    ASSERT_NOEXCEPT(std::begin(il));
    ASSERT_NOEXCEPT(std::end(il));
#if TEST_STD_VER >= 14
    ASSERT_NOEXCEPT(std::cbegin(il));
    ASSERT_NOEXCEPT(std::cend(il));
    ASSERT_NOEXCEPT(std::rbegin(il));
    ASSERT_NOEXCEPT(std::rend(il));
    ASSERT_NOEXCEPT(std::crbegin(il));
    ASSERT_NOEXCEPT(std::crend(il));
#endif
    ASSERT_SAME_TYPE(decltype(std::begin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(il)), const int*);
    assert(std::begin(il) == il.begin());
    assert(std::end(il) == il.end());
#if TEST_STD_VER >= 14
    ASSERT_SAME_TYPE(decltype(std::cbegin(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(il)), const int*);
    ASSERT_SAME_TYPE(decltype(std::rbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crbegin(il)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(il)), std::reverse_iterator<const int*>);

    assert(std::cbegin(il) == std::begin(il));
    assert(std::end(il) == std::end(il));
    assert(std::rbegin(il) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::rend(il) == std::reverse_iterator<const int*>(std::begin(il)));
    assert(std::crbegin(il) == std::reverse_iterator<const int*>(std::end(il)));
    assert(std::crend(il) == std::reverse_iterator<const int*>(std::begin(il)));

    {
      auto rb = std::rbegin(il);
      auto re = std::rend(il);
      assert(static_cast<std::size_t>(re - rb) == il.size());
      assert(*rb++ == 3);
      assert(*rb++ == 2);
      assert(*rb++ == 1);
    }
#endif

    const auto& cil = il;
    ASSERT_NOEXCEPT(std::begin(cil));
    ASSERT_NOEXCEPT(std::end(cil));
#if TEST_STD_VER >= 14
    ASSERT_NOEXCEPT(std::cbegin(cil));
    ASSERT_NOEXCEPT(std::cend(cil));
    ASSERT_NOEXCEPT(std::rbegin(cil));
    ASSERT_NOEXCEPT(std::rend(cil));
    ASSERT_NOEXCEPT(std::crbegin(cil));
    ASSERT_NOEXCEPT(std::crend(cil));
#endif
    ASSERT_SAME_TYPE(decltype(std::begin(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::end(cil)), const int*);
    assert(std::begin(cil) == il.begin());
    assert(std::end(cil) == il.end());
#if TEST_STD_VER >= 14
    ASSERT_SAME_TYPE(decltype(std::cbegin(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::cend(cil)), const int*);
    ASSERT_SAME_TYPE(decltype(std::rbegin(cil)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::rend(cil)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crbegin(cil)), std::reverse_iterator<const int*>);
    ASSERT_SAME_TYPE(decltype(std::crend(cil)), std::reverse_iterator<const int*>);

    assert(std::cbegin(cil) == std::begin(cil));
    assert(std::end(cil) == std::end(cil));
    assert(std::rbegin(cil) == std::reverse_iterator<const int*>(std::end(cil)));
    assert(std::rend(cil) == std::reverse_iterator<const int*>(std::begin(cil)));
    assert(std::crbegin(cil) == std::reverse_iterator<const int*>(std::end(cil)));
    assert(std::crend(cil) == std::reverse_iterator<const int*>(std::begin(cil)));

    {
      auto rb = std::rbegin(cil);
      auto re = std::rend(cil);
      assert(static_cast<std::size_t>(re - rb) == cil.size());
      assert(*rb++ == 3);
      assert(*rb++ == 2);
      assert(*rb++ == 1);
    }
#endif
  }

  return true;
}

int main(int, char**) {
  test<std::array<int, 3>>();
  test<std::list<int>>();
  test<std::vector<int>>();

  test_ilist();

#if TEST_STD_VER >= 17
  static_assert(test<std::array<int, 3>>());
  static_assert(test_ilist());
#endif

  // Note: Properly testing the conditional noexcept-ness propagation in std::cbegin and std::cend
  //       requires using C-style arrays, because those are the only ones with a noexcept std::begin
  //       and std::end inside namespace std
#if TEST_STD_VER >= 14
  {
    int a[]        = {1, 2, 3};
    auto const& ca = a;
    ASSERT_NOEXCEPT(std::cbegin(ca));
    ASSERT_NOEXCEPT(std::cend(ca));
    ASSERT_NOEXCEPT(std::crbegin(ca));
    ASSERT_NOEXCEPT(std::crend(ca));

    // kind of overkill, but whatever
    ASSERT_NOEXCEPT(std::cbegin(a));
    ASSERT_NOEXCEPT(std::cend(a));
    ASSERT_NOEXCEPT(std::crbegin(a));
    ASSERT_NOEXCEPT(std::crend(a));
  }

  // Make sure std::cbegin and std::cend are constexpr in C++14 too (see LWG2280).
  {
    static constexpr int a[] = {1, 2, 3};
    constexpr auto b         = std::cbegin(a);
    assert(b == a);
    constexpr auto e = std::cend(a);
    assert(e == a + 3);
  }
#endif

  return 0;
}
