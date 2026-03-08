//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// map& operator=(map&& m); // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "CopyConstructible.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <typename TKeyType >
TEST_CONSTEXPR_CXX26 bool test_move_assign() {
  {
    typedef std::pair<TKeyType, MoveOnly> V;
    typedef std::pair<const TKeyType, MoveOnly> VC;
    typedef test_less<TKeyType> C;
    typedef test_allocator<VC> A;
    typedef std::map<TKeyType, MoveOnly, C, A> M;
    typedef std::move_iterator<V*> I;
    V a1[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m1(I(a1), I(a1 + sizeof(a1) / sizeof(a1[0])), C(5), A(7));
    V a2[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m2(I(a2), I(a2 + sizeof(a2) / sizeof(a2[0])), C(5), A(7));
    M m3(C(3), A(7));
    m3 = std::move(m1);
    assert(m3 == m2);
    assert(m3.get_allocator() == A(7));
    assert(m3.key_comp() == C(5));
    assert(m1.empty());
  }
  {
    typedef std::pair<TKeyType, MoveOnly> V;
    typedef std::pair<const TKeyType, MoveOnly> VC;
    typedef test_less<TKeyType> C;
    typedef test_allocator<VC> A;
    typedef std::map<TKeyType, MoveOnly, C, A> M;
    typedef std::move_iterator<V*> I;
    V a1[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m1(I(a1), I(a1 + sizeof(a1) / sizeof(a1[0])), C(5), A(7));
    V a2[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m2(I(a2), I(a2 + sizeof(a2) / sizeof(a2[0])), C(5), A(7));
    M m3(C(3), A(5));
    m3 = std::move(m1);
    assert(m3 == m2);
    assert(m3.get_allocator() == A(5));
    assert(m3.key_comp() == C(5));
    LIBCPP_ASSERT(m1.empty());
  }
  {
    typedef std::pair<TKeyType, MoveOnly> V;
    typedef std::pair<const TKeyType, MoveOnly> VC;
    typedef test_less<TKeyType> C;
    typedef other_allocator<VC> A;
    typedef std::map<TKeyType, MoveOnly, C, A> M;
    typedef std::move_iterator<V*> I;
    V a1[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m1(I(a1), I(a1 + sizeof(a1) / sizeof(a1[0])), C(5), A(7));
    V a2[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m2(I(a2), I(a2 + sizeof(a2) / sizeof(a2[0])), C(5), A(7));
    M m3(C(3), A(5));
    m3 = std::move(m1);
    assert(m3 == m2);
    assert(m3.get_allocator() == A(7));
    assert(m3.key_comp() == C(5));
    assert(m1.empty());
  }
  {
    typedef std::pair<TKeyType, MoveOnly> V;
    typedef std::pair<const TKeyType, MoveOnly> VC;
    typedef test_less<TKeyType> C;
    typedef min_allocator<VC> A;
    typedef std::map<TKeyType, MoveOnly, C, A> M;
    typedef std::move_iterator<V*> I;
    V a1[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m1(I(a1), I(a1 + sizeof(a1) / sizeof(a1[0])), C(5), A());
    V a2[] = {V(1, 1), V(1, 2), V(1, 3), V(2, 1), V(2, 2), V(2, 3), V(3, 1), V(3, 2), V(3, 3)};
    M m2(I(a2), I(a2 + sizeof(a2) / sizeof(a2[0])), C(5), A());
    M m3(C(3), A());
    m3 = std::move(m1);
    assert(m3 == m2);
    assert(m3.get_allocator() == A());
    assert(m3.key_comp() == C(5));
    assert(m1.empty());
  }
  return true;
}

int main(int, char**) {
  test_move_assign<MoveOnly>();
#if TEST_STD_VER >= 26
  // FIXME: Within __tree, it is not allowed to move from a `const MoveOnly` which prevents this from executing during constant evaluation
// static_assert(test_move_assign<MoveOnly>());
#endif

  test_move_assign<CopyConstructible>();

#if TEST_STD_VER >= 26
  static_assert(test_move_assign<CopyConstructible>());
#endif

  return 0;
}
