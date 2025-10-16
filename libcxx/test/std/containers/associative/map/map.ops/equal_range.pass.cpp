//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// pair<iterator,iterator>             equal_range(const key_type& k); // constexpr since C++26
// pair<const_iterator,const_iterator> equal_range(const key_type& k) const; // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "private_constructor.h"
#include "is_transparent.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::pair<const int, double> V;
    typedef std::map<int, double> M;
    {
      typedef std::pair<M::iterator, M::iterator> R;
      V ar[] = {V(5, 5), V(7, 6), V(9, 7), V(11, 8), V(13, 9), V(15, 10), V(17, 11), V(19, 12)};
      M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.equal_range(5);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(7);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(9);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(11);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(13);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(15);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(17);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(19);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 8);
      r = m.equal_range(4);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 0);
      r = m.equal_range(6);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(8);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(10);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(12);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(14);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(16);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(18);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(20);
      r.first == std::next(m.begin(), 8);
      r.second == std::next(m.begin(), 8);
    }
    {
      typedef std::pair<M::const_iterator, M::const_iterator> R;
      V ar[] = {V(5, 5), V(7, 6), V(9, 7), V(11, 8), V(13, 9), V(15, 10), V(17, 11), V(19, 12)};
      const M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.equal_range(5);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(7);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(9);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(11);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(13);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(15);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(17);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(19);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 8);
      r = m.equal_range(4);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 0);
      r = m.equal_range(6);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(8);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(10);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(12);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(14);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(16);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(18);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(20);
      r.first == std::next(m.begin(), 8);
      r.second == std::next(m.begin(), 8);
    }
  }
#if TEST_STD_VER >= 11
  {
    typedef std::pair<const int, double> V;
    typedef std::map<int, double, std::less<int>, min_allocator<V>> M;
    {
      typedef std::pair<M::iterator, M::iterator> R;
      V ar[] = {V(5, 5), V(7, 6), V(9, 7), V(11, 8), V(13, 9), V(15, 10), V(17, 11), V(19, 12)};
      M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.equal_range(5);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(7);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(9);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(11);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(13);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(15);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(17);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(19);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 8);
      r = m.equal_range(4);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 0);
      r = m.equal_range(6);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(8);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(10);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(12);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(14);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(16);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(18);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(20);
      r.first == std::next(m.begin(), 8);
      r.second == std::next(m.begin(), 8);
    }
    {
      typedef std::pair<M::const_iterator, M::const_iterator> R;
      V ar[] = {V(5, 5), V(7, 6), V(9, 7), V(11, 8), V(13, 9), V(15, 10), V(17, 11), V(19, 12)};
      const M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.equal_range(5);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(7);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(9);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(11);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(13);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(15);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(17);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(19);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 8);
      r = m.equal_range(4);
      r.first == std::next(m.begin(), 0);
      r.second == std::next(m.begin(), 0);
      r = m.equal_range(6);
      r.first == std::next(m.begin(), 1);
      r.second == std::next(m.begin(), 1);
      r = m.equal_range(8);
      r.first == std::next(m.begin(), 2);
      r.second == std::next(m.begin(), 2);
      r = m.equal_range(10);
      r.first == std::next(m.begin(), 3);
      r.second == std::next(m.begin(), 3);
      r = m.equal_range(12);
      r.first == std::next(m.begin(), 4);
      r.second == std::next(m.begin(), 4);
      r = m.equal_range(14);
      r.first == std::next(m.begin(), 5);
      r.second == std::next(m.begin(), 5);
      r = m.equal_range(16);
      r.first == std::next(m.begin(), 6);
      r.second == std::next(m.begin(), 6);
      r = m.equal_range(18);
      r.first == std::next(m.begin(), 7);
      r.second == std::next(m.begin(), 7);
      r = m.equal_range(20);
      r.first == std::next(m.begin(), 8);
      r.second == std::next(m.begin(), 8);
    }
  }
#endif
#if TEST_STD_VER > 11
  {
    typedef std::pair<const int, double> V;
    typedef std::map<int, double, std::less<>> M;
    typedef std::pair<M::iterator, M::iterator> R;

    V ar[] = {V(5, 5), V(7, 6), V(9, 7), V(11, 8), V(13, 9), V(15, 10), V(17, 11), V(19, 12)};
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    R r = m.equal_range(5);
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(7);
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(9);
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(11);
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(13);
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(15);
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(17);
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(19);
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 8);
    r = m.equal_range(4);
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 0);
    r = m.equal_range(6);
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(8);
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(10);
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(12);
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(14);
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(16);
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(18);
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(20);
    r.first == std::next(m.begin(), 8);
    r.second == std::next(m.begin(), 8);

    r = m.equal_range(C2Int(5));
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(C2Int(7));
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(C2Int(9));
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(C2Int(11));
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(C2Int(13));
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(C2Int(15));
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(C2Int(17));
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(C2Int(19));
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 8);
    r = m.equal_range(C2Int(4));
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 0);
    r = m.equal_range(C2Int(6));
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(C2Int(8));
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(C2Int(10));
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(C2Int(12));
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(C2Int(14));
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(C2Int(16));
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(C2Int(18));
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(C2Int(20));
    r.first == std::next(m.begin(), 8);
    r.second == std::next(m.begin(), 8);
  }
  {
    typedef PrivateConstructor PC;
    typedef std::map<PC, double, std::less<>> M;
    typedef std::pair<M::iterator, M::iterator> R;

    M m;
    m[PC::make(5)]  = 5;
    m[PC::make(7)]  = 6;
    m[PC::make(9)]  = 7;
    m[PC::make(11)] = 8;
    m[PC::make(13)] = 9;
    m[PC::make(15)] = 10;
    m[PC::make(17)] = 11;
    m[PC::make(19)] = 12;

    R r = m.equal_range(5);
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(7);
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(9);
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(11);
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(13);
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(15);
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(17);
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(19);
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 8);
    r = m.equal_range(4);
    r.first == std::next(m.begin(), 0);
    r.second == std::next(m.begin(), 0);
    r = m.equal_range(6);
    r.first == std::next(m.begin(), 1);
    r.second == std::next(m.begin(), 1);
    r = m.equal_range(8);
    r.first == std::next(m.begin(), 2);
    r.second == std::next(m.begin(), 2);
    r = m.equal_range(10);
    r.first == std::next(m.begin(), 3);
    r.second == std::next(m.begin(), 3);
    r = m.equal_range(12);
    r.first == std::next(m.begin(), 4);
    r.second == std::next(m.begin(), 4);
    r = m.equal_range(14);
    r.first == std::next(m.begin(), 5);
    r.second == std::next(m.begin(), 5);
    r = m.equal_range(16);
    r.first == std::next(m.begin(), 6);
    r.second == std::next(m.begin(), 6);
    r = m.equal_range(18);
    r.first == std::next(m.begin(), 7);
    r.second == std::next(m.begin(), 7);
    r = m.equal_range(20);
    r.first == std::next(m.begin(), 8);
    r.second == std::next(m.begin(), 8);
  }
#endif
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
