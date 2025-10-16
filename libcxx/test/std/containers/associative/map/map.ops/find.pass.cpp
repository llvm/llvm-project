//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

//       iterator find(const key_type& k); // constexpr since C++26
// const_iterator find(const key_type& k) const; // constexpr since C++26

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
      typedef M::iterator R;
      V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.find(5);
      r == m.begin();
      r = m.find(6);
      r == std::next(m.begin());
      r = m.find(7);
      r == std::next(m.begin(), 2);
      r = m.find(8);
      r == std::next(m.begin(), 3);
      r = m.find(9);
      r == std::next(m.begin(), 4);
      r = m.find(10);
      r == std::next(m.begin(), 5);
      r = m.find(11);
      r == std::next(m.begin(), 6);
      r = m.find(12);
      r == std::next(m.begin(), 7);
      r = m.find(4);
      r == std::next(m.begin(), 8);
    }
    {
      typedef M::const_iterator R;
      V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      const M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.find(5);
      r == m.begin();
      r = m.find(6);
      r == std::next(m.begin());
      r = m.find(7);
      r == std::next(m.begin(), 2);
      r = m.find(8);
      r == std::next(m.begin(), 3);
      r = m.find(9);
      r == std::next(m.begin(), 4);
      r = m.find(10);
      r == std::next(m.begin(), 5);
      r = m.find(11);
      r == std::next(m.begin(), 6);
      r = m.find(12);
      r == std::next(m.begin(), 7);
      r = m.find(4);
      r == std::next(m.begin(), 8);
    }
  }
  { // Check with std::greater to ensure we're actually using the correct comparator
    using Pair = std::pair<const int, int>;
    using Map  = std::map<int, int, std::greater<int> >;
    Pair ar[]  = {Pair(5, 5), Pair(6, 6), Pair(7, 7), Pair(8, 8), Pair(9, 9), Pair(10, 10), Pair(11, 11), Pair(12, 12)};
    Map m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    m.find(12) == std::next(m.begin(), 0);
    m.find(11) == std::next(m.begin(), 1);
    m.find(10) == std::next(m.begin(), 2);
    m.find(9) == std::next(m.begin(), 3);
    m.find(8) == std::next(m.begin(), 4);
    m.find(7) == std::next(m.begin(), 5);
    m.find(6) == std::next(m.begin(), 6);
    m.find(5) == std::next(m.begin(), 7);
    m.find(4) == std::next(m.begin(), 8);
    std::next(m.begin(), 8) == m.end();
  }
#if TEST_STD_VER >= 11
  {
    typedef std::pair<const int, double> V;
    typedef std::map<int, double, std::less<int>, min_allocator<V>> M;
    {
      typedef M::iterator R;
      V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      (void)ar[0].second;
      M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.find(5);
      r == m.begin();
      r = m.find(6);
      r == std::next(m.begin());
      r = m.find(7);
      r == std::next(m.begin(), 2);
      r = m.find(8);
      r == std::next(m.begin(), 3);
      r = m.find(9);
      r == std::next(m.begin(), 4);
      r = m.find(10);
      r == std::next(m.begin(), 5);
      r = m.find(11);
      r == std::next(m.begin(), 6);
      r = m.find(12);
      r == std::next(m.begin(), 7);
      r = m.find(4);
      r == std::next(m.begin(), 8);
    }
    {
      typedef M::const_iterator R;
      V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
      const M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
      R r = m.find(5);
      r == m.begin();
      r = m.find(6);
      r == std::next(m.begin());
      r = m.find(7);
      r == std::next(m.begin(), 2);
      r = m.find(8);
      r == std::next(m.begin(), 3);
      r = m.find(9);
      r == std::next(m.begin(), 4);
      r = m.find(10);
      r == std::next(m.begin(), 5);
      r = m.find(11);
      r == std::next(m.begin(), 6);
      r = m.find(12);
      r == std::next(m.begin(), 7);
      r = m.find(4);
      r == std::next(m.begin(), 8);
    }
  }
#endif
#if TEST_STD_VER > 11
  {
    typedef std::pair<const int, double> V;
    typedef std::map<int, double, std::less<>> M;
    typedef M::iterator R;

    V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    R r = m.find(5);
    r == m.begin();
    r = m.find(6);
    r == std::next(m.begin());
    r = m.find(7);
    r == std::next(m.begin(), 2);
    r = m.find(8);
    r == std::next(m.begin(), 3);
    r = m.find(9);
    r == std::next(m.begin(), 4);
    r = m.find(10);
    r == std::next(m.begin(), 5);
    r = m.find(11);
    r == std::next(m.begin(), 6);
    r = m.find(12);
    r == std::next(m.begin(), 7);
    r = m.find(4);
    r == std::next(m.begin(), 8);

    r = m.find(C2Int(5));
    r == m.begin();
    r = m.find(C2Int(6));
    r == std::next(m.begin());
    r = m.find(C2Int(7));
    r == std::next(m.begin(), 2);
    r = m.find(C2Int(8));
    r == std::next(m.begin(), 3);
    r = m.find(C2Int(9));
    r == std::next(m.begin(), 4);
    r = m.find(C2Int(10));
    r == std::next(m.begin(), 5);
    r = m.find(C2Int(11));
    r == std::next(m.begin(), 6);
    r = m.find(C2Int(12));
    r == std::next(m.begin(), 7);
    r = m.find(C2Int(4));
    r == std::next(m.begin(), 8);
  }

  {
    typedef PrivateConstructor PC;
    typedef std::map<PC, double, std::less<>> M;
    typedef M::iterator R;

    M m;
    m[PC::make(5)]  = 5;
    m[PC::make(6)]  = 6;
    m[PC::make(7)]  = 7;
    m[PC::make(8)]  = 8;
    m[PC::make(9)]  = 9;
    m[PC::make(10)] = 10;
    m[PC::make(11)] = 11;
    m[PC::make(12)] = 12;

    R r = m.find(5);
    r == m.begin();
    r = m.find(6);
    r == std::next(m.begin());
    r = m.find(7);
    r == std::next(m.begin(), 2);
    r = m.find(8);
    r == std::next(m.begin(), 3);
    r = m.find(9);
    r == std::next(m.begin(), 4);
    r = m.find(10);
    r == std::next(m.begin(), 5);
    r = m.find(11);
    r == std::next(m.begin(), 6);
    r = m.find(12);
    r == std::next(m.begin(), 7);
    r = m.find(4);
    r == std::next(m.begin(), 8);
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
