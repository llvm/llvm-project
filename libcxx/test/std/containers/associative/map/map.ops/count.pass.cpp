//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// size_type count(const key_type& k) const;

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "private_constructor.h"
#include "is_transparent.h"

#if TEST_STD_VER >= 11
template <class T>
struct FinalCompare final {
  bool operator()(const T& x, const T& y) const { return x < y; }
};
#endif

template <class Map, class ArgType = typename Map::key_type>
void test() {
  typedef typename Map::value_type V;
  typedef typename Map::size_type R;

  V ar[] = {V(5, 5), V(6, 6), V(7, 7), V(8, 8), V(9, 9), V(10, 10), V(11, 11), V(12, 12)};

  const Map m(ar, ar + sizeof(ar) / sizeof(ar[0]));

  for (int i = 0; i < 5; ++i) {
    R r = m.count(ArgType(i));
    assert(r == 0);
  }

  for (int i = 5; i < 13; ++i) {
    R r = m.count(ArgType(i));
    assert(r == 1);
  }
}

int main(int, char**) {
  test<std::map<int, double> >();
#if TEST_STD_VER >= 11
  typedef std::pair<const int, double> V;
  test<std::map<int, double, std::less<int>, min_allocator<V>>>();
  test<std::map<int, double, FinalCompare<int>>>();
#endif
#if TEST_STD_VER >= 14
  typedef std::map<int, double, std::less<>> TM;
  test<TM>();
  test<TM, C2Int>();

  {
    typedef PrivateConstructor PC;
    typedef std::map<PC, double, std::less<> > M;
    typedef M::size_type R;

    M m;
    m[PC::make(5)]  = 5;
    m[PC::make(6)]  = 6;
    m[PC::make(7)]  = 7;
    m[PC::make(8)]  = 8;
    m[PC::make(9)]  = 9;
    m[PC::make(10)] = 10;
    m[PC::make(11)] = 11;
    m[PC::make(12)] = 12;

    for (int i = 0; i < 5; ++i) {
      R r = m.count(i);
      assert(r == 0);
    }

    for (int i = 5; i < 13; ++i) {
      R r = m.count(i);
      assert(r == 1);
    }
  }
#endif // TEST_STD_VER >= 14
  return 0;
}
