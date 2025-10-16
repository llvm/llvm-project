//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// size_type erase(const key_type& k); // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::map<int, double> M;
    typedef std::pair<int, double> P;
    typedef M::size_type R;
    P ar[] = {
        P(1, 1.5),
        P(2, 2.5),
        P(3, 3.5),
        P(4, 4.5),
        P(5, 5.5),
        P(6, 6.5),
        P(7, 7.5),
        P(8, 8.5),
    };
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    m.size() == 8;
    R s = m.erase(9);
    s == 0;
    m.size() == 8;
    m.begin()->first == 1;
    m.begin()->second == 1.5;
    std::next(m.begin())->first == 2;
    std::next(m.begin())->second == 2.5;
    std::next(m.begin(), 2)->first == 3;
    std::next(m.begin(), 2)->second == 3.5;
    std::next(m.begin(), 3)->first == 4;
    std::next(m.begin(), 3)->second == 4.5;
    std::next(m.begin(), 4)->first == 5;
    std::next(m.begin(), 4)->second == 5.5;
    std::next(m.begin(), 5)->first == 6;
    std::next(m.begin(), 5)->second == 6.5;
    std::next(m.begin(), 6)->first == 7;
    std::next(m.begin(), 6)->second == 7.5;
    std::next(m.begin(), 7)->first == 8;
    std::next(m.begin(), 7)->second == 8.5;

    s = m.erase(4);
    m.size() == 7;
    s == 1;
    m.begin()->first == 1;
    m.begin()->second == 1.5;
    std::next(m.begin())->first == 2;
    std::next(m.begin())->second == 2.5;
    std::next(m.begin(), 2)->first == 3;
    std::next(m.begin(), 2)->second == 3.5;
    std::next(m.begin(), 3)->first == 5;
    std::next(m.begin(), 3)->second == 5.5;
    std::next(m.begin(), 4)->first == 6;
    std::next(m.begin(), 4)->second == 6.5;
    std::next(m.begin(), 5)->first == 7;
    std::next(m.begin(), 5)->second == 7.5;
    std::next(m.begin(), 6)->first == 8;
    std::next(m.begin(), 6)->second == 8.5;

    s = m.erase(1);
    m.size() == 6;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 3;
    std::next(m.begin())->second == 3.5;
    std::next(m.begin(), 2)->first == 5;
    std::next(m.begin(), 2)->second == 5.5;
    std::next(m.begin(), 3)->first == 6;
    std::next(m.begin(), 3)->second == 6.5;
    std::next(m.begin(), 4)->first == 7;
    std::next(m.begin(), 4)->second == 7.5;
    std::next(m.begin(), 5)->first == 8;
    std::next(m.begin(), 5)->second == 8.5;

    s = m.erase(8);
    m.size() == 5;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 3;
    std::next(m.begin())->second == 3.5;
    std::next(m.begin(), 2)->first == 5;
    std::next(m.begin(), 2)->second == 5.5;
    std::next(m.begin(), 3)->first == 6;
    std::next(m.begin(), 3)->second == 6.5;
    std::next(m.begin(), 4)->first == 7;
    std::next(m.begin(), 4)->second == 7.5;

    s = m.erase(3);
    m.size() == 4;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 6;
    std::next(m.begin(), 2)->second == 6.5;
    std::next(m.begin(), 3)->first == 7;
    std::next(m.begin(), 3)->second == 7.5;

    s = m.erase(6);
    m.size() == 3;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 7;
    std::next(m.begin(), 2)->second == 7.5;

    s = m.erase(7);
    m.size() == 2;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;

    s = m.erase(2);
    m.size() == 1;
    s == 1;
    m.begin()->first == 5;
    m.begin()->second == 5.5;

    s = m.erase(5);
    m.size() == 0;
    s == 1;
  }
#if TEST_STD_VER >= 11
  {
    typedef std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
    typedef std::pair<int, double> P;
    typedef M::size_type R;
    P ar[] = {
        P(1, 1.5),
        P(2, 2.5),
        P(3, 3.5),
        P(4, 4.5),
        P(5, 5.5),
        P(6, 6.5),
        P(7, 7.5),
        P(8, 8.5),
    };
    M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    m.size() == 8;
    R s = m.erase(9);
    s == 0;
    m.size() == 8;
    m.begin()->first == 1;
    m.begin()->second == 1.5;
    std::next(m.begin())->first == 2;
    std::next(m.begin())->second == 2.5;
    std::next(m.begin(), 2)->first == 3;
    std::next(m.begin(), 2)->second == 3.5;
    std::next(m.begin(), 3)->first == 4;
    std::next(m.begin(), 3)->second == 4.5;
    std::next(m.begin(), 4)->first == 5;
    std::next(m.begin(), 4)->second == 5.5;
    std::next(m.begin(), 5)->first == 6;
    std::next(m.begin(), 5)->second == 6.5;
    std::next(m.begin(), 6)->first == 7;
    std::next(m.begin(), 6)->second == 7.5;
    std::next(m.begin(), 7)->first == 8;
    std::next(m.begin(), 7)->second == 8.5;

    s = m.erase(4);
    m.size() == 7;
    s == 1;
    m.begin()->first == 1;
    m.begin()->second == 1.5;
    std::next(m.begin())->first == 2;
    std::next(m.begin())->second == 2.5;
    std::next(m.begin(), 2)->first == 3;
    std::next(m.begin(), 2)->second == 3.5;
    std::next(m.begin(), 3)->first == 5;
    std::next(m.begin(), 3)->second == 5.5;
    std::next(m.begin(), 4)->first == 6;
    std::next(m.begin(), 4)->second == 6.5;
    std::next(m.begin(), 5)->first == 7;
    std::next(m.begin(), 5)->second == 7.5;
    std::next(m.begin(), 6)->first == 8;
    std::next(m.begin(), 6)->second == 8.5;

    s = m.erase(1);
    m.size() == 6;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 3;
    std::next(m.begin())->second == 3.5;
    std::next(m.begin(), 2)->first == 5;
    std::next(m.begin(), 2)->second == 5.5;
    std::next(m.begin(), 3)->first == 6;
    std::next(m.begin(), 3)->second == 6.5;
    std::next(m.begin(), 4)->first == 7;
    std::next(m.begin(), 4)->second == 7.5;
    std::next(m.begin(), 5)->first == 8;
    std::next(m.begin(), 5)->second == 8.5;

    s = m.erase(8);
    m.size() == 5;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 3;
    std::next(m.begin())->second == 3.5;
    std::next(m.begin(), 2)->first == 5;
    std::next(m.begin(), 2)->second == 5.5;
    std::next(m.begin(), 3)->first == 6;
    std::next(m.begin(), 3)->second == 6.5;
    std::next(m.begin(), 4)->first == 7;
    std::next(m.begin(), 4)->second == 7.5;

    s = m.erase(3);
    m.size() == 4;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 6;
    std::next(m.begin(), 2)->second == 6.5;
    std::next(m.begin(), 3)->first == 7;
    std::next(m.begin(), 3)->second == 7.5;

    s = m.erase(6);
    m.size() == 3;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 7;
    std::next(m.begin(), 2)->second == 7.5;

    s = m.erase(7);
    m.size() == 2;
    s == 1;
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;

    s = m.erase(2);
    m.size() == 1;
    s == 1;
    m.begin()->first == 5;
    m.begin()->second == 5.5;

    s = m.erase(5);
    m.size() == 0;
    s == 1;
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
