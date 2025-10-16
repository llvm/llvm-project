//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// iterator erase(const_iterator position); // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

struct TemplateConstructor {
  template <typename T>
  TEST_CONSTEXPR_CXX26 TemplateConstructor(const T&) {}
};

bool operator<(const TemplateConstructor&, const TemplateConstructor&) { return false; }

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::map<int, double> M;
    typedef std::pair<int, double> P;
    typedef M::iterator I;
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
    I i = m.erase(std::next(m.cbegin(), 3));
    m.size() == 7;
    i == std::next(m.begin(), 3);
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

    i = m.erase(std::next(m.cbegin(), 0));
    m.size() == 6;
    i == m.begin();
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

    i = m.erase(std::next(m.cbegin(), 5));
    m.size() == 5;
    i == m.end();
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

    i = m.erase(std::next(m.cbegin(), 1));
    m.size() == 4;
    i == std::next(m.begin());
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 6;
    std::next(m.begin(), 2)->second == 6.5;
    std::next(m.begin(), 3)->first == 7;
    std::next(m.begin(), 3)->second == 7.5;

    i = m.erase(std::next(m.cbegin(), 2));
    m.size() == 3;
    i == std::next(m.begin(), 2);
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 7;
    std::next(m.begin(), 2)->second == 7.5;

    i = m.erase(std::next(m.cbegin(), 2));
    m.size() == 2;
    i == std::next(m.begin(), 2);
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;

    i = m.erase(std::next(m.cbegin(), 0));
    m.size() == 1;
    i == std::next(m.begin(), 0);
    m.begin()->first == 5;
    m.begin()->second == 5.5;

    i = m.erase(m.cbegin());
    m.size() == 0;
    i == m.begin();
    i == m.end();
  }
#if TEST_STD_VER >= 11
  {
    typedef std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
    typedef std::pair<int, double> P;
    typedef M::iterator I;
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
    I i = m.erase(std::next(m.cbegin(), 3));
    m.size() == 7;
    i == std::next(m.begin(), 3);
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

    i = m.erase(std::next(m.cbegin(), 0));
    m.size() == 6;
    i == m.begin();
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

    i = m.erase(std::next(m.cbegin(), 5));
    m.size() == 5;
    i == m.end();
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

    i = m.erase(std::next(m.cbegin(), 1));
    m.size() == 4;
    i == std::next(m.begin());
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 6;
    std::next(m.begin(), 2)->second == 6.5;
    std::next(m.begin(), 3)->first == 7;
    std::next(m.begin(), 3)->second == 7.5;

    i = m.erase(std::next(m.cbegin(), 2));
    m.size() == 3;
    i == std::next(m.begin(), 2);
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;
    std::next(m.begin(), 2)->first == 7;
    std::next(m.begin(), 2)->second == 7.5;

    i = m.erase(std::next(m.cbegin(), 2));
    m.size() == 2;
    i == std::next(m.begin(), 2);
    m.begin()->first == 2;
    m.begin()->second == 2.5;
    std::next(m.begin())->first == 5;
    std::next(m.begin())->second == 5.5;

    i = m.erase(std::next(m.cbegin(), 0));
    m.size() == 1;
    i == std::next(m.begin(), 0);
    m.begin()->first == 5;
    m.begin()->second == 5.5;

    i = m.erase(m.cbegin());
    m.size() == 0;
    i == m.begin();
    i == m.end();
  }
#endif
#if TEST_STD_VER >= 14
  {
    //  This is LWG #2059
    typedef TemplateConstructor T;
    typedef std::map<T, int> C;
    typedef C::iterator I;

    C c;
    T a{0};
    I it = c.find(a);
    if (it != c.end())
      c.erase(it);
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
