//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that that the iterator types of the associative containers is initialized even when default initialized.
// This is an extension because the standard only requires initialization when an iterator is value initialized, and
// that only since C++14. We guarantee that iterators are always initialized, even in C++11.

#include <cassert>
#include <map>
#include <set>

template <class Iter>
void test() {
  {
    Iter iter1;
    Iter iter2;
    assert(iter1 == iter2);
  }
  {
    Iter iter1;
    Iter iter2{};
    assert(iter1 == iter2);
  }
  {
    Iter iter1{};
    Iter iter2;
    assert(iter1 == iter2);
  }
}

template <class Container>
void test_container() {
  test<typename Container::iterator>();
  test<typename Container::const_iterator>();
  test<typename Container::reverse_iterator>();
  test<typename Container::const_reverse_iterator>();
}

int main(int, char**) {
  test_container<std::map<int, int>>();
  test_container<std::multimap<int, int>>();
  test_container<std::set<int>>();
  test_container<std::multiset<int>>();

  return 0;
}
